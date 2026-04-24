"""
Evaluation runner for TAI-SlideQA benchmarks.

Public API
----------
evaluate_split(
    jsonl_path, retrieve_fn, variant, course_code, k=5, predict_fn=None
) -> EvalResult

run_all_variants(
    jsonl_path, retrieve_fn_factory, course_code, k=5, predict_fn_factory=None
) -> dict[str, EvalResult]

Schema bridge
-------------
Benchmark JSONL uses:
    "question_text"  — the question string sent to retrieve_fn
    "answer"         — gold answer string used by contains_answer
    "gold_page_ids"  — list of 0-based MinerU page_idx values

eval.py functions expect:
    q["question"]    → we map question_text → question internally
    q["answer_short"] → we map answer → answer_short internally

Neither the benchmark file nor the retriever is mutated.

predict_fn signature (optional)
--------------------------------
predict_fn(question_text: str, retrieved_pages: list) -> str

If predict_fn is None, contains_answer_rate is not computed
and EvalResult.contains_answer_rate is set to None.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from slideqa.eval import contains_answer, hits_at_k, mrr as _mrr, recall_at_k as _recall_at_k

logger = logging.getLogger(__name__)

_VARIANTS = ("v1", "v2", "v3")


# ---------------------------------------------------------------------------
# Result dataclass (immutable)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EvalResult:
    """Immutable evaluation result for one variant over one JSONL split."""

    variant: str
    n_questions: int
    recall_at_k: float
    mrr: float
    contains_answer_rate: Optional[float]   # None when predict_fn not provided
    k: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_variant_questions(jsonl_path: Path, variant: str) -> list[dict[str, Any]]:
    """Read jsonl_path and return records matching the given variant.

    Malformed and whitespace-only lines are silently skipped.
    """
    questions: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("run_eval: malformed JSON line in %s — skipping", jsonl_path.name)
                continue
            if obj.get("variant") == variant:
                questions.append(obj)
    return questions


def _make_eval_retrieve_fn(
    retrieve_fn: Callable,
    variant: str,
    course_code: Optional[str],
) -> Callable:
    """Wrap retrieve_fn into the signature expected by eval.py functions.

    eval.py recall_at_k / mrr call:
        retrieve_fn(q["question"], variant, course_code, top_k)

    Our retrieve_fn already has that signature; the wrapper just passes
    the question field through unchanged.
    """
    return retrieve_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_split(
    jsonl_path: Path,
    retrieve_fn: Callable,
    variant: str,
    course_code: Optional[str],
    k: int = 5,
    predict_fn: Optional[Callable] = None,
) -> EvalResult:
    """Evaluate retrieval (and optionally answer generation) for one variant.

    Args:
        jsonl_path:   Path to a merged/dev/test benchmark JSONL file.
        retrieve_fn:  Callable(query, variant, course_code, top_k) → list[result].
                      Each result must have a .page_number attribute (1-based int).
        variant:      Which variant to evaluate ("v1", "v2", or "v3").
        course_code:  Optional course filter forwarded to retrieve_fn.
        k:            Top-k cutoff for retrieval metrics.
        predict_fn:   Optional callable(question_text, retrieved_pages) → str.
                      When provided, contains_answer_rate is computed by checking
                      whether benchmark["answer"] is a substring of the prediction.
                      When None, EvalResult.contains_answer_rate is None.

    Returns:
        EvalResult with retrieval and (optionally) answer metrics.
    """
    questions = _load_variant_questions(jsonl_path, variant)

    if not questions:
        return EvalResult(
            variant=variant,
            n_questions=0,
            recall_at_k=0.0,
            mrr=0.0,
            contains_answer_rate=None,
            k=k,
        )

    # Build the eval.py-compatible question list by mapping field names.
    # eval.py recall_at_k / mrr read q["question"] and q["gold_page_ids"].
    eval_questions = [
        {"question": q["question_text"], "gold_page_ids": q["gold_page_ids"]}
        for q in questions
    ]

    recall = _recall_at_k(eval_questions, retrieve_fn, variant, course_code, k)
    mrr_score = _mrr(eval_questions, retrieve_fn, variant, course_code, k)

    # Answer-level metric — only when predict_fn is provided.
    car: Optional[float] = None
    if predict_fn is not None:
        hits = 0
        for q in questions:
            retrieved = retrieve_fn(q["question_text"], variant, course_code, k)
            predicted = predict_fn(q["question_text"], retrieved)
            if contains_answer(predicted, q["answer"]):
                hits += 1
        car = hits / len(questions)

    return EvalResult(
        variant=variant,
        n_questions=len(questions),
        recall_at_k=recall,
        mrr=mrr_score,
        contains_answer_rate=car,
        k=k,
    )


def run_all_variants(
    jsonl_path: Path,
    retrieve_fn_factory: Callable[[str], Callable],
    course_code: Optional[str],
    k: int = 5,
    predict_fn_factory: Optional[Callable[[str], Callable]] = None,
) -> dict[str, EvalResult]:
    """Run evaluate_split for all three variants and return results dict.

    Args:
        jsonl_path:            Path to benchmark JSONL (dev or test split).
        retrieve_fn_factory:   Callable(variant: str) → retrieve_fn.
                               Called once per variant so callers can return
                               variant-specific retrievers if needed.
        course_code:           Optional course filter forwarded to retrieve_fn.
        k:                     Top-k cutoff.
        predict_fn_factory:    Optional callable(variant: str) → predict_fn.
                               When None, contains_answer_rate is not computed.

    Returns:
        Dict mapping "v1" / "v2" / "v3" → EvalResult.
    """
    results: dict[str, EvalResult] = {}
    for variant in _VARIANTS:
        retrieve_fn = retrieve_fn_factory(variant)
        predict_fn = predict_fn_factory(variant) if predict_fn_factory is not None else None
        results[variant] = evaluate_split(
            jsonl_path, retrieve_fn, variant, course_code, k, predict_fn
        )
    return results
