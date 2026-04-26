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

evaluate_split_by_group(
    jsonl_path, retrieve_fn, variant, course_code, k=5,
    group_by="question_type", predict_fn=None
) -> dict[str, GroupedEvalResult]

error_analysis(
    jsonl_path, retrieve_fn, variant, course_code, k=5, predict_fn=None
) -> list[ErrorRecord]

classify_error(
    retrieved_page_numbers, gold_page_ids, predicted_answer, gold_answer, k,
    ocr_text=""
) -> ErrorCategory

Schema bridge
-------------
Benchmark JSONL uses:
    "question_text"  — the question string sent to retrieve_fn
    "answer"         — gold answer string used by contains_answer
    "gold_page_ids"  — list of 0-based MinerU page_idx values

eval.py functions expect:
    q["question"]    → we map question_text → question internally

Neither the benchmark file nor the retriever is mutated.

predict_fn signature (optional)
--------------------------------
predict_fn(question_text: str, retrieved_pages: list) -> str

If predict_fn is None, contains_answer_rate and answer-level error
categories are not computed.

Error categories
----------------
RETRIEVAL_MISS  — gold page not in top-k retrieved results
OCR_LOSS        — retrieval hit, answer wrong, OCR text very short (<80 chars)
VLM_NOISE       — retrieval hit, answer wrong, OCR adequate
                  (VLM caption / object detection likely over-interpreted)
ANSWER_MISS     — retrieval hit, answer wrong, OCR adequate, no VLM signal
                  (used as generic fallback when ocr_text is empty)

OCR length threshold: 80 characters (strips whitespace before measuring).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from slideqa.eval import contains_answer, hits_at_k, mrr as _mrr, recall_at_k as _recall_at_k

logger = logging.getLogger(__name__)

_OCR_SHORT_THRESHOLD = 80   # chars; below this → OCR_LOSS suspected

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


@dataclasses.dataclass(frozen=True)
class GroupedEvalResult:
    """Immutable evaluation result for one group (question_type or evidence_modality)."""

    group_key: str
    n_questions: int
    recall_at_k: float
    mrr: float
    k: int


class ErrorCategory(str, Enum):
    """Taxonomy of retrieval / answer failure modes."""

    RETRIEVAL_MISS = "retrieval_miss"
    ANSWER_MISS = "answer_miss"
    OCR_LOSS = "ocr_loss"
    VLM_NOISE = "vlm_noise"


@dataclasses.dataclass(frozen=True)
class ErrorRecord:
    """Immutable record describing a single failing question."""

    question_id: str
    question_text: str
    gold_page_ids: tuple[int, ...]          # 0-based
    retrieved_page_numbers: tuple[int, ...]  # 1-based, top-k
    predicted_answer: Optional[str]
    category: ErrorCategory
    notes: str = ""


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


# ---------------------------------------------------------------------------
# classify_error — single-question error taxonomy
# ---------------------------------------------------------------------------


def classify_error(
    retrieved_page_numbers: list[int],
    gold_page_ids: list[int],
    predicted_answer: Optional[str],
    gold_answer: str,
    k: int,
    ocr_text: str = "",
) -> ErrorCategory:
    """Classify why a question was answered incorrectly.

    Args:
        retrieved_page_numbers: 1-based page numbers from the retriever (ordered).
        gold_page_ids:          0-based gold page indices from the benchmark.
        predicted_answer:       Free-text prediction from the QA model, or None.
        gold_answer:            Ground-truth answer string.
        k:                      Top-k cutoff; only the first k results are considered.
        ocr_text:               OCR text of the retrieved page (empty string if unknown).

    Returns:
        ErrorCategory value describing the failure mode.

    Classification logic:
        1. If no retrieved page (within top-k) matches a gold page → RETRIEVAL_MISS.
        2. If retrieval hit but answer is wrong:
           a. If ocr_text is very short (< 80 stripped chars) → OCR_LOSS.
           b. If ocr_text is adequate (≥ 80 chars) → VLM_NOISE.
           c. If ocr_text is empty (unknown) → ANSWER_MISS (generic fallback).
    """
    gold_set = set(gold_page_ids)
    top_k = retrieved_page_numbers[:k]
    retrieval_hit = any((pn - 1) in gold_set for pn in top_k)

    if not retrieval_hit:
        return ErrorCategory.RETRIEVAL_MISS

    # Retrieval is correct — classify the answer failure.
    ocr_stripped = ocr_text.strip()
    if ocr_stripped and len(ocr_stripped) < _OCR_SHORT_THRESHOLD:
        return ErrorCategory.OCR_LOSS
    if ocr_stripped:
        return ErrorCategory.VLM_NOISE
    return ErrorCategory.ANSWER_MISS


# ---------------------------------------------------------------------------
# evaluate_split_by_group — breakdown by question_type or evidence_modality
# ---------------------------------------------------------------------------


def evaluate_split_by_group(
    jsonl_path: Path,
    retrieve_fn: Callable,
    variant: str,
    course_code: Optional[str],
    k: int = 5,
    group_by: str = "question_type",
    predict_fn: Optional[Callable] = None,
) -> dict[str, GroupedEvalResult]:
    """Compute Recall@k and MRR broken down by a grouping field.

    Args:
        jsonl_path:  Path to benchmark JSONL.
        retrieve_fn: Callable(query, variant, course_code, top_k) → list[result].
        variant:     Which variant to evaluate ("v1", "v2", or "v3").
        course_code: Optional course filter.
        k:           Top-k cutoff.
        group_by:    Field to group by — "question_type" or "evidence_modality".
                     Questions with a missing/None value go into "_unknown_".
        predict_fn:  Unused here (reserved for future contains_answer per group).

    Returns:
        Dict mapping group key → GroupedEvalResult.
        Empty dict if no questions match the variant.
    """
    questions = _load_variant_questions(jsonl_path, variant)
    if not questions:
        return {}

    # Partition questions by group key.
    groups: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        key = q.get(group_by) or "_unknown_"
        groups.setdefault(str(key), []).append(q)

    result: dict[str, GroupedEvalResult] = {}
    for group_key, group_qs in groups.items():
        eval_questions = [
            {"question": q["question_text"], "gold_page_ids": q["gold_page_ids"]}
            for q in group_qs
        ]
        recall = _recall_at_k(eval_questions, retrieve_fn, variant, course_code, k)
        mrr_score = _mrr(eval_questions, retrieve_fn, variant, course_code, k)
        result[group_key] = GroupedEvalResult(
            group_key=group_key,
            n_questions=len(group_qs),
            recall_at_k=recall,
            mrr=mrr_score,
            k=k,
        )

    return result


# ---------------------------------------------------------------------------
# error_analysis — collect all failing questions and classify them
# ---------------------------------------------------------------------------


def error_analysis(
    jsonl_path: Path,
    retrieve_fn: Callable,
    variant: str,
    course_code: Optional[str],
    k: int = 5,
    predict_fn: Optional[Callable] = None,
) -> list[ErrorRecord]:
    """Return an ErrorRecord for every question that fails retrieval or answer generation.

    A question is included when:
      - Retrieval miss: none of top-k retrieved pages matches a gold page, OR
      - Answer miss: retrieval is correct but predict_fn produces a wrong answer
        (only when predict_fn is provided).

    Correctly answered questions are NOT included.

    Args:
        jsonl_path:  Path to benchmark JSONL.
        retrieve_fn: Callable(query, variant, course_code, top_k) → list[result].
                     Each result must have `.page_number` (1-based int) and
                     optionally `.ocr_text` (str).
        variant:     Which variant to evaluate.
        course_code: Optional course filter.
        k:           Top-k cutoff.
        predict_fn:  Optional callable(question_text, retrieved_pages) → str.
                     When None, only retrieval misses are classified.

    Returns:
        List of ErrorRecord (one per failing question), in JSONL order.
    """
    questions = _load_variant_questions(jsonl_path, variant)
    records: list[ErrorRecord] = []

    for q in questions:
        question_text = q["question_text"]
        gold_page_ids: list[int] = q["gold_page_ids"]
        question_id: str = q.get("question_id", "")

        retrieved = retrieve_fn(question_text, variant, course_code, k)
        retrieved_pns: list[int] = [r.page_number for r in retrieved[:k]]

        gold_set = set(gold_page_ids)
        retrieval_hit = any((pn - 1) in gold_set for pn in retrieved_pns)

        predicted: Optional[str] = None
        answer_correct = False
        if predict_fn is not None:
            predicted = predict_fn(question_text, retrieved)
            answer_correct = contains_answer(predicted, q["answer"])

        # Skip questions where everything is correct.
        if retrieval_hit and (predict_fn is None or answer_correct):
            continue

        # Pull OCR text from the first retrieved result if available.
        ocr_text = ""
        if retrieved:
            ocr_text = getattr(retrieved[0], "ocr_text", "") or ""

        category = classify_error(
            retrieved_page_numbers=retrieved_pns,
            gold_page_ids=gold_page_ids,
            predicted_answer=predicted,
            gold_answer=q["answer"],
            k=k,
            ocr_text=ocr_text,
        )

        records.append(ErrorRecord(
            question_id=question_id,
            question_text=question_text,
            gold_page_ids=tuple(gold_page_ids),
            retrieved_page_numbers=tuple(retrieved_pns),
            predicted_answer=predicted,
            category=category,
        ))

    return records


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
