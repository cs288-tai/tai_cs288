"""
TDD — RED tests for rag/slideqa/run_eval.py.

run_eval.py provides two public functions:

    evaluate_split(
        jsonl_path: Path,
        retrieve_fn: Callable,
        variant: str,
        course_code: str | None,
        k: int = 5,
    ) -> EvalResult

    run_all_variants(
        jsonl_path: Path,
        retrieve_fn_factory: Callable[[str], Callable],
        course_code: str | None,
        k: int = 5,
    ) -> dict[str, EvalResult]

EvalResult is a dataclass:
    variant: str
    n_questions: int
    recall_at_k: float
    mrr: float
    contains_answer_rate: float
    k: int

Key bridging rules (benchmark schema → eval.py schema):
    benchmark["question_text"]  →  used as query to retrieve_fn
    benchmark["answer"]         →  used as gold for contains_answer
    benchmark["gold_page_ids"]  →  passed to hits_at_k / mrr

Only questions matching the given variant are included.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

_RAG_ROOT = Path(__file__).resolve().parents[2]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

# RED: will fail until run_eval.py exists
from slideqa.run_eval import EvalResult, evaluate_split, run_all_variants


# ---------------------------------------------------------------------------
# Minimal fake page result (mirrors SlidePageResult.page_number)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakePage:
    page_number: int  # 1-based
    rank: int = 1
    score: float = 1.0


def _page(page_number: int, rank: int = 1) -> _FakePage:
    return _FakePage(page_number=page_number, rank=rank, score=1.0 / rank)


# ---------------------------------------------------------------------------
# Fixtures — benchmark JSONL files written to tmp_path
# ---------------------------------------------------------------------------

_Q_V1_HIT = {
    "question_text": "What is attention?",
    "answer": "A weighting mechanism",
    "question_type": "type_i",
    "evidence_modality": "text_only",
    "gold_page_ids": [0],   # 0-based → page_number 1 is a hit
    "page_id": 0,
    "variant": "v1",
    "lecture": "CS288_sp26_01_Intro.pdf",
    "question_id": "aaa1",
}

_Q_V1_MISS = {
    "question_text": "Describe the diagram.",
    "answer": "A transformer block",
    "question_type": "type_ii",
    "evidence_modality": "visual",
    "gold_page_ids": [5],   # 0-based → page_number 6, never returned
    "page_id": 5,
    "variant": "v1",
    "lecture": "CS288_sp26_01_Intro.pdf",
    "question_id": "aaa2",
}

_Q_V2_HIT = {
    "question_text": "What is BERT?",
    "answer": "A pretrained language model",
    "question_type": "type_i",
    "evidence_modality": "text_only",
    "gold_page_ids": [2],
    "page_id": 2,
    "variant": "v2",
    "lecture": "CS288_sp26_01_Intro.pdf",
    "question_id": "bbb1",
}

_Q_V3_HIT = {
    "question_text": "What is GPT?",
    "answer": "A generative model",
    "question_type": "type_i",
    "evidence_modality": "text_only",
    "gold_page_ids": [3],
    "page_id": 3,
    "variant": "v3",
    "lecture": "CS288_sp26_01_Intro.pdf",
    "question_id": "ccc1",
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests: EvalResult dataclass
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_is_dataclass_with_required_fields(self):
        r = EvalResult(
            variant="v1",
            n_questions=10,
            recall_at_k=0.8,
            mrr=0.7,
            contains_answer_rate=0.6,
            k=5,
        )
        assert r.variant == "v1"
        assert r.n_questions == 10
        assert r.recall_at_k == pytest.approx(0.8)
        assert r.mrr == pytest.approx(0.7)
        assert r.contains_answer_rate == pytest.approx(0.6)
        assert r.k == 5

    def test_is_immutable(self):
        r = EvalResult("v1", 10, 0.8, 0.7, 0.6, 5)
        with pytest.raises((AttributeError, TypeError)):
            r.variant = "v2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: evaluate_split
# ---------------------------------------------------------------------------


class TestEvaluateSplit:
    """evaluate_split filters by variant, computes metrics, returns EvalResult."""

    def _always_hit_fn(self, gold_page_ids: list[int]):
        """retrieve_fn that always returns page_number = gold_page_ids[0]+1."""
        def fn(query, variant, course_code, top_k):
            return [_page(gold_page_ids[0] + 1, rank=1)]
        return fn

    def test_returns_eval_result(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code="CS288", k=5)

        assert isinstance(result, EvalResult)

    def test_variant_field_is_set(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code="CS288", k=5)

        assert result.variant == "v1"

    def test_k_field_is_set(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code="CS288", k=3)

        assert result.k == 3

    def test_n_questions_counts_only_target_variant(self, tmp_path):
        """Questions from other variants are excluded from n_questions."""
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT, _Q_V1_MISS, _Q_V2_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert result.n_questions == 2  # v1 only: HIT + MISS

    def test_recall_all_hits(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT])
        # gold_page_ids=[0] → hit when page_number=1 returned
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert result.recall_at_k == pytest.approx(1.0)

    def test_recall_no_hits(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_MISS])
        # gold=[5] → page_number=6; retrieve_fn returns page_number=1 → miss
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert result.recall_at_k == pytest.approx(0.0)

    def test_recall_partial(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT, _Q_V1_MISS])
        # HIT: gold=[0] returned page_number=1 → hit
        # MISS: gold=[5] returned page_number=1 → miss
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert result.recall_at_k == pytest.approx(0.5)

    def test_mrr_rank1_gives_1(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT])
        # gold=[0] → page_number=1 at rank 1 → MRR=1.0
        retrieve_fn = lambda q, v, c, k: [_page(1, rank=1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert result.mrr == pytest.approx(1.0)

    def test_mrr_no_hit_gives_0(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_MISS])
        # gold=[5] → page_number=6; returns page_number=1 → no hit
        retrieve_fn = lambda q, v, c, k: [_page(1, rank=1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert result.mrr == pytest.approx(0.0)

    def test_contains_answer_rate_hit(self, tmp_path):
        """answer="A weighting mechanism" → predicted contains it → rate=1.0."""
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT])
        # retrieve_fn irrelevant for contains_answer_rate;
        # but we need it to return something so recall/mrr don't crash
        retrieve_fn = lambda q, v, c, k: []

        # Patch: run_eval must accept an optional answer_fn parameter
        # OR compute contains_answer_rate using benchmark["answer"] directly
        # against the retrieved pages' text. We choose the simpler design:
        # contains_answer_rate is computed WITHOUT a QA model — it evaluates
        # whether benchmark["answer"] (the gold) is substring of a predicted
        # string supplied by an optional predict_fn. When predict_fn is None,
        # contains_answer_rate is reported as None / skipped.
        result = evaluate_split(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            predict_fn=None,
        )

        # Without predict_fn, contains_answer_rate must be None (not computed).
        assert result.contains_answer_rate is None

    def test_contains_answer_rate_with_predict_fn(self, tmp_path):
        """predict_fn returns gold verbatim → rate=1.0."""
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]
        # predict_fn(question_text, retrieved_pages) -> predicted_answer str
        predict_fn = lambda question, pages: "A weighting mechanism"

        result = evaluate_split(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            predict_fn=predict_fn,
        )

        assert result.contains_answer_rate == pytest.approx(1.0)

    def test_contains_answer_rate_miss_with_predict_fn(self, tmp_path):
        """predict_fn returns unrelated text → rate=0.0."""
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]
        predict_fn = lambda question, pages: "Something completely different"

        result = evaluate_split(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            predict_fn=predict_fn,
        )

        assert result.contains_answer_rate == pytest.approx(0.0)

    def test_retrieve_fn_receives_question_text(self, tmp_path):
        """retrieve_fn is called with question_text, not question_id or page_id."""
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT])
        captured = {}

        def retrieve_fn(query, variant, course_code, top_k):
            captured["query"] = query
            return [_page(1)]

        evaluate_split(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert captured["query"] == _Q_V1_HIT["question_text"]

    def test_retrieve_fn_receives_correct_variant_and_k(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT])
        captured = {}

        def retrieve_fn(query, variant, course_code, top_k):
            captured["variant"] = variant
            captured["top_k"] = top_k
            return [_page(1)]

        evaluate_split(jsonl, retrieve_fn, variant="v1", course_code="CS288", k=7)

        assert captured["variant"] == "v1"
        assert captured["top_k"] == 7

    def test_empty_split_returns_zero_metrics(self, tmp_path):
        """No questions matching variant → n_questions=0, all metrics=0."""
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V2_HIT])  # only v2 questions
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert result.n_questions == 0
        assert result.recall_at_k == pytest.approx(0.0)
        assert result.mrr == pytest.approx(0.0)

    def test_malformed_jsonl_lines_skipped(self, tmp_path):
        """Corrupt lines in the JSONL are skipped; valid lines are evaluated."""
        jsonl = tmp_path / "bench.jsonl"
        jsonl.write_text(
            json.dumps(_Q_V1_HIT) + "\n"
            + "NOT VALID JSON !!!\n"
            + json.dumps(_Q_V1_MISS) + "\n",
            encoding="utf-8",
        )
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert result.n_questions == 2  # two valid v1 lines

    def test_whitespace_lines_skipped(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        jsonl.write_text(
            "\n" + json.dumps(_Q_V1_HIT) + "\n" + "   \n",
            encoding="utf-8",
        )
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert result.n_questions == 1


# ---------------------------------------------------------------------------
# Tests: run_all_variants
# ---------------------------------------------------------------------------


class TestRunAllVariants:
    """run_all_variants runs evaluate_split for v1, v2, v3 and returns a dict."""

    def test_returns_dict_with_three_variants(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT, _Q_V2_HIT, _Q_V3_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]
        retrieve_fn_factory = lambda variant: retrieve_fn

        results = run_all_variants(jsonl, retrieve_fn_factory, course_code=None, k=5)

        assert set(results.keys()) == {"v1", "v2", "v3"}

    def test_each_value_is_eval_result(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT, _Q_V2_HIT, _Q_V3_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]
        retrieve_fn_factory = lambda variant: retrieve_fn

        results = run_all_variants(jsonl, retrieve_fn_factory, course_code=None, k=5)

        for v in ("v1", "v2", "v3"):
            assert isinstance(results[v], EvalResult)

    def test_variant_fields_are_correct(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT, _Q_V2_HIT, _Q_V3_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]
        retrieve_fn_factory = lambda variant: retrieve_fn

        results = run_all_variants(jsonl, retrieve_fn_factory, course_code=None, k=5)

        assert results["v1"].variant == "v1"
        assert results["v2"].variant == "v2"
        assert results["v3"].variant == "v3"

    def test_factory_called_with_correct_variant(self, tmp_path):
        """retrieve_fn_factory must receive the variant string."""
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V1_HIT, _Q_V2_HIT, _Q_V3_HIT])
        called_variants: list[str] = []

        def retrieve_fn_factory(variant: str):
            called_variants.append(variant)
            return lambda q, v, c, k: [_page(1)]

        run_all_variants(jsonl, retrieve_fn_factory, course_code=None, k=5)

        assert sorted(called_variants) == ["v1", "v2", "v3"]

    def test_n_questions_per_variant_is_correct(self, tmp_path):
        """Each variant result counts only its own questions."""
        jsonl = tmp_path / "bench.jsonl"
        # 2 v1, 1 v2, 1 v3
        _write_jsonl(jsonl, [_Q_V1_HIT, _Q_V1_MISS, _Q_V2_HIT, _Q_V3_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]
        retrieve_fn_factory = lambda variant: retrieve_fn

        results = run_all_variants(jsonl, retrieve_fn_factory, course_code=None, k=5)

        assert results["v1"].n_questions == 2
        assert results["v2"].n_questions == 1
        assert results["v3"].n_questions == 1
