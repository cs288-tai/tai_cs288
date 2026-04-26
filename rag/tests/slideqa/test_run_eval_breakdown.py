"""
TDD — tests for breakdown-by-group and error-analysis extensions in run_eval.py.

New public API being tested:

    GroupedEvalResult   — dataclass: group_key, n_questions, recall_at_k, mrr
    ErrorCategory       — enum: RETRIEVAL_MISS | ANSWER_MISS | OCR_LOSS | VLM_NOISE
    ErrorRecord         — dataclass: question_id, question_text, gold_page_ids,
                          retrieved_page_numbers, predicted_answer, category, notes

    evaluate_split_by_group(
        jsonl_path, retrieve_fn, variant, course_code, k,
        group_by="question_type",          # or "evidence_modality"
        predict_fn=None,
    ) -> dict[str, GroupedEvalResult]

    error_analysis(
        jsonl_path, retrieve_fn, variant, course_code, k,
        predict_fn=None,
    ) -> list[ErrorRecord]

    classify_error(
        retrieved_page_numbers: list[int],   # 1-based
        gold_page_ids: list[int],            # 0-based
        predicted_answer: str | None,
        gold_answer: str,
        k: int,
        ocr_text: str = "",
    ) -> ErrorCategory

Design rules:
  - classify_error returns RETRIEVAL_MISS when no retrieved page matches gold
  - classify_error returns ANSWER_MISS when retrieval is correct but answer is wrong
  - classify_error returns OCR_LOSS when retrieval hits but ocr_text is very short (<80 chars)
  - classify_error returns VLM_NOISE as fallback for other answer misses
    (caller must pass ocr_text to distinguish OCR_LOSS from VLM_NOISE)
  - GroupedEvalResult is immutable (frozen dataclass)
  - evaluate_split_by_group groups questions by the field named in group_by
  - questions with unknown/missing group key go into a "_unknown_" bucket
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_RAG_ROOT = Path(__file__).resolve().parents[2]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from slideqa.run_eval import (
    ErrorCategory,
    ErrorRecord,
    GroupedEvalResult,
    classify_error,
    error_analysis,
    evaluate_split_by_group,
)


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakePage:
    page_number: int  # 1-based
    ocr_text: str = ""
    score: float = 1.0


def _page(page_number: int, ocr_text: str = "some text") -> _FakePage:
    return _FakePage(page_number=page_number, ocr_text=ocr_text)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Representative benchmark records
_Q_TABLE_HIT = {
    "question_text": "Which model scores highest on BLEU?",
    "answer": "Transformer-large",
    "question_type": "type_iii",
    "evidence_modality": "table",
    "gold_page_ids": [0],
    "page_id": 0,
    "variant": "v1",
    "lecture": "lec01.pdf",
    "question_id": "t1",
}

_Q_TABLE_MISS = {
    "question_text": "What is the lowest WER in the table?",
    "answer": "4.2%",
    "question_type": "type_iii",
    "evidence_modality": "table",
    "gold_page_ids": [5],
    "page_id": 5,
    "variant": "v1",
    "lecture": "lec01.pdf",
    "question_id": "t2",
}

_Q_CHART_HIT = {
    "question_text": "What is the trend in the loss curve?",
    "answer": "Decreasing",
    "question_type": "type_iv",
    "evidence_modality": "chart",
    "gold_page_ids": [2],
    "page_id": 2,
    "variant": "v1",
    "lecture": "lec01.pdf",
    "question_id": "c1",
}

_Q_LAYOUT_HIT = {
    "question_text": "Which panel is on the right?",
    "answer": "Decoder",
    "question_type": "type_v",
    "evidence_modality": "layout",
    "gold_page_ids": [3],
    "page_id": 3,
    "variant": "v1",
    "lecture": "lec01.pdf",
    "question_id": "l1",
}

_Q_V2 = {
    "question_text": "V2 question",
    "answer": "V2 answer",
    "question_type": "type_iii",
    "evidence_modality": "table",
    "gold_page_ids": [0],
    "page_id": 0,
    "variant": "v2",
    "lecture": "lec01.pdf",
    "question_id": "v2_1",
}


# ---------------------------------------------------------------------------
# A. GroupedEvalResult dataclass
# ---------------------------------------------------------------------------


class TestGroupedEvalResult:
    def test_is_frozen_dataclass(self):
        r = GroupedEvalResult(
            group_key="type_iii",
            n_questions=10,
            recall_at_k=0.8,
            mrr=0.7,
            k=5,
        )
        assert r.group_key == "type_iii"
        assert r.n_questions == 10
        assert r.recall_at_k == pytest.approx(0.8)
        assert r.mrr == pytest.approx(0.7)
        assert r.k == 5

    def test_is_immutable(self):
        r = GroupedEvalResult("type_iii", 10, 0.8, 0.7, 5)
        with pytest.raises((AttributeError, TypeError)):
            r.group_key = "type_iv"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# B. ErrorCategory enum
# ---------------------------------------------------------------------------


class TestErrorCategory:
    def test_has_four_values(self):
        assert ErrorCategory.RETRIEVAL_MISS is not None
        assert ErrorCategory.ANSWER_MISS is not None
        assert ErrorCategory.OCR_LOSS is not None
        assert ErrorCategory.VLM_NOISE is not None

    def test_values_are_strings(self):
        for cat in ErrorCategory:
            assert isinstance(cat.value, str)


# ---------------------------------------------------------------------------
# C. classify_error
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_retrieval_miss_when_no_gold_in_retrieved(self):
        # gold_page_ids=[0] → page_number=1; retrieved returns page_number=3 → miss
        cat = classify_error(
            retrieved_page_numbers=[3],
            gold_page_ids=[0],
            predicted_answer=None,
            gold_answer="Transformer-large",
            k=5,
        )
        assert cat == ErrorCategory.RETRIEVAL_MISS

    def test_retrieval_miss_when_retrieved_is_empty(self):
        cat = classify_error(
            retrieved_page_numbers=[],
            gold_page_ids=[0],
            predicted_answer=None,
            gold_answer="something",
            k=5,
        )
        assert cat == ErrorCategory.RETRIEVAL_MISS

    def test_answer_miss_when_retrieval_correct_but_answer_wrong(self):
        # gold_page_ids=[0] → page_number=1 is in retrieved → retrieval hit
        # but predicted_answer doesn't contain gold
        # With adequate OCR text, the failure is attributed to VLM_NOISE
        cat = classify_error(
            retrieved_page_numbers=[1],
            gold_page_ids=[0],
            predicted_answer="completely wrong",
            gold_answer="Transformer-large",
            k=5,
            ocr_text="a" * 200,  # long OCR → VLM_NOISE, not OCR_LOSS
        )
        assert cat == ErrorCategory.VLM_NOISE

    def test_answer_miss_generic_fallback_when_no_ocr_text(self):
        # No OCR text available (empty string) → generic ANSWER_MISS fallback
        cat = classify_error(
            retrieved_page_numbers=[1],
            gold_page_ids=[0],
            predicted_answer="completely wrong",
            gold_answer="Transformer-large",
            k=5,
            ocr_text="",  # unknown → ANSWER_MISS
        )
        assert cat == ErrorCategory.ANSWER_MISS

    def test_ocr_loss_when_retrieval_correct_but_ocr_short(self):
        # Retrieval hit, wrong answer, but ocr_text is too short → OCR_LOSS
        cat = classify_error(
            retrieved_page_numbers=[1],
            gold_page_ids=[0],
            predicted_answer="wrong",
            gold_answer="Transformer-large",
            k=5,
            ocr_text="tiny",  # < 80 chars → OCR_LOSS
        )
        assert cat == ErrorCategory.OCR_LOSS

    def test_vlm_noise_when_retrieval_hit_wrong_answer_adequate_ocr(self):
        # Retrieval hit, wrong answer, OCR adequate → VLM_NOISE
        cat = classify_error(
            retrieved_page_numbers=[1],
            gold_page_ids=[0],
            predicted_answer="fabricated answer",
            gold_answer="real answer",
            k=5,
            ocr_text="a" * 200,
        )
        assert cat == ErrorCategory.VLM_NOISE

    def test_retrieval_miss_ignores_beyond_k(self):
        # Gold is at rank 6, k=5 → still a retrieval miss even though page_number=1 is present
        cat = classify_error(
            retrieved_page_numbers=[3, 4, 5, 6, 7, 1],  # 1 is rank 6, beyond k=5
            gold_page_ids=[0],
            predicted_answer=None,
            gold_answer="answer",
            k=5,
        )
        assert cat == ErrorCategory.RETRIEVAL_MISS

    def test_answer_miss_not_ocr_loss_when_ocr_exactly_at_threshold(self):
        # ocr_text of exactly 80 chars → not short → not OCR_LOSS
        cat = classify_error(
            retrieved_page_numbers=[1],
            gold_page_ids=[0],
            predicted_answer="wrong",
            gold_answer="right",
            k=5,
            ocr_text="a" * 80,
        )
        # At exactly 80 chars it's still considered adequate
        assert cat in (ErrorCategory.ANSWER_MISS, ErrorCategory.VLM_NOISE)


# ---------------------------------------------------------------------------
# D. evaluate_split_by_group
# ---------------------------------------------------------------------------


class TestEvaluateSplitByGroup:
    def _always_hit_fn(self, q_records):
        """retrieve_fn that returns the gold page_number for the question."""
        gold_map = {
            r["question_text"]: r["gold_page_ids"][0] + 1
            for r in q_records
            if r.get("gold_page_ids")
        }
        def fn(query, variant, course_code, top_k):
            pn = gold_map.get(query, 99)
            return [_page(pn)]
        return fn

    def test_returns_dict_keyed_by_group(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT, _Q_CHART_HIT, _Q_LAYOUT_HIT])
        retrieve_fn = self._always_hit_fn([_Q_TABLE_HIT, _Q_CHART_HIT, _Q_LAYOUT_HIT])

        result = evaluate_split_by_group(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            group_by="question_type",
        )

        assert "type_iii" in result
        assert "type_iv" in result
        assert "type_v" in result

    def test_each_value_is_grouped_eval_result(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT, _Q_CHART_HIT])
        retrieve_fn = self._always_hit_fn([_Q_TABLE_HIT, _Q_CHART_HIT])

        result = evaluate_split_by_group(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            group_by="question_type",
        )

        for v in result.values():
            assert isinstance(v, GroupedEvalResult)

    def test_n_questions_per_group_is_correct(self, tmp_path):
        # 2 table, 1 chart
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT, _Q_TABLE_MISS, _Q_CHART_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split_by_group(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            group_by="question_type",
        )

        assert result["type_iii"].n_questions == 2
        assert result["type_iv"].n_questions == 1

    def test_recall_per_group(self, tmp_path):
        # TABLE: 2 questions, retrieve always returns pn=1
        #   _Q_TABLE_HIT gold=[0]→pn=1 → HIT
        #   _Q_TABLE_MISS gold=[5]→pn=6, retriever returns pn=1 → MISS
        # CHART: 1 question, retrieve always returns pn=1
        #   _Q_CHART_HIT gold=[2]→pn=3, retriever returns pn=1 → MISS
        # Use a smarter retrieve_fn that returns gold page for TABLE_HIT and CHART_HIT,
        # and wrong page for TABLE_MISS.
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT, _Q_TABLE_MISS, _Q_CHART_HIT])

        gold_map = {
            _Q_TABLE_HIT["question_text"]: _Q_TABLE_HIT["gold_page_ids"][0] + 1,
            _Q_TABLE_MISS["question_text"]: 99,  # force miss
            _Q_CHART_HIT["question_text"]: _Q_CHART_HIT["gold_page_ids"][0] + 1,
        }
        retrieve_fn = lambda q, v, c, k: [_page(gold_map.get(q, 99))]

        result = evaluate_split_by_group(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            group_by="question_type",
        )

        # TABLE: 1 hit / 2 total = 0.5
        assert result["type_iii"].recall_at_k == pytest.approx(0.5)
        # CHART: 1 hit / 1 total = 1.0
        assert result["type_iv"].recall_at_k == pytest.approx(1.0)

    def test_group_by_evidence_modality(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT, _Q_CHART_HIT, _Q_LAYOUT_HIT])
        retrieve_fn = self._always_hit_fn([_Q_TABLE_HIT, _Q_CHART_HIT, _Q_LAYOUT_HIT])

        result = evaluate_split_by_group(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            group_by="evidence_modality",
        )

        assert "table" in result
        assert "chart" in result
        assert "layout" in result

    def test_only_target_variant_included(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT, _Q_V2])  # _Q_V2 is variant=v2
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split_by_group(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            group_by="question_type",
        )

        # Only v1 questions counted; type_iii should have n=1 not 2
        assert result["type_iii"].n_questions == 1

    def test_missing_group_field_goes_to_unknown(self, tmp_path):
        q_no_type = {**_Q_TABLE_HIT, "question_type": None, "question_id": "x1"}
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [q_no_type])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split_by_group(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            group_by="question_type",
        )

        assert "_unknown_" in result

    def test_group_key_field_is_set(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split_by_group(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            group_by="question_type",
        )

        assert result["type_iii"].group_key == "type_iii"

    def test_k_field_propagated(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split_by_group(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=7,
            group_by="question_type",
        )

        assert result["type_iii"].k == 7

    def test_empty_split_returns_empty_dict(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_V2])  # v2 only, asking for v1
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        result = evaluate_split_by_group(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            group_by="question_type",
        )

        assert result == {}


# ---------------------------------------------------------------------------
# E. error_analysis
# ---------------------------------------------------------------------------


class TestErrorAnalysis:
    def test_returns_list_of_error_records(self, tmp_path):
        # _Q_TABLE_MISS: gold=[5]→pn=6, retriever returns pn=1 → RETRIEVAL_MISS
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_MISS])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        records = error_analysis(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
        )

        assert isinstance(records, list)
        assert len(records) == 1

    def test_each_item_is_error_record(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_MISS])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        records = error_analysis(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert isinstance(records[0], ErrorRecord)

    def test_retrieval_miss_classified_correctly(self, tmp_path):
        # gold=[5] → pn=6, retrieve always returns pn=1 → RETRIEVAL_MISS
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_MISS])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        records = error_analysis(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert records[0].category == ErrorCategory.RETRIEVAL_MISS

    def test_correct_questions_not_included(self, tmp_path):
        # _Q_TABLE_HIT: gold=[0]→pn=1, retrieve returns pn=1 → hit → NOT in errors
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        records = error_analysis(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        assert len(records) == 0

    def test_error_record_has_required_fields(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_MISS])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        records = error_analysis(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)
        r = records[0]

        assert r.question_id == _Q_TABLE_MISS["question_id"]
        assert r.question_text == _Q_TABLE_MISS["question_text"]
        assert list(r.gold_page_ids) == _Q_TABLE_MISS["gold_page_ids"]
        assert isinstance(r.retrieved_page_numbers, (list, tuple))
        assert isinstance(r.category, ErrorCategory)

    def test_only_errors_for_target_variant(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_MISS, _Q_V2])
        retrieve_fn = lambda q, v, c, k: [_page(1)]

        records = error_analysis(jsonl, retrieve_fn, variant="v1", course_code=None, k=5)

        # Only v1 questions are analysed; _Q_V2 is v2 and must be excluded
        assert all(
            r.question_id != _Q_V2["question_id"] for r in records
        )

    def test_answer_miss_classified_when_predict_fn_provided(self, tmp_path):
        # Retrieval hit (gold=[0]→pn=1), predict_fn returns wrong answer → ANSWER_MISS or VLM_NOISE
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1, ocr_text="a" * 200)]
        predict_fn = lambda question, pages: "completely wrong answer"

        records = error_analysis(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            predict_fn=predict_fn,
        )

        assert len(records) == 1
        assert records[0].category in (ErrorCategory.ANSWER_MISS, ErrorCategory.VLM_NOISE)

    def test_no_errors_when_all_correct_with_predict_fn(self, tmp_path):
        jsonl = tmp_path / "bench.jsonl"
        _write_jsonl(jsonl, [_Q_TABLE_HIT])
        retrieve_fn = lambda q, v, c, k: [_page(1)]
        predict_fn = lambda question, pages: "Transformer-large"  # exact gold answer

        records = error_analysis(
            jsonl, retrieve_fn, variant="v1", course_code=None, k=5,
            predict_fn=predict_fn,
        )

        assert len(records) == 0
