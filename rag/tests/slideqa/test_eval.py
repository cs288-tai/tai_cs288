"""
Tests for rag/slideqa/eval.py — page-level retrieval evaluation metrics.

TDD: these tests are written BEFORE the implementation.
All tests use simple in-memory fixtures — no real DB, no real model downloads.

Key offset rule (tested explicitly below):
  MinerU content_list.json uses 0-based page_idx.
  SlidePageRecord / SlidePageResult use 1-based page_number.
  → A QA item with gold_page_ids=[0] is a hit if page_number==1 is retrieved.
  → A QA item with gold_page_ids=[4] is a hit if page_number==5 is retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pytest

# ---------------------------------------------------------------------------
# Import the module under test (will fail until eval.py exists)
# ---------------------------------------------------------------------------

from rag.slideqa.eval import (
    hits_at_k,
    mrr,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Minimal stub for SlidePageResult (mirrors retriever.SlidePageResult fields
# that eval.py needs, so tests work without importing retriever)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakePage:
    """Minimal stand-in for SlidePageResult — only fields eval.py uses."""

    page_number: int   # 1-based, as stored in DB
    page_id: str = "CS288/lec01/page_001"
    score: float = 0.9
    rank: int = 1


def _page(page_number: int, rank: int = 1) -> _FakePage:
    """Shorthand to build a fake retrieval result with a given page_number."""
    return _FakePage(
        page_number=page_number,
        page_id=f"CS288/lec01/page_{page_number:03d}",
        score=1.0 / rank,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# A. hits_at_k tests
# ---------------------------------------------------------------------------


class TestHitsAtK:
    """hits_at_k(retrieved, gold_page_ids, k) -> bool

    gold_page_ids: 0-based MinerU page_idx values.
    retrieved: list of objects with .page_number (1-based).
    Rule: hit if any of the top-k results has (page_number - 1) in gold_page_ids.
    """

    def test_hit_when_gold_page_is_in_top_k(self):
        # gold_page_ids=[0] means page_number==1 is the target.
        # top result has page_number=1 → hit.
        retrieved = [_page(1), _page(2), _page(3)]
        assert hits_at_k(retrieved, gold_page_ids=[0], k=3) is True

    def test_hit_respects_offset_zero_to_one(self):
        # gold_page_ids=[4] means page_number==5 is the target (4+1).
        retrieved = [_page(5), _page(2)]
        assert hits_at_k(retrieved, gold_page_ids=[4], k=2) is True

    def test_miss_when_gold_page_not_retrieved(self):
        # gold is page_idx=2 (page_number=3), but only page_number 1,2 retrieved.
        retrieved = [_page(1), _page(2)]
        assert hits_at_k(retrieved, gold_page_ids=[2], k=5) is False

    def test_k_cutoff_excludes_lower_results(self):
        # gold page_number=3 (page_idx=2) is at rank 3; k=2 must NOT count it.
        retrieved = [_page(1), _page(2), _page(3)]
        assert hits_at_k(retrieved, gold_page_ids=[2], k=2) is False

    def test_k_cutoff_includes_exact_k(self):
        # gold page_number=2 (page_idx=1) is rank 2; k=2 must count it.
        retrieved = [_page(1), _page(2), _page(3)]
        assert hits_at_k(retrieved, gold_page_ids=[1], k=2) is True

    def test_empty_retrieved_is_miss(self):
        assert hits_at_k([], gold_page_ids=[0], k=5) is False

    def test_empty_gold_page_ids_is_always_miss(self):
        # No gold pages → nothing to hit.
        retrieved = [_page(1), _page(2)]
        assert hits_at_k(retrieved, gold_page_ids=[], k=5) is False

    def test_multiple_gold_page_ids_any_hit_counts(self):
        # gold_page_ids=[0, 1] → page_number 1 or 2. Retrieved page_number=2 → hit.
        retrieved = [_page(3), _page(2)]
        assert hits_at_k(retrieved, gold_page_ids=[0, 1], k=2) is True

    def test_k_larger_than_retrieved_does_not_error(self):
        # k=10 but only 2 results; should still work.
        retrieved = [_page(1), _page(2)]
        assert hits_at_k(retrieved, gold_page_ids=[0], k=10) is True


# ---------------------------------------------------------------------------
# B. recall_at_k tests
# ---------------------------------------------------------------------------


class TestRecallAtK:
    """recall_at_k(questions, retrieve_fn, variant, course_code, k) -> float

    questions: list of dicts with at least "question" and "gold_page_ids" keys.
    retrieve_fn: callable(query, variant, course_code, top_k) -> list[result]
    Returns fraction of questions where hits_at_k is True.
    """

    def _make_retrieve_fn(self, results: list):
        """Build a mock retrieve_fn that always returns the given results."""
        def retrieve_fn(query, variant, course_code, top_k):
            return results
        return retrieve_fn

    def test_all_questions_hit(self):
        # Both questions have gold=[0]; retrieve_fn always returns page_number=1.
        questions = [
            {"question": "Q1", "gold_page_ids": [0]},
            {"question": "Q2", "gold_page_ids": [0]},
        ]
        retrieve_fn = self._make_retrieve_fn([_page(1)])
        result = recall_at_k(questions, retrieve_fn, variant="v1", course_code="CS288", k=5)
        assert result == pytest.approx(1.0)

    def test_no_questions_hit(self):
        # gold=[0] but retrieved=[page_number=2]; no hits.
        questions = [
            {"question": "Q1", "gold_page_ids": [0]},
        ]
        retrieve_fn = self._make_retrieve_fn([_page(2)])
        result = recall_at_k(questions, retrieve_fn, variant="v1", course_code="CS288", k=5)
        assert result == pytest.approx(0.0)

    def test_partial_hit(self):
        # 1 of 2 questions hits.
        questions = [
            {"question": "Q1", "gold_page_ids": [0]},  # page_number=1 → hit
            {"question": "Q2", "gold_page_ids": [5]},  # page_number=6, not returned → miss
        ]
        retrieve_fn = self._make_retrieve_fn([_page(1), _page(2)])
        result = recall_at_k(questions, retrieve_fn, variant="v1", course_code="CS288", k=5)
        assert result == pytest.approx(0.5)

    def test_empty_question_list_returns_zero(self):
        retrieve_fn = self._make_retrieve_fn([_page(1)])
        result = recall_at_k([], retrieve_fn, variant="v1", course_code="CS288", k=5)
        assert result == pytest.approx(0.0)

    def test_retrieve_fn_called_with_correct_variant_and_course(self):
        # Verify the correct variant and course_code are passed through.
        called_with = {}

        def retrieve_fn(query, variant, course_code, top_k):
            called_with["variant"] = variant
            called_with["course_code"] = course_code
            called_with["top_k"] = top_k
            return [_page(1)]

        questions = [{"question": "Q1", "gold_page_ids": [0]}]
        recall_at_k(questions, retrieve_fn, variant="v2", course_code="CS61A", k=3)
        assert called_with["variant"] == "v2"
        assert called_with["course_code"] == "CS61A"
        assert called_with["top_k"] == 3


# ---------------------------------------------------------------------------
# C. mrr tests
# ---------------------------------------------------------------------------


class TestMRR:
    """mrr(questions, retrieve_fn, variant, course_code, k) -> float

    Mean Reciprocal Rank: for each question, find the rank of the first hit
    (1-indexed), compute 1/rank; average over all questions.
    If no hit found within top-k, reciprocal rank = 0 for that question.
    """

    def test_mrr_all_rank_1(self):
        # Every question hits at rank 1 → MRR = 1.0
        questions = [
            {"question": "Q1", "gold_page_ids": [0]},
            {"question": "Q2", "gold_page_ids": [0]},
        ]
        retrieve_fn = lambda q, v, c, k: [_page(1, rank=1)]
        result = mrr(questions, retrieve_fn, variant="v1", course_code="CS288", k=5)
        assert result == pytest.approx(1.0)

    def test_mrr_rank_2_gives_half(self):
        # Hit at rank 2 → reciprocal rank = 0.5
        questions = [{"question": "Q1", "gold_page_ids": [1]}]
        # page_number=2 (page_idx=1) is at rank 2.
        retrieve_fn = lambda q, v, c, k: [_page(1, rank=1), _page(2, rank=2)]
        result = mrr(questions, retrieve_fn, variant="v1", course_code="CS288", k=5)
        assert result == pytest.approx(0.5)

    def test_mrr_no_hit_gives_zero(self):
        # gold=[0] but retrieved has only page_number=3 → no hit within k=2
        questions = [{"question": "Q1", "gold_page_ids": [0]}]
        retrieve_fn = lambda q, v, c, k: [_page(3, rank=1), _page(4, rank=2)]
        result = mrr(questions, retrieve_fn, variant="v1", course_code="CS288", k=2)
        assert result == pytest.approx(0.0)

    def test_mrr_mixed_ranks(self):
        # Q1: hit at rank 1 → 1.0; Q2: hit at rank 2 → 0.5; MRR = 0.75
        def retrieve_fn(query, variant, course_code, top_k):
            if query == "Q1":
                return [_page(1, rank=1), _page(2, rank=2)]   # gold=[0], hit at 1
            else:
                return [_page(3, rank=1), _page(2, rank=2)]   # gold=[1], hit at 2

        questions = [
            {"question": "Q1", "gold_page_ids": [0]},
            {"question": "Q2", "gold_page_ids": [1]},
        ]
        result = mrr(questions, retrieve_fn, variant="v1", course_code="CS288", k=5)
        assert result == pytest.approx(0.75)

    def test_mrr_empty_questions_returns_zero(self):
        retrieve_fn = lambda q, v, c, k: [_page(1)]
        result = mrr([], retrieve_fn, variant="v1", course_code="CS288", k=5)
        assert result == pytest.approx(0.0)

    def test_mrr_k_cutoff_respected(self):
        # gold is at rank 3 but k=2 → no hit within k → MRR=0
        questions = [{"question": "Q1", "gold_page_ids": [2]}]
        # page_number=3 (page_idx=2) is at rank 3
        retrieve_fn = lambda q, v, c, k: [_page(1, rank=1), _page(2, rank=2), _page(3, rank=3)]
        result = mrr(questions, retrieve_fn, variant="v1", course_code="CS288", k=2)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# D. contains_answer / contains_answer_rate tests
# ---------------------------------------------------------------------------


class TestContainsAnswer:
    """contains_answer(predicted: str, gold: str) -> bool

    Returns True if the gold answer appears (case-insensitive) as a substring
    of the predicted answer after stripping leading/trailing whitespace from both.

    This is an answer-level metric: it checks whether the model's answer
    contains the ground-truth answer string.  No retrieval involved.
    """

    def test_exact_match_is_hit(self):
        from rag.slideqa.eval import contains_answer
        assert contains_answer("transformer", "transformer") is True

    def test_case_insensitive_hit(self):
        from rag.slideqa.eval import contains_answer
        # Gold is lowercase; predicted is mixed-case → still a hit.
        assert contains_answer("The answer is Transformer.", "transformer") is True

    def test_substring_hit(self):
        from rag.slideqa.eval import contains_answer
        # Gold phrase appears inside a longer predicted answer.
        assert contains_answer(
            "Attention mechanisms allow the model to focus on relevant tokens.",
            "focus on relevant tokens",
        ) is True

    def test_miss_when_gold_not_in_predicted(self):
        from rag.slideqa.eval import contains_answer
        assert contains_answer("The model uses convolution.", "attention") is False

    def test_empty_gold_always_hits(self):
        # An empty gold string is trivially contained in any predicted string.
        from rag.slideqa.eval import contains_answer
        assert contains_answer("some answer", "") is True

    def test_empty_predicted_misses_nonempty_gold(self):
        from rag.slideqa.eval import contains_answer
        assert contains_answer("", "attention") is False

    def test_both_empty_is_hit(self):
        from rag.slideqa.eval import contains_answer
        assert contains_answer("", "") is True

    def test_leading_trailing_whitespace_stripped(self):
        from rag.slideqa.eval import contains_answer
        # Extra whitespace in gold should be stripped before matching.
        assert contains_answer("attention is all you need", "  attention  ") is True

    def test_partial_word_does_not_match_full_word_incorrectly(self):
        # "cat" is inside "concatenate" — this is expected behaviour for contains_answer.
        # Document the exact behaviour: it IS a substring match, not a word-boundary match.
        from rag.slideqa.eval import contains_answer
        assert contains_answer("we concatenate the vectors", "cat") is True


class TestContainsAnswerRate:
    """contains_answer_rate(questions, predicted_answers) -> float

    questions:         list of QA dicts with "answer_short" key (gold answer).
    predicted_answers: list of strings produced by a QA model, same order as questions.

    Returns fraction of pairs where contains_answer(predicted, gold) is True.
    """

    def test_all_hit(self):
        from rag.slideqa.eval import contains_answer_rate
        questions = [
            {"answer_short": "transformer"},
            {"answer_short": "attention"},
        ]
        predicted = ["The transformer architecture.", "It uses attention mechanisms."]
        assert contains_answer_rate(questions, predicted) == pytest.approx(1.0)

    def test_none_hit(self):
        from rag.slideqa.eval import contains_answer_rate
        questions = [{"answer_short": "transformer"}]
        predicted = ["It is a convolutional network."]
        assert contains_answer_rate(questions, predicted) == pytest.approx(0.0)

    def test_partial_hit(self):
        from rag.slideqa.eval import contains_answer_rate
        questions = [
            {"answer_short": "transformer"},   # hit
            {"answer_short": "recurrent"},     # miss
        ]
        predicted = ["The transformer model.", "Convolution is used."]
        assert contains_answer_rate(questions, predicted) == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        from rag.slideqa.eval import contains_answer_rate
        assert contains_answer_rate([], []) == pytest.approx(0.0)

    def test_case_insensitive_rate(self):
        from rag.slideqa.eval import contains_answer_rate
        questions = [{"answer_short": "BERT"}]
        predicted = ["bert is a language model."]   # lowercase predicted
        assert contains_answer_rate(questions, predicted) == pytest.approx(1.0)

    def test_uses_answer_short_key(self):
        # Verify the function reads "answer_short", not "answer" or "question".
        from rag.slideqa.eval import contains_answer_rate
        questions = [{"answer_short": "NLP", "answer": "wrong_key_value"}]
        predicted = ["NLP stands for natural language processing."]
        assert contains_answer_rate(questions, predicted) == pytest.approx(1.0)
