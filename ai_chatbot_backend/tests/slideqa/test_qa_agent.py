"""
Tests for app/services/slideqa/qa_agent.py

TDD: tests written BEFORE implementation.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup: make rag package importable from the repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from app.services.slideqa.qa_agent import answer
from app.services.slideqa.schema import QAResponse, Citation

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_page(
    page_id: str = "CS288/lec01/page_001",
    course_code: str = "CS288",
    lecture_id: str = "lec01",
    page_number: int = 1,
    ocr_text: str = "Sample OCR text about neural networks.",
    caption: Optional[str] = None,
    objects: Optional[tuple] = None,
    score: float = 0.85,
):
    """Create a SlidePageResult-compatible object for testing."""
    from rag.slideqa.retriever import SlidePageResult

    return SlidePageResult(
        page_id=page_id,
        course_code=course_code,
        lecture_id=lecture_id,
        page_number=page_number,
        image_path=f"/data/{page_id}.png",
        ocr_text=ocr_text,
        caption=caption,
        objects=objects,
        score=score,
        dense_score=score,
        bm25_score=0.0,
        rank=1,
    )


def _make_openai_client(response_text: str) -> MagicMock:
    """Create a mock OpenAI client that returns a fixed completion text."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestAbstainOnEmptyRetrievedPages:
    """Test 1: empty retrieved pages → abstained response."""

    def test_abstain_on_empty_retrieved_pages(self):
        client = _make_openai_client("some answer")
        result = answer(
            question="What is backpropagation?",
            retrieved_pages=[],
            mode="A3",
            openai_client=client,
        )
        assert result.abstained is True
        assert "could not find" in result.answer.lower()
        assert result.citations == []


class TestCitationExtractionFromCiteTag:
    """Test 2: [CITE: page_id] tag in response → citation entry."""

    def test_answer_includes_citation_when_cite_tag_present(self):
        page = _make_page(page_id="CS288/lec01/page_001")
        client = _make_openai_client(
            "Backprop computes gradients. [CITE: CS288/lec01/page_001]"
        )
        result = answer(
            question="What is backpropagation?",
            retrieved_pages=[page],
            mode="A3",
            openai_client=client,
        )
        assert result.abstained is False
        assert len(result.citations) == 1
        assert result.citations[0].page_id == "CS288/lec01/page_001"


class TestNoCitationSetsAbstained:
    """Test 3: answer with no [CITE: ...] → abstained=True."""

    def test_no_citation_sets_abstained_true(self):
        page = _make_page()
        client = _make_openai_client("I don't know the answer to this question.")
        result = answer(
            question="What is quantum computing?",
            retrieved_pages=[page],
            mode="A3",
            openai_client=client,
        )
        assert result.abstained is True


class TestCitationFieldsPopulatedCorrectly:
    """Test 4: citation fields match page_id components."""

    def test_citation_fields_populated_correctly(self):
        page = _make_page(
            page_id="CS288/lec02/page_005",
            course_code="CS288",
            lecture_id="lec02",
            page_number=5,
        )
        client = _make_openai_client(
            "The answer is here. [CITE: CS288/lec02/page_005]"
        )
        result = answer(
            question="Explain gradient descent.",
            retrieved_pages=[page],
            mode="A3",
            openai_client=client,
        )
        assert len(result.citations) == 1
        cit = result.citations[0]
        assert cit.course_code == "CS288"
        assert cit.lecture_id == "lec02"
        assert cit.page_number == 5
        assert cit.page_id == "CS288/lec02/page_005"


class TestContextIncludesOcrText:
    """Test 5: the prompt sent to OpenAI includes ocr_text of retrieved pages."""

    def test_context_includes_ocr_text(self):
        ocr = "Gradient descent minimizes the loss function iteratively."
        page = _make_page(ocr_text=ocr)
        client = _make_openai_client("[CITE: CS288/lec01/page_001]")
        answer(
            question="Explain gradient descent.",
            retrieved_pages=[page],
            mode="A3",
            openai_client=client,
        )
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else call_args.kwargs["messages"]
        # Flatten all message content to check OCR text is present
        all_content = " ".join(
            m["content"] for m in messages if isinstance(m.get("content"), str)
        )
        assert ocr in all_content


class TestContextIncludesCaption:
    """Test 6: caption is included in prompt when present."""

    def test_context_includes_caption(self):
        caption = "Figure 3: Sigmoid activation function plot."
        page = _make_page(caption=caption)
        client = _make_openai_client("[CITE: CS288/lec01/page_001]")
        answer(
            question="What activation function is shown?",
            retrieved_pages=[page],
            mode="A3",
            openai_client=client,
        )
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.kwargs["messages"]
        all_content = " ".join(
            m["content"] for m in messages if isinstance(m.get("content"), str)
        )
        assert caption in all_content


class TestOpenAIErrorReturnsAbstained:
    """Test: OpenAI API exception → abstained=True, no crash."""

    def test_openai_error_returns_abstained(self):
        page = _make_page()
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")
        result = answer(
            question="What is backpropagation?",
            retrieved_pages=[page],
            mode="A3",
            openai_client=client,
        )
        assert result.abstained is True
        assert result.citations == []
        assert "error" in result.answer.lower()


class TestMultipleCitationsExtracted:
    """Test 7: two [CITE: ...] tags → two citation entries."""

    def test_multiple_citations_extracted(self):
        page1 = _make_page(
            page_id="CS288/lec01/page_001",
            course_code="CS288",
            lecture_id="lec01",
            page_number=1,
        )
        page2 = _make_page(
            page_id="CS288/lec01/page_002",
            course_code="CS288",
            lecture_id="lec01",
            page_number=2,
            score=0.70,
        )
        page2 = _make_page.__wrapped__(page2) if hasattr(_make_page, "__wrapped__") else page2  # noqa
        from rag.slideqa.retriever import SlidePageResult
        page2 = SlidePageResult(
            page_id="CS288/lec01/page_002",
            course_code="CS288",
            lecture_id="lec01",
            page_number=2,
            image_path="/data/CS288/lec01/page_002.png",
            ocr_text="More text.",
            caption=None,
            objects=None,
            score=0.70,
            dense_score=0.70,
            bm25_score=0.0,
            rank=2,
        )
        client = _make_openai_client(
            "First point [CITE: CS288/lec01/page_001] and second point [CITE: CS288/lec01/page_002]."
        )
        result = answer(
            question="Summarize the lecture.",
            retrieved_pages=[page1, page2],
            mode="A3",
            openai_client=client,
        )
        assert len(result.citations) == 2
        page_ids = {c.page_id for c in result.citations}
        assert "CS288/lec01/page_001" in page_ids
        assert "CS288/lec01/page_002" in page_ids
