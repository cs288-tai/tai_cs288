"""
Tests for app/services/slideqa/vlm_reader.py

TDD: tests written BEFORE implementation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import os

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from app.services.slideqa.vlm_reader import (
    should_use_vlm_read,
    vlm_read_page,
    augment_answer_with_vlm,
)
from app.services.slideqa.schema import QAResponse, Citation, RetrievedPage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(
    page_id: str = "CS288/lec01/page_001",
    score: float = 0.85,
    image_path: str = "/data/CS288/lec01/page_001.png",
    ocr_text: str = "Sample slide text.",
    rank: int = 1,
):
    from rag.slideqa.retriever import SlidePageResult

    return SlidePageResult(
        page_id=page_id,
        course_code="CS288",
        lecture_id="lec01",
        page_number=1,
        image_path=image_path,
        ocr_text=ocr_text,
        caption=None,
        objects=None,
        score=score,
        dense_score=score,
        bm25_score=0.0,
        rank=rank,
    )


def _make_qa_response(answer_text: str = "The answer is X.") -> QAResponse:
    return QAResponse(
        answer=answer_text,
        citations=[],
        retrieved_pages=[],
        mode="A4",
        abstained=False,
    )


def _make_openai_client(response_text: str = "VLM image analysis result.") -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# ---------------------------------------------------------------------------
# Test 1: should_use_vlm_read returns False for A1/A2/A3
# ---------------------------------------------------------------------------


class TestShouldUseVlmReadModesA1A2A3:
    def test_should_use_vlm_read_false_for_a1_a2_a3(self):
        page = _make_page(score=0.1)  # low score, but mode is not A4
        for mode in ("A1", "A2", "A3"):
            result = should_use_vlm_read(
                retrieved_pages=[page],
                mode=mode,
                question="What is shown in the chart?",
                confidence_threshold=0.3,
            )
            assert result is False, f"Expected False for mode={mode}"


# ---------------------------------------------------------------------------
# Test 2: should_use_vlm_read True for A4 + low confidence
# ---------------------------------------------------------------------------


class TestShouldUseVlmReadA4LowConfidence:
    def test_should_use_vlm_read_true_for_a4_low_confidence(self):
        page = _make_page(score=0.1)  # below default threshold 0.3
        result = should_use_vlm_read(
            retrieved_pages=[page],
            mode="A4",
            question="Explain backpropagation.",
            confidence_threshold=0.3,
        )
        assert result is True


# ---------------------------------------------------------------------------
# Test 3: should_use_vlm_read True for A4 + visual keyword in question
# ---------------------------------------------------------------------------


class TestShouldUseVlmReadA4VisualQuestion:
    def test_should_use_vlm_read_true_for_a4_visual_question(self):
        page = _make_page(score=0.9)  # high score, but question contains "chart"
        result = should_use_vlm_read(
            retrieved_pages=[page],
            mode="A4",
            question="What does the chart show about training loss?",
            confidence_threshold=0.3,
        )
        assert result is True


# ---------------------------------------------------------------------------
# Test 4: vlm_read_page calls OpenAI with image content
# ---------------------------------------------------------------------------


class TestVlmReadPageCallsOpenAI:
    def test_vlm_read_page_calls_openai_with_image(self):
        client = _make_openai_client("The slide shows a neural network diagram.")

        # Create a real temp PNG-like file (just needs to exist and be readable)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # minimal PNG header bytes
            tmp_path = f.name

        try:
            result = vlm_read_page(
                image_path=tmp_path,
                question="What is shown?",
                openai_client=client,
            )
            assert client.chat.completions.create.called
            call_args = client.chat.completions.create.call_args
            messages = (
                call_args.kwargs.get("messages")
                or (call_args.args[0] if call_args.args else None)
            )
            assert messages is not None
            # At least one message should have image_url content
            has_image = False
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            has_image = True
            assert has_image, "Expected image_url content in OpenAI messages"
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test 5: vlm_read_page rejects non-image path
# ---------------------------------------------------------------------------


class TestVlmReadPageRejectsNonImage:
    def test_vlm_read_page_rejects_non_image_path(self):
        client = _make_openai_client()
        result = vlm_read_page(
            image_path="/some/document.txt",
            question="What is shown?",
            openai_client=client,
        )
        assert result == ""
        assert not client.chat.completions.create.called


# ---------------------------------------------------------------------------
# Test 6: augment_answer_with_vlm appends VLM result to answer
# ---------------------------------------------------------------------------


class TestAugmentAnswerAppendsVlmResult:
    def test_augment_answer_appends_vlm_result(self):
        vlm_text = "The diagram shows three layers."
        client = _make_openai_client(vlm_text)
        original = _make_qa_response("Initial answer text.")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            tmp_path = f.name

        from rag.slideqa.retriever import SlidePageResult

        page = SlidePageResult(
            page_id="CS288/lec01/page_001",
            course_code="CS288",
            lecture_id="lec01",
            page_number=1,
            image_path=tmp_path,
            ocr_text="text",
            caption=None,
            objects=None,
            score=0.9,
            dense_score=0.9,
            bm25_score=0.0,
            rank=1,
        )

        try:
            result = augment_answer_with_vlm(
                qa_response=original,
                retrieved_pages=[page],
                question="What does the diagram show?",
                openai_client=client,
            )
            assert "[VLM Image Read]" in result.answer
            assert vlm_text in result.answer
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test 7: augment_answer_with_vlm returns NEW instance (immutability)
# ---------------------------------------------------------------------------


class TestVlmReadPagePathTraversalBlocked:
    """Test: path outside allowed_root is rejected."""

    def test_path_outside_allowed_root_rejected(self, tmp_path):
        allowed = tmp_path / "data"
        allowed.mkdir()
        outside = tmp_path / "secret.png"
        outside.write_bytes(b"\x89PNG\r\n\x1a\n")
        client = _make_openai_client()
        result = vlm_read_page(
            image_path=str(outside),
            question="What is shown?",
            openai_client=client,
            allowed_root=allowed,
        )
        assert result == ""
        assert not client.chat.completions.create.called

    def test_path_inside_allowed_root_accepted(self, tmp_path):
        allowed = tmp_path / "data"
        allowed.mkdir()
        img = allowed / "slide.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        client = _make_openai_client("result")
        vlm_read_page(
            image_path=str(img),
            question="What is shown?",
            openai_client=client,
            allowed_root=allowed,
        )
        assert client.chat.completions.create.called


class TestAugmentAnswerReturnsNewInstance:
    def test_augment_answer_returns_new_instance(self):
        client = _make_openai_client("VLM result")
        original = _make_qa_response("Base answer.")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            tmp_path = f.name

        from rag.slideqa.retriever import SlidePageResult

        page = SlidePageResult(
            page_id="CS288/lec01/page_001",
            course_code="CS288",
            lecture_id="lec01",
            page_number=1,
            image_path=tmp_path,
            ocr_text="text",
            caption=None,
            objects=None,
            score=0.9,
            dense_score=0.9,
            bm25_score=0.0,
            rank=1,
        )

        try:
            result = augment_answer_with_vlm(
                qa_response=original,
                retrieved_pages=[page],
                question="What is shown?",
                openai_client=client,
            )
            assert result is not original
            assert original.answer == "Base answer."  # original unchanged
        finally:
            os.unlink(tmp_path)
