"""
Tests for app/services/slideqa/pipeline.py

TDD: tests written BEFORE implementation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
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

from app.services.slideqa.pipeline import run_pipeline
from app.services.slideqa.schema import QAResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(
    page_id: str = "CS288/lec01/page_001",
    score: float = 0.85,
):
    from rag.slideqa.retriever import SlidePageResult

    return SlidePageResult(
        page_id=page_id,
        course_code="CS288",
        lecture_id="lec01",
        page_number=1,
        image_path="/data/CS288/lec01/page_001.png",
        ocr_text="Slide content here.",
        caption=None,
        objects=None,
        score=score,
        dense_score=score,
        bm25_score=0.0,
        rank=1,
    )


def _make_mock_retriever(pages=None):
    """Create a mock Retriever that returns the given pages."""
    retriever = MagicMock()
    retriever.retrieve.return_value = pages if pages is not None else [_make_page()]
    return retriever


def _make_openai_client(response_text: str = "[CITE: CS288/lec01/page_001] answer.") -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# ---------------------------------------------------------------------------
# Test 1: invalid mode raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidModeRaisesValueError:
    def test_invalid_mode_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "slideqa.db"
            db_path.touch()
            client = _make_openai_client()
            with pytest.raises(ValueError, match="X5"):
                run_pipeline(
                    question="What is backprop?",
                    course_code="CS288",
                    mode="X5",
                    top_k=5,
                    db_path=db_path,
                    openai_client=client,
                    retriever=_make_mock_retriever(),
                )


# ---------------------------------------------------------------------------
# Test 2: mode A1 uses v1 variant
# ---------------------------------------------------------------------------


class TestModeA1UsesV1Variant:
    def test_mode_a1_uses_v1_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "slideqa.db"
            db_path.touch()
            client = _make_openai_client()
            mock_retriever = _make_mock_retriever()

            run_pipeline(
                question="What is backprop?",
                course_code="CS288",
                mode="A1",
                top_k=5,
                db_path=db_path,
                openai_client=client,
                retriever=mock_retriever,
            )

            mock_retriever.retrieve.assert_called_once()
            call_kwargs = mock_retriever.retrieve.call_args
            index_variant = (
                call_kwargs.kwargs.get("index_variant")
                or call_kwargs.args[1]
            )
            assert index_variant == "v1"


# ---------------------------------------------------------------------------
# Test 3: mode A4 uses v3 variant
# ---------------------------------------------------------------------------


class TestModeA4UsesV3Variant:
    def test_mode_a4_uses_v3_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "slideqa.db"
            db_path.touch()
            client = _make_openai_client()
            mock_retriever = _make_mock_retriever()

            run_pipeline(
                question="What is shown in this chart?",
                course_code="CS288",
                mode="A4",
                top_k=5,
                db_path=db_path,
                openai_client=client,
                retriever=mock_retriever,
            )

            mock_retriever.retrieve.assert_called_once()
            call_kwargs = mock_retriever.retrieve.call_args
            index_variant = (
                call_kwargs.kwargs.get("index_variant")
                or call_kwargs.args[1]
            )
            assert index_variant == "v3"


# ---------------------------------------------------------------------------
# Test 4: pipeline returns QAResponse instance
# ---------------------------------------------------------------------------


class TestPipelineReturnsQAResponse:
    def test_pipeline_returns_qa_response(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "slideqa.db"
            db_path.touch()
            client = _make_openai_client()
            mock_retriever = _make_mock_retriever()

            result = run_pipeline(
                question="What is gradient descent?",
                course_code="CS288",
                mode="A3",
                top_k=5,
                db_path=db_path,
                openai_client=client,
                retriever=mock_retriever,
            )

            assert isinstance(result, QAResponse)


# ---------------------------------------------------------------------------
# Test 5: A4 mode triggers vlm augment
# ---------------------------------------------------------------------------


class TestPipelineA4CallsVlmAugment:
    def test_pipeline_a4_calls_vlm_augment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "slideqa.db"
            db_path.touch()
            client = _make_openai_client()
            mock_retriever = _make_mock_retriever()

            augment_mock = MagicMock(
                return_value=QAResponse(
                    answer="augmented answer",
                    citations=[],
                    retrieved_pages=[],
                    mode="A4",
                    abstained=False,
                )
            )

            with patch(
                "app.services.slideqa.pipeline.vlm_reader.augment_answer_with_vlm",
                augment_mock,
            ):
                result = run_pipeline(
                    question="What chart is shown?",
                    course_code="CS288",
                    mode="A4",
                    top_k=5,
                    db_path=db_path,
                    openai_client=client,
                    retriever=mock_retriever,
                )

            augment_mock.assert_called_once()
            assert result.answer == "augmented answer"


# ---------------------------------------------------------------------------
# Test 6: SlideQARequest input validation
# ---------------------------------------------------------------------------


class TestSlideQARequestValidation:
    def test_empty_question_rejected(self):
        from pydantic import ValidationError
        from app.services.slideqa.schema import SlideQARequest
        with pytest.raises(ValidationError):
            SlideQARequest(question="", mode="A3", top_k=5)

    def test_invalid_mode_rejected(self):
        from pydantic import ValidationError
        from app.services.slideqa.schema import SlideQARequest
        with pytest.raises(ValidationError):
            SlideQARequest(question="What is backprop?", mode="X9", top_k=5)

    def test_top_k_out_of_range_rejected(self):
        from pydantic import ValidationError
        from app.services.slideqa.schema import SlideQARequest
        with pytest.raises(ValidationError):
            SlideQARequest(question="What is backprop?", mode="A3", top_k=0)
        with pytest.raises(ValidationError):
            SlideQARequest(question="What is backprop?", mode="A3", top_k=21)

    def test_valid_request_accepted(self):
        from app.services.slideqa.schema import SlideQARequest
        req = SlideQARequest(question="What is backprop?", mode="A2", top_k=10)
        assert req.question == "What is backprop?"
        assert req.mode == "A2"
        assert req.top_k == 10
