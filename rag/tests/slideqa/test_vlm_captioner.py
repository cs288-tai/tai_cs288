"""
Tests for rag.slideqa.vlm_captioner.

All OpenAI client calls are mocked.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.slideqa.schema import SlidePageRecord, make_page_id
from rag.slideqa.vlm_captioner import caption_lecture_pages, caption_slide_page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(page_number: int = 1, caption: str | None = None) -> SlidePageRecord:
    return SlidePageRecord(
        page_id=make_page_id("CS288", "lecture01", page_number),
        course_code="CS288",
        lecture_id="lecture01",
        page_number=page_number,
        image_path="/fake/page.png",
        caption=caption,
    )


def _make_openai_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# caption_slide_page
# ---------------------------------------------------------------------------


class TestCaptionSlidePage:
    def test_sends_correct_message_structure(self, tmp_path: Path) -> None:
        img_path = tmp_path / "slide.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            "This is a test caption."
        )

        result = caption_slide_page(client, img_path)

        assert result == "This is a test caption."
        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"]
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "user"
        content_parts = msg["content"]
        # Must have image_url part and text part
        types = [p["type"] for p in content_parts]
        assert "image_url" in types
        assert "text" in types

    def test_missing_image_returns_empty_string(self, tmp_path: Path) -> None:
        client = MagicMock()
        result = caption_slide_page(client, tmp_path / "nonexistent.png")
        assert result == ""
        client.chat.completions.create.assert_not_called()

    def test_api_error_retries_and_returns_empty(self, tmp_path: Path) -> None:
        img_path = tmp_path / "slide.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")

        with patch("rag.slideqa.vlm_captioner.time.sleep"):
            result = caption_slide_page(client, img_path)

        assert result == ""
        # Should have attempted _MAX_RETRIES times
        assert client.chat.completions.create.call_count == 3

    def test_jpeg_image_uses_jpeg_media_type(self, tmp_path: Path) -> None:
        img_path = tmp_path / "slide.jpg"
        img_path.write_bytes(b"\xff\xd8\xff")

        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response("caption")

        caption_slide_page(client, img_path)

        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"]
        image_part = next(p for p in messages[0]["content"] if p["type"] == "image_url")
        assert "data:image/jpeg" in image_part["image_url"]["url"]


# ---------------------------------------------------------------------------
# caption_lecture_pages
# ---------------------------------------------------------------------------


class TestCaptionLecturePages:
    def test_skips_records_with_existing_caption(self, tmp_path: Path) -> None:
        record = _make_record(caption="already set")
        client = MagicMock()

        result = caption_lecture_pages(client, [record])

        client.chat.completions.create.assert_not_called()
        assert result[0].caption == "already set"

    def test_fills_missing_captions(self, tmp_path: Path) -> None:
        img_path = tmp_path / "page.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        record = SlidePageRecord(
            page_id=make_page_id("CS288", "lecture01", 1),
            course_code="CS288",
            lecture_id="lecture01",
            page_number=1,
            image_path=str(img_path),
        )

        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            "Generated caption."
        )

        result = caption_lecture_pages(client, [record])

        assert result[0].caption == "Generated caption."

    def test_returns_new_instances_not_mutated(self, tmp_path: Path) -> None:
        img_path = tmp_path / "page.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        record = SlidePageRecord(
            page_id=make_page_id("CS288", "lecture01", 1),
            course_code="CS288",
            lecture_id="lecture01",
            page_number=1,
            image_path=str(img_path),
        )

        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response("New caption")

        result = caption_lecture_pages(client, [record])

        assert result[0] is not record
        assert record.caption is None  # original unchanged

    def test_api_error_sets_caption_to_none(self, tmp_path: Path) -> None:
        img_path = tmp_path / "page.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        record = SlidePageRecord(
            page_id=make_page_id("CS288", "lecture01", 1),
            course_code="CS288",
            lecture_id="lecture01",
            page_number=1,
            image_path=str(img_path),
        )

        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")

        with patch("rag.slideqa.vlm_captioner.time.sleep"):
            result = caption_lecture_pages(client, [record])

        assert result[0].caption is None

    def test_mixed_records_handled_correctly(self, tmp_path: Path) -> None:
        img_path = tmp_path / "page.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        record_with_caption = _make_record(page_number=1, caption="existing")
        record_without_caption = SlidePageRecord(
            page_id=make_page_id("CS288", "lecture01", 2),
            course_code="CS288",
            lecture_id="lecture01",
            page_number=2,
            image_path=str(img_path),
        )

        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response("new cap")

        result = caption_lecture_pages(client, [record_with_caption, record_without_caption])

        assert len(result) == 2
        assert result[0].caption == "existing"
        assert result[1].caption == "new cap"
        # Only called once for the record without caption
        assert client.chat.completions.create.call_count == 1
