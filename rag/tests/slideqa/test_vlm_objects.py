"""Unit tests for rag.slideqa.vlm_objects."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.slideqa.schema import SlidePageRecord, SlideType
from rag.slideqa.vlm_objects import (
    classify_slide_type,
    describe_lecture_objects,
    describe_objects,
    get_object_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(**kwargs) -> SlidePageRecord:
    defaults = dict(
        page_id="CS288/lecture01/page_001",
        course_code="CS288",
        lecture_id="lecture01",
        page_number=1,
        image_path="slide.png",
        ocr_text="",
    )
    defaults.update(kwargs)
    return SlidePageRecord(**defaults)


def _fake_openai_client(response_text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = response_text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


# ---------------------------------------------------------------------------
# classify_slide_type
# ---------------------------------------------------------------------------

class TestClassifySlideType:
    def test_chart_keywords(self):
        assert classify_slide_type("This figure shows a bar chart of results.") == SlideType.CHART

    def test_table_keywords(self):
        assert classify_slide_type("The table has three columns and ten rows.") == SlideType.TABLE

    def test_diagram_keywords(self):
        assert classify_slide_type("System architecture diagram with components.") == SlideType.DIAGRAM

    def test_text_slide(self):
        assert classify_slide_type("Introduction to natural language processing.") == SlideType.TEXT

    def test_empty_ocr_is_unknown(self):
        assert classify_slide_type("") == SlideType.UNKNOWN

    def test_whitespace_only_is_unknown(self):
        assert classify_slide_type("   \n  ") == SlideType.UNKNOWN

    def test_chart_beats_table(self):
        # "chart" and "table" both present — CHART wins (higher priority)
        assert classify_slide_type("chart with table data") == SlideType.CHART

    def test_table_beats_diagram(self):
        assert classify_slide_type("table and diagram") == SlideType.TABLE


# ---------------------------------------------------------------------------
# get_object_prompt
# ---------------------------------------------------------------------------

class TestGetObjectPrompt:
    def test_all_types_return_non_empty(self):
        for slide_type in SlideType:
            prompt = get_object_prompt(slide_type)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_chart_prompt_mentions_axis(self):
        assert "axis" in get_object_prompt(SlideType.CHART).lower() or \
               "legend" in get_object_prompt(SlideType.CHART).lower()

    def test_table_prompt_mentions_column(self):
        assert "column" in get_object_prompt(SlideType.TABLE).lower()

    def test_diagram_prompt_mentions_component(self):
        assert "component" in get_object_prompt(SlideType.DIAGRAM).lower()


# ---------------------------------------------------------------------------
# describe_objects
# ---------------------------------------------------------------------------

class TestDescribeObjects:
    def test_returns_tuple(self, tmp_path):
        img = tmp_path / "slide.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        client = _fake_openai_client("Arrow pointing right\nLabel: Input\nLabel: Output")
        result = describe_objects(client, img, SlideType.DIAGRAM)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_rejects_non_image_path(self, tmp_path, caplog):
        txt = tmp_path / "data.txt"
        txt.write_text("not an image")
        client = _fake_openai_client("irrelevant")
        result = describe_objects(client, txt, SlideType.TEXT)
        assert result == ()
        assert "Rejected" in caplog.text

    def test_missing_file_returns_empty(self, tmp_path, caplog):
        client = _fake_openai_client("irrelevant")
        result = describe_objects(client, tmp_path / "missing.png", SlideType.TEXT)
        assert result == ()

    def test_api_error_retries_and_returns_empty(self, tmp_path):
        img = tmp_path / "slide.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")
        with patch("time.sleep"):
            result = describe_objects(client, img, SlideType.CHART)
        assert result == ()
        assert client.chat.completions.create.call_count == 3

    def test_uses_type_specific_prompt(self, tmp_path):
        img = tmp_path / "slide.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        client = _fake_openai_client("Item A")
        describe_objects(client, img, SlideType.TABLE)
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        # The prompt text is in the message content
        content = messages[0]["content"]
        text_parts = [p["text"] for p in content if p.get("type") == "text"]
        assert any("column" in t.lower() for t in text_parts)


# ---------------------------------------------------------------------------
# describe_lecture_objects
# ---------------------------------------------------------------------------

class TestDescribeLectureObjects:
    def test_skips_already_annotated(self, tmp_path):
        img = tmp_path / "slide.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        record = _make_record(image_path=str(img), objects=("existing item",))
        client = _fake_openai_client("should not be called")
        result = describe_lecture_objects(client, [record])
        assert result[0].objects == ("existing item",)
        client.chat.completions.create.assert_not_called()

    def test_fills_missing_objects(self, tmp_path):
        img = tmp_path / "slide.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        record = _make_record(image_path=str(img))
        client = _fake_openai_client("Component A\nComponent B")
        result = describe_lecture_objects(client, [record])
        assert result[0].objects == ("Component A", "Component B")

    def test_returns_new_instances(self, tmp_path):
        img = tmp_path / "slide.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        record = _make_record(image_path=str(img))
        client = _fake_openai_client("X")
        result = describe_lecture_objects(client, [record])
        assert result[0] is not record

    def test_mixed_records(self, tmp_path):
        img = tmp_path / "slide.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        r1 = _make_record(page_id="CS288/l1/page_001", image_path=str(img), objects=("done",))
        r2 = _make_record(page_id="CS288/l1/page_002", image_path=str(img))
        client = _fake_openai_client("New item")
        result = describe_lecture_objects(client, [r1, r2])
        assert result[0].objects == ("done",)
        assert result[1].objects == ("New item",)
