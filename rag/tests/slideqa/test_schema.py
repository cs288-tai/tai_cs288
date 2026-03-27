"""
Tests for rag.slideqa.schema — SlidePageRecord, SlideType, make_page_id.
"""

from __future__ import annotations

import dataclasses

import pytest

from rag.slideqa.schema import SlidePageRecord, SlideType, make_page_id


# ---------------------------------------------------------------------------
# make_page_id
# ---------------------------------------------------------------------------


class TestMakePageId:
    def test_basic_format(self) -> None:
        result = make_page_id("CS288", "lecture01", 1)
        assert result == "CS288/lecture01/page_001"

    def test_padding_to_three_digits(self) -> None:
        assert make_page_id("CS288", "lecture02", 42) == "CS288/lecture02/page_042"

    def test_large_page_number(self) -> None:
        assert make_page_id("CS288", "lecture10", 100) == "CS288/lecture10/page_100"

    def test_empty_course_code_raises(self) -> None:
        with pytest.raises(ValueError, match="course_code"):
            make_page_id("", "lecture01", 1)

    def test_empty_lecture_id_raises(self) -> None:
        with pytest.raises(ValueError, match="lecture_id"):
            make_page_id("CS288", "", 1)

    def test_zero_page_number_raises(self) -> None:
        with pytest.raises(ValueError, match="page_number"):
            make_page_id("CS288", "lecture01", 0)


# ---------------------------------------------------------------------------
# SlideType
# ---------------------------------------------------------------------------


class TestSlideType:
    def test_enum_values(self) -> None:
        assert SlideType.TEXT.value == "text"
        assert SlideType.CHART.value == "chart"
        assert SlideType.TABLE.value == "table"
        assert SlideType.DIAGRAM.value == "diagram"
        assert SlideType.UNKNOWN.value == "unknown"

    def test_all_members_present(self) -> None:
        members = {m.name for m in SlideType}
        assert members == {"TEXT", "CHART", "TABLE", "DIAGRAM", "UNKNOWN"}


# ---------------------------------------------------------------------------
# SlidePageRecord creation and immutability
# ---------------------------------------------------------------------------


class TestSlidePageRecord:
    def _make_record(self, **kwargs) -> SlidePageRecord:
        defaults = dict(
            page_id="CS288/lecture01/page_001",
            course_code="CS288",
            lecture_id="lecture01",
            page_number=1,
            image_path="/data/slides/CS288/lecture01/page_001.png",
        )
        defaults.update(kwargs)
        return SlidePageRecord(**defaults)

    def test_creation_required_fields(self) -> None:
        rec = self._make_record()
        assert rec.page_id == "CS288/lecture01/page_001"
        assert rec.course_code == "CS288"
        assert rec.lecture_id == "lecture01"
        assert rec.page_number == 1
        assert rec.image_path == "/data/slides/CS288/lecture01/page_001.png"

    def test_default_optional_fields(self) -> None:
        rec = self._make_record()
        assert rec.ocr_text == ""
        assert rec.caption is None
        assert rec.objects is None

    def test_creation_with_all_fields(self) -> None:
        rec = self._make_record(
            ocr_text="Hello world",
            caption="A sample slide",
            objects=("bullet one", "bullet two"),
        )
        assert rec.ocr_text == "Hello world"
        assert rec.caption == "A sample slide"
        assert rec.objects == ("bullet one", "bullet two")

    def test_frozen_immutability_raises(self) -> None:
        rec = self._make_record()
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.ocr_text = "mutated"  # type: ignore[misc]

    def test_replace_returns_new_instance(self) -> None:
        rec = self._make_record()
        updated = dataclasses.replace(rec, caption="new caption")
        assert updated.caption == "new caption"
        assert rec.caption is None  # original unchanged
        assert updated is not rec

    def test_asdict_roundtrip(self) -> None:
        rec = self._make_record(ocr_text="text", caption="cap", objects=("a",))
        d = dataclasses.asdict(rec)
        assert d["page_id"] == rec.page_id
        assert d["ocr_text"] == rec.ocr_text
        assert d["caption"] == rec.caption
        assert d["objects"] == rec.objects

    def test_invalid_page_number_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="page_number"):
            self._make_record(page_number=0)

    def test_empty_page_id_raises(self) -> None:
        with pytest.raises(ValueError, match="page_id"):
            self._make_record(page_id="")

    def test_empty_course_code_raises(self) -> None:
        with pytest.raises(ValueError, match="course_code"):
            self._make_record(course_code="")

    def test_empty_lecture_id_raises(self) -> None:
        with pytest.raises(ValueError, match="lecture_id"):
            self._make_record(lecture_id="")

    def test_empty_image_path_raises(self) -> None:
        with pytest.raises(ValueError, match="image_path"):
            self._make_record(image_path="")
