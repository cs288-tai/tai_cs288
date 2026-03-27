"""
Tests for rag.slideqa.ocr_extractor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.slideqa.ocr_extractor import (
    extract_ocr_for_lecture,
    extract_ocr_from_content_list,
)


# ---------------------------------------------------------------------------
# extract_ocr_from_content_list
# ---------------------------------------------------------------------------


class TestExtractOcrFromContentList:
    def test_basic_grouping(self, tmp_path: Path) -> None:
        items = [
            {"type": "text", "text": "Hello", "page_idx": 0},
            {"type": "text", "text": "World", "page_idx": 0},
            {"type": "text", "text": "Page two text", "page_idx": 1},
        ]
        content_list = tmp_path / "content_list.json"
        content_list.write_text(json.dumps(items), encoding="utf-8")

        result = extract_ocr_from_content_list(content_list)

        assert result[0] == "Hello\nWorld"
        assert result[1] == "Page two text"

    def test_non_text_items_ignored(self, tmp_path: Path) -> None:
        items = [
            {"type": "text", "text": "Keep this", "page_idx": 0},
            {"type": "image", "text": "Drop this", "page_idx": 0},
            {"type": "table", "text": "Drop this too", "page_idx": 0},
        ]
        content_list = tmp_path / "content_list.json"
        content_list.write_text(json.dumps(items), encoding="utf-8")

        result = extract_ocr_from_content_list(content_list)

        assert result[0] == "Keep this"

    def test_empty_json_array(self, tmp_path: Path) -> None:
        content_list = tmp_path / "content_list.json"
        content_list.write_text("[]", encoding="utf-8")

        result = extract_ocr_from_content_list(content_list)

        assert result == {}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = extract_ocr_from_content_list(tmp_path / "nonexistent.json")
        assert result == {}

    def test_malformed_json_returns_empty_no_crash(self, tmp_path: Path) -> None:
        content_list = tmp_path / "content_list.json"
        content_list.write_text("{not valid json[[", encoding="utf-8")

        result = extract_ocr_from_content_list(content_list)

        assert result == {}

    def test_json_non_list_root_returns_empty(self, tmp_path: Path) -> None:
        content_list = tmp_path / "content_list.json"
        content_list.write_text('{"key": "value"}', encoding="utf-8")

        result = extract_ocr_from_content_list(content_list)

        assert result == {}

    def test_items_missing_page_idx_skipped(self, tmp_path: Path) -> None:
        items = [
            {"type": "text", "text": "No page_idx"},
            {"type": "text", "text": "Has page_idx", "page_idx": 2},
        ]
        content_list = tmp_path / "content_list.json"
        content_list.write_text(json.dumps(items), encoding="utf-8")

        result = extract_ocr_from_content_list(content_list)

        assert list(result.keys()) == [2]

    def test_multiple_pages(self, tmp_path: Path) -> None:
        items = [{"type": "text", "text": f"p{i}", "page_idx": i} for i in range(5)]
        content_list = tmp_path / "content_list.json"
        content_list.write_text(json.dumps(items), encoding="utf-8")

        result = extract_ocr_from_content_list(content_list)

        assert len(result) == 5
        for i in range(5):
            assert result[i] == f"p{i}"


# ---------------------------------------------------------------------------
# extract_ocr_for_lecture (orchestrator)
# ---------------------------------------------------------------------------


class TestExtractOcrForLecture:
    def _write_images(self, images_dir: Path, page_numbers: list[int]) -> None:
        """Create placeholder PNG files for given 1-based page numbers."""
        images_dir.mkdir(parents=True, exist_ok=True)
        for n in page_numbers:
            (images_dir / f"lecture01_page_{n:03d}.png").write_bytes(b"\x89PNG\r\n")

    def test_uses_content_list_when_provided(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        self._write_images(images_dir, [1, 2])

        items = [
            {"type": "text", "text": "Slide 1 text", "page_idx": 0},
            {"type": "text", "text": "Slide 2 text", "page_idx": 1},
        ]
        cl_path = tmp_path / "content_list.json"
        cl_path.write_text(json.dumps(items), encoding="utf-8")

        result = extract_ocr_for_lecture(images_dir, cl_path)

        assert result[1] == "Slide 1 text"
        assert result[2] == "Slide 2 text"

    def test_returns_1_based_page_numbers(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        self._write_images(images_dir, [1, 2, 3])

        items = [
            {"type": "text", "text": f"text_{i}", "page_idx": i}
            for i in range(3)
        ]
        cl_path = tmp_path / "content_list.json"
        cl_path.write_text(json.dumps(items), encoding="utf-8")

        result = extract_ocr_for_lecture(images_dir, cl_path)

        assert set(result.keys()) == {1, 2, 3}

    def test_missing_images_dir_returns_empty(self, tmp_path: Path) -> None:
        result = extract_ocr_for_lecture(tmp_path / "no_such_dir")
        assert result == {}

    def test_no_images_returns_empty(self, tmp_path: Path) -> None:
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        result = extract_ocr_for_lecture(images_dir)
        assert result == {}

    def test_no_content_list_falls_back_for_all_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without content_list, every page triggers the easyocr fallback."""
        images_dir = tmp_path / "images"
        self._write_images(images_dir, [1, 2])

        # Patch extract_ocr_from_image to avoid real easyocr call
        from rag.slideqa import ocr_extractor

        monkeypatch.setattr(
            ocr_extractor,
            "extract_ocr_from_image",
            lambda path: f"ocr:{path.name}",
        )

        result = extract_ocr_for_lecture(images_dir, content_list_path=None)

        assert set(result.keys()) == {1, 2}
        assert result[1] == "ocr:lecture01_page_001.png"
        assert result[2] == "ocr:lecture01_page_002.png"

    def test_partial_content_list_falls_back_for_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pages missing from content_list use easyocr fallback."""
        images_dir = tmp_path / "images"
        self._write_images(images_dir, [1, 2, 3])

        # content_list only has page 0 (-> page_number 1)
        items = [{"type": "text", "text": "content_list_text", "page_idx": 0}]
        cl_path = tmp_path / "content_list.json"
        cl_path.write_text(json.dumps(items), encoding="utf-8")

        from rag.slideqa import ocr_extractor

        monkeypatch.setattr(
            ocr_extractor,
            "extract_ocr_from_image",
            lambda path: f"fallback:{path.name}",
        )

        result = extract_ocr_for_lecture(images_dir, cl_path)

        assert result[1] == "content_list_text"
        assert result[2] == "fallback:lecture01_page_002.png"
        assert result[3] == "fallback:lecture01_page_003.png"
