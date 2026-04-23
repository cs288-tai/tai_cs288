"""
Tests for PdfConverter variant-markdown materialization and QA hook wiring.

Two concerns are tested here:

1. _materialize_variant_markdowns — filesystem integration test.
   Calls the real method on a real temp file and asserts that exactly three
   variant files (.v1.md, .v2.md, .v3.md) are written to disk.

2. TestToMarkdownQAHook — unit tests (all heavy deps mocked) confirming that
   _to_markdown calls generate_slideqa_for_lecture once per variant.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Make the rag package importable when running from repo root.
# ---------------------------------------------------------------------------

_RAG_ROOT = Path(__file__).resolve().parents[2]  # rag/
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from file_conversion_router.conversion.pdf_converter import PdfConverter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def converter(tmp_path):
    """A minimal PdfConverter instance — no real config needed."""
    cv = PdfConverter.__new__(PdfConverter)
    # Attributes required by BaseConverter / PdfConverter internals.
    cv.course_name = "CS 288"
    cv.course_code = "CS288"
    cv.input_path = tmp_path / "slides.pdf"
    cv.output_path = tmp_path / "out"
    cv.output_path.mkdir()
    return cv


@pytest.fixture()
def fake_md(tmp_path):
    """A placeholder markdown file so _resolve_markdown_path returns it."""
    md = tmp_path / "slides.md"
    md.write_text("# Slide content\n", encoding="utf-8")
    return md


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMaterializeVariantMarkdowns:
    """Filesystem integration tests for _materialize_variant_markdowns.

    Calls the real method — no mocking — and asserts all three variant files
    are actually written to disk with the correct names and non-empty content.
    """

    def _make_converter(self, tmp_path):
        cv = PdfConverter.__new__(PdfConverter)
        cv.course_name = "CS 288"
        cv.course_code = "CS288"
        return cv

    def test_creates_exactly_three_variant_files(self, tmp_path):
        """Exactly v1, v2, v3 markdown files must be written to disk."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\nSome text.\n", encoding="utf-8")

        cv = self._make_converter(tmp_path)
        result = cv._materialize_variant_markdowns(md)

        assert set(result.keys()) == {"v1", "v2", "v3"}

    def test_variant_files_exist_on_disk(self, tmp_path):
        """All three variant paths returned must exist as real files."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\nSome text.\n", encoding="utf-8")

        cv = self._make_converter(tmp_path)
        result = cv._materialize_variant_markdowns(md)

        for variant, path in result.items():
            assert path.exists(), f"{variant} file not found on disk: {path}"

    def test_variant_files_have_correct_names(self, tmp_path):
        """Each variant file must be named <stem>.<variant>.md."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\nSome text.\n", encoding="utf-8")

        cv = self._make_converter(tmp_path)
        result = cv._materialize_variant_markdowns(md)

        assert result["v1"].name == "lecture01.v1.md"
        assert result["v2"].name == "lecture01.v2.md"
        assert result["v3"].name == "lecture01.v3.md"

    def test_variant_files_are_non_empty(self, tmp_path):
        """Each variant file must contain non-empty content."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\nSome text.\n", encoding="utf-8")

        cv = self._make_converter(tmp_path)
        result = cv._materialize_variant_markdowns(md)

        for variant, path in result.items():
            content = path.read_text(encoding="utf-8")
            assert content.strip(), f"{variant} file is empty"

    def test_master_file_written_alongside_variants(self, tmp_path):
        """A .master.md copy of the original is also written next to variants."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\nSome text.\n", encoding="utf-8")

        cv = self._make_converter(tmp_path)
        cv._materialize_variant_markdowns(md)

        master = tmp_path / "lecture01.master.md"
        assert master.exists(), "master.md not created"

    def test_returns_dict_not_single_path(self, tmp_path):
        """The return value must be a dict, not a single Path."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\nSome text.\n", encoding="utf-8")

        cv = self._make_converter(tmp_path)
        result = cv._materialize_variant_markdowns(md)

        assert isinstance(result, dict), (
            f"Expected dict, got {type(result).__name__}. "
            "Only one path is being generated instead of three."
        )
        assert len(result) == 3, (
            f"Expected 3 variants, got {len(result)}: {list(result.keys())}"
        )


class TestToMarkdownQAHook:
    """Verify that _to_markdown wires generate_slideqa_for_lecture correctly."""

    def _run_to_markdown(self, converter, fake_md, variant_paths):
        """Patch all heavy methods and call _to_markdown; return the mock."""
        with (
            patch.object(
                type(converter),
                "_resolve_markdown_path",
                return_value=fake_md,
            ),
            patch(
                "file_conversion_router.conversion.pdf_converter.convert_pdf_to_md_by_MinerU",
                return_value={"output_file": str(fake_md)},
            ),
            patch.object(converter, "replace_images_with_vlm_descriptions"),
            patch.object(converter, "clean_markdown_content"),
            patch.object(
                converter,
                "_materialize_variant_markdowns",
                return_value=variant_paths,
            ),
            patch.object(converter, "generate_slideqa_for_lecture") as mock_qa,
        ):
            pdf_path = fake_md.with_suffix(".pdf")
            pdf_path.write_bytes(b"%PDF-1.4")  # minimal sentinel so exists() is True
            converter._to_markdown(pdf_path, fake_md.parent)
            return mock_qa

    def test_qa_hook_called_for_all_three_variants(self, converter, fake_md, tmp_path):
        """generate_slideqa_for_lecture is called once for each of v1/v2/v3."""
        v1 = tmp_path / "slides.v1.md"
        v2 = tmp_path / "slides.v2.md"
        v3 = tmp_path / "slides.v3.md"
        for p in (v1, v2, v3):
            p.write_text("", encoding="utf-8")

        variant_paths = {"v1": v1, "v2": v2, "v3": v3}
        mock_qa = self._run_to_markdown(converter, fake_md, variant_paths)

        assert mock_qa.call_count == 3

    def test_qa_hook_receives_correct_variant_md_paths(self, converter, fake_md, tmp_path):
        """Each call to generate_slideqa_for_lecture gets the right variant path."""
        v1 = tmp_path / "slides.v1.md"
        v2 = tmp_path / "slides.v2.md"
        v3 = tmp_path / "slides.v3.md"
        for p in (v1, v2, v3):
            p.write_text("", encoding="utf-8")

        variant_paths = {"v1": v1, "v2": v2, "v3": v3}
        mock_qa = self._run_to_markdown(converter, fake_md, variant_paths)

        called_variant_paths = {c.args[0] for c in mock_qa.call_args_list}
        assert called_variant_paths == {v1, v2, v3}

    def test_qa_hook_receives_correct_variant_strings(self, converter, fake_md, tmp_path):
        """Each call carries the matching variant string (v1/v2/v3)."""
        v1 = tmp_path / "slides.v1.md"
        v2 = tmp_path / "slides.v2.md"
        v3 = tmp_path / "slides.v3.md"
        for p in (v1, v2, v3):
            p.write_text("", encoding="utf-8")

        variant_paths = {"v1": v1, "v2": v2, "v3": v3}
        mock_qa = self._run_to_markdown(converter, fake_md, variant_paths)

        called_variants = {c.args[1] for c in mock_qa.call_args_list}
        assert called_variants == {"v1", "v2", "v3"}

    def test_qa_hook_receives_content_list_path(self, converter, fake_md, tmp_path):
        """The content_list_path passed to the hook is derived from the master md stem."""
        v1 = tmp_path / "slides.v1.md"
        v1.write_text("", encoding="utf-8")
        variant_paths = {"v1": v1}

        mock_qa = self._run_to_markdown(converter, fake_md, variant_paths)

        expected_content_list = fake_md.with_name(f"{fake_md.stem}_content_list.json")
        called_content_list_paths = {c.args[2] for c in mock_qa.call_args_list}
        assert expected_content_list in called_content_list_paths

    def test_returns_master_md_path(self, converter, fake_md, tmp_path):
        """_to_markdown must return the master markdown path (contract with BaseConverter)."""
        v1 = tmp_path / "slides.v1.md"
        v1.write_text("", encoding="utf-8")
        variant_paths = {"v1": v1}

        with (
            patch.object(
                type(converter),
                "_resolve_markdown_path",
                return_value=fake_md,
            ),
            patch(
                "file_conversion_router.conversion.pdf_converter.convert_pdf_to_md_by_MinerU",
                return_value={"output_file": str(fake_md)},
            ),
            patch.object(converter, "replace_images_with_vlm_descriptions"),
            patch.object(converter, "clean_markdown_content"),
            patch.object(
                converter,
                "_materialize_variant_markdowns",
                return_value=variant_paths,
            ),
            patch.object(converter, "generate_slideqa_for_lecture"),
        ):
            pdf_path = fake_md.with_suffix(".pdf")
            pdf_path.write_bytes(b"%PDF-1.4")
            result = converter._to_markdown(pdf_path, fake_md.parent)

        assert result == fake_md

    def test_qa_not_called_when_no_variants(self, converter, fake_md, tmp_path):
        """If _materialize_variant_markdowns returns empty dict, no QA is attempted."""
        with (
            patch.object(
                type(converter),
                "_resolve_markdown_path",
                return_value=fake_md,
            ),
            patch(
                "file_conversion_router.conversion.pdf_converter.convert_pdf_to_md_by_MinerU",
                return_value={"output_file": str(fake_md)},
            ),
            patch.object(converter, "replace_images_with_vlm_descriptions"),
            patch.object(converter, "clean_markdown_content"),
            patch.object(
                converter,
                "_materialize_variant_markdowns",
                return_value={},
            ),
            patch.object(converter, "generate_slideqa_for_lecture") as mock_qa,
        ):
            pdf_path = fake_md.with_suffix(".pdf")
            pdf_path.write_bytes(b"%PDF-1.4")
            converter._to_markdown(pdf_path, fake_md.parent)

        mock_qa.assert_not_called()
