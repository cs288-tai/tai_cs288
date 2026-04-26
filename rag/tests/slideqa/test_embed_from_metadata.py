"""
TDD — tests for embed_from_metadata.py

This module reads slide chunks from CS 288_metadata_new.db, extracts
per-variant text (v1/v2/v3), encodes with Qwen/Qwen3-Embedding-4B,
and writes to a slideqa.db slide_chunks table.

Text extraction rules (from chunk `text` field):
  v1 = strip all <!-- TAI_VLM_BEGIN...TAI_VLM_END --> blocks + image refs  (OCR only)
  v2 = v1 base + content inside <!-- TAI_V2_BEGIN...TAI_V2_END --> (image captions)
  v3 = v2 + content inside <!-- TAI_V3_BEGIN...TAI_V3_END --> (visual notes)

Each (file_path, chunk_index, variant) produces one slide_chunks row:
    chunk_id  = "{lecture_filename}_page_{chunk_index}_chunk_{db_idx}_{variant}"
    page_id   = "{lecture_filename}_page_{chunk_index}"
    variant   = "v1" | "v2" | "v3"
    chunk_text = extracted text for this variant
    vector    = float32 BLOB from Qwen3-Embedding-4B

page_number = chunk_index + 1  (1-based, for eval.py offset math)

slide_pages rows are also created with one row per unique (file_path, chunk_index):
    page_id      = "{lecture_filename}_page_{chunk_index}"
    course_code  = chunks.course_code
    lecture_id   = lecture_filename (basename of file_path)
    page_number  = chunk_index + 1
    image_path   = ""
    ocr_text     = v1 extracted text
"""
from __future__ import annotations

import re
import sqlite3
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_RAG_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from slideqa.embed_from_metadata import (
    extract_v1,
    extract_v2,
    extract_v3,
    build_slideqa_db_from_metadata,
    CHUNK_ID_TEMPLATE,
    PAGE_ID_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Fixture: a rich chunk text with all three variant markers
# ---------------------------------------------------------------------------

_CHUNK_TEXT_FULL = """\
CS 288 Spring 2026

## Course Instructors

![](images/abc123.jpg)
<!-- TAI_VLM_BEGIN image="images/abc123.jpg" -->
<!-- TAI_V2_BEGIN -->
[Image Caption]
A headshot of a professor in a blue background.
<!-- TAI_V2_END -->

<!-- TAI_V3_BEGIN -->
[Visual Notes]
- Portrait photo, head-and-shoulders
- Blue background, professional look
<!-- TAI_V3_END -->
<!-- TAI_VLM_END -->

### Sewon Min
"""

_CHUNK_TEXT_NO_VLM = """\
• NLP = building computer systems

### Language is an Interface
"""

# ---------------------------------------------------------------------------
# A. Text extraction functions
# ---------------------------------------------------------------------------


class TestExtractV1:
    def test_strips_vlm_blocks(self):
        text = extract_v1(_CHUNK_TEXT_FULL)
        assert "TAI_VLM_BEGIN" not in text
        assert "TAI_V2_BEGIN" not in text
        assert "TAI_V3_BEGIN" not in text
        assert "Image Caption" not in text
        assert "Visual Notes" not in text

    def test_strips_image_markdown(self):
        text = extract_v1(_CHUNK_TEXT_FULL)
        assert "![" not in text

    def test_keeps_plain_text(self):
        text = extract_v1(_CHUNK_TEXT_FULL)
        assert "CS 288" in text
        assert "Sewon Min" in text

    def test_no_vlm_block_passes_through(self):
        text = extract_v1(_CHUNK_TEXT_NO_VLM)
        assert "NLP" in text
        assert "Language is an Interface" in text

    def test_returns_string(self):
        assert isinstance(extract_v1(_CHUNK_TEXT_FULL), str)

    def test_empty_input_returns_empty(self):
        assert extract_v1("").strip() == ""


class TestExtractV2:
    def test_includes_image_caption(self):
        text = extract_v2(_CHUNK_TEXT_FULL)
        assert "headshot of a professor" in text

    def test_strips_v3_visual_notes(self):
        text = extract_v2(_CHUNK_TEXT_FULL)
        assert "Portrait photo, head-and-shoulders" not in text
        assert "TAI_V3_BEGIN" not in text

    def test_keeps_base_ocr(self):
        text = extract_v2(_CHUNK_TEXT_FULL)
        assert "CS 288" in text
        assert "Sewon Min" in text

    def test_strips_image_markdown(self):
        text = extract_v2(_CHUNK_TEXT_FULL)
        assert "![" not in text

    def test_no_vlm_same_as_v1(self):
        assert extract_v2(_CHUNK_TEXT_NO_VLM).strip() == extract_v1(_CHUNK_TEXT_NO_VLM).strip()


class TestExtractV3:
    def test_includes_image_caption(self):
        text = extract_v3(_CHUNK_TEXT_FULL)
        assert "headshot of a professor" in text

    def test_includes_visual_notes(self):
        text = extract_v3(_CHUNK_TEXT_FULL)
        assert "Portrait photo, head-and-shoulders" in text
        assert "Blue background" in text

    def test_keeps_base_ocr(self):
        text = extract_v3(_CHUNK_TEXT_FULL)
        assert "CS 288" in text
        assert "Sewon Min" in text

    def test_strips_image_markdown(self):
        text = extract_v3(_CHUNK_TEXT_FULL)
        assert "![" not in text

    def test_no_vlm_same_as_v1(self):
        assert extract_v3(_CHUNK_TEXT_NO_VLM).strip() == extract_v1(_CHUNK_TEXT_NO_VLM).strip()

    def test_v3_longer_than_v2_when_vlm_present(self):
        assert len(extract_v3(_CHUNK_TEXT_FULL)) > len(extract_v2(_CHUNK_TEXT_FULL))


# ---------------------------------------------------------------------------
# B. ID templates
# ---------------------------------------------------------------------------


class TestIdTemplates:
    def test_chunk_id_contains_lecture_page_variant(self):
        chunk_id = CHUNK_ID_TEMPLATE.format(
            lecture="CS288_sp26_01_Intro.pdf",
            chunk_index=6,
            db_idx=8,
            variant="v1",
        )
        assert "CS288_sp26_01_Intro" in chunk_id
        assert "6" in chunk_id
        assert "v1" in chunk_id

    def test_page_id_contains_lecture_and_index(self):
        page_id = PAGE_ID_TEMPLATE.format(
            lecture="CS288_sp26_01_Intro.pdf",
            chunk_index=6,
        )
        assert "CS288_sp26_01_Intro" in page_id
        assert "6" in page_id


# ---------------------------------------------------------------------------
# C. build_slideqa_db_from_metadata — integration (with mocked model)
# ---------------------------------------------------------------------------

_SOURCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_uuid  TEXT PRIMARY KEY,
    file_uuid   TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    text        TEXT NOT NULL DEFAULT '',
    file_path   TEXT,
    course_code TEXT,
    chunk_index INTEGER,
    vector      BLOB
);
"""

_DIM = 4


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_source_db(tmp_path: Path, rows: list[dict]) -> Path:
    db = tmp_path / "source.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_SOURCE_SCHEMA)
    for r in rows:
        conn.execute(
            "INSERT INTO chunks (chunk_uuid, file_uuid, idx, text, file_path, course_code, chunk_index) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (r["chunk_uuid"], "fu1", r.get("idx", 1), r.get("text", ""), r["file_path"], r.get("course_code", "CS 288"), r["chunk_index"]),
        )
    conn.commit(); conn.close()
    return db


def _make_mock_model(n_outputs: int) -> MagicMock:
    mock = MagicMock()
    vecs = np.stack([_unit_vec(i) for i in range(n_outputs)])
    mock.encode.return_value = vecs
    return mock


class TestBuildSlideqaDbFromMetadata:
    def _two_page_source(self, tmp_path: Path) -> Path:
        return _make_source_db(tmp_path, [
            {
                "chunk_uuid": "cid-1",
                "file_path": "CS 288/slides/CS288_sp26_01_Intro.pdf",
                "chunk_index": 0,
                "idx": 0,
                "text": _CHUNK_TEXT_FULL,
                "course_code": "CS 288",
            },
            {
                "chunk_uuid": "cid-2",
                "file_path": "CS 288/slides/CS288_sp26_01_Intro.pdf",
                "chunk_index": 5,
                "idx": 5,
                "text": _CHUNK_TEXT_NO_VLM,
                "course_code": "CS 288",
            },
        ])

    def test_creates_output_db(self, tmp_path):
        src = self._two_page_source(tmp_path)
        out = tmp_path / "slideqa.db"
        mock_model = _make_mock_model(6)  # 2 pages × 3 variants
        with patch("slideqa.embed_from_metadata.SentenceTransformer", return_value=mock_model):
            build_slideqa_db_from_metadata(src, out)
        assert out.exists()

    def test_slide_chunks_has_three_variants_per_page(self, tmp_path):
        src = self._two_page_source(tmp_path)
        out = tmp_path / "slideqa.db"
        mock_model = _make_mock_model(6)
        with patch("slideqa.embed_from_metadata.SentenceTransformer", return_value=mock_model):
            build_slideqa_db_from_metadata(src, out)
        conn = sqlite3.connect(str(out))
        count = conn.execute("SELECT COUNT(*) FROM slide_chunks").fetchone()[0]
        conn.close()
        assert count == 6  # 2 pages × 3 variants

    def test_slide_pages_has_one_row_per_unique_page(self, tmp_path):
        src = self._two_page_source(tmp_path)
        out = tmp_path / "slideqa.db"
        mock_model = _make_mock_model(6)
        with patch("slideqa.embed_from_metadata.SentenceTransformer", return_value=mock_model):
            build_slideqa_db_from_metadata(src, out)
        conn = sqlite3.connect(str(out))
        count = conn.execute("SELECT COUNT(*) FROM slide_pages").fetchone()[0]
        conn.close()
        assert count == 2  # 2 unique pages

    def test_page_number_is_chunk_index_plus_one(self, tmp_path):
        src = self._two_page_source(tmp_path)
        out = tmp_path / "slideqa.db"
        mock_model = _make_mock_model(6)
        with patch("slideqa.embed_from_metadata.SentenceTransformer", return_value=mock_model):
            build_slideqa_db_from_metadata(src, out)
        conn = sqlite3.connect(str(out))
        rows = conn.execute("SELECT chunk_index, page_number FROM slide_pages ORDER BY chunk_index").fetchall()
        conn.close()
        for chunk_index, page_number in rows:
            assert page_number == chunk_index + 1

    def test_vector_stored_as_float32_blob(self, tmp_path):
        src = self._two_page_source(tmp_path)
        out = tmp_path / "slideqa.db"
        mock_model = _make_mock_model(6)
        with patch("slideqa.embed_from_metadata.SentenceTransformer", return_value=mock_model):
            build_slideqa_db_from_metadata(src, out)
        conn = sqlite3.connect(str(out))
        blob = conn.execute("SELECT vector FROM slide_chunks LIMIT 1").fetchone()[0]
        conn.close()
        assert isinstance(blob, (bytes, bytearray))
        assert len(blob) % 4 == 0
        n_floats = len(blob) // 4
        vals = struct.unpack(f"{n_floats}f", blob)
        assert all(isinstance(v, float) for v in vals)

    def test_v1_chunk_text_has_no_vlm_markers(self, tmp_path):
        src = self._two_page_source(tmp_path)
        out = tmp_path / "slideqa.db"
        mock_model = _make_mock_model(6)
        with patch("slideqa.embed_from_metadata.SentenceTransformer", return_value=mock_model):
            build_slideqa_db_from_metadata(src, out)
        conn = sqlite3.connect(str(out))
        row = conn.execute(
            "SELECT chunk_text FROM slide_chunks WHERE variant='v1' LIMIT 1"
        ).fetchone()
        conn.close()
        assert "TAI_VLM" not in row[0]
        assert "TAI_V2" not in row[0]

    def test_v3_chunk_text_longer_than_v1_for_vlm_page(self, tmp_path):
        src = self._two_page_source(tmp_path)
        out = tmp_path / "slideqa.db"
        mock_model = _make_mock_model(6)
        with patch("slideqa.embed_from_metadata.SentenceTransformer", return_value=mock_model):
            build_slideqa_db_from_metadata(src, out)
        conn = sqlite3.connect(str(out))
        # Get texts for the page that has VLM content (chunk_index=0)
        rows = {r[0]: r[1] for r in conn.execute(
            "SELECT sc.variant, sc.chunk_text FROM slide_chunks sc "
            "JOIN slide_pages sp ON sc.page_id = sp.page_id "
            "WHERE sp.chunk_index = 0"
        ).fetchall()}
        conn.close()
        assert len(rows["v3"]) > len(rows["v1"])

    def test_idempotent_second_run(self, tmp_path):
        src = self._two_page_source(tmp_path)
        out = tmp_path / "slideqa.db"
        mock_model = _make_mock_model(6)
        with patch("slideqa.embed_from_metadata.SentenceTransformer", return_value=mock_model):
            build_slideqa_db_from_metadata(src, out)
            build_slideqa_db_from_metadata(src, out)  # second run — no duplicates
        conn = sqlite3.connect(str(out))
        count = conn.execute("SELECT COUNT(*) FROM slide_chunks").fetchone()[0]
        conn.close()
        assert count == 6  # still 6, not 12

    def test_course_code_filter(self, tmp_path):
        src = _make_source_db(tmp_path, [
            {"chunk_uuid": "c1", "file_path": "CS 288/slides/lec.pdf", "chunk_index": 0,
             "text": "NLP text", "course_code": "CS 288"},
            {"chunk_uuid": "c2", "file_path": "OTHER/slides/lec.pdf", "chunk_index": 0,
             "text": "other text", "course_code": "CS 61A"},
        ])
        out = tmp_path / "slideqa.db"
        mock_model = _make_mock_model(3)
        with patch("slideqa.embed_from_metadata.SentenceTransformer", return_value=mock_model):
            build_slideqa_db_from_metadata(src, out, course_code="CS 288")
        conn = sqlite3.connect(str(out))
        pages = conn.execute("SELECT COUNT(*) FROM slide_pages").fetchone()[0]
        conn.close()
        assert pages == 1  # only CS 288 page

    def test_slides_only_filter(self, tmp_path):
        """Chunks not under .../slides/... are excluded (assignments, course website, etc.)"""
        src = _make_source_db(tmp_path, [
            {"chunk_uuid": "c1", "file_path": "CS 288/slides/lec.pdf", "chunk_index": 0,
             "text": "slide text", "course_code": "CS 288"},
            {"chunk_uuid": "c2", "file_path": "CS 288/assignments/hw1.pdf", "chunk_index": 0,
             "text": "hw text", "course_code": "CS 288"},
        ])
        out = tmp_path / "slideqa.db"
        mock_model = _make_mock_model(3)
        with patch("slideqa.embed_from_metadata.SentenceTransformer", return_value=mock_model):
            build_slideqa_db_from_metadata(src, out)
        conn = sqlite3.connect(str(out))
        pages = conn.execute("SELECT COUNT(*) FROM slide_pages").fetchone()[0]
        conn.close()
        assert pages == 1  # only slide page
