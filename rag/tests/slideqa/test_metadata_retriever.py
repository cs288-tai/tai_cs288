"""
TDD — tests for MetadataDBRetriever.

MetadataDBRetriever reads chunk embeddings directly from the existing
CS 288_metadata_new.db `chunks` table (no slideqa.db required).

Schema of `chunks`:
    chunk_uuid  TEXT PK
    file_uuid   TEXT
    idx         INTEGER
    text        TEXT
    file_path   TEXT        -- e.g. "CS 288/.../CS288_sp26_01_Intro.pdf"
    course_code TEXT        -- "CS 288"
    chunk_index INTEGER     -- 1-based page number within that PDF
    vector      BLOB        -- float32, dim=2560

Mapping to eval:
    page_number = chunk_index   (1-based)
    gold_page_ids are 0-based   → offset: chunk_index - 1 == gold_page_id

retrieve_fn contract:
    (query: str, variant: str, course_code: str | None, top_k: int)
    → list of objects with .page_number (1-based int)

The variant parameter is accepted but ignored (metadata DB has no variants).
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

_RAG_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from slideqa.metadata_retriever import MetadataDBRetriever, MetadataPageResult

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_DIM = 4


def _unit_vec(seed: int, dim: int = _DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _pack(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.tolist())


_SCHEMA = """
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


def _make_metadata_db(tmp_path: Path) -> Path:
    db = tmp_path / "metadata.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    return db


def _insert_chunk(
    conn: sqlite3.Connection,
    chunk_uuid: str,
    file_path: str,
    chunk_index: int,
    vec: np.ndarray,
    course_code: str = "CS 288",
    text: str = "",
) -> None:
    conn.execute(
        "INSERT INTO chunks (chunk_uuid, file_uuid, idx, text, file_path, course_code, chunk_index, vector) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (chunk_uuid, "file-uuid-1", chunk_index, text, file_path, course_code, chunk_index, _pack(vec)),
    )


def _mock_retriever(db_path: Path, query_vec: np.ndarray) -> MetadataDBRetriever:
    r = MetadataDBRetriever(db_path=db_path, model_name="fake-model")
    mock_model = MagicMock()
    mock_model.encode.return_value = query_vec
    r._model = mock_model
    return r


# ---------------------------------------------------------------------------
# A. MetadataPageResult
# ---------------------------------------------------------------------------


class TestMetadataPageResult:
    def test_has_page_number(self):
        result = MetadataPageResult(
            page_id="CS288_sp26_01_Intro.pdf_page_3",
            course_code="CS 288",
            lecture_id="CS288_sp26_01_Intro.pdf",
            page_number=3,
            image_path="",
            ocr_text="text",
            caption=None,
            objects=None,
            score=0.9,
            dense_score=0.9,
            bm25_score=0.0,
            rank=1,
        )
        assert result.page_number == 3

    def test_is_immutable(self):
        result = MetadataPageResult(
            page_id="p", course_code="CS 288", lecture_id="lec",
            page_number=1, image_path="", ocr_text="",
            caption=None, objects=None, score=0.5,
            dense_score=0.5, bm25_score=0.0, rank=1,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.page_number = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# B. Basic retrieval
# ---------------------------------------------------------------------------


class TestMetadataRetrieval:
    def _setup(self, tmp_path: Path):
        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        v_q = _unit_vec(42)
        _insert_chunk(conn, "c1", "CS 288/slides/CS288_sp26_01_Intro.pdf", 7, v_q)
        _insert_chunk(conn, "c2", "CS 288/slides/CS288_sp26_01_Intro.pdf", 10, _unit_vec(99))
        conn.commit(); conn.close()
        return db, v_q

    def test_returns_metadata_page_results(self, tmp_path):
        db, v_q = self._setup(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("query", index_variant="v1", course_code="CS 288", top_k=2)
        assert all(isinstance(x, MetadataPageResult) for x in results)

    def test_best_matching_chunk_ranks_first(self, tmp_path):
        db, v_q = self._setup(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("query", index_variant="v1", course_code="CS 288", top_k=2)
        assert results[0].page_number == 7  # chunk_index=7 (1-based) → page_number=7

    def test_top_k_limits_results(self, tmp_path):
        db, v_q = self._setup(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("query", index_variant="v1", course_code="CS 288", top_k=1)
        assert len(results) == 1

    def test_rank_is_1indexed(self, tmp_path):
        db, v_q = self._setup(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("query", index_variant="v1", course_code="CS 288", top_k=2)
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_empty_db_returns_empty(self, tmp_path):
        db = _make_metadata_db(tmp_path)
        r = _mock_retriever(db, _unit_vec(0))
        results = r.retrieve("query", index_variant="v1", course_code="CS 288", top_k=5)
        assert results == []

    def test_score_approx_1_for_identical_vec(self, tmp_path):
        db, v_q = self._setup(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("query", index_variant="v1", course_code="CS 288", top_k=2)
        assert results[0].score == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# C. Page deduplication (multiple chunks per page → one result per page)
# ---------------------------------------------------------------------------


class TestPageDeduplication:
    def test_same_page_appears_once(self, tmp_path):
        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        v_q = _unit_vec(1)
        # Two chunks on page 3 of the same lecture
        _insert_chunk(conn, "c1", "CS 288/slides/CS288_sp26_01_Intro.pdf", 3, v_q)
        _insert_chunk(conn, "c2", "CS 288/slides/CS288_sp26_01_Intro.pdf", 3, _unit_vec(2))
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", "v1", "CS 288", top_k=5)
        page_numbers = [x.page_number for x in results]
        assert len(page_numbers) == len(set(page_numbers))
        assert len(results) == 1  # only one unique page

    def test_chunks_from_different_pages_give_multiple_results(self, tmp_path):
        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        v_q = _unit_vec(1)
        _insert_chunk(conn, "c1", "CS 288/slides/lec.pdf", 1, v_q)
        _insert_chunk(conn, "c2", "CS 288/slides/lec.pdf", 2, _unit_vec(2))
        _insert_chunk(conn, "c3", "CS 288/slides/lec.pdf", 3, _unit_vec(3))
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", "v1", "CS 288", top_k=5)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# D. lecture_id and page_number mapping
# ---------------------------------------------------------------------------


class TestLecturePageMapping:
    def test_lecture_id_is_filename(self, tmp_path):
        """lecture_id should be the PDF filename (last path component)."""
        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        v_q = _unit_vec(5)
        _insert_chunk(conn, "c1", "CS 288/course_website/sp26/assets/slides/CS288_sp26_07_Transformers.pdf", 12, v_q)
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", "v1", "CS 288", top_k=1)
        assert results[0].lecture_id == "CS288_sp26_07_Transformers.pdf"

    def test_page_number_equals_chunk_index(self, tmp_path):
        """page_number == chunk_index because chunk_index is 1-based in the metadata DB.
        eval.py checks (page_number - 1) in gold_set, where gold_page_ids are 0-based.
        chunk_index=6 (1-based) → page_number=6 → eval: (6-1)=5 in gold_set.
        """
        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        v_q = _unit_vec(7)
        _insert_chunk(conn, "c1", "CS 288/slides/lec.pdf", 6, v_q)
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", "v1", "CS 288", top_k=1)
        # chunk_index=6 (1-based) → page_number=6; eval: (6-1)=5 == gold_page_id=5
        assert results[0].page_number == 6

    def test_ocr_text_comes_from_chunk_text(self, tmp_path):
        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        v_q = _unit_vec(3)
        _insert_chunk(conn, "c1", "CS 288/slides/lec.pdf", 1, v_q, text="attention is all you need")
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", "v1", "CS 288", top_k=1)
        assert "attention" in results[0].ocr_text


# ---------------------------------------------------------------------------
# E. course_code filter
# ---------------------------------------------------------------------------


class TestCourseCodeFilter:
    def test_other_course_excluded(self, tmp_path):
        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        v_q = _unit_vec(0)
        _insert_chunk(conn, "c1", "CS 288/slides/lec.pdf", 1, v_q, course_code="CS 288")
        _insert_chunk(conn, "c2", "OTHER/slides/lec.pdf", 1, v_q, course_code="CS 61A")
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", "v1", "CS 288", top_k=5)
        assert all(x.course_code == "CS 288" for x in results)
        assert len(results) == 1

    def test_none_course_code_returns_all(self, tmp_path):
        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        v_q = _unit_vec(0)
        _insert_chunk(conn, "c1", "CS 288/slides/lec.pdf", 1, v_q, course_code="CS 288")
        _insert_chunk(conn, "c2", "OTHER/slides/lec.pdf", 2, v_q, course_code="CS 61A")
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", "v1", None, top_k=5)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# F. retrieve_fn compatibility
# ---------------------------------------------------------------------------


class TestRetrieveFnCompat:
    """Verify the retrieve method can be used as retrieve_fn in eval pipeline."""

    def test_retrieve_accepts_variant_parameter(self, tmp_path):
        """variant param is accepted (metadata DB has no variants; it's ignored)."""
        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        _insert_chunk(conn, "c1", "CS 288/slides/lec.pdf", 1, _unit_vec(0))
        conn.commit(); conn.close()

        r = _mock_retriever(db, _unit_vec(0))
        # Should not raise even though variants don't exist in this DB
        results_v1 = r.retrieve("q", "v1", "CS 288", 5)
        results_v2 = r.retrieve("q", "v2", "CS 288", 5)
        results_v3 = r.retrieve("q", "v3", "CS 288", 5)
        # All should return same results (variant ignored)
        assert len(results_v1) == len(results_v2) == len(results_v3) == 1

    def test_page_number_offset_matches_gold_page_id_convention(self, tmp_path):
        """
        chunk_index is 1-based in the metadata DB.
        gold_page_ids are 0-based (MinerU page_idx).
        eval.py checks: (page_number - 1) in gold_set
        So page_number = chunk_index, giving (chunk_index - 1) == gold_page_id.
        Example: chunk_index=7 (1-based, the 7th slide) → page_number=7
                 gold_page_id=6 (0-based, the 7th slide) → (7-1)=6 ✓
        """
        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        v_q = _unit_vec(1)
        _insert_chunk(conn, "c1", "CS 288/slides/lec.pdf", 7, v_q)  # chunk_index=7 (1-based)
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", "v1", "CS 288", top_k=1)
        pn = results[0].page_number  # should be 7
        gold_page_id = pn - 1        # eval.py offset → should be 6
        assert gold_page_id == 6

    def test_query_prefix_forwarded_to_model_encode(self, tmp_path):
        """retrieve() must prepend the Qwen3 instruction prefix before encoding."""
        from slideqa.metadata_retriever import _QUERY_PREFIX

        db = _make_metadata_db(tmp_path)
        conn = sqlite3.connect(str(db))
        v_q = _unit_vec(0)
        _insert_chunk(conn, "c1", "CS 288/slides/lec.pdf", 1, v_q)
        conn.commit(); conn.close()

        r = MetadataDBRetriever(db_path=db, model_name="fake-model")
        mock_model = MagicMock()
        mock_model.encode.return_value = v_q
        r._model = mock_model

        r.retrieve("my question", index_variant="v1", course_code="CS 288", top_k=1)

        call_arg = mock_model.encode.call_args[0][0]
        assert call_arg.startswith(_QUERY_PREFIX), (
            f"Expected encode() called with prefix, got: {call_arg!r}"
        )
        assert "my question" in call_arg
