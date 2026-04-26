"""
TDD — tests for chunk-level retrieval with page-level aggregation.

New behaviour:
  - slide_chunks table: (chunk_id, page_id, variant, text, vector)
    each chunk belongs to exactly one page_id (FK → slide_pages.page_id)
  - Retriever.retrieve() embeds the query, scores all chunks in the index,
    aggregates chunk scores to their parent page (max-pool by default),
    and returns top-k SlidePageResult sorted by aggregated score.

Aggregation options:
  - "max"  (default): page_score = max(chunk_scores for chunks on that page)
  - "sum":            page_score = sum(chunk_scores for chunks on that page)
  - "mean":           page_score = mean(chunk_scores for chunks on that page)

The existing page-level embedding path (slide_embeddings) is preserved as a
fallback when slide_chunks is empty for a given (variant, course_code).

Key invariants:
  - chunk_id is a TEXT primary key (e.g. UUID or "{page_id}_chunk_{n}")
  - A page with N chunks contributes up to N vectors to the matrix
  - After aggregation, each page_id appears exactly once in the result list
  - page_number (1-based) is carried through from slide_pages for eval offset
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_RAG_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from slideqa.retriever import Retriever, SlidePageResult

# ---------------------------------------------------------------------------
# DB schema helpers (matches what index_builder will create)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS slide_pages (
    page_id TEXT PRIMARY KEY,
    course_code TEXT NOT NULL,
    lecture_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    ocr_text TEXT NOT NULL DEFAULT '',
    caption TEXT,
    objects TEXT
);
CREATE TABLE IF NOT EXISTS slide_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id TEXT NOT NULL,
    variant TEXT NOT NULL CHECK(variant IN ('v1','v2','v3')),
    vector BLOB NOT NULL,
    FOREIGN KEY (page_id) REFERENCES slide_pages(page_id),
    UNIQUE(page_id, variant)
);
CREATE TABLE IF NOT EXISTS slide_chunks (
    chunk_id TEXT PRIMARY KEY,
    page_id  TEXT NOT NULL,
    variant  TEXT NOT NULL CHECK(variant IN ('v1','v2','v3')),
    chunk_text TEXT NOT NULL DEFAULT '',
    vector   BLOB NOT NULL,
    FOREIGN KEY (page_id) REFERENCES slide_pages(page_id)
);
"""

_DIM = 4


def _unit_vec(seed: int, dim: int = _DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _pack(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.tolist())


def _make_db(tmp_path: Path) -> Path:
    """Create an in-file SQLite DB with both page and chunk tables."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    conn.close()
    return db


def _insert_page(conn: sqlite3.Connection, page_id: str, page_number: int,
                 course_code: str = "CS288") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO slide_pages "
        "(page_id, course_code, lecture_id, page_number, image_path, ocr_text) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (page_id, course_code, "lec01", page_number, f"/img/{page_id}.png", "text"),
    )


def _insert_chunk(conn: sqlite3.Connection, chunk_id: str, page_id: str,
                  variant: str, vec: np.ndarray, text: str = "") -> None:
    conn.execute(
        "INSERT INTO slide_chunks (chunk_id, page_id, variant, chunk_text, vector) "
        "VALUES (?, ?, ?, ?, ?)",
        (chunk_id, page_id, variant, text, _pack(vec)),
    )


def _mock_retriever(db_path: Path, query_vec: np.ndarray) -> Retriever:
    """Return a Retriever whose model always returns query_vec for any input."""
    r = Retriever(db_path=db_path, model_name="fake-model")
    mock_model = MagicMock()
    mock_model.encode.return_value = query_vec
    r._model = mock_model
    return r


# ---------------------------------------------------------------------------
# A. Schema / index presence
# ---------------------------------------------------------------------------


class TestChunkTablePresence:
    def test_slide_chunks_table_created_by_init_db(self, tmp_path):
        """init_db must create slide_chunks alongside the existing tables."""
        from slideqa.index_builder import init_db
        db = tmp_path / "test.db"
        init_db(db)
        conn = sqlite3.connect(str(db))
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "slide_chunks" in tables

    def test_slide_chunks_has_required_columns(self, tmp_path):
        from slideqa.index_builder import init_db
        db = tmp_path / "test.db"
        init_db(db)
        conn = sqlite3.connect(str(db))
        cols = {row[1] for row in conn.execute("PRAGMA table_info(slide_chunks)").fetchall()}
        conn.close()
        assert {"chunk_id", "page_id", "variant", "chunk_text", "vector"} <= cols


# ---------------------------------------------------------------------------
# B. upsert_chunk_records (index_builder)
# ---------------------------------------------------------------------------


class TestUpsertChunkRecords:
    def test_inserts_chunk_rows(self, tmp_path):
        from slideqa.index_builder import init_db, upsert_chunk_records
        db = _make_db(tmp_path)
        conn = sqlite3.connect(str(db))
        _insert_page(conn, "CS288/lec01/page_001", 1)
        conn.commit(); conn.close()

        chunks = [
            {
                "chunk_id": "CS288/lec01/page_001_chunk_0",
                "page_id": "CS288/lec01/page_001",
                "variant": "v1",
                "chunk_text": "attention mechanism",
                "vector": _unit_vec(0),
            }
        ]
        count = upsert_chunk_records(db, chunks)
        assert count == 1

    def test_upsert_is_idempotent(self, tmp_path):
        from slideqa.index_builder import upsert_chunk_records
        db = _make_db(tmp_path)
        conn = sqlite3.connect(str(db))
        _insert_page(conn, "CS288/lec01/page_001", 1)
        conn.commit(); conn.close()

        chunk = {
            "chunk_id": "cid1",
            "page_id": "CS288/lec01/page_001",
            "variant": "v1",
            "chunk_text": "text",
            "vector": _unit_vec(0),
        }
        upsert_chunk_records(db, [chunk])
        count2 = upsert_chunk_records(db, [chunk])  # second upsert same chunk
        conn2 = sqlite3.connect(str(db))
        n = conn2.execute("SELECT COUNT(*) FROM slide_chunks").fetchone()[0]
        conn2.close()
        assert n == 1  # no duplicate row


# ---------------------------------------------------------------------------
# C. Retriever — chunk-level retrieval with page aggregation
# ---------------------------------------------------------------------------


class TestChunkRetrieval:
    def _setup_two_pages(self, tmp_path: Path):
        """
        Page 1 (page_number=1): two chunks — one pointing directly at query.
        Page 2 (page_number=2): one chunk — pointing away from query.
        Query is identical to page1_chunk0 → page 1 should rank #1.
        """
        db = _make_db(tmp_path)
        conn = sqlite3.connect(str(db))

        _insert_page(conn, "CS288/lec01/page_001", 1)
        _insert_page(conn, "CS288/lec01/page_002", 2)

        v_q   = _unit_vec(42)   # query vector
        v_c0  = v_q             # chunk 0 of page 1 ≡ query → score ≈ 1.0
        v_c1  = _unit_vec(7)    # chunk 1 of page 1 — different
        v_c2  = _unit_vec(99)   # chunk 0 of page 2 — different

        _insert_chunk(conn, "p1c0", "CS288/lec01/page_001", "v1", v_c0)
        _insert_chunk(conn, "p1c1", "CS288/lec01/page_001", "v1", v_c1)
        _insert_chunk(conn, "p2c0", "CS288/lec01/page_002", "v1", v_c2)
        conn.commit(); conn.close()
        return db, v_q

    def test_returns_slide_page_results(self, tmp_path):
        db, v_q = self._setup_two_pages(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("some query", index_variant="v1", course_code="CS288", top_k=2)
        assert all(isinstance(x, SlidePageResult) for x in results)

    def test_page1_ranks_first_when_chunk_matches_query(self, tmp_path):
        db, v_q = self._setup_two_pages(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("some query", index_variant="v1", course_code="CS288", top_k=2)
        assert results[0].page_number == 1

    def test_each_page_appears_at_most_once(self, tmp_path):
        db, v_q = self._setup_two_pages(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("some query", index_variant="v1", course_code="CS288", top_k=5)
        page_numbers = [x.page_number for x in results]
        assert len(page_numbers) == len(set(page_numbers))

    def test_top_k_limits_result_count(self, tmp_path):
        db, v_q = self._setup_two_pages(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", index_variant="v1", course_code="CS288", top_k=1)
        assert len(results) == 1

    def test_rank_field_is_1indexed(self, tmp_path):
        db, v_q = self._setup_two_pages(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", index_variant="v1", course_code="CS288", top_k=2)
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_score_is_max_of_chunk_scores(self, tmp_path):
        """With max-pool aggregation, page score = max chunk cosine score."""
        db, v_q = self._setup_two_pages(tmp_path)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", index_variant="v1", course_code="CS288", top_k=2)
        # Page 1's best chunk is v_q itself → cosine ≈ 1.0
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_course_code_filter_excludes_other_courses(self, tmp_path):
        db = _make_db(tmp_path)
        conn = sqlite3.connect(str(db))
        _insert_page(conn, "CS288/lec01/page_001", 1, course_code="CS288")
        _insert_page(conn, "CS61A/lec01/page_001", 1, course_code="CS61A")
        v_q = _unit_vec(1)
        _insert_chunk(conn, "c1", "CS288/lec01/page_001", "v1", v_q)
        _insert_chunk(conn, "c2", "CS61A/lec01/page_001", "v1", _unit_vec(2))
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", index_variant="v1", course_code="CS288", top_k=5)
        assert all(x.course_code == "CS288" for x in results)
        assert len(results) == 1

    def test_empty_db_returns_empty_list(self, tmp_path):
        db = _make_db(tmp_path)
        v_q = _unit_vec(0)
        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", index_variant="v1", course_code="CS288", top_k=5)
        assert results == []

    def test_aggregation_sum_different_from_max(self, tmp_path):
        """sum aggregation sums chunk scores; page with more chunks can win."""
        db = _make_db(tmp_path)
        conn = sqlite3.connect(str(db))
        _insert_page(conn, "p1", 1)
        _insert_page(conn, "p2", 2)
        v_q = _unit_vec(0)
        # p1 has 1 chunk perfectly matching query → max=1.0, sum=1.0
        # p2 has 3 chunks each scoring 0.6 → max=0.6, sum=1.8
        v_p1 = v_q
        v_p2a = _unit_vec(1); v_p2b = _unit_vec(2); v_p2c = _unit_vec(3)
        # Force p2 chunks to all score 0.6 by using a fixed dot product
        # (we just use arbitrary vecs and test that sum > max for p2)
        _insert_chunk(conn, "p1c0", "p1", "v1", v_p1)
        _insert_chunk(conn, "p2c0", "p2", "v1", v_p2a)
        _insert_chunk(conn, "p2c1", "p2", "v1", v_p2b)
        _insert_chunk(conn, "p2c2", "p2", "v1", v_p2c)
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results_max = r.retrieve("q", "v1", "CS288", top_k=2, chunk_agg="max")
        results_sum = r.retrieve("q", "v1", "CS288", top_k=2, chunk_agg="sum")

        # Max: p1 should win (cosine=1.0 vs ≤1.0)
        assert results_max[0].page_number == 1

        # Sum: verify results are lists of SlidePageResult and p2 has higher sum than max
        p2_max = next((r for r in results_max if r.page_number == 2), None)
        p2_sum = next((r for r in results_sum if r.page_number == 2), None)
        if p2_max and p2_sum:
            assert p2_sum.score > p2_max.score  # sum of 3 scores > max of 3 scores

    def test_falls_back_to_page_embeddings_when_no_chunks(self, tmp_path):
        """If slide_chunks is empty, retriever falls back to slide_embeddings."""
        db = _make_db(tmp_path)
        conn = sqlite3.connect(str(db))
        _insert_page(conn, "CS288/lec01/page_001", 1)
        v_q = _unit_vec(0)
        # Insert page-level embedding only (no chunks)
        conn.execute(
            "INSERT INTO slide_embeddings (page_id, variant, vector) VALUES (?, ?, ?)",
            ("CS288/lec01/page_001", "v1", _pack(v_q)),
        )
        conn.commit(); conn.close()

        r = _mock_retriever(db, v_q)
        results = r.retrieve("q", "v1", "CS288", top_k=5)
        assert len(results) == 1
        assert results[0].page_number == 1


# ---------------------------------------------------------------------------
# D. build_chunk_embeddings (index_builder)
# ---------------------------------------------------------------------------


class TestBuildChunkEmbeddings:
    def test_writes_chunk_rows_to_db(self, tmp_path):
        from slideqa.index_builder import init_db, build_chunk_embeddings

        db = tmp_path / "test.db"
        init_db(db)
        conn = sqlite3.connect(str(db))
        _insert_page(conn, "CS288/lec01/page_001", 1)
        conn.commit(); conn.close()

        chunks = [
            {
                "chunk_id": "CS288/lec01/page_001_chunk_0",
                "page_id": "CS288/lec01/page_001",
                "variant": "v1",
                "chunk_text": "The attention mechanism scales dot products.",
            },
            {
                "chunk_id": "CS288/lec01/page_001_chunk_1",
                "page_id": "CS288/lec01/page_001",
                "variant": "v1",
                "chunk_text": "Softmax normalises the scores.",
            },
        ]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([_unit_vec(0), _unit_vec(1)])

        with patch("slideqa.index_builder.SentenceTransformer", return_value=mock_model):
            count = build_chunk_embeddings(db, chunks, variant="v1")

        assert count == 2

        conn2 = sqlite3.connect(str(db))
        n = conn2.execute("SELECT COUNT(*) FROM slide_chunks").fetchone()[0]
        conn2.close()
        assert n == 2
