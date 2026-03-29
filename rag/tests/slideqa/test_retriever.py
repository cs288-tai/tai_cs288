"""
Tests for rag.slideqa.retriever.

All unit tests use mocked embedding models and in-memory SQLite databases
so no real model downloads or GPU are required.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
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
    variant TEXT NOT NULL CHECK(variant IN ('v1', 'v2', 'v3')),
    vector BLOB NOT NULL,
    FOREIGN KEY (page_id) REFERENCES slide_pages(page_id),
    UNIQUE(page_id, variant)
);
"""

_DIM = 4  # small fixed dimension for test vectors


def _make_unit_vec(seed: int, dim: int = _DIM) -> np.ndarray:
    """Return a deterministic L2-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _pack_vec(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.tolist())


def _build_db(
    db_path: Path,
    pages: list[dict[str, Any]],
    variant: str = "v1",
) -> None:
    """Create an SQLite DB with slide_pages + slide_embeddings rows."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA_SQL)
    for p in pages:
        conn.execute(
            "INSERT INTO slide_pages "
            "(page_id, course_code, lecture_id, page_number, image_path, ocr_text, caption, objects) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                p["page_id"],
                p["course_code"],
                p["lecture_id"],
                p["page_number"],
                p.get("image_path", "/img.png"),
                p.get("ocr_text", ""),
                p.get("caption"),
                p.get("objects"),
            ),
        )
        conn.execute(
            "INSERT INTO slide_embeddings (page_id, variant, vector) VALUES (?, ?, ?)",
            (p["page_id"], variant, _pack_vec(p["vector"])),
        )
    conn.commit()
    conn.close()


def _default_pages(n: int = 5, course_code: str = "CS101") -> list[dict[str, Any]]:
    return [
        {
            "page_id": f"{course_code}/lec01/page_{i:03d}",
            "course_code": course_code,
            "lecture_id": "lec01",
            "page_number": i,
            "image_path": f"/img_{i}.png",
            "ocr_text": f"slide content {i}",
            "caption": None,
            "objects": None,
            "vector": _make_unit_vec(i),
        }
        for i in range(1, n + 1)
    ]


def _mock_model(query_vec: np.ndarray) -> MagicMock:
    """Return a mock SentenceTransformer that encodes to query_vec."""
    model = MagicMock()
    model.encode.return_value = query_vec.reshape(1, -1)
    return model


# ---------------------------------------------------------------------------
# Test 1: retrieve returns exactly top_k results
# ---------------------------------------------------------------------------


class TestRetrieveReturnsTopK:
    def test_retrieve_returns_top_k(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        pages = _default_pages(5)
        _build_db(db_path, pages)

        query_vec = _make_unit_vec(99)
        retriever = Retriever(db_path=db_path, model_name="mock")
        with patch.object(retriever, "_model", _mock_model(query_vec)):
            retriever._model = _mock_model(query_vec)
            results = retriever.retrieve(
                query="test query", index_variant="v1", course_code=None, top_k=3
            )

        assert len(results) == 3


# ---------------------------------------------------------------------------
# Test 2: results are sorted by score descending
# ---------------------------------------------------------------------------


class TestResultsSortedByScore:
    def test_results_sorted_by_score_descending(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        pages = _default_pages(5)
        _build_db(db_path, pages)

        query_vec = _make_unit_vec(42)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        results = retriever.retrieve(
            query="test", index_variant="v1", course_code=None, top_k=5
        )

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Test 3: rank field is 1-indexed in result order
# ---------------------------------------------------------------------------


class TestResultRankField:
    def test_result_rank_field(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        pages = _default_pages(5)
        _build_db(db_path, pages)

        query_vec = _make_unit_vec(7)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        results = retriever.retrieve(
            query="test", index_variant="v1", course_code=None, top_k=5
        )

        for i, result in enumerate(results):
            assert result.rank == i + 1


# ---------------------------------------------------------------------------
# Test 4: empty index returns empty list
# ---------------------------------------------------------------------------


class TestEmptyIndexReturnsEmptyList:
    def test_empty_index_returns_empty_list(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "empty.db"
        # Create schema but insert no rows
        conn = sqlite3.connect(str(db_path))
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        conn.close()

        query_vec = _make_unit_vec(1)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        results = retriever.retrieve(
            query="test", index_variant="v1", course_code=None, top_k=5
        )

        assert results == []


# ---------------------------------------------------------------------------
# Test 5: invalid variant raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidVariantRaises:
    def test_invalid_variant_raises(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        pages = _default_pages(2)
        _build_db(db_path, pages)

        query_vec = _make_unit_vec(1)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        with pytest.raises(ValueError, match="variant"):
            retriever.retrieve(
                query="test", index_variant="v99", course_code=None, top_k=3
            )


# ---------------------------------------------------------------------------
# Test 6: top_k clamped to index size
# ---------------------------------------------------------------------------


class TestTopKClampedToIndexSize:
    def test_top_k_clamped_to_index_size(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        pages = _default_pages(3)
        _build_db(db_path, pages)

        query_vec = _make_unit_vec(5)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        results = retriever.retrieve(
            query="test", index_variant="v1", course_code=None, top_k=10
        )

        assert len(results) == 3


# ---------------------------------------------------------------------------
# Test 7: course filter restricts results to matching course
# ---------------------------------------------------------------------------


class TestCourseFilter:
    def test_course_filter(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        pages_cs101 = _default_pages(3, course_code="CS101")
        pages_cs288 = _default_pages(3, course_code="CS288")

        conn = sqlite3.connect(str(db_path))
        conn.executescript(_SCHEMA_SQL)
        for pages in (pages_cs101, pages_cs288):
            for p in pages:
                conn.execute(
                    "INSERT INTO slide_pages "
                    "(page_id, course_code, lecture_id, page_number, image_path, ocr_text, caption, objects) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        p["page_id"],
                        p["course_code"],
                        p["lecture_id"],
                        p["page_number"],
                        p["image_path"],
                        p["ocr_text"],
                        p["caption"],
                        p["objects"],
                    ),
                )
                conn.execute(
                    "INSERT INTO slide_embeddings (page_id, variant, vector) VALUES (?, ?, ?)",
                    (p["page_id"], "v1", _pack_vec(p["vector"])),
                )
        conn.commit()
        conn.close()

        query_vec = _make_unit_vec(10)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        results = retriever.retrieve(
            query="test", index_variant="v1", course_code="CS101", top_k=10
        )

        assert len(results) > 0
        assert all(r.course_code == "CS101" for r in results)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Test 8: _rrf_combine correctness
# ---------------------------------------------------------------------------


class TestRrfCombineCorrectness:
    def test_rrf_combine_correctness(self) -> None:
        from rag.slideqa.retriever import Retriever

        retriever = Retriever.__new__(Retriever)
        # dense_ranks: page A=1, B=2, C=3
        # bm25_ranks:  page A=3, B=1, C=2
        dense_ranks = {"A": 1, "B": 2, "C": 3}
        bm25_ranks = {"A": 3, "B": 1, "C": 2}
        k = 60

        scores = retriever._rrf_combine(dense_ranks, bm25_ranks, k)

        # Each score = 1/(k + rank_dense) + 1/(k + rank_bm25)
        expected_a = 1.0 / (60 + 1) + 1.0 / (60 + 3)
        expected_b = 1.0 / (60 + 2) + 1.0 / (60 + 1)
        expected_c = 1.0 / (60 + 3) + 1.0 / (60 + 2)

        assert abs(scores["A"] - expected_a) < 1e-9
        assert abs(scores["B"] - expected_b) < 1e-9
        assert abs(scores["C"] - expected_c) < 1e-9


# ---------------------------------------------------------------------------
# Test 9: retrieve with BM25 does not crash
# ---------------------------------------------------------------------------


class TestRetrieveWithBM25:
    def test_retrieve_with_bm25(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        pages = _default_pages(5)
        _build_db(db_path, pages)

        query_vec = _make_unit_vec(20)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        results = retriever.retrieve(
            query="slide content", index_variant="v1", course_code=None,
            top_k=3, use_bm25=True,
        )

        assert isinstance(results, list)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Test 10: dense_score field populated; bm25_score=0.0 when BM25 not used
# ---------------------------------------------------------------------------


class TestDenseScoreInResult:
    def test_dense_score_in_result(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        pages = _default_pages(3)
        _build_db(db_path, pages)

        query_vec = _make_unit_vec(30)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        results = retriever.retrieve(
            query="test", index_variant="v1", course_code=None, top_k=3,
            use_bm25=False,
        )

        assert len(results) > 0
        for r in results:
            assert isinstance(r.dense_score, float)
            assert r.bm25_score == 0.0


# ---------------------------------------------------------------------------
# Test 11: cache invalidation — meta timestamp change triggers fresh load
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    def test_cache_invalidates_on_meta_change(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        pages = _default_pages(3)
        _build_db(db_path, pages)

        # Write initial meta timestamp
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_modified', '2024-01-01T00:00:00.000')"
        )
        conn.commit()
        conn.close()

        query_vec = _make_unit_vec(11)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        # First call — loads and caches
        results1 = retriever.retrieve("test", "v1", None, top_k=3)
        assert len(results1) == 3

        # Bump timestamp — simulates another process writing
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_modified', '2024-01-02T00:00:00.000')"
        )
        conn.commit()
        conn.close()

        # Second call should rebuild (cache miss due to timestamp change)
        # We verify by counting _build_index calls
        with patch.object(retriever, "_build_index", wraps=retriever._build_index) as mock_build:
            retriever.retrieve("test", "v1", None, top_k=3)
            assert mock_build.call_count == 1  # rebuilt once after invalidation


# ---------------------------------------------------------------------------
# Test 12: _fallback_bm25 used when rank_bm25 is unavailable
# ---------------------------------------------------------------------------


class TestFallbackBM25:
    def test_fallback_bm25_used_on_import_error(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        pages = _default_pages(4)
        _build_db(db_path, pages)

        query_vec = _make_unit_vec(12)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        # Simulate rank_bm25 not installed
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "rank_bm25":
                raise ImportError("rank_bm25 not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            results = retriever.retrieve("slide content 1", "v1", None, top_k=3, use_bm25=True)

        assert isinstance(results, list)
        assert len(results) <= 4
        # All bm25_score fields populated (may be 0.0 for no-match pages)
        for r in results:
            assert isinstance(r.bm25_score, float)

    def test_fallback_bm25_returns_float_list(self) -> None:
        from rag.slideqa.retriever import Retriever

        corpus = [["hello", "world"], ["foo", "bar"], []]
        query = ["hello"]
        scores = Retriever._fallback_bm25(corpus, query)
        assert len(scores) == 3
        assert scores[0] > 0.0   # "hello" appears in doc 0
        assert scores[1] == 0.0  # no overlap
        assert scores[2] == 0.0  # empty doc


# ---------------------------------------------------------------------------
# Test 13: corrupt BLOB (non-multiple of 4) is skipped gracefully
# ---------------------------------------------------------------------------


class TestCorruptBlob:
    def test_corrupt_blob_skipped(self, tmp_path: Path, caplog) -> None:
        import logging
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        # Build a clean DB with 2 good pages
        pages = _default_pages(2)
        _build_db(db_path, pages)

        # Inject a corrupt blob (13 bytes — not divisible by 4)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO slide_pages (page_id, course_code, lecture_id, page_number, image_path, ocr_text) "
            "VALUES ('CS101/lec01/page_099', 'CS101', 'lec01', 99, '/img.png', '')"
        )
        conn.execute(
            "INSERT INTO slide_embeddings (page_id, variant, vector) VALUES (?, 'v1', ?)",
            ("CS101/lec01/page_099", b"\x00" * 13),  # 13 bytes — malformed
        )
        conn.commit()
        conn.close()

        query_vec = _make_unit_vec(13)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        with caplog.at_level(logging.WARNING, logger="rag.slideqa.retriever"):
            results = retriever.retrieve("test", "v1", None, top_k=5)

        # Corrupt page is silently skipped; good pages still returned
        page_ids = [r.page_id for r in results]
        assert "CS101/lec01/page_099" not in page_ids
        assert len(results) == 2
        assert any("malformed" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Test 14: zero-norm query vector logs a warning
# ---------------------------------------------------------------------------


class TestZeroNormQuery:
    def test_zero_norm_query_logs_warning(self, tmp_path: Path, caplog) -> None:
        import logging
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        _build_db(db_path, _default_pages(3))

        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(np.zeros(_DIM, dtype=np.float32))

        with caplog.at_level(logging.WARNING, logger="rag.slideqa.retriever"):
            results = retriever.retrieve("test", "v1", None, top_k=3)

        assert any("zero-norm" in msg.lower() for msg in caplog.messages)
        # Results still returned (all scores = 0.0 but no crash)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test 15: top_k == n boundary (exact match)
# ---------------------------------------------------------------------------


class TestTopKEqualsN:
    def test_top_k_equals_index_size(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        _build_db(db_path, _default_pages(3))

        query_vec = _make_unit_vec(15)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        results = retriever.retrieve("test", "v1", None, top_k=3)
        assert len(results) == 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Test 16: multi-variant filtering (v1 vs v2 embeddings)
# ---------------------------------------------------------------------------


class TestMultiVariantFiltering:
    def test_variant_filter_returns_correct_embeddings(self, tmp_path: Path) -> None:
        from rag.slideqa.retriever import Retriever

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(_SCHEMA_SQL)

        # Insert 3 pages with BOTH v1 and v2 embeddings using different vectors
        for i in range(1, 4):
            page_id = f"CS101/lec01/page_{i:03d}"
            conn.execute(
                "INSERT INTO slide_pages (page_id, course_code, lecture_id, page_number, image_path, ocr_text) "
                "VALUES (?, 'CS101', 'lec01', ?, '/img.png', 'text')",
                (page_id, i),
            )
            # v1: seed i; v2: seed i+100 (different vectors)
            conn.execute(
                "INSERT INTO slide_embeddings (page_id, variant, vector) VALUES (?, 'v1', ?)",
                (page_id, _pack_vec(_make_unit_vec(i))),
            )
            conn.execute(
                "INSERT INTO slide_embeddings (page_id, variant, vector) VALUES (?, 'v2', ?)",
                (page_id, _pack_vec(_make_unit_vec(i + 100))),
            )
        conn.commit()
        conn.close()

        query_vec = _make_unit_vec(50)
        retriever = Retriever(db_path=db_path, model_name="mock")
        retriever._model = _mock_model(query_vec)

        results_v1 = retriever.retrieve("test", "v1", None, top_k=3)
        results_v2 = retriever.retrieve("test", "v2", None, top_k=3)

        # Both return 3 results
        assert len(results_v1) == 3
        assert len(results_v2) == 3

        # Scores differ (different embedding vectors → different cosine scores)
        scores_v1 = [r.dense_score for r in results_v1]
        scores_v2 = [r.dense_score for r in results_v2]
        assert scores_v1 != scores_v2
