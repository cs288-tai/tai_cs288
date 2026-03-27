"""
Tests for rag.slideqa.index_builder.

Database interactions use tmp_path so tests are fully isolated.
SentenceTransformer is mocked to avoid GPU/model download requirements.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.slideqa.index_builder import (
    build_embeddings,
    compose_embedding_text,
    init_db,
    upsert_page_records,
)
from rag.slideqa.schema import SlidePageRecord, make_page_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    page_number: int = 1,
    ocr_text: str = "sample text",
    caption: str | None = None,
    objects: list | None = None,
) -> SlidePageRecord:
    return SlidePageRecord(
        page_id=make_page_id("CS288", "lecture01", page_number),
        course_code="CS288",
        lecture_id="lecture01",
        page_number=page_number,
        image_path=f"/fake/page_{page_number:03d}.png",
        ocr_text=ocr_text,
        caption=caption,
        objects=objects,
    )


def _fetch_all_pages(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM slide_pages").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _fetch_embeddings(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM slide_embeddings").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_creates_slide_pages_table(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        conn = sqlite3.connect(str(db))
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "slide_pages" in tables

    def test_creates_slide_embeddings_table(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        conn = sqlite3.connect(str(db))
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "slide_embeddings" in tables

    def test_idempotent_second_call(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        init_db(db)  # should not raise

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "dir" / "test.db"
        init_db(db)
        assert db.exists()


# ---------------------------------------------------------------------------
# upsert_page_records
# ---------------------------------------------------------------------------


class TestUpsertPageRecords:
    def test_inserts_records(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        records = [_make_record(1), _make_record(2)]

        count = upsert_page_records(db, records)

        assert count == 2
        rows = _fetch_all_pages(db)
        assert len(rows) == 2

    def test_idempotent_second_upsert(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        record = _make_record(1, ocr_text="original")

        upsert_page_records(db, [record])
        updated = _make_record(1, ocr_text="updated")
        upsert_page_records(db, [updated])

        rows = _fetch_all_pages(db)
        assert len(rows) == 1
        assert rows[0]["ocr_text"] == "updated"

    def test_objects_serialised_as_json(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        record = _make_record(1, objects=("item a", "item b"))

        upsert_page_records(db, [record])

        rows = _fetch_all_pages(db)
        assert json.loads(rows[0]["objects"]) == ["item a", "item b"]

    def test_none_objects_stored_as_null(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        upsert_page_records(db, [_make_record(1)])

        rows = _fetch_all_pages(db)
        assert rows[0]["objects"] is None

    def test_empty_list_returns_zero(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        assert upsert_page_records(db, []) == 0


# ---------------------------------------------------------------------------
# compose_embedding_text
# ---------------------------------------------------------------------------


class TestComposeEmbeddingText:
    def test_v1_contains_only_ocr(self) -> None:
        rec = _make_record(ocr_text="ocr text", caption="cap", objects=("obj",))
        result = compose_embedding_text(rec, "v1")
        assert "ocr text" in result
        assert "cap" not in result
        assert "obj" not in result

    def test_v2_contains_ocr_and_caption(self) -> None:
        rec = _make_record(ocr_text="ocr text", caption="cap", objects=("obj",))
        result = compose_embedding_text(rec, "v2")
        assert "ocr text" in result
        assert "cap" in result
        assert "obj" not in result

    def test_v3_contains_all_fields(self) -> None:
        rec = _make_record(ocr_text="ocr text", caption="cap", objects=("obj1", "obj2"))
        result = compose_embedding_text(rec, "v3")
        assert "ocr text" in result
        assert "cap" in result
        assert "obj1" in result

    def test_none_caption_handled_gracefully(self) -> None:
        rec = _make_record(ocr_text="ocr", caption=None)
        result = compose_embedding_text(rec, "v2")
        assert "ocr" in result

    def test_none_objects_handled_gracefully(self) -> None:
        rec = _make_record(ocr_text="ocr", caption="cap", objects=None)
        result = compose_embedding_text(rec, "v3")
        assert "ocr" in result
        assert "cap" in result

    def test_truncated_to_32000_chars(self) -> None:
        long_text = "x" * 40_000
        rec = _make_record(ocr_text=long_text)
        result = compose_embedding_text(rec, "v1")
        assert len(result) <= 32_000

    def test_empty_ocr_text(self) -> None:
        rec = _make_record(ocr_text="")
        result = compose_embedding_text(rec, "v1")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build_embeddings (mock SentenceTransformer)
# ---------------------------------------------------------------------------


class TestBuildEmbeddings:
    def _make_mock_model(self, dim: int = 4) -> MagicMock:
        import numpy as np

        model = MagicMock()
        model.encode.side_effect = lambda texts, **kw: np.random.rand(
            len(texts), dim
        ).astype("float32")
        return model

    def test_writes_correct_row_count(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        records = [_make_record(i) for i in range(1, 4)]
        upsert_page_records(db, records)

        model = self._make_mock_model()
        count = build_embeddings(db, records, variant="v1", model=model)

        assert count == 3
        rows = _fetch_embeddings(db)
        assert len(rows) == 3

    def test_variant_stored_correctly(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        records = [_make_record(1)]
        upsert_page_records(db, records)

        model = self._make_mock_model()
        build_embeddings(db, records, variant="v2", model=model)

        rows = _fetch_embeddings(db)
        assert rows[0]["variant"] == "v2"

    def test_vector_stored_as_float32_blob(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        records = [_make_record(1)]
        upsert_page_records(db, records)

        model = self._make_mock_model(dim=8)
        build_embeddings(db, records, variant="v1", model=model)

        rows = _fetch_embeddings(db)
        blob = rows[0]["vector"]
        # 8 floats * 4 bytes each
        assert len(blob) == 32
        # Should unpack without error
        values = struct.unpack("8f", blob)
        assert len(values) == 8

    def test_unique_constraint_prevents_duplicates(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        records = [_make_record(1)]
        upsert_page_records(db, records)

        model = self._make_mock_model()
        build_embeddings(db, records, variant="v1", model=model)
        # Second call should replace, not add new row
        build_embeddings(db, records, variant="v1", model=model)

        rows = _fetch_embeddings(db)
        assert len(rows) == 1

    def test_empty_records_returns_zero(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        model = self._make_mock_model()
        count = build_embeddings(db, [], variant="v1", model=model)
        assert count == 0

    def test_multiple_variants_stored_independently(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        init_db(db)
        records = [_make_record(1)]
        upsert_page_records(db, records)

        model = self._make_mock_model()
        build_embeddings(db, records, variant="v1", model=model)
        build_embeddings(db, records, variant="v2", model=model)
        build_embeddings(db, records, variant="v3", model=model)

        rows = _fetch_embeddings(db)
        assert len(rows) == 3
        variants = {r["variant"] for r in rows}
        assert variants == {"v1", "v2", "v3"}
