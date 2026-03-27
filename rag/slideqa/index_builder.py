"""
SQLite index builder for SlideQA records and embeddings.

Schema has two tables:
- slide_pages  : stores SlidePageRecord fields
- slide_embeddings : stores float32 BLOBs for v1/v2/v3 embedding variants
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Any, Optional

from rag.slideqa.schema import SlidePageRecord

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 32_000

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS slide_pages (
    page_id TEXT PRIMARY KEY, course_code TEXT NOT NULL, lecture_id TEXT NOT NULL,
    page_number INTEGER NOT NULL, image_path TEXT NOT NULL,
    ocr_text TEXT NOT NULL DEFAULT '', caption TEXT, objects TEXT,
    created_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS slide_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, page_id TEXT NOT NULL,
    variant TEXT NOT NULL CHECK(variant IN ('v1', 'v2', 'v3')),
    vector BLOB NOT NULL,
    FOREIGN KEY (page_id) REFERENCES slide_pages(page_id),
    UNIQUE(page_id, variant)
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with foreign-key enforcement enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    """
    Create the slide_pages and slide_embeddings tables if they do not exist.

    Args:
        db_path: Path to the SQLite database file.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("Database initialised at %s", db_path)
    finally:
        conn.close()


def upsert_page_records(db_path: Path, records: list[SlidePageRecord]) -> int:
    """
    Insert or replace SlidePageRecord rows in slide_pages.

    Args:
        db_path: Path to the SQLite database file.
        records: Records to upsert.

    Returns:
        Number of records written.
    """
    if not records:
        return 0

    db_path = Path(db_path)
    conn = _connect(db_path)
    try:
        rows = [
            (
                r.page_id,
                r.course_code,
                r.lecture_id,
                r.page_number,
                r.image_path,
                r.ocr_text,
                r.caption,
                json.dumps(list(r.objects)) if r.objects is not None else None,
            )
            for r in records
        ]
        conn.executemany(
            """
            INSERT INTO slide_pages
              (page_id, course_code, lecture_id, page_number, image_path,
               ocr_text, caption, objects)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(page_id) DO UPDATE SET
              course_code = excluded.course_code,
              lecture_id  = excluded.lecture_id,
              page_number = excluded.page_number,
              image_path  = excluded.image_path,
              ocr_text    = excluded.ocr_text,
              caption     = excluded.caption,
              objects     = excluded.objects
            """,
            rows,
        )
        conn.commit()
        logger.info("Upserted %d page records", len(rows))
        return len(rows)
    finally:
        conn.close()


def load_page_records(db_path: Path, course_code: str) -> dict[str, SlidePageRecord]:
    """
    Load existing SlidePageRecord rows for a course from the database.

    Args:
        db_path:     Path to the SQLite database.
        course_code: Course to filter by.

    Returns:
        Dict mapping page_id -> SlidePageRecord.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return {}

    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT page_id, course_code, lecture_id, page_number, image_path, "
            "ocr_text, caption, objects FROM slide_pages WHERE course_code = ?",
            (course_code,),
        )
        result: dict[str, SlidePageRecord] = {}
        for row in cursor.fetchall():
            page_id, cc, lid, pnum, img, ocr, cap, obj_json = row
            objects: tuple[str, ...] | None = None
            if obj_json is not None:
                try:
                    objects = tuple(json.loads(obj_json))
                except (json.JSONDecodeError, TypeError):
                    objects = None
            result[page_id] = SlidePageRecord(
                page_id=page_id,
                course_code=cc,
                lecture_id=lid,
                page_number=pnum,
                image_path=img,
                ocr_text=ocr or "",
                caption=cap,
                objects=objects,
            )
        return result
    finally:
        conn.close()


def compose_embedding_text(record: SlidePageRecord, variant: str) -> str:
    """
    Compose the text to embed for a given variant.

    Variants:
    - v1: ocr_text only
    - v2: ocr_text + caption
    - v3: ocr_text + caption + objects

    None fields are handled gracefully.  Output is truncated to 32,000 chars.

    Args:
        record:  A SlidePageRecord.
        variant: One of "v1", "v2", "v3".

    Returns:
        Text string for embedding.
    """
    if variant not in ("v1", "v2", "v3"):
        raise ValueError(f"Unknown variant: {variant!r}. Must be v1, v2, or v3.")

    parts: list[str] = [record.ocr_text or ""]

    if variant in ("v2", "v3"):
        if record.caption:
            parts.append(record.caption)

    if variant == "v3":
        if record.objects:
            parts.append(" ".join(record.objects))

    text = "\n".join(p for p in parts if p)
    return text[:_MAX_TEXT_CHARS]


def build_embeddings(
    db_path: Path,
    records: list[SlidePageRecord],
    variant: str,
    model: Optional[Any] = None,
    model_name: str = "Qwen/Qwen3-Embedding-4B",
) -> int:
    """
    Encode records and store float32 BLOBs in slide_embeddings.

    Args:
        db_path:    Path to the SQLite database.
        records:    Records to embed.
        variant:    Embedding variant ("v1", "v2", or "v3").
        model:      Pre-loaded SentenceTransformer; loaded lazily if None.
        model_name: HuggingFace model identifier (used when model is None).

    Returns:
        Number of embedding rows written.
    """
    if not records:
        return 0

    if model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info("Loading embedding model %s", model_name)
        model = SentenceTransformer(model_name)

    texts = [compose_embedding_text(r, variant) for r in records]
    page_ids = [r.page_id for r in records]

    logger.info("Encoding %d texts for variant %s", len(texts), variant)
    vectors = model.encode(texts, show_progress_bar=False)

    db_path = Path(db_path)
    conn = _connect(db_path)
    try:
        rows = [
            (page_ids[i], variant, struct.pack(f"{len(vectors[i])}f", *vectors[i].tolist()))
            for i in range(len(records))
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO slide_embeddings (page_id, variant, vector)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        logger.info("Stored %d embeddings (variant=%s)", len(rows), variant)
        return len(rows)
    finally:
        conn.close()


def build_all_variants(
    db_path: Path,
    records: list[SlidePageRecord],
    model_name: str = "Qwen/Qwen3-Embedding-4B",
) -> dict[str, int]:
    """
    Build embeddings for all three variants (v1, v2, v3), loading model once.

    Args:
        db_path:    Path to the SQLite database.
        records:    Records to embed.
        model_name: HuggingFace model identifier.

    Returns:
        Dict mapping variant -> count of rows written.
    """
    if not records:
        return {"v1": 0, "v2": 0, "v3": 0}

    from sentence_transformers import SentenceTransformer  # type: ignore

    logger.info("Loading embedding model %s (shared across variants)", model_name)
    model = SentenceTransformer(model_name)

    results: dict[str, int] = {}
    for variant in ("v1", "v2", "v3"):
        count = build_embeddings(
            db_path=db_path,
            records=records,
            variant=variant,
            model=model,
            model_name=model_name,
        )
        results[variant] = count

    return results
