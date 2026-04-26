"""
Build a slideqa.db from CS 288_metadata_new.db chunk embeddings.

Reads slide chunks from a metadata SQLite DB, extracts per-variant text
(v1/v2/v3), encodes with Qwen/Qwen3-Embedding-4B, and writes to a
slideqa.db with slide_pages and slide_chunks tables.

Text extraction per variant:
  v1 = OCR only  — strip all <!-- TAI_VLM_BEGIN...TAI_VLM_END --> + image refs
  v2 = v1 + image captions from <!-- TAI_V2_BEGIN...TAI_V2_END -->
  v3 = v2 + visual notes from <!-- TAI_V3_BEGIN...TAI_V3_END -->

page_number = chunk_index + 1  (1-based, eval.py applies pn-1 to get gold_page_id)

Usage (CLI):
    python -m slideqa.embed_from_metadata \\
        --source  /path/to/CS\\ 288_metadata_new.db \\
        --output  /path/to/slideqa.db \\
        --model   Qwen/Qwen3-Embedding-4B \\
        --variants v1,v2,v3 \\
        --batch-size 32
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_ID_TEMPLATE = "{lecture}_page_{chunk_index}_idx{db_idx}_{variant}"
PAGE_ID_TEMPLATE  = "{lecture}_page_{chunk_index}"

_IMG_RE   = re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL)
_VLM_RE   = re.compile(r"<!--\s*TAI_VLM_BEGIN.*?TAI_VLM_END\s*-->", re.DOTALL)
_V2_RE    = re.compile(r"<!--\s*TAI_V2_BEGIN\s*-->(.*?)<!--\s*TAI_V2_END\s*-->", re.DOTALL)
_V3_RE    = re.compile(r"<!--\s*TAI_V3_BEGIN\s*-->(.*?)<!--\s*TAI_V3_END\s*-->", re.DOTALL)
_MAX_TEXT = 32_000

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _base_ocr(text: str) -> str:
    """Strip all VLM blocks and image markdown refs; return plain OCR text."""
    t = _VLM_RE.sub("", text)
    t = _IMG_RE.sub("", t)
    return t


def extract_v1(text: str) -> str:
    """OCR only — no captions, no visual notes."""
    return _base_ocr(text)[:_MAX_TEXT]


def extract_v2(text: str) -> str:
    """OCR + image captions from TAI_V2_BEGIN blocks."""
    captions = [m.strip() for m in _V2_RE.findall(text)]
    base = _base_ocr(text)
    parts = [base] + captions
    return "\n".join(p for p in parts if p.strip())[:_MAX_TEXT]


def extract_v3(text: str) -> str:
    """OCR + image captions + visual notes from TAI_V2 and TAI_V3 blocks."""
    captions = [m.strip() for m in _V2_RE.findall(text)]
    visuals  = [m.strip() for m in _V3_RE.findall(text)]
    base = _base_ocr(text)
    parts = [base] + captions + visuals
    return "\n".join(p for p in parts if p.strip())[:_MAX_TEXT]


_EXTRACTORS = {"v1": extract_v1, "v2": extract_v2, "v3": extract_v3}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_DEST_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS slide_pages (
    page_id      TEXT PRIMARY KEY,
    course_code  TEXT NOT NULL,
    lecture_id   TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    page_number  INTEGER NOT NULL,
    image_path   TEXT NOT NULL DEFAULT '',
    ocr_text     TEXT NOT NULL DEFAULT '',
    caption      TEXT,
    objects      TEXT,
    created_at   TIMESTAMP DEFAULT (datetime('now', 'localtime'))
);
CREATE TABLE IF NOT EXISTS slide_chunks (
    chunk_id   TEXT PRIMARY KEY,
    page_id    TEXT NOT NULL,
    variant    TEXT NOT NULL CHECK(variant IN ('v1','v2','v3')),
    chunk_text TEXT NOT NULL DEFAULT '',
    vector     BLOB NOT NULL,
    FOREIGN KEY (page_id) REFERENCES slide_pages(page_id)
);
CREATE INDEX IF NOT EXISTS idx_slide_chunks_page_variant
    ON slide_chunks(page_id, variant);
"""


def _pack(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return struct.pack(f"{len(arr)}f", *arr.tolist())


def _init_dest(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_DEST_SCHEMA)
    conn.commit()
    conn.close()


def _load_source_rows(
    src_path: Path,
    course_code: Optional[str],
    slides_only: bool,
) -> list[dict]:
    """Load slide chunk rows from the metadata DB."""
    conn = sqlite3.connect(str(src_path))
    conn.row_factory = sqlite3.Row

    sql = (
        "SELECT chunk_uuid, idx, text, file_path, course_code, chunk_index "
        "FROM chunks WHERE chunk_index IS NOT NULL AND text IS NOT NULL"
    )
    params: list = []

    if course_code is not None:
        sql += " AND course_code = ?"
        params.append(course_code)

    if slides_only:
        sql += " AND file_path LIKE '%/slides/%'"

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _upsert_page(conn: sqlite3.Connection, page_id: str, lecture_id: str,
                 chunk_index: int, course_code: str, ocr_text: str) -> None:
    page_number = chunk_index + 1
    conn.execute(
        """
        INSERT INTO slide_pages
            (page_id, course_code, lecture_id, chunk_index, page_number, ocr_text)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(page_id) DO UPDATE SET
            ocr_text = excluded.ocr_text
        """,
        (page_id, course_code, lecture_id, chunk_index, page_number, ocr_text),
    )


def _upsert_chunk(conn: sqlite3.Connection, chunk_id: str, page_id: str,
                  variant: str, chunk_text: str, vector: np.ndarray) -> None:
    conn.execute(
        """
        INSERT INTO slide_chunks (chunk_id, page_id, variant, chunk_text, vector)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            chunk_text = excluded.chunk_text,
            vector     = excluded.vector
        """,
        (chunk_id, page_id, variant, chunk_text, _pack(vector)),
    )

# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_slideqa_db_from_metadata(
    src_path: Path,
    dest_path: Path,
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    variants: list[str] | None = None,
    course_code: Optional[str] = None,
    slides_only: bool = True,
    batch_size: int = 32,
    model=None,
) -> dict[str, int]:
    """Read chunks from src_path, encode with model, write to dest_path.

    Args:
        src_path:    Path to CS 288_metadata_new.db (source).
        dest_path:   Path to slideqa.db to create/update (destination).
        model_name:  HuggingFace model identifier (used when model is None).
        variants:    List of variants to build; defaults to ["v1", "v2", "v3"].
        course_code: Optional filter (e.g. "CS 288"); None = all courses.
        slides_only: If True, skip non-slide chunks (assignments, course website).
        batch_size:  Number of texts to encode per model.encode() call.
        model:       Pre-loaded SentenceTransformer; loaded lazily if None.

    Returns:
        Dict mapping variant → number of chunk rows written.
    """
    variants = variants or ["v1", "v2", "v3"]

    _init_dest(dest_path)
    rows = _load_source_rows(src_path, course_code, slides_only)
    if not rows:
        logger.warning("No source rows found matching filters.")
        return {v: 0 for v in variants}

    if model is None:
        logger.info("Loading model %s", model_name)
        model = SentenceTransformer(model_name)

    dest_conn = sqlite3.connect(str(dest_path))
    dest_conn.execute("PRAGMA foreign_keys = ON")

    counts: dict[str, int] = {}

    for variant in variants:
        extractor = _EXTRACTORS[variant]
        logger.info("[%s] extracting texts for %d chunks", variant, len(rows))

        texts: list[str] = []
        for r in rows:
            texts.append(extractor(r["text"] or ""))

        # Encode in batches
        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            vecs = model.encode(batch, show_progress_bar=False)
            if isinstance(vecs, np.ndarray) and vecs.ndim == 1:
                vecs = vecs[np.newaxis, :]
            all_vecs.append(np.asarray(vecs, dtype=np.float32))
            logger.debug("[%s] encoded batch %d-%d", variant, start, start + len(batch))

        matrix = np.vstack(all_vecs)  # shape [N, D]
        n_written = 0

        for i, r in enumerate(rows):
            file_path   = r["file_path"] or ""
            lecture_id  = Path(file_path).name
            chunk_index = int(r["chunk_index"])
            db_idx      = int(r["idx"])
            course      = r["course_code"] or ""

            page_id  = PAGE_ID_TEMPLATE.format(lecture=lecture_id, chunk_index=chunk_index)
            chunk_id = CHUNK_ID_TEMPLATE.format(
                lecture=lecture_id, chunk_index=chunk_index,
                db_idx=db_idx, variant=variant,
            )

            # Always upsert the slide_pages row (idempotent, uses v1 ocr_text)
            if variant == "v1":
                _upsert_page(dest_conn, page_id, lecture_id, chunk_index, course, texts[i])

            _upsert_chunk(dest_conn, chunk_id, page_id, variant, texts[i], matrix[i])
            n_written += 1

        dest_conn.commit()
        counts[variant] = n_written
        logger.info("[%s] wrote %d chunk rows", variant, n_written)

    dest_conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) "
        "VALUES ('last_modified', strftime('%Y-%m-%dT%H:%M:%f', 'now'))"
    )
    dest_conn.commit()
    dest_conn.close()
    return counts

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build slideqa.db with v1/v2/v3 chunk embeddings from metadata DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True, type=Path,
                   help="Path to CS 288_metadata_new.db (source).")
    p.add_argument("--output", required=True, type=Path,
                   help="Path to slideqa.db to create/update (destination).")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-4B",
                   help="HuggingFace embedding model identifier.")
    p.add_argument("--variants", default="v1,v2,v3",
                   help="Comma-separated variants to build.")
    p.add_argument("--course-code", default="CS 288",
                   help="Course code filter (pass 'all' to include all courses).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Texts per encode() call.")
    p.add_argument("--all-files", action="store_true",
                   help="Include non-slide files (assignments, course website).")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    p = _build_parser()
    args = p.parse_args(argv)

    if not args.source.exists():
        print(f"ERROR: source DB not found: {args.source}", file=sys.stderr)
        return 1

    course = None if args.course_code == "all" else args.course_code
    variants = [v.strip() for v in args.variants.split(",")]

    counts = build_slideqa_db_from_metadata(
        src_path=args.source,
        dest_path=args.output,
        model_name=args.model,
        variants=variants,
        course_code=course,
        slides_only=not args.all_files,
        batch_size=args.batch_size,
    )

    for v, n in counts.items():
        print(f"[{v}] {n} chunk rows written → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
