#!/usr/bin/env python3
"""
Build a slideqa.db with v1 / v2 / v3 page embeddings directly from the
on-disk MinerU output for CS 288 (or any course laid out the same way).

This is the disk-based sibling of ``rag/slideqa/embed_from_metadata.py``.
The metadata-DB version requires that the slide PDFs have already been
ingested by ``file_conversion_router.api`` — which doesn't happen for
sp26 because every PDF hits the file_uuid cache and is short-circuited
before ``PdfConverter._to_markdown`` runs (see the long discussion in
the chat that produced this script). This version skips the pipeline
and the metadata DB entirely and works straight off the
``<lecture>/<stem>.pdf/auto/`` folders MinerU already produced.

For each lecture it reads:

    <stem>.master.md            — to extract per-image (caption, visual_notes)
    <stem>_content_list.json    — to split lecture into pages and locate images

For each ``page_idx`` it builds three variant texts:

    v1 = page text from content_list (image refs stripped)
    v2 = v1 + caption blocks of every image that lives on this page
    v3 = v2 + visual-note bullets of every image that lives on this page

Each variant text is then encoded with the requested SentenceTransformer
model (default Qwen/Qwen3-Embedding-4B) and stored in the same
``slide_pages`` / ``slide_chunks`` schema that ``embed_from_metadata.py``
writes — so the existing ``Retriever`` and ``run_eval`` code can read the
output without any changes.

Usage:
    python scripts/embed_cs288_variants.py \\
        --processed-dir "/path/to/CS 288/.../sp26/assets/slides" \\
        --output        "/path/to/slideqa.db" \\
        --course-code   "CS 288" \\
        --model         "Qwen/Qwen3-Embedding-4B" \\
        --variants      v1,v2,v3 \\
        --batch-size    32

Defaults:
  --output       <processed-dir>/slideqa.db
  --variants     v1,v2,v3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Make the rag package importable when running from repo root.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse helpers from the existing slideqa embedder so the schema and
# packing logic stays in lockstep with embed_from_metadata.py.
from rag.slideqa.embed_from_metadata import (  # type: ignore
    CHUNK_ID_TEMPLATE,
    PAGE_ID_TEMPLATE,
    _init_dest,
    _pack,
    _upsert_chunk,
    _upsert_page,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown / VLM-block parsing
# ---------------------------------------------------------------------------

# Match the augmentation blocks PdfConverter inserts under each image link.
#
#   <!-- TAI_VLM_BEGIN image="path/to/img.png" -->
#       <!-- TAI_V2_BEGIN -->
#       [Image Caption]
#       ...caption text...
#       <!-- TAI_V2_END -->
#
#       <!-- TAI_V3_BEGIN -->
#       [Visual Notes]
#       - bullet 1
#       - bullet 2
#       <!-- TAI_V3_END -->
#   <!-- TAI_VLM_END -->
#
_VLM_BLOCK_RE = re.compile(
    r'<!--\s*TAI_VLM_BEGIN\s+image="(?P<img>[^"]+)"\s*-->'
    r"(?P<body>.*?)"
    r"<!--\s*TAI_VLM_END\s*-->",
    re.DOTALL,
)
_V2_BODY_RE = re.compile(
    r"<!--\s*TAI_V2_BEGIN\s*-->(?P<v2>.*?)<!--\s*TAI_V2_END\s*-->",
    re.DOTALL,
)
_V3_BODY_RE = re.compile(
    r"<!--\s*TAI_V3_BEGIN\s*-->(?P<v3>.*?)<!--\s*TAI_V3_END\s*-->",
    re.DOTALL,
)
_IMAGE_LINK_RE = re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL)
_MAX_TEXT = 32_000


def _parse_master_vlm(master_md_text: str) -> dict[str, dict[str, str]]:
    """Build a mapping ``image_relpath -> {"v2": caption, "v3": visual_notes}``.

    Both ``v2`` and ``v3`` are returned as plain text (no XML / comment
    markers). Missing pieces simply yield an empty string for that key.
    """
    out: dict[str, dict[str, str]] = {}
    for match in _VLM_BLOCK_RE.finditer(master_md_text):
        img_rel = match.group("img").strip()
        body = match.group("body")

        v2_match = _V2_BODY_RE.search(body)
        v3_match = _V3_BODY_RE.search(body)

        out[img_rel] = {
            "v2": v2_match.group("v2").strip() if v2_match else "",
            "v3": v3_match.group("v3").strip() if v3_match else "",
        }
    return out


# ---------------------------------------------------------------------------
# content_list.json → per-page text + per-page image refs
# ---------------------------------------------------------------------------


def _load_pages_and_images(
    content_list_path: Path,
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Return (page_text_by_idx, image_relpaths_by_idx).

    ``image_relpaths_by_idx`` lists the *relative* image paths exactly as
    they appear in the master markdown's image links (and as the
    ``image="..."`` attribute on each TAI_VLM_BEGIN block), so they key
    correctly into the dict produced by ``_parse_master_vlm``.
    """
    with content_list_path.open("r", encoding="utf-8") as fh:
        items = json.load(fh)

    text_parts: dict[int, list[str]] = {}
    images: dict[int, list[str]] = {}

    for item in items:
        page_idx = item.get("page_idx")
        if page_idx is None:
            continue
        if item.get("type", "") == "image":
            img_rel = (item.get("img_path") or "").strip()
            if img_rel:
                images.setdefault(page_idx, []).append(img_rel)
        else:
            text = (item.get("text") or "").strip()
            if text:
                text_parts.setdefault(page_idx, []).append(text)

    page_text = {idx: "\n\n".join(parts) for idx, parts in text_parts.items()}
    return page_text, images


# ---------------------------------------------------------------------------
# Per-page variant text builders
# ---------------------------------------------------------------------------


def _strip_image_links(text: str) -> str:
    """Remove markdown image links so v1 text is pure OCR."""
    return _IMAGE_LINK_RE.sub("", text)


def _build_page_variant_text(
    page_text: str,
    image_relpaths_on_page: list[str],
    vlm_map: dict[str, dict[str, str]],
    variant: str,
) -> str:
    """Compose v1 / v2 / v3 text for a single page.

    v1 = stripped page_text
    v2 = v1 + every caption block whose image lives on this page
    v3 = v2 + every visual-notes block whose image lives on this page
    """
    base = _strip_image_links(page_text).strip()
    if variant == "v1":
        return base[:_MAX_TEXT]

    captions: list[str] = []
    notes: list[str] = []
    for img_rel in image_relpaths_on_page:
        info = vlm_map.get(img_rel)
        if info is None:
            continue
        if info.get("v2"):
            captions.append(info["v2"])
        if info.get("v3"):
            notes.append(info["v3"])

    if variant == "v2":
        parts = [base, *captions]
        return "\n\n".join(p for p in parts if p)[:_MAX_TEXT]

    if variant == "v3":
        parts = [base, *captions, *notes]
        return "\n\n".join(p for p in parts if p)[:_MAX_TEXT]

    raise ValueError(f"Unknown variant: {variant!r}")


# ---------------------------------------------------------------------------
# Lecture discovery
# ---------------------------------------------------------------------------


def _discover_lectures(processed_dir: Path) -> list[tuple[Path, str]]:
    """Find every lecture under processed_dir.

    Returns a list of ``(work_dir, stem)`` pairs, where ``work_dir`` is the
    directory that contains the variant markdown files (the ``auto/``
    folder for MinerU layouts) and ``stem`` is the prefix used for those
    files (e.g. ``CS288_sp26_01_Intro.pdf``).
    """
    lectures: list[tuple[Path, str]] = []
    for v1_path in sorted(processed_dir.rglob("*.v1.md")):
        work_dir = v1_path.parent
        stem = v1_path.name[: -len(".v1.md")]
        lectures.append((work_dir, stem))
    return lectures


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_slideqa_db_from_disk(
    processed_dir: Path,
    dest_path: Path,
    course_code: str,
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    variants: Optional[list[str]] = None,
    batch_size: int = 32,
    model=None,
) -> dict[str, int]:
    """Build a slideqa.db by walking ``processed_dir`` recursively.

    Args:
        processed_dir:  Root to scan for ``*.v1.md`` (each found file's
                        parent is treated as a lecture work dir).
        dest_path:      Output slideqa.db path.
        course_code:    Stored in slide_pages.course_code (e.g. "CS 288").
        model_name:     SentenceTransformer model id.
        variants:       Subset of ["v1","v2","v3"]; default = all three.
        batch_size:     Encode batch size.
        model:          Pre-loaded SentenceTransformer (loaded lazily if None).

    Returns:
        dict ``variant -> chunk rows written``.
    """
    variants = variants or ["v1", "v2", "v3"]
    for v in variants:
        if v not in {"v1", "v2", "v3"}:
            raise ValueError(f"Invalid variant: {v}")

    lectures = _discover_lectures(processed_dir)
    if not lectures:
        logger.warning("No *.v1.md files found under %s", processed_dir)
        return {v: 0 for v in variants}

    _init_dest(dest_path)

    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required to build embeddings; "
                "pip install sentence-transformers"
            ) from exc
        logger.info("Loading model %s", model_name)
        model = SentenceTransformer(model_name)

    # ----- Phase 1: collect per-(lecture, page_idx) variant texts -------------
    # We collect every (lecture_id, page_idx, variant_text) up front so the
    # encoder can run on big batches across lectures — much faster than
    # encoding 1 lecture at a time.

    # Per variant, parallel arrays we'll feed to model.encode():
    items_by_variant: dict[str, list[dict]] = {v: [] for v in variants}

    for work_dir, stem in lectures:
        master_md = work_dir / f"{stem}.master.md"
        content_list_path = work_dir / f"{stem}_content_list.json"

        if not content_list_path.exists():
            # Fallback: pick any *_content_list.json in the work dir.
            cands = sorted(work_dir.glob("*_content_list.json"))
            if not cands:
                logger.warning("[skip] %s: no *_content_list.json", work_dir)
                continue
            content_list_path = cands[0]

        page_text, page_images = _load_pages_and_images(content_list_path)

        if master_md.exists():
            vlm_map = _parse_master_vlm(master_md.read_text(encoding="utf-8"))
        else:
            logger.info(
                "[%s] no master.md found; v2/v3 will fall back to text-only",
                stem,
            )
            vlm_map = {}

        all_page_ids = sorted(set(page_text.keys()) | set(page_images.keys()))
        if not all_page_ids:
            logger.warning("[skip] %s: empty content_list.json", stem)
            continue

        for page_idx in all_page_ids:
            ptext = page_text.get(page_idx, "")
            pimgs = page_images.get(page_idx, [])
            for variant in variants:
                vtext = _build_page_variant_text(ptext, pimgs, vlm_map, variant)
                items_by_variant[variant].append(
                    {
                        "lecture_id": stem,
                        "page_idx": page_idx,
                        "text": vtext,
                    }
                )

        logger.info(
            "[lecture] %s: %d pages, %d images mapped",
            stem,
            len(all_page_ids),
            len(vlm_map),
        )

    # ----- Phase 2: encode + upsert ------------------------------------------

    counts: dict[str, int] = {}
    dest_conn = sqlite3.connect(str(dest_path))
    dest_conn.execute("PRAGMA foreign_keys = ON")

    try:
        for variant in variants:
            items = items_by_variant[variant]
            if not items:
                counts[variant] = 0
                logger.info("[%s] no items to embed", variant)
                continue

            logger.info("[%s] encoding %d page texts", variant, len(items))
            texts = [it["text"] for it in items]

            all_vecs: list[np.ndarray] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start: start + batch_size]
                vecs = model.encode(batch, show_progress_bar=False)
                if isinstance(vecs, np.ndarray) and vecs.ndim == 1:
                    vecs = vecs[np.newaxis, :]
                all_vecs.append(np.asarray(vecs, dtype=np.float32))

            matrix = np.vstack(all_vecs)
            n_written = 0

            for i, it in enumerate(items):
                lecture_id = it["lecture_id"]
                chunk_index = int(it["page_idx"])
                page_id = PAGE_ID_TEMPLATE.format(
                    lecture=lecture_id, chunk_index=chunk_index
                )
                chunk_id = CHUNK_ID_TEMPLATE.format(
                    lecture=lecture_id,
                    chunk_index=chunk_index,
                    db_idx=chunk_index,
                    variant=variant,
                )

                # Upsert the slide_pages row exactly once per page (use the
                # v1 text as the canonical ocr_text). For variants 2/3 the
                # row is already there from the v1 loop.
                if variant == "v1":
                    _upsert_page(
                        dest_conn,
                        page_id=page_id,
                        lecture_id=lecture_id,
                        chunk_index=chunk_index,
                        course_code=course_code,
                        ocr_text=it["text"],
                    )

                _upsert_chunk(
                    dest_conn,
                    chunk_id=chunk_id,
                    page_id=page_id,
                    variant=variant,
                    chunk_text=it["text"],
                    vector=matrix[i],
                )
                n_written += 1

            dest_conn.commit()
            counts[variant] = n_written
            logger.info("[%s] wrote %d chunk rows", variant, n_written)

        dest_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) "
            "VALUES ('last_modified', strftime('%Y-%m-%dT%H:%M:%f', 'now'))"
        )
        dest_conn.commit()
    finally:
        dest_conn.close()

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build slideqa.db with v1/v2/v3 page embeddings directly from "
            "MinerU on-disk output (no metadata DB required)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--processed-dir",
        required=True,
        type=Path,
        help=(
            "Root directory to scan recursively for *.v1.md. The folder "
            "containing each *.v1.md is treated as one lecture's work dir "
            "and must also contain *.master.md and *_content_list.json."
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        type=Path,
        help=(
            "Path to write slideqa.db. "
            "Default: <processed-dir>/slideqa.db (lives next to the lectures)."
        ),
    )
    p.add_argument(
        "--course-code",
        default="CS 288",
        help="Value stored in slide_pages.course_code (e.g. 'CS 288').",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-4B",
        help="HuggingFace SentenceTransformer model id.",
    )
    p.add_argument(
        "--variants",
        default="v1,v2,v3",
        help="Comma-separated subset of v1,v2,v3 to build.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Texts per encode() call.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    processed_dir: Path = args.processed_dir
    if not processed_dir.exists():
        print(f"ERROR: --processed-dir does not exist: {processed_dir}", file=sys.stderr)
        return 1

    output_path: Path = args.output or (processed_dir / "slideqa.db")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    counts = build_slideqa_db_from_disk(
        processed_dir=processed_dir,
        dest_path=output_path,
        course_code=args.course_code,
        model_name=args.model,
        variants=variants,
        batch_size=args.batch_size,
    )

    print()
    for v in ("v1", "v2", "v3"):
        if v in counts:
            print(f"[{v}] {counts[v]} chunk rows written → {output_path}")
    print(f"\nslideqa.db: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
