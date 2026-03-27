"""
CLI preprocessing pipeline for TAI-SlideQA.

Usage:
    python -m rag.slideqa.preprocess \\
      --course CS288 \\
      --variant all \\
      --data-dir data/slides \\
      --db-path data/slideqa.db \\
      --openai-model gpt-4o \\
      --dpi 150 \\
      --skip-vlm
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

_COURSE_CODE_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

from rag.slideqa.index_builder import (
    build_all_variants,
    build_embeddings,
    init_db,
    load_page_records,
    upsert_page_records,
)
from rag.slideqa.ocr_extractor import extract_ocr_for_lecture
from rag.slideqa.schema import SlidePageRecord, make_page_id
from rag.slideqa.vlm_captioner import caption_lecture_pages
from rag.slideqa.vlm_objects import describe_lecture_objects

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _ensure_images(pdf_path: Path, images_dir: Path, dpi: int) -> bool:
    """
    Convert a PDF to images if images_dir has no PNG files yet.

    Returns True if images are available after this call.
    """
    existing = list(images_dir.glob("*_page_*.png"))
    if existing:
        logger.info("Images already exist in %s (%d files)", images_dir, len(existing))
        return True

    try:
        from rag.file_conversion_router.utils.pdf_to_image import pdf_to_images  # type: ignore

        pdf_to_images(
            pdf_path=pdf_path,
            output_dir=images_dir,
            dpi=dpi,
            image_format="png",
        )
        return True
    except Exception as exc:
        logger.error("pdf_to_images failed for %s: %s", pdf_path, exc)
        return False


def _derive_lecture_id(pdf_path: Path) -> str:
    """Derive a lecture identifier from the PDF stem."""
    return pdf_path.stem


def _build_records_for_lecture(
    course_code: str,
    lecture_id: str,
    images_dir: Path,
    content_list_path: Optional[Path],
) -> list[SlidePageRecord]:
    """
    Build initial SlidePageRecord list for one lecture.

    Args:
        course_code:       Course identifier.
        lecture_id:        Lecture identifier.
        images_dir:        Directory of page images.
        content_list_path: Optional MinerU content_list.json.

    Returns:
        List of SlidePageRecord (one per page).
    """
    ocr_map = extract_ocr_for_lecture(images_dir, content_list_path)
    image_files: dict[int, Path] = {}
    for img in sorted(images_dir.glob("*_page_*.png")):
        parts = img.stem.rsplit("_page_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            image_files[int(parts[1])] = img

    records: list[SlidePageRecord] = []
    for page_number, img_path in sorted(image_files.items()):
        try:
            record = SlidePageRecord(
                page_id=make_page_id(course_code, lecture_id, page_number),
                course_code=course_code,
                lecture_id=lecture_id,
                page_number=page_number,
                image_path=str(img_path),
                ocr_text=ocr_map.get(page_number, ""),
            )
            records.append(record)
        except Exception as exc:
            logger.error(
                "Failed to build record for page %d of %s: %s",
                page_number,
                lecture_id,
                exc,
            )

    return records


def _validate_course_code(course: str) -> None:
    """Reject course codes containing path separators or glob metacharacters."""
    if not _COURSE_CODE_RE.match(course):
        raise ValueError(
            f"Invalid course code {course!r}. "
            "Only alphanumeric characters, hyphens, and underscores are permitted."
        )


def _assert_within(path: Path, base: Path, label: str) -> None:
    """Raise ValueError if resolved path is not under base."""
    resolved = path.resolve()
    base_resolved = base.resolve()
    if not str(resolved).startswith(str(base_resolved) + "/") and resolved != base_resolved:
        raise ValueError(
            f"{label} {resolved} is outside the permitted base directory {base_resolved}"
        )


def _discover_pdfs(data_dir: Path, course: str) -> list[Path]:
    """Glob PDFs under data_dir/course at depth 1 and 2."""
    _validate_course_code(course)
    shallow = list(data_dir.glob(f"{course}/*.pdf"))
    nested = list(data_dir.glob(f"{course}/*/*.pdf"))
    all_pdfs: list[Path] = []
    for p in sorted(set(shallow + nested)):
        try:
            _assert_within(p, data_dir, "PDF")
            all_pdfs.append(p)
        except ValueError as exc:
            logger.warning("Skipping PDF outside data_dir: %s", exc)
    return all_pdfs


def _run_vlm(
    records: list[SlidePageRecord],
    openai_model: str,
) -> list[SlidePageRecord]:
    """Run captioning and object description using GPT-4o."""
    try:
        import openai  # type: ignore

        client = openai.OpenAI()
    except ImportError:
        logger.error("openai package not installed; skipping VLM step")
        return records

    logger.info("Running VLM captioning (%d records)", len(records))
    records = caption_lecture_pages(client, records, model=openai_model)
    logger.info("Running VLM object description (%d records)", len(records))
    records = describe_lecture_objects(client, records, model=openai_model)
    return records


def run_pipeline(
    course: str,
    data_dir: Path,
    db_path: Path,
    variant: str,
    openai_model: str,
    dpi: int,
    skip_vlm: bool,
) -> None:
    """
    Main pipeline: discover -> images -> OCR -> DB -> VLM -> embeddings.
    """
    data_dir = Path(data_dir).resolve()
    db_path = Path(db_path).resolve()

    # Validate db_path is within data_dir to prevent writes to arbitrary locations.
    # Allow db_path to live anywhere under data_dir's parent (one level up) so that
    # the default "data/slideqa.db" alongside "data/slides/" is accepted.
    allowed_db_root = data_dir.parent
    _assert_within(db_path, allowed_db_root, "--db-path")

    pdfs = _discover_pdfs(data_dir, course)
    if not pdfs:
        logger.warning("No PDFs found for course %s under %s", course, data_dir)

    init_db(db_path)

    all_records: list[SlidePageRecord] = []
    error_count = 0

    for pdf_path in pdfs:
        lecture_id = _derive_lecture_id(pdf_path)
        images_dir = pdf_path.parent / pdf_path.stem

        try:
            ok = _ensure_images(pdf_path, images_dir, dpi)
            if not ok:
                error_count += 1
                continue

            content_list_path = pdf_path.parent / "content_list.json"
            if not content_list_path.exists():
                content_list_path = None

            records = _build_records_for_lecture(
                course_code=course,
                lecture_id=lecture_id,
                images_dir=images_dir,
                content_list_path=content_list_path,
            )

            if records:
                upsert_page_records(db_path, records)
                all_records.extend(records)
                logger.info("Processed %d pages for %s", len(records), lecture_id)
        except Exception as exc:
            logger.error("Error processing %s: %s", pdf_path, exc)
            error_count += 1

    if not skip_vlm and all_records:
        try:
            # Merge with DB-persisted records so that already-captioned pages
            # are not re-sent to the OpenAI API (true idempotency across runs).
            db_records = load_page_records(db_path, course)
            merged = [
                db_records.get(r.page_id, r) for r in all_records
            ]
            merged = _run_vlm(merged, openai_model)
            upsert_page_records(db_path, merged)
            all_records = merged
        except Exception as exc:
            logger.error("VLM pipeline failed: %s", exc)
            error_count += 1

    if all_records:
        try:
            if variant == "all":
                counts = build_all_variants(db_path, all_records)
                logger.info("Embeddings built: %s", counts)
            else:
                count = build_embeddings(db_path, all_records, variant=variant)
                logger.info("Embeddings built for variant=%s: %d", variant, count)
        except Exception as exc:
            logger.error("Embedding step failed: %s", exc)
            error_count += 1

    print(
        f"\nSummary: course={course}, pdfs={len(pdfs)}, "
        f"pages={len(all_records)}, errors={error_count}"
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag.slideqa.preprocess",
        description="TAI-SlideQA preprocessing pipeline",
    )
    parser.add_argument("--course", required=True, help="Course code (e.g. CS288)")
    parser.add_argument(
        "--variant",
        default="all",
        choices=["v1", "v2", "v3", "all"],
        help="Embedding variant to build (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/slides",
        help="Root data directory (default: data/slides)",
    )
    parser.add_argument(
        "--db-path",
        default="data/slideqa.db",
        help="SQLite database path (default: data/slideqa.db)",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o",
        help="OpenAI model for VLM calls (default: gpt-4o)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PDF-to-image conversion (default: 150)",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM captioning and object description steps",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        course=args.course,
        data_dir=Path(args.data_dir),
        db_path=Path(args.db_path),
        variant=args.variant,
        openai_model=args.openai_model,
        dpi=args.dpi,
        skip_vlm=args.skip_vlm,
    )
    sys.exit(0)
