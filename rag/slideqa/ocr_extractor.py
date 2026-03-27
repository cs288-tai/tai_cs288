"""
OCR extraction utilities for slide pages.

Supports two backends:
1. MinerU content_list.json (preferred when available)
2. easyocr fallback for individual images
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_ocr_from_content_list(content_list_path: Path) -> dict[int, str]:
    """
    Parse a MinerU content_list.json and group text by 0-based page index.

    Args:
        content_list_path: Path to content_list.json produced by MinerU.

    Returns:
        Dict mapping 0-based page_idx -> concatenated text.
        Returns empty dict on parse failure.
    """
    content_list_path = Path(content_list_path)
    if not content_list_path.exists():
        logger.warning("content_list.json not found: %s", content_list_path)
        return {}

    try:
        raw = content_list_path.read_text(encoding="utf-8")
        items = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to parse content_list.json %s: %s", content_list_path, exc)
        return {}

    if not isinstance(items, list):
        logger.error("content_list.json root must be a list, got %s", type(items).__name__)
        return {}

    page_texts: dict[int, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        page_idx = item.get("page_idx")
        text = item.get("text", "")
        if page_idx is None or item.get("type") != "text":
            continue
        try:
            page_texts.setdefault(int(page_idx), []).append(str(text))
        except (ValueError, TypeError):
            logger.warning("Skipping item with non-integer page_idx: %r", page_idx)
            continue

    return {idx: "\n".join(parts) for idx, parts in page_texts.items()}


_easyocr_reader = None


def _get_easyocr_reader():
    """Return a module-level cached easyocr.Reader (loaded once)."""
    global _easyocr_reader  # noqa: PLW0603
    if _easyocr_reader is None:
        import easyocr  # type: ignore
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def extract_ocr_from_image(image_path: Path) -> str:
    """
    Extract text from a single image using easyocr.

    Args:
        image_path: Path to the image file.

    Returns:
        Extracted text string, or empty string on failure.
    """
    try:
        import easyocr as _easyocr_check  # type: ignore  # noqa: F401
    except ImportError:
        logger.error("easyocr is not installed; cannot perform image-based OCR")
        return ""

    image_path = Path(image_path)
    if not image_path.exists():
        logger.warning("Image not found for OCR: %s", image_path)
        return ""

    try:
        reader = _get_easyocr_reader()
        results = reader.readtext(str(image_path), detail=0)
        return "\n".join(results)
    except Exception as exc:
        logger.error("easyocr failed on %s: %s", image_path, exc)
        return ""


def extract_ocr_for_lecture(
    images_dir: Path,
    content_list_path: Optional[Path] = None,
) -> dict[int, str]:
    """
    Orchestrate OCR for all pages in a lecture directory.

    Uses content_list.json when provided; falls back to easyocr for
    pages that are missing from the content list.

    Images must follow naming convention: ``<prefix>_page_NNN.png``
    (1-indexed page numbers, zero-padded to 3 digits).

    Args:
        images_dir:        Directory containing page images.
        content_list_path: Optional path to MinerU content_list.json.

    Returns:
        Dict mapping 1-based page_number -> ocr_text.
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        logger.error("images_dir does not exist: %s", images_dir)
        return {}

    # Collect available page images (1-indexed)
    image_files: dict[int, Path] = {}
    for img in sorted(images_dir.glob("*_page_*.png")):
        stem = img.stem  # e.g. "lecture01_page_003"
        parts = stem.rsplit("_page_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            image_files[int(parts[1])] = img

    if not image_files:
        logger.warning("No page images found in %s", images_dir)
        return {}

    # 0-based index from content list (if provided)
    content_list_map: dict[int, str] = {}
    if content_list_path is not None:
        content_list_map = extract_ocr_from_content_list(content_list_path)

    result: dict[int, str] = {}
    for page_number, img_path in sorted(image_files.items()):
        zero_idx = page_number - 1
        if zero_idx in content_list_map:
            result[page_number] = content_list_map[zero_idx]
        else:
            logger.debug("Falling back to easyocr for page %d (%s)", page_number, img_path)
            result[page_number] = extract_ocr_from_image(img_path)

    return result
