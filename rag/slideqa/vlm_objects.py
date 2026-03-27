"""
VLM-based object/element description for slide pages.

classify_slide_type uses keyword heuristics on OCR text.
describe_objects calls GPT-4o with a type-specific prompt.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rag.slideqa.schema import SlidePageRecord, SlideType

if TYPE_CHECKING:
    import openai as _openai_type

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0

_CHART_KEYWORDS = {"figure", "chart", "graph", "plot", "histogram", "bar", "pie", "line"}
_TABLE_KEYWORDS = {"table", "row", "column", "cell", "grid"}
_DIAGRAM_KEYWORDS = {"diagram", "algorithm", "architecture", "flowchart", "uml", "schema"}

_OBJECT_PROMPTS: dict[SlideType, str] = {
    SlideType.CHART: (
        "List the key visual elements of this chart or figure. "
        "Include axis labels, legend entries, notable data points, and any annotations. "
        "Return one element per line."
    ),
    SlideType.TABLE: (
        "List the column headers and describe the data contained in this table. "
        "Note any notable values or patterns. "
        "Return one element per line."
    ),
    SlideType.DIAGRAM: (
        "List the components, nodes, arrows, and labels visible in this diagram. "
        "Describe the relationships shown. "
        "Return one element per line."
    ),
    SlideType.TEXT: (
        "List the main bullet points, headings, and key terms on this text slide. "
        "Return one item per line."
    ),
    SlideType.UNKNOWN: (
        "List the distinct visual and textual elements on this slide. "
        "Return one element per line."
    ),
}


def classify_slide_type(ocr_text: str) -> SlideType:
    """
    Heuristic classification based on OCR text keywords.

    Priority order: CHART > TABLE > DIAGRAM > TEXT.

    Args:
        ocr_text: Raw OCR text for a slide page.

    Returns:
        SlideType classification.
    """
    lower = ocr_text.lower()
    tokens = set(lower.split())

    if tokens & _CHART_KEYWORDS:
        return SlideType.CHART
    if tokens & _TABLE_KEYWORDS:
        return SlideType.TABLE
    if tokens & _DIAGRAM_KEYWORDS:
        return SlideType.DIAGRAM
    if lower.strip():
        return SlideType.TEXT
    return SlideType.UNKNOWN


def get_object_prompt(slide_type: SlideType) -> str:
    """
    Return the type-specific VLM prompt for object description.

    Args:
        slide_type: Classified slide type.

    Returns:
        Prompt string.
    """
    return _OBJECT_PROMPTS.get(slide_type, _OBJECT_PROMPTS[SlideType.UNKNOWN])


def describe_objects(
    client: "_openai_type.OpenAI",
    image_path: Path,
    slide_type: SlideType,
    model: str = "gpt-4o",
) -> tuple[str, ...]:
    """
    Call GPT-4o to enumerate visual/textual objects on a slide.

    Args:
        client:     Initialised OpenAI client.
        image_path: Path to the slide image.
        slide_type: Pre-classified slide type.
        model:      OpenAI model name.

    Returns:
        List of object description strings (one per line from the model).
        Returns empty list on failure.
    """
    import base64

    image_path = Path(image_path)
    suffix = image_path.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        logger.error("Rejected non-image path for object description: %s", image_path)
        return ()

    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        return ()

    try:
        raw_bytes = image_path.read_bytes()
        b64_image = base64.b64encode(raw_bytes).decode("utf-8")
    except OSError as exc:
        logger.error("Failed to read image %s: %s", image_path, exc)
        return ()

    media_type = "jpeg" if suffix in (".jpg", ".jpeg") else "png"
    prompt = get_object_prompt(slide_type)

    message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{media_type};base64,{b64_image}",
                },
            },
            {"type": "text", "text": prompt},
        ],
    }

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[message],
                max_tokens=512,
            )
            content = response.choices[0].message.content or ""
            return tuple(line.strip() for line in content.splitlines() if line.strip())
        except Exception as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "describe_objects attempt %d/%d failed for %s: %s; retrying in %.1fs",
                attempt + 1,
                _MAX_RETRIES,
                image_path,
                exc,
                wait,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait)

    logger.error("All retries exhausted for describe_objects on %s", image_path)
    return ()


def describe_lecture_objects(
    client: "_openai_type.OpenAI",
    page_records: list[SlidePageRecord],
    model: str = "gpt-4o",
) -> list[SlidePageRecord]:
    """
    Describe objects for all pages, skipping those already annotated.

    Idempotent: records where ``objects`` is not None are returned
    unchanged.  Returns a new list; original records are never mutated.

    Args:
        client:       Initialised OpenAI client.
        page_records: Input slide records.
        model:        OpenAI model name.

    Returns:
        New list of SlidePageRecord with objects filled in.
    """
    updated: list[SlidePageRecord] = []
    for record in page_records:
        if record.objects is not None:
            updated.append(record)
            continue

        slide_type = classify_slide_type(record.ocr_text)
        objects: tuple[str, ...] = describe_objects(
            client=client,
            image_path=Path(record.image_path),
            slide_type=slide_type,
            model=model,
        )
        updated.append(dataclasses.replace(record, objects=objects))

    return updated
