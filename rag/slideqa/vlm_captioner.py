"""
VLM-based slide captioning using GPT-4o vision.

All functions follow the immutability pattern: new SlidePageRecord
instances are created via dataclasses.replace rather than mutation.
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rag.slideqa.schema import SlidePageRecord

if TYPE_CHECKING:
    import openai as _openai_type

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Describe this lecture slide in 2-4 sentences. "
    "Focus on the main concept and any figures, charts, or diagrams present."
)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


def caption_slide_page(
    client: "_openai_type.OpenAI",
    image_path: Path,
    model: str = "gpt-4o",
) -> str:
    """
    Caption a single slide image using GPT-4o vision.

    Args:
        client:     Initialised OpenAI client.
        image_path: Path to the slide image.
        model:      OpenAI model name.

    Returns:
        Caption string, or empty string on unrecoverable failure.
    """
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        logger.error("Rejected non-image path for captioning: %s", image_path)
        return ""

    if not image_path.exists():
        logger.error("Image not found for captioning: %s", image_path)
        return ""

    try:
        raw_bytes = image_path.read_bytes()
        b64_image = base64.b64encode(raw_bytes).decode("utf-8")
    except OSError as exc:
        logger.error("Failed to read image %s: %s", image_path, exc)
        return ""

    media_type = "jpeg" if suffix in (".jpg", ".jpeg") else "png"

    message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{media_type};base64,{b64_image}",
                },
            },
            {"type": "text", "text": _CAPTION_PROMPT},
        ],
    }

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[message],
                max_tokens=256,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "caption_slide_page attempt %d/%d failed for %s: %s; retrying in %.1fs",
                attempt + 1,
                _MAX_RETRIES,
                image_path,
                exc,
                wait,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait)

    logger.error("All retries exhausted for captioning %s", image_path)
    return ""


def caption_lecture_pages(
    client: "_openai_type.OpenAI",
    page_records: list[SlidePageRecord],
    model: str = "gpt-4o",
) -> list[SlidePageRecord]:
    """
    Caption all slide pages, skipping those that already have captions.

    Idempotent: records where ``caption`` is already set are returned
    unchanged.  Returns a new list; original records are never mutated.

    Args:
        client:       Initialised OpenAI client.
        page_records: Input slide records.
        model:        OpenAI model name.

    Returns:
        New list of SlidePageRecord with captions filled in.
    """
    updated: list[SlidePageRecord] = []
    for record in page_records:
        if record.caption is not None:
            updated.append(record)
            continue

        caption_text = caption_slide_page(
            client=client,
            image_path=Path(record.image_path),
            model=model,
        )
        # caption_text may be "" on failure; store None to keep idempotency
        new_caption: str | None = caption_text if caption_text else None
        updated.append(dataclasses.replace(record, caption=new_caption))

    return updated
