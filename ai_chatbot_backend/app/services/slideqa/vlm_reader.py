"""
VLM (Vision Language Model) reader for TAI-SlideQA Phase 4.

Provides image-based augmentation for mode A4 using GPT-4o vision.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from app.services.slideqa.schema import QAResponse

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_VISUAL_KEYWORDS = frozenset({
    "chart", "graph", "plot", "diagram", "figure", "image",
    "picture", "table", "illustration", "visual", "show",
})


def should_use_vlm_read(
    retrieved_pages: list,
    mode: str,
    question: str,
    confidence_threshold: float = 0.3,
) -> bool:
    """Return True if VLM image read should be triggered.

    Always False for modes A1/A2/A3.
    For A4: True if top page score < confidence_threshold OR
            question contains visual keywords.
    """
    if mode != "A4":
        return False

    question_lower = question.lower()
    has_visual_keyword = any(kw in question_lower for kw in _VISUAL_KEYWORDS)
    if has_visual_keyword:
        return True

    if retrieved_pages:
        top_score = retrieved_pages[0].score
        if top_score < confidence_threshold:
            return True

    return False


def vlm_read_page(
    image_path: str,
    question: str,
    openai_client,
    openai_model: str = "gpt-4o",
    allowed_root: Optional[Path] = None,
) -> str:
    """Read a slide image with GPT-4o vision and return an answer string.

    Args:
        image_path: Absolute path to the image file.
        question: The student's question to answer from the image.
        openai_client: An openai.OpenAI sync client instance.
        openai_model: Model identifier (must support vision).
        allowed_root: Optional directory that image_path must reside under.
                      When provided, paths outside this root are rejected to
                      prevent path-traversal via DB-controlled image_path values.

    Returns:
        Response text from VLM, or "" on failure / non-image path.
    """
    path = Path(image_path).resolve()

    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            logger.error(
                "vlm_read_page: image path %s is outside allowed root %s", path, root
            )
            return ""

    if path.suffix.lower() not in _IMAGE_EXTENSIONS:
        logger.warning("vlm_read_page: non-image path rejected: %s", path)
        return ""

    try:
        with open(path, "rb") as f:
            image_bytes = f.read()
    except OSError as exc:
        logger.error("vlm_read_page: cannot read image %s: %s", image_path, exc)
        return ""

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"

    prompt_text = (
        f"Look at this slide image and answer the question: {question}. "
        "Cite specific elements you see."
    )

    try:
        completion = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{encoded}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                    ],
                }
            ],
            max_tokens=512,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        logger.error("vlm_read_page: OpenAI call failed: %s", exc)
        return ""


def augment_answer_with_vlm(
    qa_response: QAResponse,
    retrieved_pages: list,
    question: str,
    openai_client,
    openai_model: str = "gpt-4o",
    allowed_root: Optional[Path] = None,
) -> QAResponse:
    """For A4 mode: augment the answer with VLM image read of the top page.

    Only augments if retrieved_pages is non-empty and the top page has an
    image file. Returns a new QAResponse (immutable — original unchanged).
    """
    if not retrieved_pages:
        return qa_response

    top_page = retrieved_pages[0]
    image_path = top_page.image_path or ""

    if not image_path:
        return qa_response

    vlm_text = vlm_read_page(
        image_path=image_path,
        question=question,
        openai_client=openai_client,
        openai_model=openai_model,
        allowed_root=allowed_root,
    )

    if not vlm_text:
        return qa_response

    augmented_answer = qa_response.answer + f"\n\n[VLM Image Read]:\n{vlm_text}"

    return qa_response.model_copy(update={"answer": augmented_answer})
