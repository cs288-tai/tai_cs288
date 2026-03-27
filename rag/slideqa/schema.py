"""
SlidePageRecord schema and helpers for TAI-SlideQA.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Optional


class SlideType(Enum):
    """Slide content type classification."""

    TEXT = "text"
    CHART = "chart"
    TABLE = "table"
    DIAGRAM = "diagram"
    UNKNOWN = "unknown"


def make_page_id(course_code: str, lecture_id: str, page_number: int) -> str:
    """
    Build a canonical page ID.

    Format: "COURSE/lectureNN/page_NNN"

    Args:
        course_code: Course identifier (e.g. "CS288").
        lecture_id:  Lecture identifier (e.g. "lecture01").
        page_number: 1-indexed page number.

    Returns:
        Formatted page ID string.
    """
    if not course_code:
        raise ValueError("course_code must not be empty")
    if not lecture_id:
        raise ValueError("lecture_id must not be empty")
    if page_number < 1:
        raise ValueError(f"page_number must be >= 1, got {page_number}")
    return f"{course_code}/{lecture_id}/page_{page_number:03d}"


@dataclasses.dataclass(frozen=True)
class SlidePageRecord:
    """
    Immutable record representing a single slide page.

    page_id format: "COURSE/lectureNN/page_NNN"
    page_number is 1-indexed.
    """

    page_id: str
    course_code: str
    lecture_id: str
    page_number: int
    image_path: str
    ocr_text: str = ""
    caption: Optional[str] = None
    objects: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        if not self.page_id:
            raise ValueError("page_id must not be empty")
        if not self.course_code:
            raise ValueError("course_code must not be empty")
        if not self.lecture_id:
            raise ValueError("lecture_id must not be empty")
        if not self.image_path:
            raise ValueError("image_path must not be empty")
