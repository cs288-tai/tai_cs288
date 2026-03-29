"""
Pydantic schemas for TAI-SlideQA Phase 4.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    course_code: str
    lecture_id: str
    page_number: int
    page_id: str
    score: float


class RetrievedPage(BaseModel):
    page_id: str
    score: float
    rank: int


class QAResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_pages: List[RetrievedPage]
    mode: str           # "A1", "A2", "A3", "A4"
    abstained: bool     # True if no citations found


class SlideQARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    course_code: Optional[str] = Field(None, max_length=64)
    mode: str = Field("A3", pattern="^(A1|A2|A3|A4)$")
    top_k: int = Field(5, ge=1, le=20)
