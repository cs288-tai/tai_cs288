"""
FastAPI route for TAI-SlideQA Phase 4.

Exposes a POST /slideqa/query endpoint that runs the full QA pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import openai as _openai_lib
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import verify_api_token
from app.services.slideqa.pipeline import run_pipeline
from app.services.slideqa.schema import QAResponse, SlideQARequest

logger = logging.getLogger(__name__)

router = APIRouter()

_DEFAULT_DB_PATH = "data/slideqa.db"

# Module-level OpenAI client singleton — avoids re-creating connection pool per request.
_openai_client: Optional[_openai_lib.OpenAI] = None


def _get_openai_client() -> _openai_lib.OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable is not set.",
            )
        _openai_client = _openai_lib.OpenAI(api_key=api_key)
    return _openai_client


@router.post("/query", response_model=QAResponse)
async def slideqa_query(
    request: SlideQARequest,
    _: bool = Depends(verify_api_token),
) -> QAResponse:
    """Run the SlideQA pipeline for a student question.

    Reads SLIDEQA_DB_PATH from environment variable (defaults to data/slideqa.db).
    OPENAI_API_KEY must be set; client is a process-level singleton.
    """
    db_path_str = os.environ.get("SLIDEQA_DB_PATH", _DEFAULT_DB_PATH)
    db_path = Path(db_path_str)

    openai_client = _get_openai_client()

    try:
        return run_pipeline(
            question=request.question,
            course_code=request.course_code,
            mode=request.mode,
            top_k=request.top_k,
            db_path=db_path,
            openai_client=openai_client,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("SlideQA pipeline error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the SlideQA query.",
        ) from exc
