"""
End-to-end pipeline for TAI-SlideQA Phase 4.

Orchestrates retrieval → answer generation → optional VLM augmentation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from app.services.slideqa import vlm_reader
from app.services.slideqa.qa_agent import answer as qa_answer
from app.services.slideqa.schema import QAResponse

logger = logging.getLogger(__name__)

_MODE_TO_VARIANT = {"A1": "v1", "A2": "v2", "A3": "v3", "A4": "v3"}
VALID_MODES = frozenset({"A1", "A2", "A3", "A4"})


def _ensure_rag_importable() -> None:
    """Add the repo root to sys.path so rag.slideqa is importable."""
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def run_pipeline(
    question: str,
    course_code: Optional[str],
    mode: str,
    top_k: int,
    db_path: Path,
    openai_client,
    openai_model: str = "gpt-4o",
    retriever=None,
) -> QAResponse:
    """End-to-end pipeline: retrieve → answer → optionally augment with VLM.

    Args:
        question: The student's question.
        course_code: Optional course filter for retrieval.
        mode: Pipeline mode ("A1", "A2", "A3", "A4").
        top_k: Number of pages to retrieve.
        db_path: Path to the SlideQA SQLite database.
        openai_client: An openai.OpenAI sync client instance.
        openai_model: OpenAI model identifier.
        retriever: Optional Retriever instance (injectable for testing).

    Returns:
        QAResponse with answer, citations, and metadata.

    Raises:
        ValueError: If mode is not one of the valid modes.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode {mode!r}. Must be one of {sorted(VALID_MODES)}."
        )

    index_variant = _MODE_TO_VARIANT[mode]

    if retriever is None:
        _ensure_rag_importable()
        from rag.slideqa.retriever import Retriever

        retriever = Retriever(db_path=db_path)

    retrieved_pages = retriever.retrieve(
        query=question,
        index_variant=index_variant,
        course_code=course_code,
        top_k=top_k,
    )

    qa_response = qa_answer(
        question=question,
        retrieved_pages=retrieved_pages,
        mode=mode,
        openai_client=openai_client,
        openai_model=openai_model,
    )

    if mode == "A4":
        from pathlib import Path as _Path
        try:
            from app.config import settings as _settings
            _allowed_root: Optional[_Path] = _Path(_settings.DATA_DIR).resolve()
        except Exception:
            _allowed_root = None
        qa_response = vlm_reader.augment_answer_with_vlm(
            qa_response=qa_response,
            retrieved_pages=retrieved_pages,
            question=question,
            openai_client=openai_client,
            openai_model=openai_model,
            allowed_root=_allowed_root,
        )

    return qa_response
