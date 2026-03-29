"""
QA Agent for TAI-SlideQA Phase 4.

Generates answers with citations from retrieved slide pages using OpenAI.
"""

import re
import logging
from app.services.slideqa.schema import Citation, QAResponse, RetrievedPage

logger = logging.getLogger(__name__)

_CITE_PATTERN = re.compile(r"\[CITE:\s*([^\]]+)\]")

_SYSTEM_PROMPT = (
    "You are a teaching assistant answering questions about lecture slides. "
    "Answer based only on the provided slide context. "
    "Always cite the specific slide pages you used in your answer "
    "using the format [CITE: page_id]."
)


def _build_context(retrieved_pages: list) -> str:
    """Build a context string from retrieved slide pages."""
    parts: list[str] = []
    for page in retrieved_pages:
        section = f"--- Page {page.page_id} ---\n{page.ocr_text}"
        if page.caption:
            section += f"\n{page.caption}"
        if page.objects:
            section += "\n" + "\n".join(page.objects)
        parts.append(section)
    return "\n\n".join(parts)


def _parse_citations(response_text: str, retrieved_pages: list) -> list[Citation]:
    """Extract [CITE: page_id] tags and map to Citation objects."""
    page_map = {p.page_id: p for p in retrieved_pages}
    found_ids = _CITE_PATTERN.findall(response_text)
    citations: list[Citation] = []
    seen: set[str] = set()
    for raw_id in found_ids:
        page_id = raw_id.strip()
        if page_id in seen:
            continue
        seen.add(page_id)
        page = page_map.get(page_id)
        if page is None:
            continue
        citations.append(
            Citation(
                course_code=page.course_code,
                lecture_id=page.lecture_id,
                page_number=page.page_number,
                page_id=page.page_id,
                score=page.score,
            )
        )
    return citations


def answer(
    question: str,
    retrieved_pages: list,
    mode: str,
    openai_client,
    openai_model: str = "gpt-4o",
) -> QAResponse:
    """Generate an answer with citations from retrieved slide pages.

    Args:
        question: The student's question.
        retrieved_pages: List of SlidePageResult objects from the retriever.
        mode: Pipeline mode string ("A1", "A2", "A3", "A4").
        openai_client: An openai.OpenAI sync client instance.
        openai_model: Model identifier for OpenAI completions.

    Returns:
        QAResponse with answer, citations, and abstained flag.
    """
    retrieved_page_refs = [
        RetrievedPage(page_id=p.page_id, score=p.score, rank=p.rank)
        for p in retrieved_pages
    ]

    if not retrieved_pages:
        return QAResponse(
            answer="I could not find relevant slides to answer your question.",
            citations=[],
            retrieved_pages=retrieved_page_refs,
            mode=mode,
            abstained=True,
        )

    context = _build_context(retrieved_pages)
    user_prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"
    )

    try:
        completion = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
        )
    except Exception as exc:
        logger.error("qa_agent.answer: OpenAI API call failed: %s", exc)
        return QAResponse(
            answer="An error occurred while generating the answer.",
            citations=[],
            retrieved_pages=retrieved_page_refs,
            mode=mode,
            abstained=True,
        )

    response_text: str = completion.choices[0].message.content or ""
    citations = _parse_citations(response_text, retrieved_pages)
    abstained = len(citations) == 0

    return QAResponse(
        answer=response_text,
        citations=citations,
        retrieved_pages=retrieved_page_refs,
        mode=mode,
        abstained=abstained,
    )
