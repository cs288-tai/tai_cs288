"""
Page-level retrieval evaluation metrics for TAI-SlideQA.

OFFSET RULE (critical — read this first):
  MinerU content_list.json stores page numbers as 0-based "page_idx".
  SlidePageResult.page_number is 1-based (as stored in the SQLite DB).

  So a QA item with gold_page_ids=[0] is a hit when page_number==1 is retrieved.
  Formula: retrieved_page is a hit  ←→  (retrieved.page_number - 1) in gold_page_ids

All functions in this module apply this offset consistently.
No function mutates its inputs.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Type alias for the retrieve_fn signature used by recall_at_k / mrr.
# retrieve_fn(query, variant, course_code, top_k) -> list[result]
# Each result must have a .page_number attribute (1-based int).
# ---------------------------------------------------------------------------
RetrieveFn = Callable[[str, str, Optional[str], int], list]


# ---------------------------------------------------------------------------
# A. hits_at_k — single question, single retrieval result list
# ---------------------------------------------------------------------------


def hits_at_k(
    retrieved: list,
    gold_page_ids: list[int],
    k: int,
) -> bool:
    """Return True if any of the top-k retrieved results matches a gold page.

    Args:
        retrieved:     Ordered list of retrieval results (best first).
                       Each item must have a `.page_number` attribute (1-based int).
        gold_page_ids: List of 0-based MinerU page_idx values that count as correct.
        k:             How many top results to consider.

    Returns:
        True if at least one result in retrieved[:k] has
        (result.page_number - 1) in gold_page_ids.
        False if retrieved is empty, gold_page_ids is empty, or no hit found.
    """
    if not retrieved or not gold_page_ids:
        return False

    gold_set = set(gold_page_ids)
    # Only look at the first k results.
    for result in retrieved[:k]:
        # Convert 1-based page_number back to 0-based page_idx for comparison.
        if (result.page_number - 1) in gold_set:
            return True
    return False


# ---------------------------------------------------------------------------
# B. recall_at_k — over a full question set
# ---------------------------------------------------------------------------


def recall_at_k(
    questions: list[dict[str, Any]],
    retrieve_fn: RetrieveFn,
    variant: str,
    course_code: Optional[str],
    k: int,
) -> float:
    """Compute Recall@k: fraction of questions where hits_at_k is True.

    Args:
        questions:    List of QA dicts. Each must have:
                        "question"      — query string
                        "gold_page_ids" — list of 0-based page_idx values
        retrieve_fn:  Callable with signature
                        retrieve_fn(query: str, variant: str,
                                    course_code: str | None, top_k: int)
                        → list of result objects with .page_number (1-based)
        variant:      Index variant to retrieve from ("v1", "v2", or "v3").
        course_code:  Optional course filter passed to retrieve_fn.
        k:            Top-k cutoff for hits_at_k.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for empty question list.
    """
    if not questions:
        return 0.0

    hits = 0
    for q in questions:
        retrieved = retrieve_fn(q["question"], variant, course_code, k)
        if hits_at_k(retrieved, q["gold_page_ids"], k):
            hits += 1

    return hits / len(questions)


# ---------------------------------------------------------------------------
# C. mrr — Mean Reciprocal Rank over a full question set
# ---------------------------------------------------------------------------


def mrr(
    questions: list[dict[str, Any]],
    retrieve_fn: RetrieveFn,
    variant: str,
    course_code: Optional[str],
    k: int,
) -> float:
    """Compute Mean Reciprocal Rank (MRR) at cutoff k.

    For each question, find the rank (1-indexed) of the first retrieved result
    that matches a gold page. Reciprocal rank = 1/rank. If no hit within top-k,
    reciprocal rank = 0. MRR is the average over all questions.

    Args:
        questions:    List of QA dicts with "question" and "gold_page_ids" keys.
        retrieve_fn:  Same signature as in recall_at_k.
        variant:      Index variant ("v1", "v2", or "v3").
        course_code:  Optional course filter.
        k:            Top-k cutoff; results beyond rank k are ignored.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for empty question list.
    """
    if not questions:
        return 0.0

    total_rr = 0.0

    for q in questions:
        retrieved = retrieve_fn(q["question"], variant, course_code, k)
        gold_set = set(q["gold_page_ids"])

        # Walk top-k results to find the first hit.
        for rank, result in enumerate(retrieved[:k], start=1):
            if (result.page_number - 1) in gold_set:
                total_rr += 1.0 / rank
                break
        # If no hit found, reciprocal rank = 0 (add nothing).

    return total_rr / len(questions)


# ---------------------------------------------------------------------------
# D. citation_correctness — checks that cited pages are gold pages
# ---------------------------------------------------------------------------


def citation_correctness(
    questions: list[dict[str, Any]],
    qa_responses: list,
) -> float:
    """Compute citation correctness: fraction of responses where at least one
    cited page is in the gold page set.

    A citation is correct if:
        (citation.page_number - 1) in question["gold_page_ids"]

    This is a looser metric than recall_at_k because it only checks pages
    that the QA model chose to cite, not the full retrieval set.

    Args:
        questions:    List of QA dicts with "gold_page_ids" key (0-based page_idx).
        qa_responses: List of response objects. Each must have a `.citations`
                      attribute — an iterable of objects with `.page_number` (1-based).
                      Must be the same length as questions and in the same order.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for empty input.
    """
    if not questions:
        return 0.0

    hits = 0
    for q, response in zip(questions, qa_responses):
        gold_set = set(q["gold_page_ids"])
        # A hit means at least one citation.page_number - 1 is in gold_set.
        if any((c.page_number - 1) in gold_set for c in response.citations):
            hits += 1

    return hits / len(questions)


# ---------------------------------------------------------------------------
# E. contains_answer — answer-level metric (no retrieval needed)
# ---------------------------------------------------------------------------


def contains_answer(predicted: str, gold: str) -> bool:
    """Return True if the gold answer appears (case-insensitive) in the predicted answer.

    Both strings are stripped of leading/trailing whitespace before comparison.
    This is a substring match, not a word-boundary match.

    Args:
        predicted: The model's generated answer string.
        gold:      The ground-truth short answer string (from "answer_short" field).

    Returns:
        True if gold.strip().lower() is a substring of predicted.strip().lower().
        An empty gold string is trivially contained in any predicted string.
    """
    # Normalise both sides: strip whitespace and lower-case.
    gold_norm = gold.strip().lower()
    pred_norm = predicted.strip().lower()
    return gold_norm in pred_norm


def contains_answer_rate(
    questions: list[dict[str, Any]],
    predicted_answers: list[str],
) -> float:
    """Compute the fraction of questions where contains_answer is True.

    This is a pure answer-level metric — it does not involve retrieval.
    Use it to evaluate whether a QA model's free-text output covers the
    ground-truth answer string.

    Args:
        questions:         List of QA dicts. Each must have an "answer_short" key
                           containing the ground-truth short answer string.
        predicted_answers: List of model-generated answer strings, same length
                           and same order as questions.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for an empty question list.
    """
    if not questions:
        return 0.0

    hits = sum(
        1
        for q, pred in zip(questions, predicted_answers)
        if contains_answer(pred, q["answer_short"])
    )
    return hits / len(questions)
