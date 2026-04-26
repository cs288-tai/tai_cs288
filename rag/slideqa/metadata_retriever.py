"""
MetadataDBRetriever — reads chunk embeddings from the existing
CS 288_metadata_new.db `chunks` table (no slideqa.db required).

chunks table columns used:
    chunk_uuid   TEXT PK
    text         TEXT      — OCR/slide text (used as ocr_text)
    file_path    TEXT      — full relative path including PDF filename
    course_code  TEXT
    chunk_index  INTEGER   — 0-based page index (same numbering as gold_page_ids)
    vector       BLOB      — float32 embedding

Mapping to eval pipeline:
    chunk_index  == gold_page_id  (both 0-based)
    page_number  = chunk_index + 1  (1-based, as eval.py expects)
    eval.py checks: (page_number - 1) in gold_set
                  = (chunk_index + 1 - 1) in gold_set
                  = chunk_index in gold_set  ✓

The `variant` parameter is accepted for API compatibility but ignored
(the metadata DB stores a single embedding per chunk).
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MAX_EMBEDDING_DIM = 16_384
_MAX_TOP_K = 200
_MAX_CACHE_ENTRIES = 4


@dataclass(frozen=True)
class MetadataPageResult:
    """Immutable result record for a single retrieved slide page."""

    page_id: str
    course_code: str
    lecture_id: str
    page_number: int       # 1-based; equals chunk_index
    image_path: str
    ocr_text: str
    caption: Optional[str]
    objects: Optional[tuple[str, ...]]
    score: float
    dense_score: float
    bm25_score: float
    rank: int


class MetadataDBRetriever:
    """Dense retriever over the `chunks` table in an existing metadata SQLite DB.

    Designed to plug into the eval_benchmark.py retrieve_fn interface:
        retrieve(query, index_variant, course_code, top_k) → list[MetadataPageResult]

    Chunk scores are aggregated to page level by max-pool before ranking.
    """

    def __init__(self, db_path: Path, model_name: str = "Qwen/Qwen3-Embedding-4B") -> None:
        self._db_path = Path(db_path).resolve()
        self._model_name = model_name
        self._model: Any = None
        self._index_cache: dict[Optional[str], dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    def retrieve(
        self,
        query: str,
        index_variant: str,
        course_code: Optional[str],
        top_k: int = 5,
        use_bm25: bool = False,
        rrf_k: int = 60,
        chunk_agg: str = "max",
    ) -> list[MetadataPageResult]:
        """Return top_k MetadataPageResult sorted by score descending.

        Args:
            query:         Question text to embed and score against.
            index_variant: Accepted but ignored (metadata DB has one embedding per chunk).
            course_code:   Filter to chunks where chunks.course_code == course_code.
                           Pass None to include all courses.
            top_k:         Number of page-level results to return.
            use_bm25:      Not implemented; accepted for API compatibility.
            rrf_k:         Not implemented; accepted for API compatibility.
            chunk_agg:     Aggregation over chunks sharing the same (file_path, chunk_index).
                           "max" (default), "sum", or "mean".
        """
        if not isinstance(top_k, int) or top_k < 1 or top_k > _MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {_MAX_TOP_K}, got {top_k!r}")

        model = self._get_model()
        raw = model.encode(query)
        qv = np.array(raw, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm
        else:
            logger.warning("Query vector is zero-norm; all scores will be 0.0")

        idx = self._load_index(course_code)
        if idx["M"] is None:
            return []

        matrix: np.ndarray = idx["M"]
        row_meta: list[dict[str, Any]] = idx["row_meta"]

        dense_scores: np.ndarray = matrix @ qv

        page_scores, page_meta = self._aggregate(row_meta, dense_scores, chunk_agg)

        unique_pids = list(page_scores.keys())
        agg_vals = np.array([page_scores[pid] for pid in unique_pids], dtype=np.float32)

        n = len(unique_pids)
        k = min(top_k, n)
        if k == n:
            top_idx = np.argsort(agg_vals)[::-1]
        else:
            top_idx = np.argpartition(agg_vals, -k)[-k:]
            top_idx = top_idx[np.argsort(agg_vals[top_idx])[::-1]]

        results: list[MetadataPageResult] = []
        for rank_pos, i in enumerate(top_idx, start=1):
            pid = unique_pids[i]
            meta = page_meta[pid]
            results.append(MetadataPageResult(
                page_id=pid,
                course_code=meta["course_code"],
                lecture_id=meta["lecture_id"],
                page_number=meta["page_number"],
                image_path="",
                ocr_text=meta["ocr_text"],
                caption=None,
                objects=None,
                score=float(agg_vals[i]),
                dense_score=float(agg_vals[i]),
                bm25_score=0.0,
                rank=rank_pos,
            ))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info("Loading embedding model %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _load_index(self, course_code: Optional[str]) -> dict[str, Any]:
        with self._cache_lock:
            cached = self._index_cache.get(course_code)
            if cached is not None:
                return cached

        idx = self._build_index(course_code)

        with self._cache_lock:
            if len(self._index_cache) >= _MAX_CACHE_ENTRIES:
                oldest = next(iter(self._index_cache))
                del self._index_cache[oldest]
            self._index_cache[course_code] = idx

        return idx

    def _build_index(self, course_code: Optional[str]) -> dict[str, Any]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            sql = (
                "SELECT chunk_uuid, text, file_path, course_code, chunk_index, vector "
                "FROM chunks WHERE vector IS NOT NULL AND chunk_index IS NOT NULL"
            )
            params: list[Any] = []
            if course_code is not None:
                sql += " AND course_code = ?"
                params.append(course_code)
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        row_meta: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []

        for row in rows:
            blob = row["vector"]
            if not isinstance(blob, (bytes, bytearray)) or len(blob) % 4 != 0:
                continue
            n_floats = len(blob) // 4
            if n_floats == 0 or n_floats > _MAX_EMBEDDING_DIM:
                continue
            vec = np.array(struct.unpack(f"{n_floats}f", blob), dtype=np.float32)

            file_path = row["file_path"] or ""
            lecture_id = Path(file_path).name  # last component: "CS288_sp26_01_Intro.pdf"
            chunk_index = int(row["chunk_index"])  # 0-based, same as gold_page_ids
            page_number = chunk_index + 1           # eval.py expects 1-based
            page_id = f"{lecture_id}_page_{chunk_index}"

            row_meta.append({
                "page_id": page_id,
                "course_code": row["course_code"] or "",
                "lecture_id": lecture_id,
                "page_number": page_number,
                "ocr_text": row["text"] or "",
            })
            vectors.append(vec)

        if not vectors:
            return {"M": None, "row_meta": []}

        matrix = np.vstack(vectors).astype(np.float32)
        return {"M": matrix, "row_meta": row_meta}

    @staticmethod
    def _aggregate(
        row_meta: list[dict[str, Any]],
        scores: np.ndarray,
        agg: str,
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        """Aggregate per-row scores to unique page_id scores.

        Returns:
            page_scores: {page_id: aggregated_score}
            page_meta:   {page_id: metadata dict}  (first-seen metadata kept)
        """
        page_score_lists: dict[str, list[float]] = {}
        page_meta: dict[str, dict[str, Any]] = {}

        for i, meta in enumerate(row_meta):
            pid = meta["page_id"]
            if pid not in page_score_lists:
                page_score_lists[pid] = []
                page_meta[pid] = meta
            page_score_lists[pid].append(float(scores[i]))

        page_scores: dict[str, float] = {}
        for pid, s_list in page_score_lists.items():
            if agg == "max":
                page_scores[pid] = max(s_list)
            elif agg == "sum":
                page_scores[pid] = sum(s_list)
            else:  # mean
                page_scores[pid] = sum(s_list) / len(s_list)

        return page_scores, page_meta
