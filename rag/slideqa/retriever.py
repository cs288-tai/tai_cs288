"""
Retrieval module for TAI-SlideQA.

Provides dense cosine-similarity retrieval over slide embeddings stored in
SQLite, with optional BM25 re-ranking via Reciprocal Rank Fusion (RRF).
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

_VALID_VARIANTS = frozenset({"v1", "v2", "v3"})
_MAX_EMBEDDING_DIM = 16_384   # hard cap on vector dimension
_MAX_TOP_K = 200
_MIN_RRF_K = 1
_MAX_CACHE_ENTRIES = 12       # 3 variants × expected course codes

# Qwen3-Embedding asymmetric query prefix — must match encoding of slide chunks.
# Slide chunks were encoded as: "document_hierarchy_path: {path}\ndocument: {text}\n"
# Queries must use this instruction prefix so both live in the same embedding space.
_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


# ---------------------------------------------------------------------------
# Result dataclass (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlidePageResult:
    """Immutable result record for a single retrieved slide page."""

    page_id: str
    course_code: str
    lecture_id: str
    page_number: int
    image_path: str
    ocr_text: str
    caption: Optional[str]
    objects: Optional[tuple[str, ...]]
    score: float         # final combined score (higher = better)
    dense_score: float   # raw dense cosine score
    bm25_score: float    # raw BM25 score (0.0 if BM25 not used)
    rank: int            # 1-indexed rank in result list


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------


class Retriever:
    """Dense (+ optional BM25) retriever over a SlideQA SQLite index."""

    def __init__(self, db_path: Path, model_name: str = "Qwen/Qwen3-Embedding-4B") -> None:
        resolved = Path(db_path).resolve()
        if not resolved.exists():
            # Allow non-existent paths (will be created later), but reject traversal
            # by ensuring the parent exists and is a real directory.
            if not resolved.parent.exists():
                raise ValueError(f"db_path parent does not exist: {resolved.parent}")
        self._db_path = resolved
        self._model_name = model_name
        self._model: Any = None  # lazy-loaded SentenceTransformer
        # Bounded LRU cache: (variant, course_code) -> index dict
        self._index_cache: dict[tuple[str, Optional[str]], dict[str, Any]] = {}
        self._cache_order: list[tuple[str, Optional[str]]] = []  # LRU eviction order
        self._cache_lock = threading.Lock()  # guards _index_cache and _cache_order

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        index_variant: str,
        course_code: Optional[str],
        top_k: int = 5,
        use_bm25: bool = False,
        rrf_k: int = 60,
        chunk_agg: str = "max",
    ) -> list[SlidePageResult]:
        """Return top_k SlidePageResult sorted by score descending.

        When the index contains chunk-level embeddings (slide_chunks), each
        chunk is scored independently and then aggregated to its parent page
        using chunk_agg ("max", "sum", or "mean").  Falls back to page-level
        slide_embeddings when no chunks are present for (variant, course_code).
        """
        if index_variant not in _VALID_VARIANTS:
            raise ValueError(
                f"Invalid variant {index_variant!r}. Must be one of {sorted(_VALID_VARIANTS)}."
            )
        if not isinstance(top_k, int) or top_k < 1 or top_k > _MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {_MAX_TOP_K}, got {top_k!r}")
        if not isinstance(rrf_k, int) or rrf_k < _MIN_RRF_K:
            raise ValueError(f"rrf_k must be >= {_MIN_RRF_K}, got {rrf_k!r}")
        if chunk_agg not in ("max", "sum", "mean"):
            raise ValueError(f"chunk_agg must be 'max', 'sum', or 'mean', got {chunk_agg!r}")

        # 1. Embed query
        model = self._get_model()
        raw = model.encode(_QUERY_PREFIX + query)
        qv = np.array(raw, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm
        else:
            logger.warning("Query vector is zero-norm; all dense scores will be 0.0")

        # 2. Load (or use cached) index
        idx = self._load_index(index_variant, course_code)
        if idx["M"] is None:
            return []

        matrix: np.ndarray = idx["M"]   # shape [N, D]
        page_ids: list[str] = idx["page_ids"]      # one entry per row (chunk or page)
        metadata: list[dict[str, Any]] = idx["metadata"]  # parallel to page_ids
        is_chunk_index: bool = idx.get("is_chunk_index", False)

        # 3. Dense cosine scores (dot product on L2-normalised vectors)
        dense_scores: np.ndarray = matrix @ qv  # shape [N]

        # 4. If chunk index: aggregate per-chunk scores to page-level scores.
        if is_chunk_index:
            page_ids, metadata, dense_scores = self._aggregate_chunks(
                page_ids, metadata, dense_scores, chunk_agg
            )

        # 5. Optionally compute BM25 and combine via RRF (page-level only)
        if use_bm25:
            page_idx = {"page_ids": page_ids, "metadata": metadata}
            bm25_raw = self._compute_bm25(query, page_idx)
            final_scores, bm25_scores_map = self._combine_rrf(
                dense_scores, page_ids, bm25_raw, rrf_k
            )
        else:
            final_scores = dense_scores
            bm25_scores_map: dict[str, float] = {}

        # 6. Rank and return top_k
        n = len(page_ids)
        k = min(top_k, n)
        if k == n:
            top_idx = np.argsort(final_scores)[::-1]
        else:
            top_idx = np.argpartition(final_scores, -k)[-k:]
            top_idx = top_idx[np.argsort(final_scores[top_idx])[::-1]]

        results: list[SlidePageResult] = []
        for rank_pos, i in enumerate(top_idx, start=1):
            pid = page_ids[i]
            meta = metadata[i]
            results.append(
                SlidePageResult(
                    page_id=pid,
                    course_code=meta["course_code"],
                    lecture_id=meta["lecture_id"],
                    page_number=meta["page_number"],
                    image_path=meta["image_path"],
                    ocr_text=meta["ocr_text"],
                    caption=meta["caption"],
                    objects=meta["objects"],
                    score=float(final_scores[i]),
                    dense_score=float(dense_scores[i]),
                    bm25_score=bm25_scores_map.get(pid, 0.0),
                    rank=rank_pos,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info("Loading embedding model %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _read_db_version(self) -> str:
        """
        Read the last-modified timestamp from the meta table.

        More reliable than PRAGMA data_version, which is connection-local and
        does not reflect writes made by other processes or connections.
        Returns "0" if the meta table does not exist yet.
        """
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'last_modified'"
            ).fetchone()
            return row[0] if row else "0"
        except sqlite3.OperationalError:
            # meta table does not exist (older DB or first run)
            return "0"
        finally:
            conn.close()

    def _load_index(
        self, variant: str, course_code: Optional[str]
    ) -> dict[str, Any]:
        """Load and cache the embedding index for (variant, course_code).

        Thread-safe: uses self._cache_lock for all cache reads and writes.
        Staleness detection uses a meta table timestamp written by index_builder.
        """
        cache_key = (variant, course_code)
        dv_now = self._read_db_version()

        with self._cache_lock:
            cached = self._index_cache.get(cache_key)
            if cached is not None and cached["dv"] == dv_now:
                # Move to most-recently-used position
                if cache_key in self._cache_order:
                    self._cache_order.remove(cache_key)
                self._cache_order.append(cache_key)
                return cached

        # Build outside the lock (expensive DB + numpy work)
        idx = self._build_index(variant, course_code, dv_now)

        with self._cache_lock:
            # Evict LRU entry if at capacity
            if len(self._index_cache) >= _MAX_CACHE_ENTRIES and self._cache_order:
                oldest = self._cache_order.pop(0)
                self._index_cache.pop(oldest, None)
            self._index_cache[cache_key] = idx
            self._cache_order.append(cache_key)

        return idx

    def _build_index(
        self, variant: str, course_code: Optional[str], dv: int
    ) -> dict[str, Any]:
        """Query SQLite and build a numpy matrix + metadata lists.

        Prefers chunk-level embeddings (slide_chunks) when available for
        (variant, course_code).  Falls back to page-level slide_embeddings.
        The returned dict includes is_chunk_index=True when chunk rows are used
        so retrieve() knows to aggregate before ranking.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            # Check whether slide_chunks has any rows for this (variant, course_code)
            chunk_check_sql = (
                "SELECT COUNT(*) FROM slide_chunks sc "
                "JOIN slide_pages sp ON sc.page_id = sp.page_id "
                "WHERE sc.variant = ?"
            )
            chunk_params: list[Any] = [variant]
            if course_code is not None:
                chunk_check_sql += " AND sp.course_code = ?"
                chunk_params.append(course_code)

            try:
                chunk_count = conn.execute(chunk_check_sql, chunk_params).fetchone()[0]
            except sqlite3.OperationalError:
                chunk_count = 0  # slide_chunks table does not exist yet

            if chunk_count > 0:
                rows, is_chunk_index = self._fetch_chunk_rows(conn, variant, course_code)
            else:
                rows, is_chunk_index = self._fetch_page_rows(conn, variant, course_code)
        finally:
            conn.close()

        page_ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []

        for row in rows:
            blob = row["vector"]
            if not isinstance(blob, (bytes, bytearray)) or len(blob) % 4 != 0:
                logger.warning("Skipping row for page_id %s: malformed vector blob", row["page_id"])
                continue
            n_floats = len(blob) // 4
            if n_floats == 0 or n_floats > _MAX_EMBEDDING_DIM:
                logger.warning(
                    "Skipping row for page_id %s: vector dimension %d out of range",
                    row["page_id"], n_floats,
                )
                continue
            vec = np.array(struct.unpack(f"{n_floats}f", blob), dtype=np.float32)

            obj_raw = row["objects"] if "objects" in row.keys() else None
            objects: Optional[tuple[str, ...]] = None
            if obj_raw is not None:
                import json
                try:
                    objects = tuple(json.loads(obj_raw))
                except (json.JSONDecodeError, TypeError):
                    objects = None

            page_ids.append(row["page_id"])
            metadata.append(
                {
                    "course_code": row["course_code"],
                    "lecture_id": row["lecture_id"],
                    "page_number": row["page_number"],
                    "image_path": row["image_path"],
                    "ocr_text": row["ocr_text"] or "",
                    "caption": row["caption"] if "caption" in row.keys() else None,
                    "objects": objects,
                }
            )
            vectors.append(vec)

        if not vectors:
            return {
                "dv": dv, "M": None, "page_ids": [], "metadata": [],
                "is_chunk_index": is_chunk_index,
            }

        matrix = np.vstack(vectors).astype(np.float32)
        return {
            "dv": dv, "M": matrix, "page_ids": page_ids, "metadata": metadata,
            "is_chunk_index": is_chunk_index,
        }

    @staticmethod
    def _fetch_chunk_rows(
        conn: sqlite3.Connection, variant: str, course_code: Optional[str]
    ) -> tuple[list[Any], bool]:
        """Fetch rows from slide_chunks joined to slide_pages."""
        sql = (
            "SELECT sp.page_id, sp.course_code, sp.lecture_id, "
            "sp.page_number, sp.image_path, sp.ocr_text, sp.caption, sp.objects, "
            "sc.vector "
            "FROM slide_chunks sc "
            "JOIN slide_pages sp ON sc.page_id = sp.page_id "
            "WHERE sc.variant = ?"
        )
        params: list[Any] = [variant]
        if course_code is not None:
            sql += " AND sp.course_code = ?"
            params.append(course_code)
        return conn.execute(sql, params).fetchall(), True

    @staticmethod
    def _fetch_page_rows(
        conn: sqlite3.Connection, variant: str, course_code: Optional[str]
    ) -> tuple[list[Any], bool]:
        """Fetch rows from slide_embeddings joined to slide_pages (page-level fallback)."""
        sql = (
            "SELECT sp.page_id, sp.course_code, sp.lecture_id, "
            "sp.page_number, sp.image_path, sp.ocr_text, sp.caption, sp.objects, "
            "se.vector "
            "FROM slide_pages sp "
            "JOIN slide_embeddings se ON sp.page_id = se.page_id "
            "WHERE se.variant = ?"
        )
        params: list[Any] = [variant]
        if course_code is not None:
            sql += " AND sp.course_code = ?"
            params.append(course_code)
        return conn.execute(sql, params).fetchall(), False

    @staticmethod
    def _aggregate_chunks(
        page_ids: list[str],
        metadata: list[dict[str, Any]],
        scores: np.ndarray,
        agg: str,
    ) -> tuple[list[str], list[dict[str, Any]], np.ndarray]:
        """Aggregate chunk-level scores to unique page-level scores.

        Returns parallel (page_ids, metadata, scores) lists with one entry
        per unique page_id, in the order pages are first encountered.
        """
        # Collect scores per page, preserving first-seen metadata
        page_scores: dict[str, list[float]] = {}
        page_meta: dict[str, dict[str, Any]] = {}
        for i, pid in enumerate(page_ids):
            if pid not in page_scores:
                page_scores[pid] = []
                page_meta[pid] = metadata[i]
            page_scores[pid].append(float(scores[i]))

        unique_pids = list(page_scores.keys())
        agg_meta = [page_meta[pid] for pid in unique_pids]
        agg_vals: list[float] = []
        for pid in unique_pids:
            s = page_scores[pid]
            if agg == "max":
                agg_vals.append(max(s))
            elif agg == "sum":
                agg_vals.append(sum(s))
            else:  # mean
                agg_vals.append(sum(s) / len(s))

        return unique_pids, agg_meta, np.array(agg_vals, dtype=np.float32)

    def _compute_bm25(
        self, query: str, idx: dict[str, Any]
    ) -> dict[str, float]:
        """
        Compute BM25 scores for all pages in idx.

        Uses rank_bm25 library if available, falls back to simple TF-IDF-like
        word-overlap scoring.
        """
        metadata: list[dict[str, Any]] = idx["metadata"]
        page_ids: list[str] = idx["page_ids"]
        corpus = [m["ocr_text"] for m in metadata]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        tokenized_query = query.lower().split()

        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            bm25 = BM25Okapi(tokenized_corpus)
            raw_scores = bm25.get_scores(tokenized_query)
        except ImportError:
            raw_scores = self._fallback_bm25(tokenized_corpus, tokenized_query)

        return {page_ids[i]: float(raw_scores[i]) for i in range(len(page_ids))}

    @staticmethod
    def _fallback_bm25(
        tokenized_corpus: list[list[str]], tokenized_query: list[str]
    ) -> list[float]:
        """Simple word-overlap TF score when rank_bm25 is unavailable."""
        query_terms = set(tokenized_query)
        scores: list[float] = []
        for doc_tokens in tokenized_corpus:
            if not doc_tokens:
                scores.append(0.0)
                continue
            tf = sum(1 for t in doc_tokens if t in query_terms) / len(doc_tokens)
            scores.append(tf)
        return scores

    def _combine_rrf(
        self,
        dense_scores: np.ndarray,
        page_ids: list[str],
        bm25_scores: dict[str, float],
        rrf_k: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Combine dense and BM25 scores via Reciprocal Rank Fusion."""
        n = len(page_ids)

        # Build dense rank dict (1-indexed, rank 1 = highest score)
        dense_order = np.argsort(dense_scores)[::-1]
        dense_ranks: dict[str, int] = {page_ids[i]: int(r) + 1 for r, i in enumerate(dense_order)}

        # Build bm25 rank dict
        bm25_vals = [(pid, bm25_scores.get(pid, 0.0)) for pid in page_ids]
        bm25_sorted = sorted(bm25_vals, key=lambda x: x[1], reverse=True)
        bm25_ranks: dict[str, int] = {pid: r + 1 for r, (pid, _) in enumerate(bm25_sorted)}

        rrf_scores = self._rrf_combine(dense_ranks, bm25_ranks, rrf_k)

        combined = np.array(
            [rrf_scores.get(pid, 0.0) for pid in page_ids], dtype=np.float32
        )
        return combined, {pid: bm25_scores.get(pid, 0.0) for pid in page_ids}

    @staticmethod
    def _rrf_combine(
        dense_ranks: dict[str, int],
        bm25_ranks: dict[str, int],
        k: int,
    ) -> dict[str, float]:
        """
        Reciprocal Rank Fusion over two rank dicts.

        Score = 1/(k + rank_dense) + 1/(k + rank_bm25)
        """
        all_ids = set(dense_ranks) | set(bm25_ranks)
        scores: dict[str, float] = {}
        for pid in all_ids:
            r_dense = dense_ranks.get(pid, len(dense_ranks) + 1)
            r_bm25 = bm25_ranks.get(pid, len(bm25_ranks) + 1)
            scores[pid] = 1.0 / (k + r_dense) + 1.0 / (k + r_bm25)
        return scores
