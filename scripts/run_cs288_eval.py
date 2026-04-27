#!/usr/bin/env python3
"""
Local ablation eval runner for CS 288 SlideQA.

Loads the benchmark JSONL produced by generate_cs288_qa.py, runs retrieval
for each variant (v1, v2, v3) using the local SQLite embedding index, computes
page-level metrics, and saves a results table.

Metrics computed:
  Recall@1, Recall@3, Recall@5  — fraction of questions with a gold page in top-k
  MRR@5                         — Mean Reciprocal Rank (within top-5)
  contains_answer_rate          — fraction where gold answer_short ⊆ predicted answer
                                  (only computed when --predictions-file is supplied)
  citation_hit_rate             — None (requires a QA agent, not in local path)

Gold page matching rule (OFFSET):
  gold_page_ids uses 0-based MinerU page_idx.
  SlidePageResult.page_number is 1-based.
  Hit condition: (result.page_number - 1) in gold_page_ids.

Usage:
  conda activate eecs-rag
  python scripts/run_cs288_eval.py \\
      --benchmark cs288_benchmark.jsonl \\
      --db-path data/slideqa.db \\
      --output-dir results/

Output:
  results/cs288_eval_results.json  — per-variant metrics
  results/cs288_eval_results.csv   — same data as CSV (easy to paste into a table)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Make rag package importable when running from repo root.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag.slideqa.eval import contains_answer_rate, mrr, recall_at_k
from rag.slideqa.retriever import Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_benchmark(benchmark_path: Path) -> list[dict]:
    """Load the benchmark JSONL into a list of QA dicts."""
    questions: list[dict] = []
    with benchmark_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _filter_by_variant(questions: list[dict], variant: str) -> list[dict]:
    """Return only questions that belong to a given variant."""
    return [q for q in questions if q.get("variant") == variant]


def _filter_by_types(
    questions: list[dict], allowed_types: Optional[set[str]]
) -> list[dict]:
    """Drop questions whose ``question_type`` is not in ``allowed_types``."""
    if not allowed_types:
        return questions
    return [q for q in questions if q.get("question_type") in allowed_types]


def _stratified_sample(
    questions: list[dict],
    max_per_type: Optional[int],
    seed: int,
) -> list[dict]:
    """Stratified down-sample by ``question_type``.

    Each ``question_type`` bucket is independently shuffled (deterministically,
    using ``seed``) and truncated to ``max_per_type`` items. When ``max_per_type``
    is None the input is returned unchanged.

    The sample is balanced across types — useful when one type dominates the
    benchmark (e.g. type_i = 11K but type_ii = 45 in CS 288). Stratifying
    keeps rare types fully represented while capping the common ones.
    """
    if max_per_type is None or max_per_type <= 0:
        return questions

    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q.get("question_type", "unknown")].append(q)

    sampled: list[dict] = []
    for qtype, bucket in by_type.items():
        rng.shuffle(bucket)
        sampled.extend(bucket[:max_per_type])

    rng.shuffle(sampled)
    return sampled


def _type_breakdown(questions: list[dict]) -> dict[str, int]:
    """Count questions by ``question_type``; convenient for logging."""
    return dict(Counter(q.get("question_type", "unknown") for q in questions))


def _dump_predictions(
    questions: list[dict],
    retrieve_fn,
    variant: str,
    course_code: Optional[str],
    out_path: Path,
    top_k: int = 5,
    snippet_chars: int = 240,
) -> None:
    """Write per-question retrieval details to a JSONL file.

    For each question we run retrieval once at ``top_k`` and record the
    question, gold page ids, the top-k retrieved pages, and the rank of
    the first gold hit (None if no hit in top-k). Useful for eyeballing
    *why* a particular question missed.

    Each output line is a JSON object:

        {
          "variant": "v1",
          "question_id": "...",
          "question_type": "type_iii",
          "question": "...",
          "answer_short": "...",
          "gold_page_ids": [3],
          "lecture_id": "...",
          "first_hit_rank": 2,
          "retrieved": [
            {"rank": 1, "page_id": "...", "lecture_id": "...",
             "page_number": 4, "page_idx": 3, "is_hit": false,
             "score": 0.812, "snippet": "..."}, ...
          ]
        }
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for q in questions:
            results = retrieve_fn(q["question"], variant, course_code, top_k)
            gold_set = set(q.get("gold_page_ids") or [])

            retrieved: list[dict] = []
            first_hit_rank: Optional[int] = None
            for i, r in enumerate(results, start=1):
                page_idx = int(r.page_number) - 1  # 1-based → 0-based for comparison
                is_hit = page_idx in gold_set
                if is_hit and first_hit_rank is None:
                    first_hit_rank = i
                snippet = (r.ocr_text or "")[:snippet_chars].replace("\n", " ").strip()
                retrieved.append(
                    {
                        "rank": i,
                        "page_id": r.page_id,
                        "lecture_id": r.lecture_id,
                        "page_number": r.page_number,  # 1-based
                        "page_idx": page_idx,          # 0-based, comparable to gold
                        "is_hit": is_hit,
                        "score": round(float(r.score), 4),
                        "dense_score": round(float(r.dense_score), 4),
                        "snippet": snippet,
                    }
                )

            fh.write(
                json.dumps(
                    {
                        "variant": variant,
                        "question_id": q.get("question_id"),
                        "question_type": q.get("question_type"),
                        "question": q.get("question"),
                        "answer_short": q.get("answer_short"),
                        "gold_page_ids": sorted(gold_set),
                        "lecture_id": q.get("lecture_id"),
                        "first_hit_rank": first_hit_rank,
                        "retrieved": retrieved,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _make_retrieve_fn(
    retriever: Retriever,
    course_code: Optional[str],
    use_bm25: bool = False,
    rrf_k: int = 60,
    chunk_agg: str = "max",
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
):
    """Wrap retriever.retrieve into the callable expected by eval functions.

    Forwards all retrieval-tuning knobs (BM25 toggle, RRF k, chunk aggregator,
    weighted RRF) so a single eval invocation can sweep these values without
    touching the retriever code.
    """
    def retrieve_fn(query: str, variant: str, course_code: Optional[str], top_k: int):
        return retriever.retrieve(
            query=query,
            index_variant=variant,
            course_code=course_code,
            top_k=top_k,
            use_bm25=use_bm25,
            rrf_k=rrf_k,
            chunk_agg=chunk_agg,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
    return retrieve_fn


def _eval_subset(
    questions: list[dict],
    retrieve_fn,
    variant: str,
    course_code: Optional[str],
    k_list: tuple[int, ...] = (1, 3, 5),
    mrr_k: int = 5,
) -> dict[str, float]:
    """Compute retrieval metrics over an arbitrary subset of questions.

    Returns a dict with keys n, recall@<k> for each k in k_list, and mrr@<mrr_k>.
    """
    if not questions:
        out: dict[str, float] = {"n": 0}
        for k in k_list:
            out[f"recall@{k}"] = 0.0
        out[f"mrr@{mrr_k}"] = 0.0
        return out
    out = {"n": len(questions)}
    for k in k_list:
        out[f"recall@{k}"] = round(
            recall_at_k(questions, retrieve_fn, variant, course_code, k=k), 4
        )
    out[f"mrr@{mrr_k}"] = round(
        mrr(questions, retrieve_fn, variant, course_code, k=mrr_k), 4
    )
    return out


def _eval_variant(
    questions: list[dict],
    retrieve_fn,
    variant: str,
    course_code: Optional[str],
    predicted_answers: Optional[list[str]] = None,
    k_list: tuple[int, ...] = (1, 3, 5),
    mrr_k: int = 5,
) -> dict:
    """Compute all metrics for one variant. Returns a dict of metric → value.

    Args:
        questions:          QA dicts with "question", "gold_page_ids", "answer_short".
        retrieve_fn:        Callable(query, variant, course_code, top_k) -> results.
        variant:            "v1", "v2", or "v3".
        course_code:        Optional course filter for retrieval.
        predicted_answers:  Optional list of model-generated answer strings (same
                            order as questions).  When provided, contains_answer_rate
                            is computed; otherwise reported as None (not run).
    """
    if not questions:
        zero: dict = {"variant": variant, "n_questions": 0}
        for k in k_list:
            zero[f"recall@{k}"] = 0.0
        zero[f"mrr@{mrr_k}"] = 0.0
        zero["citation_hit_rate"] = None
        zero["contains_answer_rate"] = None
        return zero

    out: dict = {"variant": variant, "n_questions": len(questions)}
    for k in k_list:
        out[f"recall@{k}"] = round(
            recall_at_k(questions, retrieve_fn, variant, course_code, k=k), 4
        )
    out[f"mrr@{mrr_k}"] = round(
        mrr(questions, retrieve_fn, variant, course_code, k=mrr_k), 4
    )

    # Answer-level metric: only computed when predicted answers are supplied.
    ca_rate: Optional[float] = None
    if predicted_answers is not None:
        ca_rate = round(contains_answer_rate(questions, predicted_answers), 4)

    # Per-type breakdown (R@k for each k in k_list, MRR@<mrr_k> per type).
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q.get("question_type", "unknown")].append(q)
    per_type = {
        qtype: _eval_subset(qs, retrieve_fn, variant, course_code, k_list, mrr_k)
        for qtype, qs in sorted(by_type.items())
    }

    out["citation_hit_rate"] = None
    out["contains_answer_rate"] = ca_rate
    out["per_type"] = per_type
    return out


def _fmt(val: Optional[float], width: int = 8) -> str:
    """Format a metric value for table display; show '-' when not computed."""
    if val is None:
        return f"{'—':>{width}}"
    return f"{val:>{width}.4f}"


def _print_table(
    results: list[dict],
    k_list: tuple[int, ...] = (1, 3, 5),
    mrr_k: int = 5,
) -> None:
    """Print a readable summary table (overall + per-type) to stdout."""
    rk_keys = [f"recall@{k}" for k in k_list]
    mrr_key = f"mrr@{mrr_k}"

    header_parts = [f"{'Variant':<10}", f"{'N':>6}"]
    header_parts += [f"{'R@'+str(k):>7}" for k in k_list]
    header_parts += [f"{mrr_key.upper():>8}", f"{'ContainsAns':>12}"]
    header = " ".join(header_parts)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        row = [f"{r['variant']:<10}", f"{r['n_questions']:>6}"]
        row += [_fmt(r.get(k), 7) for k in rk_keys]
        row += [_fmt(r.get(mrr_key), 8), _fmt(r.get("contains_answer_rate"), 12)]
        print(" ".join(row))

    # Per-type breakdown — only print when we have it.
    have_per_type = any(r.get("per_type") for r in results)
    if not have_per_type:
        print()
        return

    sub_parts = [f"{'Variant':<10}", f"{'Type':<10}", f"{'N':>6}"]
    sub_parts += [f"{'R@'+str(k):>7}" for k in k_list]
    sub_parts += [f"{mrr_key.upper():>8}"]
    sub_header = " ".join(sub_parts)
    print("\n" + sub_header)
    print("-" * len(sub_header))
    for r in results:
        per_type = r.get("per_type") or {}
        for qtype in sorted(per_type.keys()):
            m = per_type[qtype]
            row = [f"{r['variant']:<10}", f"{qtype:<10}", f"{m['n']:>6}"]
            row += [_fmt(m.get(k), 7) for k in rk_keys]
            row += [_fmt(m.get(mrr_key), 8)]
            print(" ".join(row))
    print()


def _save_json(results: list[dict], output_dir: Path) -> Path:
    out = output_dir / "cs288_eval_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _save_csv(results: list[dict], output_dir: Path) -> Path:
    """Write the overall (per-variant) metrics to CSV.

    The ``per_type`` nested dict is dropped from the CSV (it's preserved in
    the JSON output); a separate per-type CSV is written by ``_save_per_type_csv``.
    """
    out = output_dir / "cs288_eval_results.csv"
    if not results:
        return out
    flat = [{k: v for k, v in r.items() if k != "per_type"} for r in results]
    fields = list(flat[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    return out


def _save_per_type_csv(results: list[dict], output_dir: Path) -> Optional[Path]:
    """Write the per-(variant, type) metrics to a separate CSV, if present."""
    rows: list[dict] = []
    for r in results:
        per_type = r.get("per_type") or {}
        for qtype, m in per_type.items():
            rows.append(
                {
                    "variant": r["variant"],
                    "question_type": qtype,
                    "n": m["n"],
                    "recall@1": m["recall@1"],
                    "recall@3": m["recall@3"],
                    "recall@5": m["recall@5"],
                    "mrr@5": m["mrr@5"],
                }
            )
    if not rows:
        return None
    out = output_dir / "cs288_eval_results_by_type.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_predictions(predictions_path: Path) -> dict[str, list[str]]:
    """Load a predictions JSONL where each line has "question_id" and "predicted_answer".

    Returns a dict: { variant -> [predicted_answer, ...] } in benchmark order.
    The benchmark is filtered per-variant, so this dict is keyed by variant.

    File format (one JSON object per line):
        {"question_id": "lecture01_v1_p0_0", "predicted_answer": "Transformers use attention."}
    """
    predictions: dict[str, str] = {}
    with predictions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                predictions[obj["question_id"]] = obj["predicted_answer"]
            except (json.JSONDecodeError, KeyError):
                continue  # skip malformed lines
    return predictions


def run_eval(
    benchmark_path: Path,
    db_path: Path,
    course_code: Optional[str],
    output_dir: Path,
    embedding_model: str,
    predictions_path: Optional[Path] = None,
    max_per_type: Optional[int] = None,
    question_types: Optional[set[str]] = None,
    seed: int = 0,
    dump_predictions: bool = True,
    dump_top_k: int = 5,
    use_bm25: bool = False,
    rrf_k: int = 60,
    chunk_agg: str = "max",
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    k_list: tuple[int, ...] = (1, 3, 5),
    mrr_k: int = 5,
) -> None:
    """Load benchmark, run retrieval for each variant, compute metrics, save results.

    Args:
        benchmark_path:   Path to the merged benchmark JSONL.
        db_path:          Path to the SlideQA SQLite embedding database.
        course_code:      Optional course filter (e.g. "CS288"). Pass None to
                          evaluate across all courses in the DB.
        output_dir:       Directory to write result files.
        embedding_model:  SentenceTransformer model name to embed queries.
        predictions_path: Optional JSONL with predicted answers (question_id →
                          predicted_answer).  When supplied, contains_answer_rate
                          is computed for each variant.
    """
    print(f"Loading benchmark: {benchmark_path}")
    all_questions = _load_benchmark(benchmark_path)
    print(f"  {len(all_questions)} total QA entries")

    # Build a lookup from question_id → predicted answer (if predictions provided).
    pred_lookup: dict[str, str] = {}
    if predictions_path is not None:
        pred_lookup = _load_predictions(predictions_path)
        print(f"  Loaded {len(pred_lookup)} predicted answers from {predictions_path}")

    print(f"Loading retriever from: {db_path}")
    retriever = Retriever(db_path=db_path, model_name=embedding_model)
    retrieve_fn = _make_retrieve_fn(
        retriever,
        course_code,
        use_bm25=use_bm25,
        rrf_k=rrf_k,
        chunk_agg=chunk_agg,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    print(
        f"  Retrieval config: dense_weight={dense_weight} bm25_weight={bm25_weight} "
        f"use_bm25={use_bm25} rrf_k={rrf_k} chunk_agg={chunk_agg}"
    )
    print(f"  k_list={list(k_list)} mrr_k={mrr_k}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for variant in ("v1", "v2", "v3"):
        variant_qs = _filter_by_variant(all_questions, variant)
        variant_qs = _filter_by_types(variant_qs, question_types)
        before = len(variant_qs)
        variant_qs = _stratified_sample(variant_qs, max_per_type, seed)
        breakdown = _type_breakdown(variant_qs)
        sampled_note = (
            f" (stratified sample {len(variant_qs)}/{before}, max_per_type={max_per_type})"
            if max_per_type
            else ""
        )
        print(f"\nEvaluating {variant} ({len(variant_qs)} questions){sampled_note}")
        print(f"  type breakdown: {breakdown}")

        # Build aligned predicted-answer list for this variant (None entries for missing).
        predicted: Optional[list[str]] = None
        if pred_lookup:
            predicted = [
                pred_lookup.get(q["question_id"], "")
                for q in variant_qs
            ]

        metrics = _eval_variant(
            variant_qs, retrieve_fn, variant, course_code, predicted,
            k_list=k_list, mrr_k=mrr_k,
        )
        results.append(metrics)
        rk_str = "  ".join(
            f"R@{k}={metrics.get(f'recall@{k}')}" for k in k_list
        )
        print(
            f"  {rk_str}  MRR@{mrr_k}={metrics.get(f'mrr@{mrr_k}')}  "
            f"ContainsAns={metrics.get('contains_answer_rate')}"
        )

        if dump_predictions:
            dump_path = (output_dir / f"predictions_{variant}.jsonl").resolve()
            print(f"  Dumping per-question retrieval details -> {dump_path}")
            _dump_predictions(
                variant_qs,
                retrieve_fn,
                variant,
                course_code,
                dump_path,
                top_k=dump_top_k,
            )
            try:
                size = dump_path.stat().st_size
                print(f"    wrote {size:,} bytes ({len(variant_qs)} questions)")
            except OSError as e:
                print(f"    WARNING: dump file not found after write: {e}")

    _print_table(results, k_list=k_list, mrr_k=mrr_k)

    json_out = _save_json(results, output_dir)
    csv_out = _save_csv(results, output_dir)
    per_type_csv = _save_per_type_csv(results, output_dir)
    print(f"Results saved:\n  {json_out}\n  {csv_out}")
    if per_type_csv is not None:
        print(f"  {per_type_csv}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run page-level SlideQA ablation evaluation (local only)."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        type=Path,
        help="Path to the benchmark JSONL produced by generate_cs288_qa.py.",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        type=Path,
        help="Path to the SlideQA SQLite embedding database.",
    )
    parser.add_argument(
        "--course-code",
        default="CS288",
        help="Course filter for retrieval (default: CS288). Pass 'all' to skip filtering.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        type=Path,
        help="Directory to write cs288_eval_results.json and .csv (default: results/).",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-4B",
        help="SentenceTransformer model for query embedding (default: Qwen/Qwen3-Embedding-4B).",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=None,
        help=(
            "If set, stratified down-sample each question_type bucket to at most "
            "this many questions per variant before evaluating. Useful when the "
            "benchmark is large/skewed (e.g. CS288 has 11K type_i but only 45 type_ii)."
        ),
    )
    parser.add_argument(
        "--question-types",
        default=None,
        help=(
            "Comma-separated list of question_type values to keep "
            "(e.g. 'type_i,type_iii'). Default: keep all types."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for stratified sampling (default: 0).",
    )
    parser.add_argument(
        "--no-dump-predictions",
        dest="dump_predictions",
        action="store_false",
        default=True,
        help=(
            "Disable the default per-question retrieval dump. By default the "
            "script writes <output-dir>/predictions_<variant>.jsonl for each "
            "variant, with the question, gold page ids, top-K retrieved pages "
            "(plus snippets), and the rank of the first gold hit."
        ),
    )
    parser.add_argument(
        "--dump-top-k",
        type=int,
        default=5,
        help="How many retrieved pages to include in --dump-predictions (default 5).",
    )
    parser.add_argument(
        "--use-bm25",
        action="store_true",
        help=(
            "Combine dense retrieval with BM25 using Reciprocal Rank Fusion. "
            "Often a free win on short-text corpora like slides."
        ),
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k parameter when --use-bm25 is set (default 60).",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the dense ranker in weighted RRF (default 1.0). "
            "Only used when --use-bm25 is set. Combine with --bm25-weight to "
            "trade off the two rankers; e.g. 2.0 / 1.0 favours dense, "
            "1.0 / 0.0 disables BM25 (same as omitting --use-bm25)."
        ),
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
        help="Weight on the BM25 ranker in weighted RRF (default 1.0).",
    )
    parser.add_argument(
        "--chunk-agg",
        choices=("max", "sum", "mean"),
        default="max",
        help=(
            "How to aggregate per-chunk scores into a per-page score when the "
            "index is chunk-level (default: max). 'mean' averages chunk scores; "
            "'sum' adds them (favours pages with many chunks)."
        ),
    )
    parser.add_argument(
        "--k-list",
        default="1,3,5",
        help=(
            "Comma-separated list of k values for Recall@k (default '1,3,5'). "
            "Add larger k to inspect tail behaviour, e.g. '1,3,5,10,20'."
        ),
    )
    parser.add_argument(
        "--mrr-k",
        type=int,
        default=5,
        help="Cutoff for MRR@k (default 5).",
    )
    parser.add_argument(
        "--predictions-file",
        default=None,
        type=Path,
        help=(
            "Optional JSONL file with predicted answers for contains_answer_rate. "
            "Each line: {\"question_id\": \"...\", \"predicted_answer\": \"...\"}."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.benchmark.exists():
        print(f"ERROR: --benchmark not found: {args.benchmark}", file=sys.stderr)
        sys.exit(1)
    if not args.db_path.exists():
        print(f"ERROR: --db-path not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)
    if args.predictions_file is not None and not args.predictions_file.exists():
        print(f"ERROR: --predictions-file not found: {args.predictions_file}", file=sys.stderr)
        sys.exit(1)

    course_code = None if args.course_code == "all" else args.course_code

    qtypes: Optional[set[str]] = None
    if args.question_types:
        qtypes = {t.strip() for t in args.question_types.split(",") if t.strip()}

    run_eval(
        benchmark_path=args.benchmark,
        db_path=args.db_path,
        course_code=course_code,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        predictions_path=args.predictions_file,
        max_per_type=args.max_per_type,
        question_types=qtypes,
        seed=args.seed,
        dump_predictions=args.dump_predictions,
        dump_top_k=args.dump_top_k,
        use_bm25=args.use_bm25,
        rrf_k=args.rrf_k,
        chunk_agg=args.chunk_agg,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        k_list=tuple(int(k) for k in args.k_list.split(",") if k.strip()),
        mrr_k=args.mrr_k,
    )
