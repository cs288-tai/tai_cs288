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
import sys
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


def _make_retrieve_fn(retriever: Retriever, course_code: Optional[str]):
    """Wrap retriever.retrieve into the callable expected by eval functions.

    Signature expected by eval: (query, variant, course_code, top_k) -> results
    """
    def retrieve_fn(query: str, variant: str, course_code: Optional[str], top_k: int):
        return retriever.retrieve(
            query=query,
            index_variant=variant,
            course_code=course_code,
            top_k=top_k,
        )
    return retrieve_fn


def _eval_variant(
    questions: list[dict],
    retrieve_fn,
    variant: str,
    course_code: Optional[str],
    predicted_answers: Optional[list[str]] = None,
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
    _zero: dict = {
        "variant": variant,
        "n_questions": 0,
        "recall@1": 0.0,
        "recall@3": 0.0,
        "recall@5": 0.0,
        "mrr@5": 0.0,
        # citation_hit_rate needs a QA agent — placeholder for now.
        "citation_hit_rate": None,
        # contains_answer_rate needs predicted answers from a QA model.
        "contains_answer_rate": None,
    }
    if not questions:
        return _zero

    r1 = recall_at_k(questions, retrieve_fn, variant, course_code, k=1)
    r3 = recall_at_k(questions, retrieve_fn, variant, course_code, k=3)
    r5 = recall_at_k(questions, retrieve_fn, variant, course_code, k=5)
    mrr5 = mrr(questions, retrieve_fn, variant, course_code, k=5)

    # Answer-level metric: only computed when predicted answers are supplied.
    ca_rate: Optional[float] = None
    if predicted_answers is not None:
        ca_rate = round(contains_answer_rate(questions, predicted_answers), 4)

    return {
        "variant": variant,
        "n_questions": len(questions),
        "recall@1": round(r1, 4),
        "recall@3": round(r3, 4),
        "recall@5": round(r5, 4),
        "mrr@5": round(mrr5, 4),
        # citation_hit_rate needs a QA agent — placeholder for now.
        "citation_hit_rate": None,
        "contains_answer_rate": ca_rate,
    }


def _fmt(val: Optional[float], width: int = 8) -> str:
    """Format a metric value for table display; show '-' when not computed."""
    if val is None:
        return f"{'—':>{width}}"
    return f"{val:>{width}.4f}"


def _print_table(results: list[dict]) -> None:
    """Print a readable summary table to stdout."""
    header = (
        f"{'Variant':<10} {'N':>6} {'R@1':>7} {'R@3':>7} {'R@5':>7}"
        f" {'MRR@5':>8} {'ContainsAns':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['variant']:<10} {r['n_questions']:>6} "
            f"{_fmt(r['recall@1'], 7)} {_fmt(r['recall@3'], 7)} {_fmt(r['recall@5'], 7)}"
            f" {_fmt(r['mrr@5'], 8)} {_fmt(r['contains_answer_rate'], 12)}"
        )
    print()


def _save_json(results: list[dict], output_dir: Path) -> Path:
    out = output_dir / "cs288_eval_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _save_csv(results: list[dict], output_dir: Path) -> Path:
    out = output_dir / "cs288_eval_results.csv"
    if not results:
        return out
    fields = list(results[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
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
    retrieve_fn = _make_retrieve_fn(retriever, course_code)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for variant in ("v1", "v2", "v3"):
        variant_qs = _filter_by_variant(all_questions, variant)
        print(f"\nEvaluating {variant} ({len(variant_qs)} questions)...")

        # Build aligned predicted-answer list for this variant (None entries for missing).
        predicted: Optional[list[str]] = None
        if pred_lookup:
            predicted = [
                pred_lookup.get(q["question_id"], "")
                for q in variant_qs
            ]

        metrics = _eval_variant(variant_qs, retrieve_fn, variant, course_code, predicted)
        results.append(metrics)
        print(f"  Recall@1={metrics['recall@1']}  Recall@3={metrics['recall@3']}  "
              f"Recall@5={metrics['recall@5']}  MRR@5={metrics['mrr@5']}  "
              f"ContainsAns={metrics['contains_answer_rate']}")

    _print_table(results)

    json_out = _save_json(results, output_dir)
    csv_out = _save_csv(results, output_dir)
    print(f"Results saved:\n  {json_out}\n  {csv_out}")


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

    run_eval(
        benchmark_path=args.benchmark,
        db_path=args.db_path,
        course_code=course_code,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        predictions_path=args.predictions_file,
    )
