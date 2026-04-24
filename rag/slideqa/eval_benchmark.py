#!/usr/bin/env python3
"""
CLI entry-point: evaluate the SlideQA retrieval system on a benchmark split.

Usage — dev tuning (iterate freely):
    python -m slideqa.eval_benchmark \\
        --split data/cs288_benchmark_dev.jsonl \\
        --db    path/to/slideqa.db \\
        --k     5

Usage — official test evaluation (run once, report results):
    python -m slideqa.eval_benchmark \\
        --split data/cs288_benchmark_test.jsonl \\
        --db    path/to/slideqa.db \\
        --k     5

Options:
    --split         Path to benchmark JSONL (dev or test).
    --db            Path to SlideQA SQLite index built by index_builder.py.
    --k             Top-k retrieval cutoff (default: 5).
    --course-code   Optional course filter passed to the retriever (e.g. CS288).
    --use-bm25      Enable BM25 hybrid retrieval (default: dense-only).
    --model         Embedding model name (default: Qwen/Qwen3-Embedding-4B).
    --variants      Comma-separated variants to evaluate (default: v1,v2,v3).
    --json          Print results as JSON instead of a human-readable table.

Outputs three metrics per variant:
    Recall@k          — fraction of questions where gold page is in top-k results
    MRR               — Mean Reciprocal Rank over top-k
    Contains-Answer   — NOT computed here (no QA model); shown as "—"
                        Pass --predict-with-context to enable (requires OpenAI key).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python -m slideqa.eval_benchmark` from the rag/ root.
_RAG_ROOT = Path(__file__).resolve().parents[1]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from slideqa.run_eval import EvalResult, evaluate_split, run_all_variants


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(val: float | None, pct: bool = True) -> str:
    if val is None:
        return "    —"
    if pct:
        return f"{val * 100:5.1f}%"
    return f"{val:6.4f}"


def _print_table(results: dict[str, EvalResult], k: int) -> None:
    header = f"{'variant':<8} {'n_q':>6}  {'Recall@'+str(k):>9}  {'MRR@'+str(k):>7}  {'ContainsAns':>11}"
    print()
    print(header)
    print("-" * len(header))
    for variant in ("v1", "v2", "v3"):
        if variant not in results:
            continue
        r = results[variant]
        print(
            f"{r.variant:<8} {r.n_questions:>6}  "
            f"{_fmt(r.recall_at_k):>9}  "
            f"{_fmt(r.mrr):>7}  "
            f"{_fmt(r.contains_answer_rate):>11}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate TAI-SlideQA retrieval on a benchmark split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--split", required=True, type=Path, help="Benchmark JSONL path.")
    p.add_argument("--db", required=True, type=Path, help="SlideQA SQLite index path.")
    p.add_argument("--k", type=int, default=5, help="Top-k cutoff.")
    p.add_argument("--course-code", default=None, help="Optional course filter (e.g. CS288).")
    p.add_argument("--use-bm25", action="store_true", help="Enable BM25 hybrid retrieval.")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedding model name for the Retriever.",
    )
    p.add_argument(
        "--variants",
        default="v1,v2,v3",
        help="Comma-separated variants to evaluate.",
    )
    p.add_argument("--json", action="store_true", dest="as_json", help="Output results as JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    split_path: Path = args.split
    if not split_path.exists():
        print(f"ERROR: split file not found: {split_path}", file=sys.stderr)
        return 1

    db_path: Path = args.db
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 1

    # Lazy import — keeps startup fast when running tests without heavy deps.
    from slideqa.retriever import Retriever

    retriever = Retriever(db_path=db_path, model_name=args.model)
    use_bm25: bool = args.use_bm25
    course_code: str | None = args.course_code

    def retrieve_fn(query: str, variant: str, _course_code: str | None, top_k: int):
        return retriever.retrieve(
            query=query,
            index_variant=variant,
            course_code=course_code,
            top_k=top_k,
            use_bm25=use_bm25,
        )

    variants_to_run = [v.strip() for v in args.variants.split(",")]
    results: dict[str, EvalResult] = {}

    print(f"Evaluating {split_path.name}  (k={args.k})")
    for variant in variants_to_run:
        print(f"  [{variant}] running...", end=" ", flush=True)
        result = evaluate_split(
            jsonl_path=split_path,
            retrieve_fn=retrieve_fn,
            variant=variant,
            course_code=course_code,
            k=args.k,
        )
        results[variant] = result
        print(
            f"n={result.n_questions}  "
            f"Recall@{args.k}={result.recall_at_k:.3f}  "
            f"MRR={result.mrr:.3f}"
        )

    if args.as_json:
        out = {
            v: {
                "variant": r.variant,
                "n_questions": r.n_questions,
                f"recall_at_{args.k}": r.recall_at_k,
                "mrr": r.mrr,
                "contains_answer_rate": r.contains_answer_rate,
                "k": r.k,
            }
            for v, r in results.items()
        }
        print(json.dumps(out, indent=2))
    else:
        _print_table(results, args.k)

    return 0


if __name__ == "__main__":
    sys.exit(main())
