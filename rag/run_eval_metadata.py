#!/usr/bin/env python3
"""
Evaluate SlideQA retrieval using chunk embeddings from CS 288_metadata_new.db.

Usage:
    # Dev eval (iterate freely):
    python run_eval_metadata.py --split dev

    # Test eval (run once, report results):
    python run_eval_metadata.py --split test --breakdown --error-analysis

    # Both splits, k=10, JSON output:
    python run_eval_metadata.py --split dev  --k 10 --json
    python run_eval_metadata.py --split test --k 10 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_RAG_ROOT = Path(__file__).resolve().parent
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

_DATA_DIR = _RAG_ROOT.parent / "data"
_METADATA_DB = _DATA_DIR / "CS 288_metadata_new.db"
_SPLITS = {
    "dev":  _DATA_DIR / "cs288_benchmark_dev.jsonl",
    "test": _DATA_DIR / "cs288_benchmark_test.jsonl",
}
_COURSE_CODE = "CS 288"


def _fmt(val: float | None, pct: bool = True) -> str:
    if val is None:
        return "    —"
    return f"{val * 100:5.1f}%" if pct else f"{val:6.4f}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Evaluate TAI-SlideQA with metadata DB chunk embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--db", type=Path, default=_METADATA_DB, help="Path to metadata SQLite DB.")
    p.add_argument("--k", type=int, default=5, help="Top-k cutoff.")
    p.add_argument("--variants", default="v1,v2,v3", help="Comma-separated variants to evaluate.")
    p.add_argument("--breakdown", action="store_true",
                   help="Print per-question_type and per-evidence_modality breakdown.")
    p.add_argument("--error-analysis", action="store_true", dest="error_analysis",
                   help="Print error category counts per variant.")
    p.add_argument("--json", action="store_true", dest="as_json", help="Output results as JSON.")
    args = p.parse_args(argv)

    split_path = _SPLITS[args.split]
    if not split_path.exists():
        print(f"ERROR: split file not found: {split_path}", file=sys.stderr)
        return 1
    if not args.db.exists():
        print(f"ERROR: DB not found: {args.db}", file=sys.stderr)
        return 1

    from slideqa.metadata_retriever import MetadataDBRetriever
    from slideqa.run_eval import (
        EvalResult, ErrorCategory,
        evaluate_split, evaluate_split_by_group, error_analysis,
    )

    retriever = MetadataDBRetriever(db_path=args.db)

    def retrieve_fn(query: str, variant: str, course_code: str | None, top_k: int):
        return retriever.retrieve(
            query=query,
            index_variant=variant,
            course_code=course_code,
            top_k=top_k,
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
            course_code=_COURSE_CODE,
            k=args.k,
        )
        results[variant] = result
        print(f"n={result.n_questions}  Recall@{args.k}={result.recall_at_k:.3f}  MRR={result.mrr:.3f}")

        if args.breakdown:
            for group_field in ("question_type", "evidence_modality"):
                grouped = evaluate_split_by_group(
                    jsonl_path=split_path,
                    retrieve_fn=retrieve_fn,
                    variant=variant,
                    course_code=_COURSE_CODE,
                    k=args.k,
                    group_by=group_field,
                )
                if not args.as_json:
                    print(f"\n  Breakdown by {group_field}:")
                    for key in sorted(grouped):
                        r = grouped[key]
                        print(f"    {r.group_key:<22} n={r.n_questions:5d}  "
                              f"Recall@{args.k}={_fmt(r.recall_at_k)}  MRR={_fmt(r.mrr)}")

        if args.error_analysis:
            err_records = error_analysis(
                jsonl_path=split_path,
                retrieve_fn=retrieve_fn,
                variant=variant,
                course_code=_COURSE_CODE,
                k=args.k,
            )
            counts: Counter = Counter(rec.category for rec in err_records)
            if not args.as_json:
                total = sum(counts.values())
                print(f"\n  [{variant}] error analysis — {total} failing questions:")
                for cat in ErrorCategory:
                    n = counts.get(cat, 0)
                    pct = 100 * n / total if total else 0.0
                    print(f"    {cat.value:<30} {n:>5}  ({pct:5.1f}%)")

    if args.as_json:
        out: dict = {}
        for v, r in results.items():
            out[v] = {
                "variant": r.variant,
                "n_questions": r.n_questions,
                f"recall_at_{args.k}": r.recall_at_k,
                "mrr": r.mrr,
                "contains_answer_rate": r.contains_answer_rate,
                "k": r.k,
            }
        print(json.dumps(out, indent=2))
    else:
        print()
        header = f"{'variant':<8} {'n_q':>6}  {'Recall@'+str(args.k):>9}  {'MRR@'+str(args.k):>7}"
        print(header)
        print("-" * len(header))
        for v in variants_to_run:
            if v not in results:
                continue
            r = results[v]
            print(f"{r.variant:<8} {r.n_questions:>6}  "
                  f"{_fmt(r.recall_at_k):>9}  {_fmt(r.mrr):>7}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
