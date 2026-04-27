"""Grid-search retrieval parameters for CS 288 SlideQA.

Iterates over a parameter grid (BM25 on/off, weights, RRF k, chunk_agg,
embedding model) and runs ``run_cs288_eval.py`` for each cell. After every run
it parses the per-variant ``cs288_eval_results.json`` and accumulates a single
summary CSV at ``<base-output-dir>/sweep_summary.csv``.

Why a wrapper rather than nested loops in ``run_cs288_eval.py``? Different
embedding models need fresh ``Retriever`` instances (and reloading a 1B+ model
inside one process accumulates VRAM). Spawning a subprocess per cell is the
simplest way to get clean state.

Example
-------

    python scripts/sweep_cs288_eval.py \
        --benchmark data/cs288_merged_deduped.jsonl \
        --db-path   data/slideqa.db \
        --course-code "CS 288" \
        --max-per-type 50 \
        --base-output-dir results/sweep_v1 \
        --models BAAI/bge-m3 \
        --bm25 off,on \
        --dense-weights 1.0 \
        --bm25-weights 0.5,1.0,2.0 \
        --chunk-aggs max,mean \
        --rrf-ks 60

Each grid cell writes a sub-directory like
``results/sweep_v1/m=bge-m3_bm25=on_dw=1.0_bw=2.0_rrf=60_agg=max/`` containing
the standard ``cs288_eval_results.json`` plus prediction dumps.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from itertools import product
from pathlib import Path


def _slug(value: object) -> str:
    """Filename-safe slug for a parameter value."""
    s = str(value).replace("/", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)


def _parse_csv_list(s: str, cast=str) -> list:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True, type=Path)
    p.add_argument("--db-path", required=True, type=Path)
    p.add_argument("--course-code", default="CS 288")
    p.add_argument("--max-per-type", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-output-dir", required=True, type=Path)
    p.add_argument("--k-list", default="1,3,5")
    p.add_argument("--mrr-k", type=int, default=5)
    p.add_argument(
        "--models",
        default="BAAI/bge-m3",
        help="Comma-separated SentenceTransformer model ids to sweep over.",
    )
    p.add_argument(
        "--bm25",
        default="off,on",
        help="Comma-separated 'off'/'on' values for --use-bm25.",
    )
    p.add_argument("--dense-weights", default="1.0")
    p.add_argument("--bm25-weights", default="1.0")
    p.add_argument("--rrf-ks", default="60")
    p.add_argument("--chunk-aggs", default="max")
    p.add_argument(
        "--question-types",
        default=None,
        help="Optional comma-separated question_type filter passed to each cell.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run, without executing them.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.base_output_dir.mkdir(parents=True, exist_ok=True)

    models       = _parse_csv_list(args.models)
    bm25_flags   = _parse_csv_list(args.bm25)
    dense_w_list = _parse_csv_list(args.dense_weights, float)
    bm25_w_list  = _parse_csv_list(args.bm25_weights, float)
    rrf_k_list   = _parse_csv_list(args.rrf_ks, int)
    agg_list     = _parse_csv_list(args.chunk_aggs)
    k_list_tuple = tuple(int(k) for k in args.k_list.split(",") if k.strip())

    summary_rows: list[dict] = []
    cells = list(product(models, bm25_flags, dense_w_list, bm25_w_list, rrf_k_list, agg_list))
    print(f"Sweep grid: {len(cells)} cells")

    for cell_idx, (model, bm25_flag, dw, bw, rrf_k, agg) in enumerate(cells, start=1):
        # If BM25 is off, weight/rrf_k variations don't matter — dedup by skipping
        # everything except the canonical "off" cell.
        if bm25_flag == "off" and (dw != dense_w_list[0] or bw != bm25_w_list[0] or rrf_k != rrf_k_list[0]):
            continue

        cell_name = (
            f"m={_slug(Path(model).name)}_bm25={bm25_flag}"
            f"_dw={dw}_bw={bw}_rrf={rrf_k}_agg={agg}"
        )
        cell_dir = args.base_output_dir / cell_name
        print(f"\n[{cell_idx}/{len(cells)}] {cell_name}")
        print(f"  -> {cell_dir}")

        cmd = [
            sys.executable,
            "scripts/run_cs288_eval.py",
            "--benchmark", str(args.benchmark),
            "--db-path", str(args.db_path),
            "--course-code", args.course_code,
            "--embedding-model", model,
            "--max-per-type", str(args.max_per_type),
            "--seed", str(args.seed),
            "--output-dir", str(cell_dir),
            "--chunk-agg", agg,
            "--k-list", args.k_list,
            "--mrr-k", str(args.mrr_k),
        ]
        if bm25_flag == "on":
            cmd += [
                "--use-bm25",
                "--rrf-k", str(rrf_k),
                "--dense-weight", str(dw),
                "--bm25-weight", str(bw),
            ]
        if args.question_types:
            cmd += ["--question-types", args.question_types]

        if args.dry_run:
            print("  DRY RUN:", " ".join(cmd))
            continue

        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  cell failed (rc={rc}); continuing")
            continue

        results_json = cell_dir / "cs288_eval_results.json"
        if not results_json.exists():
            print(f"  WARNING: missing {results_json}")
            continue

        per_variant = json.loads(results_json.read_text(encoding="utf-8"))
        for vrow in per_variant:
            srow: dict = {
                "cell": cell_name,
                "model": model,
                "use_bm25": bm25_flag == "on",
                "dense_weight": dw,
                "bm25_weight": bw,
                "rrf_k": rrf_k,
                "chunk_agg": agg,
                "variant": vrow["variant"],
                "n_questions": vrow["n_questions"],
            }
            for k in k_list_tuple:
                srow[f"recall@{k}"] = vrow.get(f"recall@{k}")
            srow[f"mrr@{args.mrr_k}"] = vrow.get(f"mrr@{args.mrr_k}")
            summary_rows.append(srow)

    if args.dry_run or not summary_rows:
        return

    summary_csv = args.base_output_dir / "sweep_summary.csv"
    fields = list(summary_rows[0].keys())
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSweep complete. Summary: {summary_csv}")

    # Pretty-print the best cell per variant by R@5 (or last R@k in list).
    target_key = f"recall@{k_list_tuple[-1]}"
    print(f"\nBest cell per variant by {target_key}:")
    by_variant: dict[str, list[dict]] = {}
    for r in summary_rows:
        by_variant.setdefault(r["variant"], []).append(r)
    for variant, rows in sorted(by_variant.items()):
        rows = [r for r in rows if r.get(target_key) is not None]
        if not rows:
            continue
        best = max(rows, key=lambda r: r[target_key])
        print(
            f"  {variant}: {target_key}={best[target_key]:.4f}  "
            f"mrr@{args.mrr_k}={best[f'mrr@{args.mrr_k}']:.4f}  "
            f"({best['cell']})"
        )


if __name__ == "__main__":
    main()
