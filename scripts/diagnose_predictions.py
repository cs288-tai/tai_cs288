"""Diagnose retrieval failure modes from ``predictions_*.jsonl`` dumps.

Run::

    python scripts/diagnose_predictions.py results/<run-dir>

Reads ``predictions_v1.jsonl``, ``predictions_v2.jsonl``, ``predictions_v3.jsonl``
out of the given directory and reports, per variant:

  1. **Top-1 lecture-correct rate** — fraction of questions where the rank-1
     retrieved page even comes from the gold lecture. Low values (<30%) point
     at lecture-level leakage — most queries are pulling pages from the wrong
     deck entirely. Fix: add a lecture/page header to chunk text and re-embed.

  2. **Top-1 attractor pages** — pages that show up as rank-1 across many
     unrelated queries. A page that takes >10% of all top-1 slots is acting as
     an attractor (boilerplate text, "Outline" slides, "Questions?" slides).

  3. **Near-miss rate** — among questions that missed the top-K, how often
     a page within ±1 slide of gold *was* retrieved. High values suggest the
     model is in the right neighbourhood but ranking the wrong slide first;
     adjacent-slide info would help (longer chunk windows or a reranker).

  4. **Sample misses** with retrieved snippets, for human inspection.

Pure-stdlib (no jq), so it runs anywhere Python runs.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _top1_lecture_correct(records: list[dict]) -> tuple[int, int]:
    correct = sum(
        1
        for r in records
        if r.get("retrieved")
        and r["retrieved"][0].get("lecture_id") == r.get("lecture_id")
    )
    return correct, len(records)


def _topk_lecture_correct(records: list[dict], k: int) -> tuple[int, int]:
    """Rate at which the gold lecture appears anywhere in top-k retrieved."""
    correct = 0
    for r in records:
        gold_lec = r.get("lecture_id")
        if not gold_lec:
            continue
        retrieved_lecs = [p.get("lecture_id") for p in (r.get("retrieved") or [])[:k]]
        if gold_lec in retrieved_lecs:
            correct += 1
    return correct, len(records)


def _top1_attractors(records: list[dict], n: int = 5) -> list[tuple[int, str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for r in records:
        if r.get("retrieved"):
            top = r["retrieved"][0]
            key = (top.get("page_id", ""), top.get("lecture_id", ""))
            counter[key] += 1
    return [(c, pid, lec) for (pid, lec), c in counter.most_common(n)]


def _near_miss_rate(records: list[dict]) -> tuple[int, int]:
    """Fraction of misses (no top-K hit) where some retrieved page is within
    +/-1 of a gold page index *and* in the right lecture."""
    near = 0
    total_misses = 0
    for r in records:
        if r.get("first_hit_rank") is not None:
            continue
        total_misses += 1
        gold = set(r.get("gold_page_ids") or [])
        gold_lec = r.get("lecture_id")
        if not gold:
            continue
        for hit in r.get("retrieved") or []:
            if hit.get("lecture_id") != gold_lec:
                continue
            if any(abs(int(hit.get("page_idx", -999)) - g) <= 1 for g in gold):
                near += 1
                break
    return near, total_misses


def _sample_misses(records: list[dict], k: int = 3) -> list[dict]:
    misses = [r for r in records if r.get("first_hit_rank") is None]
    return misses[:k]


def _diagnose_variant(records: list[dict]) -> dict:
    c1, n = _top1_lecture_correct(records)
    c5, _ = _topk_lecture_correct(records, k=5)
    near, miss_total = _near_miss_rate(records)
    return {
        "n": n,
        "top1_lecture_correct": (c1, n),
        "top5_lecture_correct": (c5, n),
        "near_miss": (near, miss_total),
        "attractors": _top1_attractors(records, n=5),
    }


def _print_variant(variant: str, recs: list[dict]) -> None:
    if not recs:
        print(f"=== {variant}: empty ===")
        return
    d = _diagnose_variant(recs)
    n = d["n"]
    c1, _ = d["top1_lecture_correct"]
    c5, _ = d["top5_lecture_correct"]
    near, miss_total = d["near_miss"]
    nrate = near / miss_total if miss_total else 0.0

    print(f"=== {variant} : {n} records ===")
    print(f"  top-1 lecture correct: {c1}/{n}  ({c1/n:.1%})")
    print(f"  top-5 lecture correct: {c5}/{n}  ({c5/n:.1%})")
    print(
        f"  near-miss (correct lecture, gold +/-1) among {miss_total} misses: "
        f"{near} ({nrate:.1%})"
    )
    print("  top-1 attractor pages (count : lecture / page_id):")
    for count, pid, lec in d["attractors"]:
        share = count / n if n else 0.0
        flag = "  <-- ATTRACTOR" if share > 0.10 else ""
        print(f"    {count:>3} ({share:.1%})  {lec} :: {pid}{flag}")
    print()


def main(results_dir: Path) -> None:
    if not results_dir.exists():
        print(f"ERROR: directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Diagnosing predictions in: {results_dir}\n")

    for variant in ("v1", "v2", "v3"):
        path = results_dir / f"predictions_{variant}.jsonl"
        if not path.exists():
            print(f"[skip] {path.name} missing")
            continue
        recs = _load_jsonl(path)
        _print_variant(variant, recs)

    # Show a few sample misses from v1 with retrieved snippets, for eyeballing.
    v1_path = results_dir / "predictions_v1.jsonl"
    if v1_path.exists():
        print("=== sample v1 misses (top-2 retrieved per question) ===")
        for r in _sample_misses(_load_jsonl(v1_path), k=3):
            gold_lec = r.get("lecture_id")
            print(f"\nQ: {r.get('question')}")
            print(
                f"  gold lecture: {gold_lec}  "
                f"gold page_ids (0-based): {r.get('gold_page_ids')}"
            )
            for hit in (r.get("retrieved") or [])[:2]:
                same = (
                    "same lecture"
                    if hit.get("lecture_id") == gold_lec
                    else "WRONG LECTURE"
                )
                print(
                    f"  rank {hit.get('rank')} | page_idx={hit.get('page_idx')} | "
                    f"score={hit.get('score'):.3f}  [{same}]"
                )
                print(f"     lecture: {hit.get('lecture_id')}")
                snip = (hit.get("snippet") or "")[:200]
                print(f"     snippet: {snip}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "usage: python scripts/diagnose_predictions.py <results-dir>",
            file=sys.stderr,
        )
        sys.exit(1)
    main(Path(sys.argv[1]))
