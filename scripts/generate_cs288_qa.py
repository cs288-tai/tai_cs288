#!/usr/bin/env python3
"""
Local benchmark QA generation for CS 288 SlideQA.

This script generates PAGE-level QA pairs from the ablation markdown files
(v1 / v2 / v3) that the pdf_converter already produced.  It does NOT touch
the production backend or any online service.

How it works:
  1. Walk PROCESSED_DIR for lecture folders.
  2. For each lecture, load the MinerU content_list.json (page_idx → text).
  3. For each variant (v1, v2, v3) and each slide page, call GPT to generate QA.
  4. Write per-variant sidecar files:  <lecture>.<variant>.qa.jsonl
  5. Merge all sidecars into a single benchmark file: cs288_benchmark.jsonl

Each QA entry has:
  question_id       — unique string: <lecture>_<variant>_p<page_idx>_<n>
  course_code       — e.g. "CS288"
  lecture_id        — folder name, e.g. "lecture01"
  question          — the question text
  answer_short      — the short answer
  gold_page_ids     — [page_idx] (0-based MinerU page_idx)
  question_type     — type_i … type_v
  evidence_modality — text_only | visual | table | chart | layout
  variant           — "v1", "v2", or "v3"

Idempotent: skips any sidecar that already exists.

Usage:
  conda activate eecs-rag
  export OPENAI_API_KEY=sk-...
  python scripts/generate_cs288_qa.py \\
      --processed-dir /path/to/processed \\
      --course-code CS288 \\
      --output-benchmark /path/to/cs288_benchmark.jsonl

The --processed-dir should contain one subdirectory per lecture, e.g.:
  processed/
  └── lecture01/
      ├── lecture01.v1.md
      ├── lecture01.v2.md
      ├── lecture01.v3.md
      └── lecture01_content_list.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make rag package importable when running from repo root.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag.file_conversion_router.utils.title_handle import get_slideqa_pairs_for_page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_content_list(lecture_dir: Path, stem: str) -> Path | None:
    """Return the MinerU content_list.json path for a lecture, or None.

    Tries ``<stem>_content_list.json`` first (matches the folder name), then
    falls back to any ``*_content_list.json`` in the directory so we also
    handle layouts where the markdown stems include the source extension
    (e.g. ``CS288_sp26_01_Intro.pdf_content_list.json``).
    """
    candidate = lecture_dir / f"{stem}_content_list.json"
    if candidate.exists():
        return candidate
    matches = sorted(lecture_dir.glob("*_content_list.json"))
    if matches:
        return matches[0]
    return None


def _discover_stem(lecture_dir: Path, default_stem: str) -> str:
    """Discover the markdown stem used by the variant files.

    Looks for ``*.v1.md`` and returns the prefix before ``.v1.md``.
    Falls back to the folder name if no variant file is found yet.
    """
    matches = sorted(lecture_dir.glob("*.v1.md"))
    if matches:
        return matches[0].name[: -len(".v1.md")]
    return default_stem


def _load_pages(content_list_path: Path) -> dict[int, str]:
    """Load content_list.json and group text snippets by 0-based page_idx.

    Returns a dict: { page_idx (int, 0-based): combined_text (str) }
    Items with no text or no page_idx are skipped.
    """
    with content_list_path.open("r", encoding="utf-8") as fh:
        items = json.load(fh)

    pages: dict[int, list[str]] = {}
    for item in items:
        page_idx = item.get("page_idx")
        text = item.get("text", "").strip()
        if page_idx is None or not text:
            continue
        pages.setdefault(page_idx, []).append(text)

    # Join multiple text snippets on the same page into one block.
    return {idx: "\n\n".join(parts) for idx, parts in pages.items()}


def _generate_for_lecture_variant(
    lecture_dir: Path,
    stem: str,
    variant: str,
    course_code: str,
    pages: dict[int, str],
    force: bool = False,
) -> Path | None:
    """Generate QA pairs for one lecture × one variant, write a sidecar .qa.jsonl.

    Returns the sidecar path if QA was generated, None if skipped or failed.
    By default skips silently if the sidecar already exists (idempotent).
    Pass ``force=True`` to overwrite — useful when the QA prompt changes and
    the sidecars need to be regenerated without re-running the rest of the
    RAG pipeline.
    """
    variant_md = lecture_dir / f"{stem}.{variant}.md"
    if not variant_md.exists():
        print(f"    [skip] {variant_md.name} not found")
        return None

    # Sidecar path uses .qa.jsonl (not .md.qa.jsonl).
    sidecar = lecture_dir / f"{stem}.{variant}.qa.jsonl"
    if sidecar.exists() and not force:
        print(f"    [skip] {sidecar.name} already exists (use --force to overwrite)")
        return sidecar
    if sidecar.exists() and force:
        print(f"    [force] regenerating {sidecar.name}")

    all_pairs: list[dict] = []
    for page_idx in sorted(pages.keys()):
        page_text = pages[page_idx]
        pairs = get_slideqa_pairs_for_page(
            page_text=page_text,
            page_id=page_idx,   # 0-based, stored as gold_page_ids
            variant=variant,
        )
        # Rename question_text → question and answer → answer_short
        # to match the required QA schema.
        for i, pair in enumerate(pairs):
            entry = {
                "question_id": f"{stem}_{variant}_p{page_idx}_{i}",
                "course_code": course_code,
                "lecture_id": stem,
                "question": pair["question_text"],
                "answer_short": pair["answer"],
                "gold_page_ids": pair["gold_page_ids"],   # [page_idx], 0-based
                "question_type": pair["question_type"],
                "evidence_modality": pair["evidence_modality"],
                "variant": variant,
            }
            all_pairs.append(entry)
        if pairs:
            print(f"      page_idx={page_idx}: {len(pairs)} QA pairs ({variant})")

    if not all_pairs:
        print(f"    [warn] No QA pairs generated for {stem} / {variant}")
        return None

    sidecar.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in all_pairs) + "\n",
        encoding="utf-8",
    )
    print(f"    → wrote {sidecar.name} ({len(all_pairs)} pairs)")
    return sidecar


def generate_all(
    processed_dir: Path,
    course_code: str,
    output_benchmark: Path,
    force: bool = False,
) -> None:
    """Walk processed_dir, generate QA for all lectures × variants, merge.

    Args:
        processed_dir:    Directory containing one subfolder per lecture.
        course_code:      Course identifier, e.g. "CS288".
        output_benchmark: Path to write the merged benchmark JSONL file.
    """
    variants = ("v1", "v2", "v3")
    all_sidecars: list[Path] = []

    # Discover every lecture by recursively finding *.v1.md anywhere under
    # processed_dir. The directory containing the v1 file is the work dir
    # (so layouts like ``<lecture>/<stem>.pdf/auto/<stem>.pdf.v1.md`` from
    # MinerU are handled the same as flat ``<lecture>/<stem>.v1.md``).
    v1_files = sorted(processed_dir.rglob("*.v1.md"))
    if not v1_files:
        print(f"[warn] no *.v1.md files found anywhere under {processed_dir}")
        return

    for v1_path in v1_files:
        work_dir = v1_path.parent
        stem = v1_path.name[: -len(".v1.md")]
        content_list = _find_content_list(work_dir, stem)
        if content_list is None:
            print(f"[skip] {v1_path}: no *_content_list.json next to it")
            continue

        print(f"[lecture] {work_dir}  (stem={stem})")
        pages = _load_pages(content_list)
        if not pages:
            print(f"  [warn] {stem}: content_list.json has no usable pages")
            continue
        print(f"  found {len(pages)} slide pages")

        for variant in variants:
            sidecar = _generate_for_lecture_variant(
                work_dir, stem, variant, course_code, pages, force=force
            )
            if sidecar is not None:
                all_sidecars.append(sidecar)

    # --- Merge all sidecars into one benchmark file ---
    output_benchmark.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_benchmark.open("w", encoding="utf-8") as fout:
        for sidecar in sorted(all_sidecars):
            lecture_id = sidecar.parent.name
            for line in sidecar.read_text(encoding="utf-8").strip().splitlines():
                obj = json.loads(line)
                obj["lecture"] = lecture_id   # add top-level lecture field
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total += 1

    print(f"\nBenchmark written: {output_benchmark} ({total} total QA pairs)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate page-level SlideQA benchmark for CS 288 (local only)."
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        type=Path,
        help=(
            "Root directory to scan recursively for *.v1.md files. The "
            "directory containing each *.v1.md is treated as that lecture's "
            "work dir (must also contain *.v2.md, *.v3.md, and "
            "*_content_list.json). Both flat and MinerU's "
            "<lecture>/<stem>.pdf/auto/ layouts are supported."
        ),
    )
    parser.add_argument(
        "--course-code",
        default="CS288",
        help="Course code to embed in each QA entry (default: CS288).",
    )
    parser.add_argument(
        "--output-benchmark",
        default="cs288_benchmark.jsonl",
        type=Path,
        help="Path to write the merged benchmark JSONL (default: cs288_benchmark.jsonl).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate every <stem>.<variant>.qa.jsonl sidecar even if it "
            "already exists. Use this after changing the QA-generation prompt."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    processed = Path(args.processed_dir)
    if not processed.exists():
        print(f"ERROR: --processed-dir does not exist: {processed}", file=sys.stderr)
        sys.exit(1)
    generate_all(
        processed_dir=processed,
        course_code=args.course_code,
        output_benchmark=Path(args.output_benchmark),
        force=args.force,
    )
