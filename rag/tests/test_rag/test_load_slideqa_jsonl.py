"""
TDD — RED tests for load_slideqa_jsonl().

load_slideqa_jsonl(output_md_path: Path) -> list[dict]

Scans for all variant sidecar files (.v1.qa.jsonl, .v2.qa.jsonl, .v3.qa.jsonl)
next to the given master markdown path, reads every line from each, and returns
a flat list of QA pair dicts.

Each returned dict must have at minimum:
    question_text, answer, question_type, evidence_modality,
    gold_page_ids, page_id, variant

Behaviour:
  - Missing sidecar file → silently skipped (not an error)
  - All three sidecars present → all pairs merged into one list
  - Malformed JSON lines → skipped with a warning, not raised
  - Empty sidecar → contributes zero pairs (not an error)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_RAG_ROOT = Path(__file__).resolve().parents[2]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

# RED: this import will fail until we add the function
from file_conversion_router.services.directory_service import load_slideqa_jsonl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PAIR_V1 = {
    "question_text": "What is attention?",
    "answer": "A mechanism to weight tokens.",
    "question_type": "type_i",
    "evidence_modality": "text_only",
    "gold_page_ids": [0],
    "page_id": 0,
    "variant": "v1",
}

_VALID_PAIR_V2 = {
    "question_text": "Describe the diagram.",
    "answer": "A transformer block.",
    "question_type": "type_ii",
    "evidence_modality": "visual",
    "gold_page_ids": [1],
    "page_id": 1,
    "variant": "v2",
}

_VALID_PAIR_V3 = {
    "question_text": "What is in the table?",
    "answer": "Hyperparameters.",
    "question_type": "type_iii",
    "evidence_modality": "table",
    "gold_page_ids": [2],
    "page_id": 2,
    "variant": "v3",
}


def _write_sidecar(directory: Path, stem: str, variant: str, pairs: list[dict]) -> Path:
    """Write a .vN.qa.jsonl sidecar file and return its path."""
    path = directory / f"{stem}.{variant}.qa.jsonl"
    lines = [json.dumps(p, ensure_ascii=False) for p in pairs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadSlideqaJsonl:

    def test_returns_empty_list_when_no_sidecars(self, tmp_path):
        """No .vN.qa.jsonl sidecars → empty list, no error."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\n", encoding="utf-8")

        result = load_slideqa_jsonl(md)

        assert result == []

    def test_loads_single_variant_sidecar(self, tmp_path):
        """One sidecar present → its pairs returned."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\n", encoding="utf-8")
        _write_sidecar(tmp_path, "lecture01", "v1", [_VALID_PAIR_V1])

        result = load_slideqa_jsonl(md)

        assert len(result) == 1
        assert result[0]["question_type"] == "type_i"
        assert result[0]["evidence_modality"] == "text_only"
        assert result[0]["variant"] == "v1"

    def test_loads_all_three_variant_sidecars(self, tmp_path):
        """All three sidecars present → pairs from all three merged."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\n", encoding="utf-8")
        _write_sidecar(tmp_path, "lecture01", "v1", [_VALID_PAIR_V1])
        _write_sidecar(tmp_path, "lecture01", "v2", [_VALID_PAIR_V2])
        _write_sidecar(tmp_path, "lecture01", "v3", [_VALID_PAIR_V3])

        result = load_slideqa_jsonl(md)

        assert len(result) == 3
        variants_seen = {p["variant"] for p in result}
        assert variants_seen == {"v1", "v2", "v3"}

    def test_missing_sidecar_silently_skipped(self, tmp_path):
        """If v2 sidecar is absent, only v1 and v3 pairs are returned."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\n", encoding="utf-8")
        _write_sidecar(tmp_path, "lecture01", "v1", [_VALID_PAIR_V1])
        # v2 intentionally not written
        _write_sidecar(tmp_path, "lecture01", "v3", [_VALID_PAIR_V3])

        result = load_slideqa_jsonl(md)

        assert len(result) == 2
        assert {p["variant"] for p in result} == {"v1", "v3"}

    def test_empty_sidecar_contributes_zero_pairs(self, tmp_path):
        """Empty sidecar file → no error, zero pairs from that variant."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\n", encoding="utf-8")
        empty = tmp_path / "lecture01.v1.qa.jsonl"
        empty.write_text("", encoding="utf-8")

        result = load_slideqa_jsonl(md)

        assert result == []

    def test_malformed_json_line_is_skipped(self, tmp_path):
        """A corrupt line in the sidecar is skipped; valid lines are kept."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\n", encoding="utf-8")
        sidecar = tmp_path / "lecture01.v1.qa.jsonl"
        sidecar.write_text(
            json.dumps(_VALID_PAIR_V1) + "\n"
            + "NOT VALID JSON !!!\n"
            + json.dumps({**_VALID_PAIR_V1, "page_id": 1}) + "\n",
            encoding="utf-8",
        )

        result = load_slideqa_jsonl(md)

        assert len(result) == 2  # 2 valid lines, 1 malformed skipped

    def test_each_pair_carries_required_fields(self, tmp_path):
        """Every returned dict must have all required SlideQA fields."""
        required = {
            "question_text", "answer", "question_type",
            "evidence_modality", "gold_page_ids", "page_id", "variant",
        }
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\n", encoding="utf-8")
        _write_sidecar(tmp_path, "lecture01", "v1", [_VALID_PAIR_V1])

        result = load_slideqa_jsonl(md)

        for pair in result:
            assert required <= set(pair.keys()), (
                f"Pair missing fields: {required - set(pair.keys())}"
            )

    def test_multiple_pairs_per_sidecar(self, tmp_path):
        """Multiple QA pairs in one sidecar are all returned."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\n", encoding="utf-8")
        pairs = [
            {**_VALID_PAIR_V1, "page_id": i, "gold_page_ids": [i]}
            for i in range(5)
        ]
        _write_sidecar(tmp_path, "lecture01", "v1", pairs)

        result = load_slideqa_jsonl(md)

        assert len(result) == 5

    def test_uses_master_md_stem_not_variant_stem(self, tmp_path):
        """Sidecars are derived from the master .md stem, not from a variant path."""
        # Master md is lecture01.md → sidecars are lecture01.v1.qa.jsonl etc.
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\n", encoding="utf-8")
        _write_sidecar(tmp_path, "lecture01", "v1", [_VALID_PAIR_V1])

        result = load_slideqa_jsonl(md)

        assert len(result) == 1

    def test_whitespace_only_lines_skipped(self, tmp_path):
        """Blank/whitespace lines in the JSONL file are silently skipped."""
        md = tmp_path / "lecture01.md"
        md.write_text("# Slide\n", encoding="utf-8")
        sidecar = tmp_path / "lecture01.v1.qa.jsonl"
        sidecar.write_text(
            "\n"
            + json.dumps(_VALID_PAIR_V1) + "\n"
            + "   \n",
            encoding="utf-8",
        )

        result = load_slideqa_jsonl(md)

        assert len(result) == 1
