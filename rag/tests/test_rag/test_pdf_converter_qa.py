"""
Tests for PdfConverter QA-pair generation additions.

TDD — RED phase: these tests are written before the implementation exists.

Covers:
  - _build_qa_generation_prompt() produces a prompt containing all 5 question types
    (type i text-only, type ii image-dependent, type iii table-centric,
     type iv chart/graph, type v layout-dependent)
  - _call_vlm_for_qa_pairs() returns a list of dicts with the required keys
  - generate_qa_pairs_for_variant() writes a .qa.jsonl file next to the variant .md
  - generate_qa_pairs_for_variant() is idempotent (skips already-written pairs)
  - generate_qa_pairs_for_variant() handles missing OPENAI_API_KEY gracefully
  - generate_qa_pairs_for_variant() handles a malformed VLM JSON response
  - Each QA entry has: question_text, answer, question_type, evidence_modality,
    gold_page_ids, variant
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make file_conversion_router importable when running from the rag/ directory.
_RAG_ROOT = Path(__file__).resolve().parents[3]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from file_conversion_router.conversion.pdf_converter import PdfConverter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_QUESTION_TYPES = {"type_i", "type_ii", "type_iii", "type_iv", "type_v"}


def _make_converter() -> PdfConverter:
    return PdfConverter(course_name="CS 288", course_code="CS288")


def _make_minimal_variant_md(tmp_path: Path, variant: str) -> Path:
    """Write a tiny .v1.md / .v2.md / .v3.md for testing."""
    p = tmp_path / f"lecture01.{variant}.md"
    p.write_text(
        textwrap.dedent(
            f"""\
            # Lecture 01
            This is slide text for variant {variant}.
            """
        ),
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# Tests for _build_qa_generation_prompt
# ---------------------------------------------------------------------------


class TestBuildQAGenerationPrompt:
    """_build_qa_generation_prompt() must mention all five question types."""

    def test_returns_a_string(self):
        conv = _make_converter()
        prompt = conv._build_qa_generation_prompt(
            slide_content="Some slide text.",
            variant="v1",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_mentions_all_five_question_types(self):
        conv = _make_converter()
        prompt = conv._build_qa_generation_prompt(
            slide_content="Some slide text.",
            variant="v1",
        )
        prompt_lower = prompt.lower()
        # Each type label must appear
        for label in ("type i", "type ii", "type iii", "type iv", "type v"):
            assert label in prompt_lower, f"Missing '{label}' in prompt"

    def test_mentions_type_descriptions(self):
        conv = _make_converter()
        prompt = conv._build_qa_generation_prompt(
            slide_content="Some slide text.",
            variant="v1",
        )
        prompt_lower = prompt.lower()
        # Key terms that must be in the prompt per the paper
        for term in ("text-only", "diagram", "table", "chart", "layout"):
            assert term in prompt_lower, f"Missing '{term}' in prompt"

    def test_includes_slide_content_in_prompt(self):
        conv = _make_converter()
        slide_text = "Unique_marker_abc123"
        prompt = conv._build_qa_generation_prompt(
            slide_content=slide_text,
            variant="v3",
        )
        assert slide_text in prompt

    def test_requests_json_output(self):
        conv = _make_converter()
        prompt = conv._build_qa_generation_prompt(
            slide_content="text",
            variant="v2",
        )
        assert "json" in prompt.lower()

    def test_prompt_differs_by_variant(self):
        conv = _make_converter()
        p_v1 = conv._build_qa_generation_prompt(slide_content="text", variant="v1")
        p_v3 = conv._build_qa_generation_prompt(slide_content="text", variant="v3")
        # v3 prompt should mention captions/visual notes; at minimum the variant label
        assert p_v1 != p_v3


# ---------------------------------------------------------------------------
# Tests for _call_vlm_for_qa_pairs
# ---------------------------------------------------------------------------

_MOCK_QA_RESPONSE = [
    {
        "question_text": "What is the purpose of backpropagation?",
        "answer": "To compute gradients for training neural networks.",
        "question_type": "type_i",
        "evidence_modality": "text_only",
        "gold_page_ids": [],
    },
    {
        "question_text": "What does the diagram on this slide depict?",
        "answer": "A computational graph showing forward and backward passes.",
        "question_type": "type_ii",
        "evidence_modality": "visual",
        "gold_page_ids": [],
    },
]


class TestCallVlmForQaPairs:
    """_call_vlm_for_qa_pairs() wraps the OpenAI call and returns a typed list."""

    def test_returns_list_on_success(self, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(_MOCK_QA_RESPONSE)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            result = conv._call_vlm_for_qa_pairs(
                slide_content="Some slide text.",
                variant="v1",
            )

        assert isinstance(result, list)
        assert len(result) == 2

    def test_each_entry_has_required_keys(self, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(_MOCK_QA_RESPONSE)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            result = conv._call_vlm_for_qa_pairs(
                slide_content="Some slide text.",
                variant="v1",
            )

        required_keys = {"question_text", "answer", "question_type", "evidence_modality", "gold_page_ids"}
        for entry in result:
            assert required_keys <= set(entry.keys()), f"Missing keys in {entry}"

    def test_question_type_is_valid(self, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(_MOCK_QA_RESPONSE)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            result = conv._call_vlm_for_qa_pairs(
                slide_content="Some slide text.",
                variant="v1",
            )

        for entry in result:
            assert entry["question_type"] in VALID_QUESTION_TYPES

    def test_returns_empty_list_when_no_api_key(self, monkeypatch):
        conv = _make_converter()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = conv._call_vlm_for_qa_pairs(
            slide_content="Some slide text.",
            variant="v1",
        )
        assert result == []

    def test_returns_empty_list_on_malformed_json(self, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not valid JSON {{ broken"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            result = conv._call_vlm_for_qa_pairs(
                slide_content="Some slide text.",
                variant="v1",
            )
        assert result == []

    def test_returns_empty_list_on_openai_exception(self, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            result = conv._call_vlm_for_qa_pairs(
                slide_content="Some slide text.",
                variant="v1",
            )
        assert result == []


# ---------------------------------------------------------------------------
# Tests for generate_qa_pairs_for_variant
# ---------------------------------------------------------------------------


class TestGenerateQaPairsForVariant:
    """generate_qa_pairs_for_variant() reads a .vN.md and writes a .vN.qa.jsonl."""

    def test_writes_jsonl_file_next_to_variant_md(self, tmp_path, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        md_path = _make_minimal_variant_md(tmp_path, "v1")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(_MOCK_QA_RESPONSE)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            out_path = conv.generate_qa_pairs_for_variant(md_path, variant="v1")

        expected = tmp_path / "lecture01.v1.qa.jsonl"
        assert expected.exists(), "Expected .qa.jsonl was not written"
        assert out_path == expected

    def test_jsonl_lines_are_valid_json(self, tmp_path, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        md_path = _make_minimal_variant_md(tmp_path, "v2")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(_MOCK_QA_RESPONSE)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            out_path = conv.generate_qa_pairs_for_variant(md_path, variant="v2")

        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(_MOCK_QA_RESPONSE)
        for line in lines:
            obj = json.loads(line)
            assert "question_text" in obj

    def test_each_entry_carries_variant_field(self, tmp_path, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        md_path = _make_minimal_variant_md(tmp_path, "v3")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(_MOCK_QA_RESPONSE)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            out_path = conv.generate_qa_pairs_for_variant(md_path, variant="v3")

        for line in out_path.read_text(encoding="utf-8").strip().splitlines():
            obj = json.loads(line)
            assert obj.get("variant") == "v3"

    def test_idempotent_does_not_overwrite_existing_file(self, tmp_path, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        md_path = _make_minimal_variant_md(tmp_path, "v1")

        # Pre-write the .qa.jsonl
        existing = tmp_path / "lecture01.v1.qa.jsonl"
        sentinel = json.dumps({"question_text": "sentinel", "variant": "v1"})
        existing.write_text(sentinel + "\n", encoding="utf-8")
        mtime_before = existing.stat().st_mtime

        mock_client = MagicMock()
        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            out_path = conv.generate_qa_pairs_for_variant(md_path, variant="v1")

        # File should not have been touched
        assert out_path.read_text(encoding="utf-8").startswith('{"question_text": "sentinel"')
        assert not mock_client.chat.completions.create.called

    def test_returns_none_when_no_api_key(self, tmp_path, monkeypatch):
        conv = _make_converter()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        md_path = _make_minimal_variant_md(tmp_path, "v1")

        result = conv.generate_qa_pairs_for_variant(md_path, variant="v1")
        assert result is None

    def test_returns_none_when_vlm_produces_no_pairs(self, tmp_path, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        md_path = _make_minimal_variant_md(tmp_path, "v2")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps([])  # empty list
        mock_client.chat.completions.create.return_value = mock_response

        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            result = conv.generate_qa_pairs_for_variant(md_path, variant="v2")

        assert result is None
        assert not (tmp_path / "lecture01.v2.qa.jsonl").exists()


# ---------------------------------------------------------------------------
# Tests for generate_all_qa_pairs (convenience wrapper)
# ---------------------------------------------------------------------------


class TestGenerateAllQaPairs:
    """generate_all_qa_pairs() runs QA generation for v1, v2, and v3."""

    def test_generates_all_three_variants(self, tmp_path, monkeypatch):
        conv = _make_converter()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Write v1/v2/v3 markdown stubs
        for v in ("v1", "v2", "v3"):
            _make_minimal_variant_md(tmp_path, v)

        # Stub the master md path (stem = "lecture01")
        master_md = tmp_path / "lecture01.md"
        master_md.write_text("# Master\n", encoding="utf-8")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(_MOCK_QA_RESPONSE)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("file_conversion_router.conversion.pdf_converter.OpenAI", return_value=mock_client):
            results = conv.generate_all_qa_pairs(master_md)

        # Should return a dict keyed by variant
        assert set(results.keys()) == {"v1", "v2", "v3"}
        for v, path in results.items():
            if path is not None:
                assert path.suffix == ".jsonl"
                assert v in path.name
