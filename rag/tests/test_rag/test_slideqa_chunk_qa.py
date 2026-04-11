"""
Tests for get_slideqa_pairs_for_page() in title_handle.py.

QA pairs are generated per whole slide page (not per chunk).
Gold evidence is page_id only — no chunk_id in the schema.

Covers:
  - Returns a list of dicts with required keys
  - question_type is one of the five valid values
  - gold_page_ids is a list seeded with the caller's page_id
  - page_id and variant are carried through
  - No chunk_id field in output
  - Returns [] when OPENAI_API_KEY is absent
  - Returns [] on malformed JSON response
  - Returns [] on API exception
  - Malformed/missing-key objects are dropped
  - Invalid question_type objects are dropped
  - Prompt uses gold_page_ids (not gold_chunk_ids)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_RAG_ROOT = Path(__file__).resolve().parents[3]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from file_conversion_router.utils.title_handle import get_slideqa_pairs_for_page

VALID_QUESTION_TYPES = {"type_i", "type_ii", "type_iii", "type_iv", "type_v"}

_MOCK_PAIRS = [
    {
        "question_text": "What does backpropagation compute?",
        "answer": "Gradients for training neural networks.",
        "question_type": "type_i",
        "evidence_modality": "text_only",
        "gold_page_ids": [],
    },
    {
        "question_text": "What does the diagram depict?",
        "answer": "A computational graph.",
        "question_type": "type_ii",
        "evidence_modality": "visual",
        "gold_page_ids": [],
    },
]


class TestGetSlideqaPairsForPage:

    def test_returns_list_on_success(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(_MOCK_PAIRS)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_page(
                page_text="Some slide text.",
                page_id=3,
                variant="v2",
            )

        assert isinstance(result, list)
        assert len(result) == 2

    def test_each_entry_has_required_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(_MOCK_PAIRS)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_page(
                page_text="Some slide text.",
                page_id=3,
                variant="v1",
            )

        required = {
            "question_text", "answer", "question_type",
            "evidence_modality", "gold_page_ids", "page_id", "variant",
        }
        for entry in result:
            assert required <= set(entry.keys()), f"Missing keys in {entry}"

    def test_no_chunk_id_in_output(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(_MOCK_PAIRS)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_page(
                page_text="text", page_id=1, variant="v1"
            )

        for entry in result:
            assert "chunk_id" not in entry
            assert "gold_chunk_ids" not in entry

    def test_page_id_and_variant_carried_through(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(_MOCK_PAIRS)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_page(
                page_text="text", page_id=7, variant="v3"
            )

        for entry in result:
            assert entry["page_id"] == 7
            assert entry["variant"] == "v3"

    def test_gold_page_ids_seeded_with_page_id(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(_MOCK_PAIRS)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_page(
                page_text="text", page_id=5, variant="v2"
            )

        for entry in result:
            assert isinstance(entry["gold_page_ids"], list)
            assert 5 in entry["gold_page_ids"]

    def test_question_type_is_valid(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(_MOCK_PAIRS)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_page(
                page_text="text", page_id=1, variant="v1"
            )

        for entry in result:
            assert entry["question_type"] in VALID_QUESTION_TYPES

    def test_returns_empty_list_when_no_api_key(self, monkeypatch):
        with patch("file_conversion_router.utils.title_handle.get_openai_api_key", return_value=None):
            result = get_slideqa_pairs_for_page(
                page_text="text", page_id=1, variant="v1"
            )
        assert result == []

    def test_returns_empty_list_on_malformed_json(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "not valid JSON {{"
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_page(
                page_text="text", page_id=1, variant="v1"
            )
        assert result == []

    def test_returns_empty_list_on_api_exception(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_page(
                page_text="text", page_id=1, variant="v1"
            )
        assert result == []

    def test_filters_out_pairs_missing_required_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        bad_pairs = [
            {"question_text": "Q?", "question_type": "type_i",
             "evidence_modality": "text_only", "gold_page_ids": []},  # missing answer
            {"answer": "A", "question_type": "type_ii",
             "evidence_modality": "visual", "gold_page_ids": []},     # missing question_text
            {"question_text": "Valid Q?", "answer": "Valid A",
             "question_type": "type_i", "evidence_modality": "text_only",
             "gold_page_ids": []},
        ]
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(bad_pairs)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_page(
                page_text="text", page_id=1, variant="v1"
            )

        assert len(result) == 1
        assert result[0]["question_text"] == "Valid Q?"

    def test_filters_out_pairs_with_invalid_question_type(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        bad_pairs = [
            {"question_text": "Q?", "answer": "A",
             "question_type": "type_vi",  # invalid
             "evidence_modality": "text_only", "gold_page_ids": []},
            {"question_text": "Good Q?", "answer": "Good A",
             "question_type": "type_iii",
             "evidence_modality": "table", "gold_page_ids": []},
        ]
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(bad_pairs)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_page(
                page_text="text", page_id=1, variant="v1"
            )

        assert len(result) == 1
        assert result[0]["question_type"] == "type_iii"

    def test_prompt_uses_gold_page_ids_not_gold_chunk_ids(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        captured = {}

        def fake_create(**kwargs):
            captured["content"] = kwargs["messages"][0]["content"]
            m = MagicMock()
            m.choices[0].message.content = json.dumps([])
            return m

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            get_slideqa_pairs_for_page(page_text="text", page_id=1, variant="v1")

        prompt = captured.get("content", "")
        assert "gold_page_ids" in prompt
        assert "gold_chunk_ids" not in prompt
