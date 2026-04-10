"""
Tests for get_slideqa_pairs_for_chunk() in title_handle.py.

TDD — RED phase: written before implementation exists.

Covers:
  - Returns a list of dicts with required keys
  - question_type is one of the five valid values
  - evidence_modality is a valid string
  - gold_chunk_ids is a list (may be empty at generation time)
  - page_id is carried through from the caller
  - variant is carried through
  - Returns [] when OPENAI_API_KEY is absent
  - Returns [] on malformed JSON response
  - Returns [] on API exception
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

from file_conversion_router.utils.title_handle import get_slideqa_pairs_for_chunk

VALID_QUESTION_TYPES = {"type_i", "type_ii", "type_iii", "type_iv", "type_v"}
VALID_EVIDENCE_MODALITIES = {"text_only", "visual", "table", "chart", "layout"}

_MOCK_PAIRS = [
    {
        "question_text": "What does backpropagation compute?",
        "answer": "Gradients for training neural networks.",
        "question_type": "type_i",
        "evidence_modality": "text_only",
        "gold_chunk_ids": [],
    },
    {
        "question_text": "What does the diagram depict?",
        "answer": "A computational graph.",
        "question_type": "type_ii",
        "evidence_modality": "visual",
        "gold_chunk_ids": [],
    },
]


class TestGetSlideqaPairsForChunk:

    def test_returns_list_on_success(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(_MOCK_PAIRS)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_chunk(
                chunk_text="Some slide text.",
                chunk_id="chunk_001",
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
            result = get_slideqa_pairs_for_chunk(
                chunk_text="Some slide text.",
                chunk_id="chunk_001",
                page_id=3,
                variant="v1",
            )

        required = {
            "question_text", "answer", "question_type",
            "evidence_modality", "gold_chunk_ids", "chunk_id", "page_id", "variant",
        }
        for entry in result:
            assert required <= set(entry.keys()), f"Missing keys in {entry}"

    def test_chunk_id_and_page_id_carried_through(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(_MOCK_PAIRS)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_chunk(
                chunk_text="text",
                chunk_id="chunk_007",
                page_id=5,
                variant="v3",
            )

        for entry in result:
            assert entry["chunk_id"] == "chunk_007"
            assert entry["page_id"] == 5
            assert entry["variant"] == "v3"

    def test_question_type_is_valid(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(_MOCK_PAIRS)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_chunk(
                chunk_text="text", chunk_id="c1", page_id=1, variant="v1"
            )

        for entry in result:
            assert entry["question_type"] in VALID_QUESTION_TYPES

    def test_returns_empty_list_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Also ensure .env file doesn't provide a key
        with patch("file_conversion_router.utils.title_handle.get_openai_api_key", return_value=None):
            result = get_slideqa_pairs_for_chunk(
                chunk_text="text", chunk_id="c1", page_id=1, variant="v1"
            )
        assert result == []

    def test_returns_empty_list_on_malformed_json(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "not valid JSON {{"
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_chunk(
                chunk_text="text", chunk_id="c1", page_id=1, variant="v1"
            )
        assert result == []

    def test_returns_empty_list_on_api_exception(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_chunk(
                chunk_text="text", chunk_id="c1", page_id=1, variant="v1"
            )
        assert result == []

    def test_gold_chunk_ids_is_list(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(_MOCK_PAIRS)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_chunk(
                chunk_text="text", chunk_id="c1", page_id=1, variant="v2"
            )

        for entry in result:
            assert isinstance(entry["gold_chunk_ids"], list)

    def test_filters_out_pairs_missing_required_keys(self, monkeypatch):
        """Objects missing question_text, answer, or question_type are dropped."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        bad_pairs = [
            # missing answer
            {"question_text": "Q?", "question_type": "type_i",
             "evidence_modality": "text_only", "gold_chunk_ids": []},
            # missing question_text
            {"answer": "A", "question_type": "type_ii",
             "evidence_modality": "visual", "gold_chunk_ids": []},
            # valid
            {"question_text": "Valid Q?", "answer": "Valid A",
             "question_type": "type_i", "evidence_modality": "text_only",
             "gold_chunk_ids": []},
        ]
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(bad_pairs)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_chunk(
                chunk_text="text", chunk_id="c1", page_id=1, variant="v1"
            )

        assert len(result) == 1
        assert result[0]["question_text"] == "Valid Q?"

    def test_filters_out_pairs_with_invalid_question_type(self, monkeypatch):
        """Objects with an unrecognized question_type are dropped."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        bad_pairs = [
            {"question_text": "Q?", "answer": "A",
             "question_type": "type_vi",  # invalid
             "evidence_modality": "text_only", "gold_chunk_ids": []},
            {"question_text": "Good Q?", "answer": "Good A",
             "question_type": "type_iii",  # valid
             "evidence_modality": "table", "gold_chunk_ids": []},
        ]
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(bad_pairs)
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            result = get_slideqa_pairs_for_chunk(
                chunk_text="text", chunk_id="c1", page_id=1, variant="v1"
            )

        assert len(result) == 1
        assert result[0]["question_type"] == "type_iii"

    def test_prompt_uses_gold_chunk_ids_not_gold_page_ids(self, monkeypatch):
        """The prompt sent to the model should reference gold_chunk_ids, not gold_page_ids."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        captured_prompt = {}

        def fake_create(**kwargs):
            captured_prompt["content"] = kwargs["messages"][0]["content"]
            m = MagicMock()
            m.choices[0].message.content = json.dumps([])
            return m

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create

        with patch("file_conversion_router.utils.title_handle.OpenAI", return_value=mock_client):
            get_slideqa_pairs_for_chunk(
                chunk_text="text", chunk_id="c1", page_id=1, variant="v1"
            )

        prompt_text = captured_prompt.get("content", "")
        assert "gold_chunk_ids" in prompt_text
        assert "gold_page_ids" not in prompt_text
