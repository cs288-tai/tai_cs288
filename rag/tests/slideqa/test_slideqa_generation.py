"""
TDD — RED tests for image-aware SlideQA generation.

Target functions:
    get_slideqa_pairs_for_page(page_text, page_id, variant, image_paths=None)
        → list[dict]

    generate_slideqa_for_lecture(variant_md_path, variant, content_list_path)
        (method on BaseConverter subclass)

Key behaviour being tested:
    1. When image_paths is provided, the OpenAI call uses a vision-capable model
       and includes image_url content blocks for each image.
    2. When image_paths is None or [], the call falls back to text-only (original
       behaviour — no regression).
    3. generate_slideqa_for_lecture collects image paths from content_list.json
       items with type=="image" and passes them to get_slideqa_pairs_for_page.
    4. page_id is 0-based (equals page_idx from MinerU content_list.json).
    5. gold_page_ids is seeded with [page_id] (same 0-based integer).

MinerU content_list.json image item schema:
    {
        "type": "image",
        "page_idx": 0,          # 0-based
        "img_path": "images/page_0_img_0.png"   # relative to the output dir
    }
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_RAG_ROOT = Path(__file__).resolve().parents[2]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

from file_conversion_router.utils.title_handle import get_slideqa_pairs_for_page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PAIR = {
    "question_text": "What is attention?",
    "answer": "A weighting mechanism.",
    "question_type": "type_iii",
    "evidence_modality": "table",
    "gold_page_ids": [],
}


def _make_openai_response(pairs: list[dict]) -> MagicMock:
    """Build a minimal mock OpenAI response whose .choices[0].message.content
    is the JSON-serialised pairs list."""
    msg = MagicMock()
    msg.content = json.dumps(pairs)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# A. page_id index base
# ---------------------------------------------------------------------------


class TestPageIdIndexBase:
    """page_id passed in is echoed into gold_page_ids and the pair's page_id field."""

    def test_page_id_0_is_stored_as_gold(self):
        """page_id=0 → gold_page_ids=[0] (0-based, matches MinerU page_idx)."""
        mock_resp = _make_openai_response([_VALID_PAIR])

        with (
            patch(
                "file_conversion_router.utils.title_handle.get_openai_api_key",
                return_value="sk-test",
            ),
            patch("file_conversion_router.utils.title_handle.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            MockOpenAI.return_value = mock_client

            result = get_slideqa_pairs_for_page("some text", page_id=0, variant="v1")

        assert len(result) == 1
        assert result[0]["gold_page_ids"] == [0]
        assert result[0]["page_id"] == 0

    def test_page_id_4_is_stored_as_gold(self):
        """page_id=4 → gold_page_ids=[4]."""
        mock_resp = _make_openai_response([_VALID_PAIR])

        with (
            patch(
                "file_conversion_router.utils.title_handle.get_openai_api_key",
                return_value="sk-test",
            ),
            patch("file_conversion_router.utils.title_handle.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            MockOpenAI.return_value = mock_client

            result = get_slideqa_pairs_for_page("some text", page_id=4, variant="v1")

        assert result[0]["gold_page_ids"] == [4]
        assert result[0]["page_id"] == 4


# ---------------------------------------------------------------------------
# B. text-only call (no images) — no regression
# ---------------------------------------------------------------------------


class TestTextOnlyCall:
    """When image_paths is None or [], behaviour is identical to the original."""

    def test_no_images_uses_text_message_only(self):
        """Without images, the messages list has a single user text message."""
        mock_resp = _make_openai_response([_VALID_PAIR])

        with (
            patch(
                "file_conversion_router.utils.title_handle.get_openai_api_key",
                return_value="sk-test",
            ),
            patch("file_conversion_router.utils.title_handle.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            MockOpenAI.return_value = mock_client

            get_slideqa_pairs_for_page("text content", page_id=0, variant="v1")

            call_kwargs = mock_client.chat.completions.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["messages"]
            # The user message content must be a plain string (not a list of content blocks)
            user_msg = next(m for m in messages if m["role"] == "user")
            assert isinstance(user_msg["content"], str)

    def test_empty_image_paths_behaves_as_no_images(self):
        """image_paths=[] is equivalent to image_paths=None."""
        mock_resp = _make_openai_response([_VALID_PAIR])

        with (
            patch(
                "file_conversion_router.utils.title_handle.get_openai_api_key",
                return_value="sk-test",
            ),
            patch("file_conversion_router.utils.title_handle.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            MockOpenAI.return_value = mock_client

            get_slideqa_pairs_for_page(
                "text content", page_id=0, variant="v1", image_paths=[]
            )

            call_kwargs = mock_client.chat.completions.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs.kwargs["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            assert isinstance(user_msg["content"], str)


# ---------------------------------------------------------------------------
# C. vision call (with images)
# ---------------------------------------------------------------------------


class TestVisionCall:
    """When image_paths is non-empty, the model call uses vision content blocks."""

    def test_image_paths_causes_content_list_in_message(self, tmp_path):
        """With images, user message content is a list (OpenAI vision format)."""
        # Create a tiny valid PNG (1x1 pixel) so base64 encoding works
        img = tmp_path / "slide_0.png"
        img.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        mock_resp = _make_openai_response([_VALID_PAIR])

        with (
            patch(
                "file_conversion_router.utils.title_handle.get_openai_api_key",
                return_value="sk-test",
            ),
            patch("file_conversion_router.utils.title_handle.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            MockOpenAI.return_value = mock_client

            result = get_slideqa_pairs_for_page(
                "text content", page_id=0, variant="v1", image_paths=[img]
            )

            call_kwargs = mock_client.chat.completions.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs.kwargs["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")

            # Content must be a list of blocks (vision format)
            assert isinstance(user_msg["content"], list), (
                "Expected list of content blocks when images provided"
            )

    def test_image_content_block_has_correct_type(self, tmp_path):
        """Each image produces a content block with type='image_url'."""
        img = tmp_path / "slide_0.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        mock_resp = _make_openai_response([_VALID_PAIR])

        with (
            patch(
                "file_conversion_router.utils.title_handle.get_openai_api_key",
                return_value="sk-test",
            ),
            patch("file_conversion_router.utils.title_handle.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            MockOpenAI.return_value = mock_client

            get_slideqa_pairs_for_page(
                "text content", page_id=0, variant="v1", image_paths=[img]
            )

            messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            img_blocks = [b for b in user_msg["content"] if b.get("type") == "image_url"]
            assert len(img_blocks) == 1

    def test_image_content_block_uses_base64_data_url(self, tmp_path):
        """Image block uses data: URI with base64-encoded PNG content."""
        img = tmp_path / "slide_0.png"
        raw_bytes = b"\x89PNG\r\n\x1a\n" + b"\xAB" * 20
        img.write_bytes(raw_bytes)

        mock_resp = _make_openai_response([_VALID_PAIR])

        with (
            patch(
                "file_conversion_router.utils.title_handle.get_openai_api_key",
                return_value="sk-test",
            ),
            patch("file_conversion_router.utils.title_handle.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            MockOpenAI.return_value = mock_client

            get_slideqa_pairs_for_page(
                "text content", page_id=0, variant="v1", image_paths=[img]
            )

            messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            img_block = next(b for b in user_msg["content"] if b.get("type") == "image_url")
            url = img_block["image_url"]["url"]

            expected_b64 = base64.b64encode(raw_bytes).decode("ascii")
            assert url == f"data:image/png;base64,{expected_b64}"

    def test_text_block_present_alongside_image(self, tmp_path):
        """The prompt text also appears as a 'text' block in the content list."""
        img = tmp_path / "slide_0.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        mock_resp = _make_openai_response([_VALID_PAIR])

        with (
            patch(
                "file_conversion_router.utils.title_handle.get_openai_api_key",
                return_value="sk-test",
            ),
            patch("file_conversion_router.utils.title_handle.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            MockOpenAI.return_value = mock_client

            get_slideqa_pairs_for_page(
                "text content", page_id=0, variant="v1", image_paths=[img]
            )

            messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            text_blocks = [b for b in user_msg["content"] if b.get("type") == "text"]
            assert len(text_blocks) >= 1

    def test_two_images_produce_two_image_blocks(self, tmp_path):
        """One image per Path in image_paths."""
        img1 = tmp_path / "img1.png"
        img2 = tmp_path / "img2.png"
        img1.write_bytes(b"\x89PNG" + b"\x00" * 10)
        img2.write_bytes(b"\x89PNG" + b"\x00" * 10)

        mock_resp = _make_openai_response([_VALID_PAIR])

        with (
            patch(
                "file_conversion_router.utils.title_handle.get_openai_api_key",
                return_value="sk-test",
            ),
            patch("file_conversion_router.utils.title_handle.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            MockOpenAI.return_value = mock_client

            get_slideqa_pairs_for_page(
                "text", page_id=0, variant="v1", image_paths=[img1, img2]
            )

            messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            img_blocks = [b for b in user_msg["content"] if b.get("type") == "image_url"]
            assert len(img_blocks) == 2

    def test_nonexistent_image_is_skipped_gracefully(self, tmp_path):
        """A missing image file is skipped; valid images and text still processed."""
        good_img = tmp_path / "good.png"
        good_img.write_bytes(b"\x89PNG" + b"\x00" * 10)
        missing = tmp_path / "does_not_exist.png"

        mock_resp = _make_openai_response([_VALID_PAIR])

        with (
            patch(
                "file_conversion_router.utils.title_handle.get_openai_api_key",
                return_value="sk-test",
            ),
            patch("file_conversion_router.utils.title_handle.OpenAI") as MockOpenAI,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            MockOpenAI.return_value = mock_client

            # Must not raise
            result = get_slideqa_pairs_for_page(
                "text", page_id=0, variant="v1", image_paths=[good_img, missing]
            )

            messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            img_blocks = [b for b in user_msg["content"] if b.get("type") == "image_url"]
            # Only 1 good image was encoded; missing is skipped
            assert len(img_blocks) == 1


# ---------------------------------------------------------------------------
# D. generate_slideqa_for_lecture — image collection from content_list.json
# ---------------------------------------------------------------------------


class TestGenerateSlideqaImageCollection:
    """generate_slideqa_for_lecture must collect img_path values per page_idx
    and pass them to get_slideqa_pairs_for_page."""

    def _make_content_list(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Write a content_list.json with 1 text + 1 image item on page_idx=0."""
        # Create a fake image file
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_file = img_dir / "page_0_img_0.png"
        img_file.write_bytes(b"\x89PNG" + b"\x00" * 10)

        content_list = [
            {"type": "text", "page_idx": 0, "text": "Slide text here."},
            {"type": "image", "page_idx": 0, "img_path": str(img_file)},
        ]
        cl_path = tmp_path / "lecture_content_list.json"
        cl_path.write_text(json.dumps(content_list), encoding="utf-8")

        # Variant markdown
        variant_md = tmp_path / "lecture.v1.md"
        variant_md.write_text("# Lecture\n\nSlide text here.", encoding="utf-8")

        return cl_path, variant_md, img_file

    def test_image_paths_passed_to_get_slideqa_pairs(self, tmp_path):
        """For a page with an image item, image_paths is non-empty in the call."""
        cl_path, variant_md, img_file = self._make_content_list(tmp_path)

        with patch(
            "file_conversion_router.conversion.base_converter.get_slideqa_pairs_for_page"
        ) as mock_fn:
            mock_fn.return_value = []

            from file_conversion_router.conversion.base_converter import BaseConverter

            # Use a minimal concrete subclass — we only need generate_slideqa_for_lecture
            class _FakeConverter(BaseConverter):
                def _to_markdown(self, inp, out):
                    pass

            converter = _FakeConverter.__new__(_FakeConverter)
            converter.generate_slideqa_for_lecture(variant_md, "v1", cl_path)

            assert mock_fn.called, "get_slideqa_pairs_for_page was never called"
            call_kwargs = mock_fn.call_args.kwargs or {}
            image_paths = call_kwargs.get("image_paths") or (
                mock_fn.call_args.args[3] if len(mock_fn.call_args.args) > 3 else None
            )
            assert image_paths is not None and len(image_paths) >= 1, (
                "Expected image_paths to be passed with at least 1 path"
            )

    def test_page_without_images_passes_empty_image_paths(self, tmp_path):
        """For a page with no image items, image_paths is [] or None."""
        content_list = [
            {"type": "text", "page_idx": 0, "text": "Text only slide."},
        ]
        cl_path = tmp_path / "lecture_content_list.json"
        cl_path.write_text(json.dumps(content_list), encoding="utf-8")
        variant_md = tmp_path / "lecture.v1.md"
        variant_md.write_text("# Lecture\nText only.", encoding="utf-8")

        with patch(
            "file_conversion_router.conversion.base_converter.get_slideqa_pairs_for_page"
        ) as mock_fn:
            mock_fn.return_value = []

            from file_conversion_router.conversion.base_converter import BaseConverter

            class _FakeConverter(BaseConverter):
                def _to_markdown(self, inp, out):
                    pass

            converter = _FakeConverter.__new__(_FakeConverter)
            converter.generate_slideqa_for_lecture(variant_md, "v1", cl_path)

            call_kwargs = mock_fn.call_args.kwargs or {}
            image_paths = call_kwargs.get("image_paths") or (
                mock_fn.call_args.args[3] if len(mock_fn.call_args.args) > 3 else []
            )
            assert not image_paths, (
                f"Expected empty image_paths for text-only page, got {image_paths}"
            )
