from __future__ import annotations

from file_conversion_router.conversion.base_converter import BaseConverter
from file_conversion_router.services.tai_MinerU_service.api import (
    convert_pdf_to_md_by_MinerU,
)

from pathlib import Path
from typing import Any, Optional
import base64
import json
import os
import re

from loguru import logger

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]


class PdfConverter(BaseConverter):
    """PDF -> Markdown converter with optional VLM augmentation.

    This version keeps the original image links in the Markdown and inserts
    VLM-generated blocks below each image link. It also materializes three
    ablation-ready Markdown views:
      - v1: OCR / original text only
      - v2: OCR + image captions
      - v3: OCR + image captions + structured visual notes
    """

    # Markers used for idempotent insertion and later filtering.
    VLM_BEGIN_PREFIX = '<!-- TAI_VLM_BEGIN image="'
    VLM_END = "<!-- TAI_VLM_END -->"

    V2_BEGIN = "<!-- TAI_V2_BEGIN -->"
    V2_END = "<!-- TAI_V2_END -->"

    V3_BEGIN = "<!-- TAI_V3_BEGIN -->"
    V3_END = "<!-- TAI_V3_END -->"

    # Match standard markdown image links.
    IMAGE_LINK_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")

    def __init__(
        self,
        course_name: str,
        course_code: str,
        file_uuid: Optional[str] = None,
    ) -> None:
        """Initialize converter with course metadata."""
        self.course_name = course_name
        self.course_code = course_code
        self.file_uuid = file_uuid

    def remove_image_links(self, text: str) -> str:
        """Remove all markdown image links from text."""
        image_link_pattern = r"!\[.*?\]\(.*?\)"
        return re.sub(image_link_pattern, "", text)

    def _read_text(self, path: Path) -> str:
        """Read UTF-8 text from file."""
        return path.read_text(encoding="utf-8")

    def _write_text(self, path: Path, content: str) -> None:
        """Write UTF-8 text to file."""
        path.write_text(content, encoding="utf-8")

    def _resolve_markdown_path(self, result: Any, source_pdf: Path) -> Path:
        """Resolve MinerU output into a markdown path.

        This helper accepts several common return shapes so the code is more
        robust to service wrapper differences.
        """
        if isinstance(result, Path):
            return result

        if isinstance(result, str):
            return Path(result)

        if isinstance(result, dict):
            for key in (
                "md_file_path",
                "markdown_path",
                "target",
                "output_path",
                "md_path",
            ):
                value = result.get(key)
                if value:
                    return Path(value)

        # Conservative fallback: common local naming convention.
        candidate = source_pdf.with_suffix(".md")
        if candidate.exists():
            return candidate

        raise FileNotFoundError(
            "Could not resolve markdown output path from MinerU result."
        )

    def _build_vlm_prompt(self, context_before: str, context_after: str) -> str:
        """Build the prompt used for caption + structured notes generation."""
        return (
            "You are analyzing an image extracted from a course slide or document.\n\n"
            f"Markdown context before the image:\n{context_before}\n\n"
            f"Markdown context after the image:\n{context_after}\n\n"
            "Return JSON only, with this exact schema:\n"
            '{\n'
            '  "caption": "2-4 sentences describing the image and the key idea it conveys",\n'
            '  "visual_notes": ["bullet 1", "bullet 2", "bullet 3"]\n'
            '}\n\n'
            "Rules:\n"
            "1. The caption should be concise but informative.\n"
            "2. visual_notes should capture concrete details useful for retrieval.\n"
            "3. If the image is a chart, include axes, legend, and main trend.\n"
            "4. If the image is a table, include headers and key values/comparisons.\n"
            "5. If the image is a diagram, include components, arrows, and relationships.\n"
            "6. If the image is code/equations/layout, summarize the most important structure.\n"
            "7. Do not wrap the JSON in markdown fences.\n"
        )

    def _call_vlm_for_image_description(
        self,
        image_path: Path,
        context_before: str,
        context_after: str,
    ) -> tuple[str, list[str]]:
        """Call OpenAI vision model and return (caption, visual_notes)."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.info("OPENAI_API_KEY not set; skipping VLM call.")
            return "", []

        client = OpenAI(api_key=api_key)

        ext = image_path.suffix.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"

        with image_path.open("rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = self._build_vlm_prompt(context_before, context_after)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{ext};base64,{image_b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=700,
            )

            raw = (response.choices[0].message.content or "").strip()
            data = json.loads(raw)

            caption = str(data.get("caption", "") or "").strip()
            visual_notes_raw = data.get("visual_notes", [])

            if not isinstance(visual_notes_raw, list):
                visual_notes_raw = []

            visual_notes = [
                str(item).strip()
                for item in visual_notes_raw
                if str(item).strip()
            ]
            return caption, visual_notes

        except Exception as e:
            logger.warning(f"VLM call failed for {image_path}: {e}")
            return "", []

    def _render_vlm_block(
        self,
        image_rel_path: str,
        caption: str,
        visual_notes: list[str],
    ) -> str:
        """Render a single VLM augmentation block under one image link."""
        lines: list[str] = [
            f'<!-- TAI_VLM_BEGIN image="{image_rel_path}" -->',
        ]

        if caption:
            lines.extend(
                [
                    self.V2_BEGIN,
                    "[Image Caption]",
                    caption,
                    self.V2_END,
                    "",
                ]
            )

        if visual_notes:
            lines.extend(
                [
                    self.V3_BEGIN,
                    "[Visual Notes]",
                    *[f"- {note}" for note in visual_notes],
                    self.V3_END,
                ]
            )

        lines.append(self.VLM_END)
        return "\n".join(lines)

    def _has_vlm_block(self, content: str, image_rel_path: str) -> bool:
        """Check whether content already has a VLM block for this image."""
        marker = f'{self.VLM_BEGIN_PREFIX}{image_rel_path}" -->'
        return marker in content

    def replace_images_with_vlm_descriptions(self, markdown_path: Path) -> str:
        """Insert V2/V3 blocks under each image link, while keeping the link.

        Behavior:
        - Keeps original markdown image links untouched.
        - Inserts a VLM block directly below each image link.
        - Skips images that already have a block (idempotent).
        - If image file is missing or VLM fails, leaves the link in place.
        """
        content = self._read_text(markdown_path)

        # If there is no API key, keep markdown unchanged.
        if not os.environ.get("OPENAI_API_KEY"):
            logger.info("OPENAI_API_KEY not set – skipping VLM augmentation.")
            return content

        lines = content.split("\n")
        result_lines: list[str] = []

        for i, line in enumerate(lines):
            result_lines.append(line)

            match = self.IMAGE_LINK_PATTERN.search(line)
            if not match:
                continue

            image_rel_path = match.group(2).strip()
            if not image_rel_path:
                continue

            # Idempotency check against original full content.
            if self._has_vlm_block(content, image_rel_path):
                continue

            image_path = markdown_path.parent / image_rel_path
            if not image_path.exists():
                logger.warning(f"Image not found, keeping link unchanged: {image_path}")
                continue

            context_before = "\n".join(lines[max(0, i - 5): i])
            context_after = "\n".join(lines[i + 1: min(len(lines), i + 6)])

            logger.info(f"Calling VLM for image: {image_path.name}")
            caption, visual_notes = self._call_vlm_for_image_description(
                image_path=image_path,
                context_before=context_before,
                context_after=context_after,
            )

            if not caption and not visual_notes:
                # Keep link unchanged if VLM call produced nothing useful.
                continue

            block = self._render_vlm_block(
                image_rel_path=image_rel_path,
                caption=caption,
                visual_notes=visual_notes,
            )
            result_lines.append(block)

        new_content = "\n".join(result_lines)
        self._write_text(markdown_path, new_content)
        return new_content

    def _remove_vlm_blocks(self, text: str) -> str:
        """Remove entire VLM augmentation blocks."""
        pattern = (
            r'<!-- TAI_VLM_BEGIN image=".*?" -->'
            r".*?"
            r"<!-- TAI_VLM_END -->"
        )
        return re.sub(pattern, "", text, flags=re.DOTALL)

    def _strip_v3_only(self, text: str) -> str:
        """Remove V3 blocks but keep V2 blocks."""
        pattern = (
            re.escape(self.V3_BEGIN)
            + r".*?"
            + re.escape(self.V3_END)
        )
        return re.sub(pattern, "", text, flags=re.DOTALL)

    def _normalize_blank_lines(self, text: str) -> str:
        """Normalize extra blank lines for cleaner markdown output."""
        # Remove lines that contain only hash marks and spaces.
        text = re.sub(r"^[ \t]*#+[ \t]*$", "", text, flags=re.MULTILINE)

        # Collapse 3+ blank lines into 2.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip() + "\n"

    def clean_markdown_content(self, markdown_path: Path) -> str:
        """Clean markdown while preserving image links and VLM blocks.

        Important:
        - We no longer remove image links here.
        - This keeps the master markdown faithful to the original content.
        """
        content = self._read_text(markdown_path)
        cleaned_content = self._normalize_blank_lines(content)
        self._write_text(markdown_path, cleaned_content)
        return cleaned_content

    def build_markdown_for_variant(self, content: str, variant: str) -> str:
        """Build ablation-specific markdown text from one master markdown.

        Variants:
        - v1: keep original OCR/original markdown only; remove all VLM blocks
        - v2: keep V2 caption blocks, remove V3 visual notes
        - v3: keep both V2 and V3
        """
        variant = variant.lower().strip()
        if variant not in {"v1", "v2", "v3"}:
            raise ValueError(f"Unknown variant: {variant}. Expected one of v1/v2/v3.")

        if variant == "v1":
            text = self._remove_vlm_blocks(content)
            return self._normalize_blank_lines(text)

        if variant == "v2":
            text = self._strip_v3_only(content)
            return self._normalize_blank_lines(text)

        # v3 keeps both V2 and V3 blocks
        return self._normalize_blank_lines(content)

    def _materialize_variant_markdowns(self, markdown_path: Path) -> dict[str, Path]:
        """Create .v1.md / .v2.md / .v3.md files next to the master markdown."""
        content = self._read_text(markdown_path)

        # Keep the original as the master file.
        master_path = markdown_path.with_name(f"{markdown_path.stem}.master.md")
        self._write_text(master_path, content)

        variant_paths: dict[str, Path] = {}
        for variant in ("v1", "v2", "v3"):
            variant_content = self.build_markdown_for_variant(content, variant)
            variant_path = markdown_path.with_name(f"{markdown_path.stem}.{variant}.md")
            self._write_text(variant_path, variant_content)
            variant_paths[variant] = variant_path

        return variant_paths

    def _load_content_list_json(self, markdown_path: Path) -> Any:
        """Load MinerU content list JSON if present."""
        json_file_path = markdown_path.with_name(f"{markdown_path.stem}_content_list.json")
        if not json_file_path.exists():
            logger.warning(f"content_list JSON not found: {json_file_path}")
            return None

        with json_file_path.open("r", encoding="utf-8") as f_json:
            return json.load(f_json)

    def _to_markdown(self, input_path: Path, output_path: Path) -> Optional[Path]:
        """Convert a PDF to markdown, augment it, and materialize V1/V2/V3 views.

        Matches the BaseConverter._to_markdown(input_path, output_path) -> Path contract.

        Returns:
            Path to the master markdown file (used by BaseConverter._to_page for
            further structured-content processing), or None on failure.
        """
        source_path = Path(input_path)

        if not source_path.exists():
            raise FileNotFoundError(f"PDF file not found: {source_path}")

        result = convert_pdf_to_md_by_MinerU(
            source_path,
            self.course_name,
            self.course_code,
        )
        md_file_path = self._resolve_markdown_path(result, source_path)

        if not md_file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_file_path}")

        # 1. Insert VLM blocks in-place into the original markdown.
        self.replace_images_with_vlm_descriptions(md_file_path)

        # 2. Clean while preserving image links and VLM blocks.
        self.clean_markdown_content(md_file_path)

        # 3. Materialize ablation-ready markdown files (v1/v2/v3).
        variant_paths = self._materialize_variant_markdowns(md_file_path)

        # 4. Generate SlideQA pairs per slide page for each variant.
        content_list_path = md_file_path.with_name(
            f"{md_file_path.stem}_content_list.json"
        )
        for variant, variant_md_path in variant_paths.items():
            self.generate_slideqa_for_lecture(variant_md_path, variant, content_list_path)

        # Return the master markdown path for BaseConverter to use.
        return md_file_path

    def generate_index_helper(self, data, md=None):
        self.index_helper = []
        for item in data:
            if item.get('text_level') == 1:
                title = item['text'].strip()
                if title.startswith('# '):
                    title = title[2:]

                skip_patterns = [
                    re.compile(r'^\s*ROAR ACADEMY EXERCISES\s*$', re.I),
                    re.compile(r'^\s*(?:#+\s*)+$')  # lines that are only # + spaces
                ]
                if any(p.match(title) for p in skip_patterns):
                    continue

                # Check if title appears after any number of # symbols
                if md:
                    lines = md.split('\n')
                    title_found = False
                    for line in lines:
                        stripped_line = line.strip()
                        if stripped_line.startswith('#') and title in stripped_line:
                            # More precise check: extract the heading text
                            heading_text = re.sub(r'^#+\s*', '', stripped_line).strip()
                            if heading_text == title:
                                title_found = True
                                break

                    if title_found:
                        page_index = item['page_idx'] + 1  # 1-based
                        self.index_helper.append({title: page_index})