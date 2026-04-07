from file_conversion_router.conversion.base_converter import BaseConverter
from file_conversion_router.services.tai_MinerU_service.api import convert_pdf_to_md_by_MinerU
from pathlib import Path
import base64
import json
import os
import re

from loguru import logger


class PdfConverter(BaseConverter):
    def __init__(self, course_name, course_code, file_uuid: str = None):
        super().__init__(course_name, course_code, file_uuid)
        self.available_tools = ["MinerU"]
        self.index_helper = None
        self.file_name = ""


    def remove_image_links(self, text):
        """
        Remove image links from the text.
        """
        # Regular expression to match image links
        image_link_pattern = r"!\[.*?\]\(.*?\)"
        # Remove all image links
        return re.sub(image_link_pattern, "", text)

    def _call_vlm_for_image_description(self, image_path: Path, context_before: str, context_after: str) -> str:
        """
        Call OpenAI vision model to generate a text description and QA for an image,
        given surrounding markdown context.
        Returns a text block to insert in place of the image link.
        """
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        ext = image_path.suffix.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "You are analyzing an image from a course slide or document.\n\n"
            f"Markdown context before the image:\n{context_before}\n\n"
            f"Markdown context after the image:\n{context_after}\n\n"
            "Tasks:\n"
            "1. Classify this image into exactly one of these types:\n"
            "   - chart: bar charts, line graphs, pie charts, scatter plots, histograms\n"
            "   - table: grids, tabular data (even if rendered as an image)\n"
            "   - diagram: flowcharts, architecture diagrams, UML, circuit diagrams, algorithms\n"
            "   - image: photos, illustrations, screenshots, or other visual content\n"
            "2. Write a concise but complete description of what this image shows "
            "and the key information it conveys.\n"
            "3. Write one Q&A pair (question and answer) that tests understanding of this image.\n\n"
            "Output format (use exactly these labels):\n"
            "[Image Type]\n"
            "<one of: chart, table, diagram, image>\n\n"
            "[Image Description]\n"
            "<description here>\n\n"
            "[Image Q&A]\n"
            "Q: <question>\n"
            "A: <answer>\n\n"
            "Output only the formatted text above, nothing else."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/{ext};base64,{image_b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"VLM call failed for {image_path}: {e}")
            return ""

    def replace_images_with_vlm_descriptions(self, markdown_path: Path) -> str:
        """
        Find every image link in the markdown, call VLM to get a description + QA,
        and replace the image link with that text in-place.
        Falls back to removing the link if the image file is missing or VLM fails.
        Only runs when OPENAI_API_KEY is set.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            logger.info("OPENAI_API_KEY not set – skipping VLM image description.")
            return None

        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
        result_lines = []

        for i, line in enumerate(lines):
            match = image_pattern.search(line)
            if not match:
                result_lines.append(line)
                continue

            image_rel_path = match.group(1)
            image_path = markdown_path.parent / image_rel_path

            if not image_path.exists():
                logger.warning(f"Image not found, removing link: {image_path}")
                result_lines.append(image_pattern.sub("", line))
                continue

            context_before = "\n".join(lines[max(0, i - 5) : i])
            context_after = "\n".join(lines[i + 1 : min(len(lines), i + 6)])

            logger.info(f"Calling VLM for image: {image_path.name}")
            description = self._call_vlm_for_image_description(image_path, context_before, context_after)

            if description:
                # Replace the image link with the VLM-generated text
                replaced = image_pattern.sub("", line).strip()
                if replaced:
                    result_lines.append(replaced)
                result_lines.append(description)
            else:
                # VLM failed – fall back to removing the link
                result_lines.append(image_pattern.sub("", line))

        new_content = "\n".join(result_lines)
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return new_content

    def clean_markdown_content(self, markdown_path):
        with open(markdown_path, "r", encoding="utf-8") as file:
            content = file.read()
        # Remove any remaining image links (fallback if VLM step was skipped)
        cleaned_content = self.remove_image_links(content)
        # Remove lines that contain only hash symbols and spaces
        # This pattern matches lines with only #, spaces, and optionally newlines
        cleaned_content = re.sub(r'^[ #]+$', '-------------', cleaned_content, flags=re.MULTILINE)
        with open(markdown_path, "w", encoding="utf-8") as file:
            file.write(cleaned_content)
        return cleaned_content

    # Override
    def _to_markdown(
        self, input_path: Path, output_path: Path, conversion_method: str = "MinerU"
    ) -> Path:
        self.file_name = input_path.name
        output_dir = output_path.parent

        if conversion_method == "MinerU":
            md_file_path = convert_pdf_to_md_by_MinerU(input_path, output_dir)

            if md_file_path.exists():
                print(f"Markdown file found: {md_file_path}")
            else:
                raise FileNotFoundError(f"Markdown file not found: {md_file_path}")
            # Set the target to this markdown path
            target = md_file_path
            # Replace image links with VLM-generated descriptions (runs only if OPENAI_API_KEY is set)
            self.replace_images_with_vlm_descriptions(target)
            cleaned_content = self.clean_markdown_content(target)
            json_file_path = md_file_path.with_name(f"{md_file_path.stem}_content_list.json")
            with open(json_file_path, "r", encoding="utf-8") as f_json:
                data = json.load(f_json)
            self.generate_index_helper(data, md=cleaned_content)
            return target

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