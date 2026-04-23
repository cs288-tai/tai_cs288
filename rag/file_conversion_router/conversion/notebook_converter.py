from pathlib import Path
import re
import nbformat
from nbconvert import MarkdownExporter
from file_conversion_router.conversion.base_converter import BaseConverter
from file_conversion_router.utils.title_handle import normalize_title_for_match
from nbformat.validator import normalize
import uuid

# Same heading pattern used by apply_structure_for_one_title, so extraction
# and matching agree on what counts as a heading.
_HEADING_PATTERN = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>\S.*?)$")


class NotebookConverter(BaseConverter):
    def __init__(self, course_name, course_code, file_uuid: str = None):
        super().__init__(course_name, course_code, file_uuid)
        self.index_helper = None

    def extract_all_markdown_titles(self, content):
        """
        Extract titles from markdown content that start with #
        Returns a list of all found titles
        """
        if not content.strip():
            return []
        titles = []
        # Process line by line to match BaseConverter logic
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                if title == "":
                    continue
                # Remove * characters from title
                title = title.replace('*', '')
                if title.strip():
                    titles.append(title.strip())
        return titles


    def generate_index_helper_from_markdown(self, final_md: str, notebook_content) -> None:
        """Build index_helper from the post-processed markdown.

        Titles are drawn from the final markdown (what the downstream matcher
        sees), then attributed back to a notebook cell index by normalized
        lookup against each cell's own titles. Titles without a cell match
        inherit the most recent matched cell index.
        """
        # Build normalized-title -> cell_index map from notebook cells
        cell_lookup = {}
        for i, cell in enumerate(notebook_content.cells):
            for title in self.extract_all_markdown_titles(cell.source):
                key = normalize_title_for_match(title)
                if key and key not in cell_lookup:
                    cell_lookup[key] = i + 1

        self.index_helper = []
        last_cell_index = 1
        for line in final_md.split("\n"):
            match = _HEADING_PATTERN.match(line.strip())
            if not match:
                continue
            title = match.group("title").strip().replace("*", "")
            if not title:
                continue
            key = normalize_title_for_match(title)
            cell_index = cell_lookup.get(key, last_cell_index)
            last_cell_index = cell_index
            self.index_helper.append({title: cell_index})

    # Override
    def _to_markdown(self, input_path: Path, output_path: Path) -> Path:
        output_path = output_path.with_suffix(".md")

        with open(input_path, "r") as input_file, open(output_path, "w") as output_file:
            # Read and normalize notebook in memory (don't modify source file)
            content = nbformat.read(input_file, as_version=4)
            content = self.pre_process_notebook(content)
            normalize(content)
            for cell in getattr(content, "cells", []):
                cell.setdefault("id", uuid.uuid4().hex)
            markdown_converter = MarkdownExporter()
            (markdown_content, resources) = markdown_converter.from_notebook_node(
                content
            )
            final_md = self._post_process_markdown(markdown_content)
            self.generate_index_helper_from_markdown(final_md, content)
            output_file.write(final_md)
        return output_path

    def _post_process_markdown(self, markdown_content: str) -> str:
        lines = markdown_content.split("\n")[
            1:
        ]  # first line is the title of the course section

        processed_lines = []
        for i, line in enumerate(lines):
            if i == 1:  # convert lecture title to h1
                processed_lines.append(f"# {line.lstrip('#').strip()}")
            elif line.startswith("#"):  # convert all other heading down one level
                processed_lines.append(f"#{line.strip()}")
            else:
                processed_lines.append(line.strip())
        # Fix title levels in markdown content
        md_content = self.fix_markdown_title_levels("\n".join(processed_lines))
        return md_content

    def pre_process_notebook(self, content):
        """
        Pre-process the notebook in memory to clean up cell IDs.
        Works on notebook object without modifying the source file.
        Returns the cleaned notebook object.
        """
        for cell in content.cells:
            if 'id' in cell:
                del cell['id']
        return content

    def fix_markdown_title_levels(self, md_content):
        """
        Fix title levels in markdown content to ensure they are sequential.
        Takes markdown content as string and returns the fixed content.
        """
        lines = md_content.split('\n')
        title_info = []

        # Extract all title information with line numbers
        for i, line in enumerate(lines):
            # Match markdown headers (# ## ### etc.)
            header_match = re.match(r'^(#+)\s*(.*)', line.strip())
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                title_info.append({
                    'line_index': i,
                    'level_of_title': level,
                    'title': title,
                    'original_line': line
                })

        # Apply your fixing logic
        last_level = 0
        for i in range(len(title_info)):
            current_level = title_info[i]["level_of_title"]

            if current_level > last_level + 1:
                diff = current_level - (last_level + 1)
                j = i
                while j < len(title_info) and title_info[j]["level_of_title"] >= current_level:
                    title_info[j]["level_of_title"] -= diff
                    j += 1

            last_level = title_info[i]["level_of_title"]

        # Reconstruct the markdown content with fixed levels
        result_lines = lines.copy()
        for info in title_info:
            new_header = '#' * info['level_of_title'] + ' ' + info['title']
            result_lines[info['line_index']] = new_header
        return '\n'.join(result_lines)

