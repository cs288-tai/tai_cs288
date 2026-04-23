# Copyright (c) Opendatalab. All rights reserved.
import os
from pathlib import Path

from loguru import logger

from mineru.cli.common import do_parse as mineru_do_parse, prepare_env, read_fn


def parse_doc(
        pdf_path: Path,
        output_folder: Path,
        lang="en",
        backend="pipeline",
        method="auto",
        server_url=None,
        start_page_id=0,
        end_page_id=None):
    """
        Parameter description:
        pdf_path: Path to the PDF file to be parsed.
        output_folder: Output folder where the markdown file will be saved.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka'].
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-transformers: More general.
            vlm-sglang-engine: Faster(engine).
            vlm-sglang-client: Faster(client).
            without method specified, pipeline will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)

        Returns:
        Path: Path to the output markdown file
    """

    # Ensure output folder exists
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Get the PDF file name (used as the task stem by MinerU)
    file_name = pdf_path.name

    # Read the PDF file
    pdf_bytes = read_fn(pdf_path)

    # Use MinerU's built-in do_parse
    mineru_do_parse(
        output_dir=str(output_folder),
        pdf_file_names=[file_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[lang],
        backend=backend,
        parse_method=method,
        server_url=server_url,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )

    # MinerU outputs to: output_folder/{file_name}/{method}/{file_name}.md
    md_file_path = output_folder / file_name / method / f"{file_name}.md"

    if md_file_path.exists():
        logger.info(f"Markdown saved to: {md_file_path}")
    else:
        logger.warning(f"Expected markdown file not found at: {md_file_path}")

    return md_file_path


if __name__ == '__main__':
    from file_conversion_router.config import get_test_data_path, get_test_output_path

    doc_path = get_test_data_path('testing/pdfs/disc01.pdf')
    output_dir = get_test_output_path('disc01')
    parse_doc(doc_path, output_dir, backend="pipeline")
