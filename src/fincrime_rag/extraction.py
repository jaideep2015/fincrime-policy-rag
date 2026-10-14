"""Stage 2: PDF -> per-page raw text, with a text-coverage check and OCR merge/fallback.

One JSONL file per document (``data/extracted/<doc_id>.jsonl``), one record per PDF page::

    {"doc_id", "pdf_page" (1-based), "page_label" (printed label or null), "text",
     "extraction_method" ("text" | "merged" | "ocr"), "text_layer_words", "ocr_words",
     "coverage", "box_coverage"}

``pdf_page`` is always present and is the key used for citations and retrieval metrics.
``page_label`` is the label the PDF itself defines (e.g. "iii", "4-12"); many PDFs define none.

Text-coverage check: when OCR results are supplied, every page's text-layer word count is compared
with the OCR word count. A page below ``ocr.coverage_threshold`` is flagged and handled by what
its text layer is worth:

* ``merged``: the text layer is usable (at least ``merge_min_layer_words`` words, and the text layer
  already covers at least ``merge_min_box_coverage`` of the OCR boxes). The text layer stays as the
  base and only OCR boxes it does not already cover are added, in vertical order.
* ``ocr``: no usable text layer (image-only covers, scanned forms). OCR text alone is used.

Unflagged pages keep the text layer untouched (``text``). Cleaning happens later, in ``cleaning.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pymupdf

from fincrime_rag.config import Config, OCRConfig, load_config
from fincrime_rag.ingestion import read_manifest
from fincrime_rag.ocr import (
    PageOCR,
    assess_page,
    layer_lines,
    make_engine,
    merge_layer_and_ocr,
    ocr_document,
    ocr_text,
    split_by_coverage,
)

Page = dict  # {"doc_id", "pdf_page", "page_label", "text", ...}


def extract_pdf(
    path: Path,
    doc_id: str,
    ocr_pages: dict[int, PageOCR] | None = None,
    ocr_cfg: OCRConfig | None = None,
    recover: Callable[[list[int]], dict[int, PageOCR]] | None = None,
) -> list[Page]:
    """Text layer per page; pages flagged by the coverage check are merged with, or replaced by, OCR.

    ``ocr_pages`` (default-settings OCR of every page) drives the coverage check. For flagged pages
    the OCR used for merging/replacing comes from ``recover(flagged_page_numbers)`` (a higher-recall
    pass) when given, else from the same OCR result.
    """
    pages: list[Page] = []
    flagged: list[int] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            record: Page = {
                "doc_id": doc_id,
                "pdf_page": index,
                "page_label": page.get_label() or None,
                "text": text,
                "extraction_method": "text",
                "text_layer_words": len(text.split()),
                "ocr_words": None,
                "coverage": None,
                "box_coverage": None,
            }
            if ocr_pages is not None and ocr_cfg is not None and index in ocr_pages:
                assessment = assess_page(text, ocr_pages[index], ocr_cfg)
                record["ocr_words"] = assessment.ocr_words
                record["coverage"] = assessment.coverage
                if assessment.flagged:
                    flagged.append(index)
            pages.append(record)

        if flagged and ocr_cfg is not None:
            source = recover(flagged) if recover is not None else {}
            for number in flagged:
                page_ocr = source.get(number) or ocr_pages[number]
                _merge_or_replace(pages[number - 1], doc[number - 1], page_ocr, ocr_cfg)
    return pages


def _merge_or_replace(record: Page, page: pymupdf.Page, page_ocr: PageOCR, cfg: OCRConfig) -> None:
    """Set ``record['text']`` / ``extraction_method`` for a page the coverage check flagged."""
    words = page.get_text("words")
    covered, ocr_only = split_by_coverage(page_ocr, words, cfg)
    usable_boxes = len(covered) + len(ocr_only)
    box_coverage = len(covered) / usable_boxes if usable_boxes else 0.0
    record["box_coverage"] = round(box_coverage, 3)
    if record["text_layer_words"] >= cfg.merge_min_layer_words and box_coverage >= cfg.merge_min_box_coverage:
        record["text"] = merge_layer_and_ocr(layer_lines(page), ocr_only, cfg)
        record["extraction_method"] = "merged"
    else:
        record["text"] = ocr_text(page_ocr, cfg)
        record["extraction_method"] = "ocr"


def write_pages(path: Path, pages: list[Page]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")


def read_pages(path: Path) -> list[Page]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_all(cfg: Config, only: str | None = None, use_ocr: bool | None = None) -> int:
    rows = read_manifest(cfg.paths.manifest)
    if only:
        rows = [r for r in rows if r.doc_id == only]
        if not rows:
            print(f"unknown doc_id: {only}")
            return 1
    use_ocr = cfg.ocr.enabled if use_ocr is None else use_ocr
    engine = make_engine(cfg.ocr) if use_ocr else None
    for row in rows:
        src = cfg.paths.raw_dir / row.filename
        if not src.exists():
            print(f"[skip] {row.doc_id}: {src.name} not in data/raw")
            continue
        ocr_pages = ocr_document(src, engine, cfg.ocr, cfg.paths.cache_dir) if engine else None

        def recover(numbers: list[int], src=src) -> dict[int, PageOCR]:
            return ocr_document(
                src, engine, cfg.ocr, cfg.paths.cache_dir, verbose=True, profile="recover", only_pages=numbers
            )

        pages = extract_pdf(src, row.doc_id, ocr_pages, cfg.ocr, recover if engine else None)
        write_pages(cfg.paths.extracted_dir / f"{row.doc_id}.jsonl", pages)
        labelled = sum(1 for p in pages if p["page_label"])
        merged = sum(1 for p in pages if p["extraction_method"] == "merged")
        ocr_n = sum(1 for p in pages if p["extraction_method"] == "ocr")
        print(
            f"[ok]   {row.doc_id}: {len(pages)} pages, {labelled} with PDF page labels, "
            f"{merged} merged with OCR, {ocr_n} OCR only"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.extraction", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--doc", help="extract a single doc_id")
    parser.add_argument("--no-ocr", action="store_true", help="skip coverage check and OCR")
    args = parser.parse_args(argv)
    return extract_all(load_config(args.config), only=args.doc, use_ocr=False if args.no_ocr else None)


if __name__ == "__main__":
    sys.exit(main())
