# Fraud & Financial Crime Policy Assistant

A RAG pipeline over public fraud/AML documents (FinCEN, FATF, FFIEC BSA/AML manual, CFPB Reg E) that answers with document + page citations, refuses when retrieved context is insufficient, and includes a quantitative evaluation and validation layer.

Everything runs locally on CPU; no API key is required.

**Status:** Phase 1 (ingestion, extraction, cleaning, QA) built. See [PLAN.md](PLAN.md) for the phased plan and [DECISIONS.md](DECISIONS.md) for design choices.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev]"
ruff check .
pytest
```

## Phase 1 commands
```bash
python -m fincrime_rag.ingestion download   # fetch script-eligible PDFs into data/raw, record sha256
python -m fincrime_rag.ingestion verify     # re-check data/raw against data/manifest.csv; fills sha256 for manual files
python -m fincrime_rag.extraction           # PDF -> data/extracted/<doc_id>.jsonl (one record per page)
python -m fincrime_rag.cleaning             # -> data/processed/<doc_id>.pages.jsonl + logs/<doc_id>_removed_lines.csv
python -m fincrime_rag.qa                   # -> reports/extraction_qa.md, data/processed/qa_pages.csv
```
FATF and FFIEC block scripted downloads: save those four PDFs by hand under the filenames in `data/manifest.csv`
(`source=manual`), then run `verify`.

## Eval set (Phase 1.5)
The questions are hand-written in `data/eval/qa_set.jsonl` (format example: `data/eval/EXAMPLE_FORMAT.md`).
```bash
python -m fincrime_rag.evalset validate         # schema check + counts; exit 1 on errors
python -m fincrime_rag.evalset pages <doc_id>   # pdf_page / page_label / printed_page_guess (no page text)
```
Gold pages are PDF page indexes (the `pdf_page` column), not printed numbers.

## OCR (coverage check and fallback)
`pip install -e ".[ocr]"` installs RapidOCR (pip-only, models bundled). `python -m fincrime_rag.extraction`
OCRs every page once (cached under `.cache/ocr`, so the first run takes a while), compares OCR and text-layer word
counts. For pages below `ocr.coverage_threshold` it keeps the text layer and adds OCR-only lines (`extraction_method: merged`), or uses OCR text alone when the page has no usable text layer (`ocr`). Reg E model-form pages are tagged `content_type: model_form` and excluded from the index by default (`indexing.exclude_content_types`).
`python -m fincrime_rag.ocr` warms the cache on its own; `--no-ocr` on extraction skips all of it.
