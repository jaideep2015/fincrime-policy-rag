"""Shared fixtures: a tiny generated PDF with known headers, footers, labels, and a blank page."""

from pathlib import Path

import pymupdf
import pytest

from fincrime_rag.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]

# Body text per page index (0-based). Page 3 (index 3) is intentionally blank.
BODY = {
    0: [
        "Money mule accounts move funds for criminals.",
        "Banks should report contin-",
        "ued activity promptly.",
    ],
    1: [
        "Strong anti-money laundering controls help.",
        "Analysts review the anti-",
        "money laundering alerts daily.",
    ],
    2: ["Structuring keeps deposits under thresholds."],
    4: ["Elder financial exploitation is rising."],
    5: ["Report within 30 calendar days."],
    6: ["Retain records for five years."],
    7: ["Closing page with a final note."],
}
N_PAGES = 8


def make_pdf(path: Path, n_pages: int = N_PAGES, header: str = "ACME Advisory") -> Path:
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 40), header, fontsize=9)
        y = 120
        for line in BODY.get(i, []) if n_pages == N_PAGES else [f"Body of page {i + 1}."]:
            page.insert_text((72, y), line, fontsize=11)
            y += 18
        page.insert_text((72, 800), f"Page {i + 1}", fontsize=9)
    doc.set_page_labels([{"startpage": 0, "prefix": "A-", "style": "D", "firstpagenum": 1}])
    doc.save(path)
    doc.close()
    return path


def blank_out_page(path: Path, index: int) -> None:
    """Rewrite the PDF with the given page's content removed (keeps header/footer off that page)."""
    src = pymupdf.open(path)
    page = src[index]
    page.clean_contents()
    for xref in page.get_contents():
        src.update_stream(xref, b" ")
    src.save(path.with_suffix(".tmp"))
    src.close()
    path.with_suffix(".tmp").replace(path)


@pytest.fixture()
def fixture_pdf(tmp_path: Path) -> Path:
    pdf = make_pdf(tmp_path / "fixture.pdf")
    blank_out_page(pdf, 3)
    return pdf


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    """Repo config with every data path redirected into tmp_path."""
    base = load_config(ROOT / "config.yaml")
    paths = base.paths.model_copy(
        update={
            "manifest": tmp_path / "manifest.csv",
            "raw_dir": tmp_path / "raw",
            "extracted_dir": tmp_path / "extracted",
            "processed_dir": tmp_path / "processed",
            "reports_dir": tmp_path / "reports",
        }
    )
    (tmp_path / "raw").mkdir()
    return base.model_copy(update={"paths": paths})


def extract_fixture_pages(pdf: Path) -> list[dict]:
    """Text-only page records for a generated PDF (no OCR)."""
    from fincrime_rag.extraction import extract_pdf

    return extract_pdf(pdf, "fx")
