"""Shared fixtures: a tiny generated PDF with known headers, footers, labels, and a blank page."""

import hashlib
import re
from pathlib import Path

import numpy as np
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


# ---- index / retrieval fakes (no model download) ----
DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words, L2-normalised. Counts calls so cache behaviour can be checked."""

    name = "fake"

    def __init__(self):
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        self.calls += len(texts)
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def chunk(cid, doc, pages, text, section=None):
    return {
        "chunk_id": cid,
        "doc_id": doc,
        "pdf_pages": pages,
        "printed_page_guess": str(pages[0] + 1),
        "section": section,
        "section_hint": None,
        "content_type": "body",
        "n_tokens": len(text.split()),
        "text_hash": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }


CHUNKS = [
    chunk("a:p1:c0", "a", [1], "money mule accounts receive and forward illicit funds quickly"),
    chunk("a:p2:c1", "a", [2, 3], "elder financial exploitation red flags include sudden account changes"),
    chunk("b:p1:c0", "b", [1], "section 1005.11 error resolution procedures for consumers", "1005.11"),
    chunk("b:p5:c1", "b", [5], "suspicious activity reports must be filed within thirty calendar days"),
]


@pytest.fixture()
def index_cfg(cfg, tmp_path):
    paths = cfg.paths.model_copy(update={"index_dir": tmp_path / "chroma", "cache_dir": tmp_path / "cache"})
    return cfg.model_copy(update={"paths": paths})
