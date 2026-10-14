"""OCR support for the text-coverage check and the OCR fallback.

Some PDFs carry part of their content only as images (for example FinCEN advisories whose
red-flag bullets are icons next to text that is not in the text layer). To catch this, every page
is rendered, OCR'd once (results are cached on disk, keyed by the file's sha256), and the OCR word
count is compared with the text-layer word count:

    coverage = text-layer words / OCR words        (flagged when below ``coverage_threshold``)

Pages that are flagged get their text replaced by the OCR text (``extraction_method = "ocr"``),
after removing icon noise: very short, low-confidence boxes in the left gutter (a red-flag icon is
read as "13" or similar).

The engine sits behind a tiny protocol so tests can use a fake; the real one is RapidOCR
(``pip install rapidocr-onnxruntime``), which needs no system install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pymupdf

from fincrime_rag.config import Config, OCRConfig, load_config
from fincrime_rag.ingestion import read_manifest


@dataclass(frozen=True)
class OCRBox:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    confidence: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class PageOCR:
    pdf_page: int
    width: int  # rendered pixels
    height: int
    boxes: list[OCRBox]


class OCREngine(Protocol):
    name: str

    def recognize(self, image, high_recall: bool = False) -> list[OCRBox]:
        """OCR one rendered page (a NumPy HxWx3 uint8 array) and return line-level boxes.

        ``high_recall`` uses looser detection settings that find lines the default pass can miss
        (slower); it is used only to produce replacement text for pages the coverage check flagged.
        """


class RapidOCREngine:
    name = "rapidocr-onnxruntime"

    def __init__(self, cfg: OCRConfig | None = None) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "OCR needs rapidocr-onnxruntime: pip install -e '.[ocr]' (or run with --no-ocr)"
            ) from exc
        self._engine = RapidOCR()
        self._recall_kwargs = (
            {}
            if cfg is None
            else {
                "box_thresh": cfg.recover_box_thresh,
                "text_score": cfg.recover_text_score,
                "unclip_ratio": cfg.recover_unclip_ratio,
            }
        )

    def recognize(self, image, high_recall: bool = False) -> list[OCRBox]:
        result, _ = self._engine(image, **(self._recall_kwargs if high_recall else {}))
        boxes = []
        for corners, text, confidence in result or []:
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            boxes.append(OCRBox(min(xs), min(ys), max(xs), max(ys), text, float(confidence)))
        return boxes


def make_engine(cfg: OCRConfig) -> OCREngine:
    if cfg.engine == "rapidocr":
        return RapidOCREngine(cfg)
    raise ValueError(f"unknown OCR engine: {cfg.engine}")


# --------------------------------------------------------------------------- rendering + cache
def render_page(page: pymupdf.Page, dpi: int):
    import numpy as np

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 16):
            h.update(block)
    return h.hexdigest()


def cache_path(cache_dir: Path, sha256: str, dpi: int, engine_name: str, profile: str = "scan") -> Path:
    suffix = "" if profile == "scan" else f"_{profile}"
    return cache_dir / "ocr" / f"{sha256[:16]}_{engine_name}_{dpi}dpi{suffix}.jsonl"


def _box_to_json(b: OCRBox) -> list:
    return [round(b.x0, 1), round(b.y0, 1), round(b.x1, 1), round(b.y1, 1), b.text, round(b.confidence, 3)]


def _read_cache(path: Path) -> dict[int, PageOCR]:
    pages: dict[int, PageOCR] = {}
    if not path.exists():
        return pages
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pages[rec["pdf_page"]] = PageOCR(
                rec["pdf_page"], rec["width"], rec["height"], [OCRBox(*b) for b in rec["boxes"]]
            )
    return pages


def ocr_document(
    pdf_path: Path,
    engine: OCREngine,
    cfg: OCRConfig,
    cache_dir: Path,
    verbose: bool = False,
    profile: str = "scan",
    only_pages: list[int] | None = None,
) -> dict[int, PageOCR]:
    """OCR pages of a PDF, resuming from and appending to the on-disk cache.

    ``profile="scan"`` (default) OCRs every page with the engine's default settings, for the
    coverage check. ``profile="recover"`` uses high-recall settings and is normally limited to
    ``only_pages`` (the pages that the coverage check flagged). Each profile has its own cache.
    """
    high_recall = profile == "recover"
    path = cache_path(cache_dir, file_sha256(pdf_path), cfg.dpi, engine.name, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = _read_cache(path)
    with pymupdf.open(pdf_path) as doc, open(path, "a", encoding="utf-8", newline="\n") as out:
        wanted = only_pages if only_pages is not None else list(range(1, doc.page_count + 1))
        todo = [i for i in wanted if i not in pages]
        started = time.time()
        for done, number in enumerate(todo, start=1):
            image = render_page(doc[number - 1], cfg.dpi)
            boxes = engine.recognize(image, high_recall=high_recall)
            pages[number] = PageOCR(number, image.shape[1], image.shape[0], boxes)
            out.write(
                json.dumps(
                    {
                        "pdf_page": number,
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "boxes": [_box_to_json(b) for b in boxes],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            if verbose and (done % 10 == 0 or done == len(todo)):
                elapsed = time.time() - started
                print(f"    {pdf_path.name}: {done}/{len(todo)} pages OCR'd ({elapsed:.0f}s)", flush=True)
    return pages if only_pages is None else {n: pages[n] for n in wanted if n in pages}


# --------------------------------------------------------------------------- text from boxes
def filter_icon_noise(page: PageOCR, cfg: OCRConfig) -> tuple[list[OCRBox], list[OCRBox]]:
    """Split boxes into (kept, dropped noise).

    Dropped: boxes below ``min_confidence`` (garbage readings), and icon noise, i.e. short,
    low-confidence boxes in the left gutter (a red-flag icon read as "13").
    """
    kept, dropped = [], []
    for box in page.boxes:
        in_gutter = page.width > 0 and box.x0 / page.width < cfg.icon_gutter_fraction
        is_noise = box.confidence < cfg.min_confidence or (
            in_gutter
            and len(box.text.strip()) <= cfg.icon_max_chars
            and box.confidence <= cfg.icon_max_confidence
        )
        (dropped if is_noise else kept).append(box)
    return kept, dropped


def boxes_to_text(boxes: list[OCRBox], cfg: OCRConfig) -> str:
    """Reading-order text: boxes on the same row are joined, big vertical gaps become blank lines."""
    if not boxes:
        return ""
    tolerance = 0.5 * statistics.median(b.height for b in boxes)
    rows: list[list[OCRBox]] = []
    for box in sorted(boxes, key=lambda b: b.y_center):
        if rows and abs(box.y_center - statistics.mean(b.y_center for b in rows[-1])) <= tolerance:
            rows[-1].append(box)
        else:
            rows.append([box])
    tops = [min(b.y0 for b in row) for row in rows]
    pitches = [b - a for a, b in zip(tops, tops[1:], strict=False)]
    median_pitch = statistics.median(pitches) if pitches else 0.0
    lines: list[str] = []
    for index, row in enumerate(rows):
        gap = tops[index] - tops[index - 1] if index else 0.0
        if median_pitch and gap > cfg.paragraph_gap_factor * median_pitch:
            lines.append("")
        lines.append(" ".join(b.text.strip() for b in sorted(row, key=lambda b: b.x0)))
    return "\n".join(lines)


@dataclass
class PageAssessment:
    text_layer_words: int
    ocr_words: int
    coverage: float | None  # None when the page has too few OCR words to judge
    flagged: bool
    ocr_text: str  # icon noise removed
    dropped_icon_noise: list[str]


def ocr_text(page_ocr: PageOCR, cfg: OCRConfig) -> str:
    """Reading-order text of an OCR'd page, icon noise removed."""
    kept, _ = filter_icon_noise(page_ocr, cfg)
    return boxes_to_text(kept, cfg)


def assess_page(text_layer: str, page_ocr: PageOCR, cfg: OCRConfig) -> PageAssessment:
    kept, dropped = filter_icon_noise(page_ocr, cfg)
    text = boxes_to_text(kept, cfg)
    ocr_words = len(text.split())
    layer_words = len(text_layer.split())
    coverage = layer_words / ocr_words if ocr_words >= max(cfg.coverage_min_ocr_words, 1) else None
    return PageAssessment(
        text_layer_words=layer_words,
        ocr_words=ocr_words,
        coverage=None if coverage is None else round(coverage, 3),
        flagged=coverage is not None and coverage < cfg.coverage_threshold,
        ocr_text=text,
        dropped_icon_noise=[b.text.strip() for b in dropped],
    )


# --------------------------------------------------------------------------- merge with the text layer
@dataclass(frozen=True)
class LayerLine:
    """One line of the PDF text layer, in PDF points."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def layer_lines(page: pymupdf.Page) -> list[LayerLine]:
    """Text-layer lines with their boxes (empty / whitespace-only lines, e.g. icon placeholders, skipped)."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            raw = "".join(span["text"] for span in line["spans"])
            if raw.strip():
                x0, y0, x1, y1 = line["bbox"]
                # one leading space is kept: it marks a line that continues the previous row's text
                # ("...another.<sup>14</sup> Money mule..."), so it is not glued to a superscript
                lines.append(LayerLine(x0, y0, x1, y1, (" " if raw[:1].isspace() else "") + raw.strip()))
    return lines


def to_points(box: OCRBox, dpi: int) -> OCRBox:
    """OCR pixels -> PDF points (72 per inch)."""
    k = 72 / dpi
    return OCRBox(box.x0 * k, box.y0 * k, box.x1 * k, box.y1 * k, box.text, box.confidence)


def covered_share(box_pt: OCRBox, words: list[tuple]) -> float:
    """Share of the box's area lying on text-layer word boxes (``page.get_text("words")`` tuples)."""
    area = (box_pt.x1 - box_pt.x0) * (box_pt.y1 - box_pt.y0)
    if area <= 0:
        return 0.0
    inter = 0.0
    for w in words:
        ix = min(box_pt.x1, w[2]) - max(box_pt.x0, w[0])
        iy = min(box_pt.y1, w[3]) - max(box_pt.y0, w[1])
        if ix > 0 and iy > 0:
            inter += ix * iy
    return inter / area


def split_by_coverage(
    page_ocr: PageOCR, words: list[tuple], cfg: OCRConfig
) -> tuple[list[OCRBox], list[OCRBox]]:
    """Usable OCR boxes (noise filters applied, in points) -> (covered by the text layer, OCR-only)."""
    kept, _ = filter_icon_noise(page_ocr, cfg)
    covered, only = [], []
    for box in kept:
        pt = to_points(box, cfg.dpi)
        (covered if covered_share(pt, words) >= cfg.merge_covered_fraction else only).append(pt)
    return covered, only


@dataclass
class _Row:
    y0: float
    y1: float
    x0: float
    text: str
    source: str  # "layer" | "ocr"


def _rows(items: list[tuple[float, float, float, float, str]], source: str, join_gap: float) -> list[_Row]:
    """Group (x0, y0, x1, y1, text) items on the same visual row, left to right."""
    rows: list[list[tuple[float, float, float, float, str]]] = []
    for item in sorted(items, key=lambda i: (i[1] + i[3]) / 2):
        if rows:
            ref = rows[-1][0]
            overlap = min(ref[3], item[3]) - max(ref[1], item[1])
            if overlap > 0.5 * min(ref[3] - ref[1], item[3] - item[1]):
                rows[-1].append(item)
                continue
        rows.append([item])
    out = []
    for row in rows:
        row.sort(key=lambda i: i[0])
        text = row[0][4].lstrip()
        for prev, cur in zip(row, row[1:], strict=False):
            tight = cur[0] - prev[2] < join_gap and not cur[4].startswith(" ")
            text += ("" if tight else " ") + cur[4].lstrip()
        out.append(_Row(min(i[1] for i in row), max(i[3] for i in row), row[0][0], text, source))
    return out


def merge_layer_and_ocr(lines: list[LayerLine], ocr_only: list[OCRBox], cfg: OCRConfig) -> str:
    """Text layer as the base, plus OCR-only boxes, in vertical reading order.

    Rows are ordered by vertical position, which also moves footnotes to the bottom. A paragraph
    break is inserted where the row pitch exceeds ``paragraph_gap_factor`` x the median pitch, and
    always where the source switches between text layer and OCR, so a run of OCR-only lines (for
    example a missing red-flag bullet) is its own paragraph. Assumes single-column pages.
    """
    rows = _rows([(ln.x0, ln.y0, ln.x1, ln.y1, ln.text) for ln in lines], "layer", join_gap=1.5)
    rows += _rows([(b.x0, b.y0, b.x1, b.y1, b.text.strip()) for b in ocr_only], "ocr", join_gap=4.0)
    rows.sort(key=lambda r: ((r.y0 + r.y1) / 2, r.x0))
    pitches = [b.y0 - a.y0 for a, b in zip(rows, rows[1:], strict=False) if b.y0 - a.y0 > 0.5]
    median_pitch = statistics.median(pitches) if pitches else 0.0
    out: list[str] = []
    for index, row in enumerate(rows):
        if index:
            prev = rows[index - 1]
            pitch = row.y0 - prev.y0
            too_far = bool(median_pitch) and pitch > cfg.paragraph_gap_factor * median_pitch
            if row.source != prev.source or too_far:
                out.append("")
        out.append(row.text)
    return "\n".join(out)


# --------------------------------------------------------------------------- CLI: warm the cache
def warm_cache(cfg: Config, docs: list[str] | None = None) -> int:
    engine = make_engine(cfg.ocr)
    for row in read_manifest(cfg.paths.manifest):
        if docs and row.doc_id not in docs:
            continue
        src = cfg.paths.raw_dir / row.filename
        if not src.exists():
            print(f"[skip] {row.doc_id}: not in data/raw")
            continue
        pages = ocr_document(src, engine, cfg.ocr, cfg.paths.cache_dir, verbose=True)
        print(f"[ok]   {row.doc_id}: {len(pages)} pages OCR'd/cached", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.ocr", description="OCR documents into the cache.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--docs", nargs="*", help="doc_ids to OCR (default: all)")
    args = parser.parse_args(argv)
    return warm_cache(load_config(args.config), args.docs)


if __name__ == "__main__":
    sys.exit(main())
