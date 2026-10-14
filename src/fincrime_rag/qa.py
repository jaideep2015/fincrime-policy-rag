"""Extraction QA: per-page and per-document statistics, flags, and a Markdown report.

Per page (computed on the extracted text, i.e. OCR text where the OCR fallback replaced the text
layer): characters, words, empty-page flag, ``bad_char_ratio`` (U+FFFD, control and private-use
characters) and ``odd_token_ratio`` (vowel-less alphabetic tokens of 5+ letters or tokens over 25
letters, a symptom of broken font encoding; tokens with digits, slashes, '@' or '_' are ignored so
URLs and codes do not count, and all-caps acronyms such as CMLN, NMLRA or TCOs are ignored when
``qa.odd_token_ignore_acronyms`` is set). Coverage fields (``extraction_method``,
``text_layer_words``, ``ocr_words``, ``coverage``) come from the extraction step.

Per document, the numbers are aggregated and a document is flagged when it crosses a threshold
from ``config.yaml`` (``qa:``). Flags are prompts for a human look, not verdicts. The report also
counts the running-head / page-number lines still present after cleaning ("residual edge lines").

Outputs: ``data/processed/qa_pages.csv`` (per page) and ``reports/extraction_qa.md``.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

from fincrime_rag.cleaning import count_residual_edge_lines, normalize_line
from fincrime_rag.config import Config, QAConfig, load_config
from fincrime_rag.extraction import Page, read_pages
from fincrime_rag.ingestion import ManifestRow, read_manifest

_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")
_SKIP_TOKEN_RE = re.compile(r"[\d/@_\\]|https?|www\.")
_ACRONYM_RE = re.compile(r"^[A-Z]{2,}s?$")  # CMLN, NMLRA, TCOs
_MAX_TOKEN_LEN = 25
_MIN_VOWELLESS_LEN = 5


def bad_char_count(text: str) -> int:
    bad = 0
    for ch in text:
        if ch == "�":
            bad += 1
        elif ch not in "\n\t\r" and unicodedata.category(ch) in {"Cc", "Co"}:
            bad += 1
    return bad


def token_counts(text: str, ignore_acronyms: bool = False) -> tuple[int, int]:
    """(alphabetic tokens considered, odd tokens)."""
    total = odd = 0
    for raw in text.split():
        if _SKIP_TOKEN_RE.search(raw):
            continue
        token = re.sub(r"[^A-Za-z]", "", raw)
        if not token:
            continue
        if ignore_acronyms and _ACRONYM_RE.match(token):
            continue
        total += 1
        if len(token) > _MAX_TOKEN_LEN or (
            len(token) >= _MIN_VOWELLESS_LEN and not _VOWEL_RE.search(token)
        ):
            odd += 1
    return total, odd


def page_stats(page: Page, cfg: QAConfig) -> dict:
    text = page["text"]
    chars = len(text.strip())
    non_ws = sum(1 for c in text if not c.isspace())
    bad = bad_char_count(text)
    tokens, odd = token_counts(text, cfg.odd_token_ignore_acronyms)
    return {
        "doc_id": page["doc_id"],
        "pdf_page": page["pdf_page"],
        "page_label": page["page_label"] or "",
        "extraction_method": page.get("extraction_method", "text"),
        "text_layer_words": page.get("text_layer_words", ""),
        "ocr_words": "" if page.get("ocr_words") is None else page["ocr_words"],
        "coverage": "" if page.get("coverage") is None else page["coverage"],
        "box_coverage": "" if page.get("box_coverage") is None else page["box_coverage"],
        "chars": chars,
        "words": len(text.split()),
        "is_empty": chars <= cfg.empty_page_max_chars,
        "bad_chars": bad,
        "bad_char_ratio": round(bad / non_ws, 5) if non_ws else 0.0,
        "tokens": tokens,
        "odd_tokens": odd,
        "odd_token_ratio": round(odd / tokens, 5) if tokens else 0.0,
    }


def _norm(text: str) -> str:
    """Comparison key: NFKC, lower-case, letters and digits only (so hyphen repair, ligatures, quote
    styles, bullets and whitespace do not count as differences)."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).lower())


def text_layer_lines_lost(
    pdf_path: Path, cleaned: list[Page], removed: list[dict], min_words: int = 5
) -> dict[str, list[int]]:
    """Text-layer lines of ``min_words``+ words that are absent from the final (cleaned) page text.

    Lines that the cleaning step deliberately removed (running heads, page numbers; see the removed-
    lines log) do not count as lost. Returns {"all": pdf pages with a lost line (one entry per
    line), "merged": the same restricted to pages whose extraction_method is "merged"}.
    """
    removed_by_page: dict[int, str] = {}
    for r in removed:
        removed_by_page[int(r["pdf_page"])] = removed_by_page.get(int(r["pdf_page"]), "") + _norm(r["line"])
    by_page = {p["pdf_page"]: p for p in cleaned}
    lost: dict[str, list[int]] = {"all": [], "merged": []}
    with pymupdf.open(pdf_path) as doc:
        for number, page in enumerate(doc, start=1):
            final = by_page.get(number)
            if final is None:
                continue
            kept = _norm(final["text"])
            dropped = removed_by_page.get(number, "")
            for line in page.get_text("text").splitlines():
                if len(line.split()) < min_words:
                    continue
                key = _norm(line)
                if key and key not in kept and key not in dropped:
                    lost["all"].append(number)
                    if final["extraction_method"] == "merged":
                        lost["merged"].append(number)
    return lost


def doc_stats(
    row: ManifestRow,
    stats: list[dict],
    cleaned: list[Page],
    removed: list[dict],
    cfg: QAConfig,
    residual: dict[str, int] | None = None,
    lines_lost: dict[str, list[int]] | None = None,
    bad_ratio_cleaned: float | None = None,
) -> dict:
    n = len(stats)
    non_ws_chars = sum(s["chars"] for s in stats)
    tokens = sum(s["tokens"] for s in stats)
    empty_pages = [s["pdf_page"] for s in stats if s["is_empty"]]
    ocr_pages = [s["pdf_page"] for s in stats if s["extraction_method"] == "ocr"]
    merged_pages = [s["pdf_page"] for s in stats if s["extraction_method"] == "merged"]
    coverages = [s["coverage"] for s in stats if s["coverage"] != ""]
    bad_ratio = sum(s["bad_chars"] for s in stats) / non_ws_chars if non_ws_chars else 0.0
    odd_ratio = sum(s["odd_tokens"] for s in stats) / tokens if tokens else 0.0
    median_chars = statistics.median(s["chars"] for s in stats) if stats else 0
    kinds = Counter(
        "page_number"
        if r["reason"] == "page_number"
        else "head"
        if r["reason"] in {"printed_page_head", "running_head"}
        else "repeated"
        for r in removed
    )
    with_label = sum(1 for p in cleaned if p["page_label"])
    with_guess = sum(1 for p in cleaned if not p["page_label"] and p.get("printed_page_guess"))
    with_section = sum(1 for p in cleaned if p.get("section_hint"))
    nulled = sum(1 for p in cleaned if p.get("printed_page_guess_rejected") is not None)
    tagged = sum(1 for p in cleaned if p.get("content_type", "body") != "body")

    flags = []
    if len(empty_pages) >= cfg.flag_empty_page_min_count and len(empty_pages) / n > cfg.flag_empty_page_share:
        flags.append("many empty pages")
    if bad_ratio > cfg.flag_bad_char_ratio:
        flags.append("bad characters")
    if odd_ratio > cfg.flag_odd_token_ratio:
        flags.append("odd tokens")
    if median_chars < cfg.flag_min_chars_per_page:
        flags.append("low text density")
    if lines_lost and lines_lost["merged"]:
        flags.append("text-layer lines lost on merged pages")
    return {
        "doc_id": row.doc_id,
        "pages": n,
        "empty_pages": empty_pages,
        "ocr_pages": ocr_pages,
        "merged_pages": merged_pages,
        "lines_lost": None if lines_lost is None else {k: len(v) for k, v in lines_lost.items()},
        "bad_char_ratio_cleaned": bad_ratio_cleaned,
        "guesses_nulled": nulled,
        "pages_tagged": tagged,
        "min_coverage": min(coverages) if coverages else None,
        "median_chars": int(median_chars),
        "bad_char_ratio": bad_ratio,
        "odd_token_ratio": odd_ratio,
        "removed_page_numbers": kinds["page_number"],
        "removed_heads": kinds["head"],
        "removed_repeated": kinds["repeated"],
        "pages_with_pdf_label": with_label,
        "pages_with_printed_guess": with_guess,
        "pages_with_section_hint": with_section,
        "residual": residual,
        "flags": flags,
    }


def top_removed_patterns(removed: list[dict], limit: int = 3) -> list[tuple[str, int]]:
    counts = Counter(
        normalize_line(r["line"]) for r in removed if r["reason"].startswith("repeated_on_")
    )
    return counts.most_common(limit)


def _read_removed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _short_list(values: list[int], limit: int = 6) -> str:
    text = ", ".join(map(str, values[:limit]))
    return text + (" ..." if len(values) > limit else "")


def build_report(docs: list[dict], patterns: dict[str, list[tuple[str, int]]]) -> str:
    lines = [
        "# Extraction QA report",
        "",
        "Generated by `python -m fincrime_rag.qa`. A flag is a prompt for a human look, not a verdict; "
        "thresholds live in `config.yaml` (`qa:`, `ocr:`).",
        "",
        "- **Merged pages**: the text layer covered less than the threshold of the OCR word count but is "
        "usable, so it is kept as the base and OCR-only lines are added. **OCR-only pages**: no usable "
        "text layer (image-only covers, scanned forms); OCR text alone. **Min coverage**: lowest "
        "text-layer/OCR word ratio on a judged page.",
        "- **Lines lost**: text-layer lines of 5+ words absent from the final page text (running heads and "
        "page numbers that cleaning removed on purpose do not count), as total / on merged pages. It must "
        "be 0 on merged pages.",
        "- **Bad-char %** is computed on the raw extracted text (before cleaning, so it still counts "
        "Symbol-font private-use bullets such as U+F0B7) and on the cleaned text, shown as raw / cleaned.",
        "- **Residual edge lines**: running-head/page-number lines still present after cleaning (total, "
        "split page numbers / head patterns / repeated lines).",
        "- **Printed guess**: pages with a `printed_page_guess`; the number in parentheses is how many "
        "guesses were nulled because they broke the document's modal printed-minus-PDF page offset.",
        "",
        "| doc_id | pages | empty pages | merged pages | OCR-only pages | min coverage | lines lost | "
        "median chars/page | bad-char % raw / cleaned | odd-token % | residual edge lines | "
        "removed: nums / heads / repeated | PDF labels / printed guess (nulled) / section hint | "
        "pages tagged non-body | flags |",
        "|---|---:|---|---|---|---:|---|---:|---|---:|---|---|---|---:|---|",
    ]
    for d in docs:
        empty = f"{len(d['empty_pages'])} ({_short_list(d['empty_pages'])})" if d["empty_pages"] else "0"
        ocr = f"{len(d['ocr_pages'])} ({_short_list(d['ocr_pages'])})" if d["ocr_pages"] else "0"
        merged = f"{len(d['merged_pages'])} ({_short_list(d['merged_pages'])})" if d["merged_pages"] else "0"
        lost = "-" if d["lines_lost"] is None else f"{d['lines_lost']['all']} / {d['lines_lost']['merged']}"
        cleaned_bad = d["bad_char_ratio_cleaned"]
        bad_cleaned = "-" if cleaned_bad is None else f"{cleaned_bad * 100:.2f}"
        bad = f"{d['bad_char_ratio'] * 100:.2f} / {bad_cleaned}"
        cov = "-" if d["min_coverage"] is None else f"{d['min_coverage']:.2f}"
        res = d["residual"]
        residual = (
            "-"
            if res is None
            else f"{res['total']} ({res['page_numbers']} / {res['head_patterns']} / {res['repeated']})"
        )
        lines.append(
            f"| {d['doc_id']} | {d['pages']} | {empty} | {merged} | {ocr} | {cov} | {lost} | "
            f"{d['median_chars']} | {bad} | {d['odd_token_ratio'] * 100:.2f} | {residual} | "
            f"{d['removed_page_numbers']} / {d['removed_heads']} / {d['removed_repeated']} | "
            f"{d['pages_with_pdf_label']} / {d['pages_with_printed_guess']} ({d['guesses_nulled']}) / "
            f"{d['pages_with_section_hint']} | {d['pages_tagged']} | {', '.join(d['flags']) or '-'} |"
        )
    lines += ["", "## Most frequent repeated header/footer patterns removed", ""]
    for d in docs:
        pats = patterns.get(d["doc_id"], [])
        if pats:
            lines.append(f"- **{d['doc_id']}**: " + "; ".join(f"`{p}` x{n}" for p, n in pats))
    lines += [
        "",
        "Patterns are shown in normalised form (lower-case, digit runs as `#`, whitespace removed). "
        "Full per-line logs: `data/processed/logs/<doc_id>_removed_lines.csv`. "
        "Per-page stats: `data/processed/qa_pages.csv`.",
        "",
    ]
    return "\n".join(lines)


def run_qa(cfg: Config) -> int:
    rows = read_manifest(cfg.paths.manifest)
    page_rows: list[dict] = []
    docs: list[dict] = []
    patterns: dict[str, list[tuple[str, int]]] = {}
    for row in rows:
        extracted = cfg.paths.extracted_dir / f"{row.doc_id}.jsonl"
        cleaned_path = cfg.paths.processed_dir / f"{row.doc_id}.pages.jsonl"
        if not extracted.exists():
            print(f"[skip] {row.doc_id}: not extracted")
            continue
        raw_pages = read_pages(extracted)
        cleaned = read_pages(cleaned_path) if cleaned_path.exists() else []
        removed = _read_removed(cfg.paths.processed_dir / "logs" / f"{row.doc_id}_removed_lines.csv")
        stats = [page_stats(p, cfg.qa) for p in raw_pages]
        page_rows.extend(stats)
        residual = count_residual_edge_lines(cleaned, cfg.cleaning) if cleaned else None
        pdf = cfg.paths.raw_dir / row.filename
        lost = text_layer_lines_lost(pdf, cleaned, removed) if cleaned and pdf.exists() else None
        cleaned_text = "".join(p["text"] for p in cleaned)
        cleaned_non_ws = sum(1 for c in cleaned_text if not c.isspace())
        bad_cleaned = bad_char_count(cleaned_text) / cleaned_non_ws if cleaned_non_ws else None
        docs.append(doc_stats(row, stats, cleaned, removed, cfg.qa, residual, lost, bad_cleaned))
        patterns[row.doc_id] = top_removed_patterns(removed)

    if not docs:
        print("nothing to report: run extraction first")
        return 1
    out_pages = cfg.paths.processed_dir / "qa_pages.csv"
    out_pages.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pages, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(page_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(page_rows)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.reports_dir / "extraction_qa.md"
    report_path.write_text(build_report(docs, patterns), encoding="utf-8", newline="\n")
    flagged = [d["doc_id"] for d in docs if d["flags"]]
    print(f"wrote {report_path} and {out_pages}; {len(docs)} docs, {len(flagged)} flagged: {flagged}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.qa", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    return run_qa(load_config(args.config))


if __name__ == "__main__":
    sys.exit(main())
