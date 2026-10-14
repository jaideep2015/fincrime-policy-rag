"""Stage 3: clean extracted page text and log every header/footer line removed.

Cleaning is deliberately conservative and fully logged:

1. Running heads and page numbers, detected from the data and from configured patterns:
   * bare page numbers ("12", "iv", "Page 3 of 9", "F-3") in the first/last
     ``page_number_scan_lines`` lines;
   * heads shaped like "N | TITLE" or "TITLE | N" (``printed_page_head_patterns``) in the first
     ``running_head_scan_lines`` lines; the printed page number is recorded;
   * running heads matching ``running_head_patterns`` (e.g. CFR "§ 1005.18", "Pt. 1005, Supp. I")
     in the first ``running_head_scan_lines`` lines; a named group ``section`` fills
     ``section_hint``;
   * any other line whose normalised form (see ``normalize_line``) appears among the first/last
     ``header_footer_scan_lines`` lines of at least ``repeat_fraction`` of the pages. Skipped for
     documents shorter than ``min_pages_for_detection``.
   Printed page numbers found this way are kept as ``printed_page_guess`` for pages without a
   PDF page label.
2. Text tidying: ligatures expanded, soft hyphens and non-breaking spaces removed, dot leaders
   collapsed, whitespace normalised, line-end hyphenation repaired (a word split as
   "anti-\\nmoney" keeps its hyphen if "anti-money" occurs elsewhere in the document).

Every removed line goes to ``data/processed/logs/<doc_id>_removed_lines.csv`` for review.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from fincrime_rag.config import CleaningConfig, Config, IndexingConfig, load_config
from fincrime_rag.extraction import Page, read_pages, write_pages
from fincrime_rag.ingestion import read_manifest

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_LIGATURE_RE = re.compile("[" + "".join(_LIGATURES) + "]")

# Bare page numbers: arabic, lower- or upper-case roman, appendix style (4-12, F-3, F<en dash>3),
# with an optional "Page" prefix and "of N" suffix.
_DASH = "-‐‑–−"
_PAGE_NUMBER_RE = re.compile(
    rf"^(?:page\s+)?(?:\d{{1,4}}|[ivxlcdm]{{1,6}}|[IVXLCDM]{{1,6}}"
    rf"|[A-Za-z][{_DASH}]\d{{1,3}}|\d{{1,3}}[{_DASH}]\d{{1,3}})(?:\s+of\s+\d{{1,4}})?$",
    re.IGNORECASE,
)
_ROMAN_OK_RE = re.compile(r"^(?:[ivxlcdm]+|[IVXLCDM]+)$")  # reject mixed-case words like "Civil"

_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z]+)-[ \t]*\n[ \t]*([a-z][A-Za-z]*)")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_DOT_LEADER_RE = re.compile(r"(?:\. ?){4,}")
_DIGIT_RUN_RE = re.compile(r"\d+")
_PRIVATE_PIPE = ""  # Symbol-font vertical bar used in some running heads


def prep_line(line: str) -> str:
    """Line as matched against patterns: private-use bar mapped to '|', ends stripped."""
    return line.replace(_PRIVATE_PIPE, "|").strip()


def normalize_line(line: str) -> str:
    """Pattern key for repeat detection.

    Lower-cased; each run of digits becomes one '#' (so "Page 9" and "Page 10" match); U+F07C
    becomes '|'; for "A | B" lines the two sides are sorted (so "6 | TITLE" matches "TITLE | 7");
    all whitespace is removed (so a letter-spaced text-layer head "F I N C E N" matches the OCR
    reading "FINCEN").
    """
    s = _DIGIT_RUN_RE.sub("#", prep_line(line).lower())
    if "|" in s:
        s = "|".join(sorted(part.strip() for part in s.split("|")))
    return re.sub(r"\s+", "", s)


def is_page_number(line: str) -> bool:
    s = prep_line(line)
    if not _PAGE_NUMBER_RE.match(s):
        return False
    core = re.sub(r"^(?:page\s+)", "", s, flags=re.IGNORECASE)
    core = re.sub(r"\s+of\s+\d{1,4}$", "", core, flags=re.IGNORECASE)
    if core.isalpha():  # roman numeral candidates must be consistently cased
        return bool(_ROMAN_OK_RE.match(core))
    return True


def _page_number_token(line: str) -> str:
    core = re.sub(r"^(?:page\s+)", "", prep_line(line), flags=re.IGNORECASE)
    core = re.sub(r"\s+of\s+\d{1,4}$", "", core, flags=re.IGNORECASE)
    return re.sub(f"[{_DASH}]", "-", core)


def _edge_positions(lines: list[str], k: int) -> dict[int, str]:
    """Indexes of the first/last k non-empty lines -> 'header' / 'footer'."""
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    positions = {i: "footer" for i in nonempty[-k:]}
    positions.update({i: "header" for i in nonempty[:k]})  # header wins on short pages
    return positions


def _first_lines(lines: list[str], k: int) -> list[int]:
    return [i for i, line in enumerate(lines) if line.strip()][:k]


def _match_group(patterns: list[re.Pattern[str]], line: str, group: str) -> str | None:
    """First pattern that fully matches the line -> value of `group` ('' if the pattern has none)."""
    for pattern in patterns:
        match = pattern.match(line)
        if match:
            return match.groupdict().get(group) or ""
    return None


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def detect_repeated_patterns(pages: list[Page], cfg: CleaningConfig) -> dict[str, int]:
    """Normalised edge-line patterns appearing on >= repeat_fraction of pages -> page count."""
    text_pages = [p for p in pages if p["text"].strip()]
    if len(text_pages) < cfg.min_pages_for_detection:
        return {}
    # Lines the configured head regexes handle (first lines only) must not also become "repeated"
    # patterns, or a bare "§ 1005.99" at the bottom of a page would be removed with the heads.
    handled = _compile(cfg.running_head_patterns + cfg.printed_page_head_patterns)
    counts: Counter[str] = Counter()
    for page in text_pages:
        lines = page["text"].split("\n")
        seen = {
            normalize_line(lines[i])
            for i in _edge_positions(lines, cfg.header_footer_scan_lines)
            if not is_page_number(lines[i]) and _match_group(handled, prep_line(lines[i]), "") is None
        }
        # a head must contain a letter: "#." (numbered-list markers such as "3.") is content, not a head
        counts.update(p for p in seen if len(p) >= 2 and any(c.isalpha() for c in p))
    threshold = cfg.repeat_fraction * len(text_pages)
    return {pattern: n for pattern, n in counts.items() if n >= threshold}


def _tidy(text: str) -> str:
    text = _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group()], text)
    text = text.replace("­", "").replace("\xa0", " ").replace("", "•")  # Symbol bullet
    text = _DOT_LEADER_RE.sub(" ... ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^•[ ]*\n(?=\S)", "• ", text)  # bullet extracted on its own line
    return text.strip()


def _build_vocab(texts: list[str]) -> set[str]:
    """Lower-cased words of the document, plus hyphenated compounds and their parts."""
    vocab: set[str] = set()
    for text in texts:
        # drop line-break fragments ("contin-" / "ued") so they never count as words
        for token in _TOKEN_RE.findall(_HYPHEN_BREAK_RE.sub(" ", text)):
            token = token.lower()
            vocab.add(token)
            vocab.update(token.split("-"))
    return vocab


def _repair_hyphenation(text: str, vocab: set[str]) -> str:
    """Rejoin words split across lines, keeping the hyphen of genuine compounds.

    Decision for "left-\\nright", in order: the joined word occurs elsewhere -> join; the hyphenated
    compound occurs elsewhere -> keep the hyphen; both parts are standalone words of 3+ letters
    -> keep the hyphen ("account-holding"); otherwise join ("contin-ued").
    """

    def join(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        lo, ro = left.lower(), right.lower()
        if lo + ro in vocab:
            return left + right
        if f"{lo}-{ro}" in vocab or (len(lo) >= 3 and len(ro) >= 3 and lo in vocab and ro in vocab):
            return f"{left}-{right}"
        return left + right

    return _HYPHEN_BREAK_RE.sub(join, text)


# Priority of sources for printed_page_guess (lowest number wins).
_PRIORITY = {"printed_page_head": 0, "footer_number": 1, "header_number": 2}


def clean_document(pages: list[Page], cfg: CleaningConfig) -> tuple[list[Page], list[dict]]:
    """Return (cleaned pages, removed-line log rows)."""
    patterns = detect_repeated_patterns(pages, cfg)
    head_patterns = _compile(cfg.running_head_patterns)
    page_head_patterns = _compile(cfg.printed_page_head_patterns)
    n_text_pages = sum(1 for p in pages if p["text"].strip())
    log: list[dict] = []
    stripped: list[tuple[Page, str, str | None, str | None]] = []

    for page in pages:
        lines = page["text"].split("\n")
        keep = [True] * len(lines)
        repeat_pos = _edge_positions(lines, cfg.header_footer_scan_lines)
        number_pos = _edge_positions(lines, cfg.page_number_scan_lines)
        head_pos = set(_first_lines(lines, cfg.running_head_scan_lines))
        guesses: list[tuple[int, str]] = []
        section_hint: str | None = None

        for idx in sorted(set(repeat_pos) | set(number_pos) | head_pos):
            line = prep_line(lines[idx])
            position = repeat_pos.get(idx) or number_pos.get(idx) or "header"
            reason = None
            if idx in head_pos and (token := _match_group(page_head_patterns, line, "page")) is not None:
                reason = "printed_page_head"
                if token:
                    guesses.append((_PRIORITY["printed_page_head"], token))
            elif idx in head_pos and (section := _match_group(head_patterns, line, "section")) is not None:
                reason = "running_head"
                if section and section_hint is None:
                    section_hint = section
            elif cfg.strip_edge_page_numbers and idx in number_pos and is_page_number(line):
                reason = "page_number"
                guesses.append((_PRIORITY[f"{position}_number"], _page_number_token(line)))
            elif idx in repeat_pos and normalize_line(line) in patterns:
                reason = f"repeated_on_{patterns[normalize_line(line)]}_of_{n_text_pages}_pages"
            if reason:
                keep[idx] = False
                log.append(
                    {"pdf_page": page["pdf_page"], "position": position, "reason": reason, "line": line}
                )
        guess = min(guesses)[1] if guesses else None
        body = "\n".join(line for line, k in zip(lines, keep, strict=True) if k)
        stripped.append((page, body, guess, section_hint))

    vocab = _build_vocab([text for _, text, _, _ in stripped])
    cleaned: list[Page] = []
    for page, text, guess, section_hint in stripped:
        cleaned.append(
            {
                "doc_id": page["doc_id"],
                "pdf_page": page["pdf_page"],
                "page_label": page["page_label"],
                "printed_page_guess": None if page["page_label"] else guess,
                "printed_page_guess_rejected": None,
                "section_hint": section_hint,
                "content_type": _content_type(page["doc_id"], page["pdf_page"], cfg),
                "extraction_method": page.get("extraction_method", "text"),
                "text": _tidy(_repair_hyphenation(text, vocab)),
            }
        )
    reject_inconsistent_page_guesses(cleaned)
    return cleaned, log


def _content_type(doc_id: str, pdf_page: int, cfg: CleaningConfig) -> str:
    """"body" unless a configured page tag covers the page (e.g. Reg E model forms)."""
    for tag in cfg.page_tags:
        if tag.doc_id == doc_id and tag.first_page <= pdf_page <= tag.last_page:
            return tag.content_type
    return "body"


def reject_inconsistent_page_guesses(pages: list[Page]) -> int:
    """Null numeric ``printed_page_guess`` values that break the document's page-number offset.

    Printed numbers advance with the PDF index, so ``printed - pdf_page`` should be constant. The
    modal offset over numeric guesses wins; guesses that deviate (a stray number read from a
    figure or footnote) are set to None and kept in ``printed_page_guess_rejected`` for audit.
    Needs at least 3 numeric guesses, with the modal offset on at least half of them.
    Non-numeric guesses ("F-1", "iv") are left alone. Returns how many were nulled.
    """
    numeric = [
        (p, int(p["printed_page_guess"]) - p["pdf_page"])
        for p in pages
        if str(p.get("printed_page_guess") or "").isdigit()
    ]
    if len(numeric) < 3:
        return 0
    offset, support = Counter(o for _, o in numeric).most_common(1)[0]
    if support < 3 or support * 2 < len(numeric):
        return 0
    nulled = 0
    for page, page_offset in numeric:
        if page_offset != offset:
            page["printed_page_guess_rejected"] = page["printed_page_guess"]
            page["printed_page_guess"] = None
            nulled += 1
    return nulled


def indexable_pages(pages: list[Page], cfg: IndexingConfig) -> list[Page]:
    """Pages to index: everything except content types listed in ``indexing.exclude_content_types``."""
    return [p for p in pages if p.get("content_type", "body") not in cfg.exclude_content_types]


def count_residual_edge_lines(pages: list[Page], cfg: CleaningConfig) -> dict[str, int]:
    """How many running-head / page-number lines are still present in (cleaned) pages.

    Applies the same detection as ``clean_document`` to the given pages and counts what it
    would still remove: bare page numbers, configured head patterns, repeated edge lines.
    """
    patterns = detect_repeated_patterns(pages, cfg)
    head_patterns = _compile(cfg.running_head_patterns + cfg.printed_page_head_patterns)
    counts = {"page_numbers": 0, "head_patterns": 0, "repeated": 0}
    for page in pages:
        lines = page["text"].split("\n")
        repeat_pos = _edge_positions(lines, cfg.header_footer_scan_lines)
        number_pos = _edge_positions(lines, cfg.page_number_scan_lines)
        head_pos = set(_first_lines(lines, cfg.running_head_scan_lines))
        for idx in set(repeat_pos) | set(number_pos) | head_pos:
            line = prep_line(lines[idx])
            if idx in head_pos and _match_group(head_patterns, line, "section") is not None:
                counts["head_patterns"] += 1
            elif idx in number_pos and is_page_number(line):
                counts["page_numbers"] += 1
            elif idx in repeat_pos and normalize_line(line) in patterns:
                counts["repeated"] += 1
    counts["total"] = sum(counts.values())
    return counts


def write_removed_log(path: Path, log: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pdf_page", "position", "reason", "line"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(log)


def clean_all(cfg: Config, only: str | None = None) -> int:
    rows = read_manifest(cfg.paths.manifest)
    if only:
        rows = [r for r in rows if r.doc_id == only]
    for row in rows:
        src = cfg.paths.extracted_dir / f"{row.doc_id}.jsonl"
        if not src.exists():
            print(f"[skip] {row.doc_id}: not extracted yet")
            continue
        cleaned, log = clean_document(read_pages(src), cfg.cleaning)
        write_pages(cfg.paths.processed_dir / f"{row.doc_id}.pages.jsonl", cleaned)
        write_removed_log(cfg.paths.processed_dir / "logs" / f"{row.doc_id}_removed_lines.csv", log)
        nulled = sum(1 for p in cleaned if p["printed_page_guess_rejected"] is not None)
        tagged = sum(1 for p in cleaned if p["content_type"] != "body")
        print(
            f"[ok]   {row.doc_id}: {len(cleaned)} pages, {len(log)} header/footer lines removed, "
            f"{nulled} printed-page guesses nulled, {tagged} pages tagged non-body"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.cleaning", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--doc", help="clean a single doc_id")
    args = parser.parse_args(argv)
    return clean_all(load_config(args.config), only=args.doc)


if __name__ == "__main__":
    sys.exit(main())
