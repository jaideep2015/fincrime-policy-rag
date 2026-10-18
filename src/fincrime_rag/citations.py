"""Citation parsing and validation.

Citations look like ``[doc_id p.N]`` where N is the PDF page index. ``[doc_id p.4, 5]`` and
``[doc_id p.4-6]`` are also read (expanded to single pages). A cited (doc_id, page) is *valid* only if some
retrieved chunk of that document lists that page in ``pdf_pages``. Invalid citations are stripped from the
answer text and counted; an answer with no valid citation is not grounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CITATION = re.compile(
    r"\[\s*(?P<doc>[A-Za-z0-9_]+)\s+p\.?\s*(?P<pages>\d+(?:\s*(?:,|-|–|and)\s*\d+)*)\s*\]",
    re.IGNORECASE,
)
MAX_RANGE_PAGES = 50  # a cited range wider than this is not expanded


@dataclass(frozen=True)
class Citation:
    doc_id: str
    page: int


def _expand(pages: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"\s*(?:,|and)\s*", pages):
        bounds = [int(n) for n in re.split(r"\s*[-–]\s*", part) if n]
        if len(bounds) == 2 and 0 <= bounds[1] - bounds[0] < MAX_RANGE_PAGES:
            out.extend(range(bounds[0], bounds[1] + 1))
        else:
            out.extend(bounds[:1])
    return out


def parse_citations(text: str) -> list[Citation]:
    found: list[Citation] = []
    for match in CITATION.finditer(text):
        found.extend(Citation(match.group("doc"), p) for p in _expand(match.group("pages")))
    return found


def context_pages(chunks: list[dict]) -> set[tuple[str, int]]:
    return {(c["doc_id"], int(p)) for c in chunks for p in c["pdf_pages"]}


@dataclass(frozen=True)
class CitationCheck:
    valid: list[Citation]
    invalid: list[Citation]
    text: str  # answer with invalid citations removed

    @property
    def grounded(self) -> bool:
        return bool(self.valid)


def validate_citations(text: str, chunks: list[dict]) -> CitationCheck:
    allowed = context_pages(chunks)
    valid: list[Citation] = []
    invalid: list[Citation] = []

    def keep(match: re.Match) -> str:
        cites = [Citation(match.group("doc"), p) for p in _expand(match.group("pages"))]
        good = [c for c in cites if (c.doc_id, c.page) in allowed]
        valid.extend(good)
        invalid.extend(c for c in cites if (c.doc_id, c.page) not in allowed)
        return match.group(0) if good and len(good) == len(cites) else ""

    cleaned = re.sub(r"[ \t]{2,}", " ", CITATION.sub(keep, text)).strip()
    # identical citations are counted once everywhere (order kept)
    return CitationCheck(list(dict.fromkeys(valid)), list(dict.fromkeys(invalid)), cleaned)


# Lenient reading (reported next to the strict bracket format, never the primary metric): also "(doc_id p.N)".
LENIENT = re.compile(
    r"\(\s*(?P<doc>[A-Za-z0-9_]+)\s+p\.?\s*(?P<pages>\d+(?:\s*(?:,|-|–|and)\s*\d+)*)\s*\)",
    re.IGNORECASE,
)


def lenient_valid_citations(text: str, chunks: list[dict]) -> list[Citation]:
    allowed = context_pages(chunks)
    found: list[Citation] = []
    for pattern in (CITATION, LENIENT):
        for match in pattern.finditer(text):
            found.extend(
                Citation(match.group("doc"), p)
                for p in _expand(match.group("pages"))
                if (match.group("doc"), p) in allowed
            )
    return list(dict.fromkeys(found))


def strip_citation_text(text: str, doc_ids: list[str]) -> str:
    """Answer text without citation tags, ``cite_as`` labels and document ids (for keyword/year checks)."""
    text = LENIENT.sub(" ", CITATION.sub(" ", text))
    text = re.sub(r"cite_as\s*:?", " ", text, flags=re.IGNORECASE)
    for doc_id in sorted(doc_ids, key=len, reverse=True):
        text = text.replace(doc_id, " ")
    return re.sub(r"\s+", " ", text).strip()
