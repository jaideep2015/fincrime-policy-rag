"""LLM-free citation support checks, defined before measuring and not tuned (D-030).

Page-level precision only says that a cited page is a gold page. These checks ask, for each *cited sentence*
of an answer, whether the cited page's own text supports it:

* numeric containment: every number in the sentence appears in the text of the cited page (commas ignored);
  sentences without numbers are not counted;
* content-word overlap: share of the sentence's content words (lower-case letters only, at least 4 letters,
  fixed stop list below) that occur on the cited page.

Unit = a (sentence, cited document, cited page) pair; identical pairs are counted once. Only citations that
were validated against the retrieved context are used. Both checks are crude (a page can contain the right
numbers and words without supporting the claim) and say nothing about correctness.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from fincrime_rag.citations import CITATION, Citation, _expand
from fincrime_rag.guardrails import sentences

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
CONTENT_WORD = re.compile(r"[a-z]{4,}")
STOP_WORDS = frozenset(
    "that this with from have been which their there were will would could should about into other such"
    " than also these those them they when what where while being does each more most some only over under"
    " within between because through however including".split()
)
PageText = Callable[[str, int], str | None]


@dataclass(frozen=True)
class PairSupport:
    doc_id: str
    page: int
    numeric_ok: bool | None  # None: the sentence has no numbers
    overlap: float | None  # None: the sentence has no content words


def cited_sentences(text: str, valid: set[tuple[str, int]]) -> list[tuple[str, list[Citation]]]:
    """(sentence without tags, valid citations on it), only for sentences that carry a citation.

    A segment that is only a citation tag (the model wrote "claim. [doc p.4]") joins the previous sentence.
    """
    entries: list[tuple[str, list[Citation]]] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)
        for segment in sentences(line):
            tags = [
                Citation(m.group("doc"), p)
                for m in CITATION.finditer(segment)
                for p in _expand(m.group("pages"))
                if (m.group("doc"), p) in valid
            ]
            body = CITATION.sub("", segment).strip()
            if not re.search(r"[A-Za-z0-9]", body):
                if entries:
                    entries[-1][1].extend(tags)
                continue
            entries.append((body, list(tags)))
    return [(body, tags) for body, tags in entries if tags]


def numbers_in(sentence: str) -> list[str]:
    return [n.rstrip(",.") for n in NUMBER.findall(sentence) if n.rstrip(",.")]


def numeric_containment(sentence: str, page_text: str) -> bool | None:
    numbers = numbers_in(sentence)
    if not numbers:
        return None
    plain = page_text.replace(",", "")
    return all(n in page_text or n.replace(",", "") in plain for n in numbers)


def content_overlap(sentence: str, page_text: str) -> float | None:
    words = set(CONTENT_WORD.findall(sentence.lower())) - STOP_WORDS
    if not words:
        return None
    return len(words & set(CONTENT_WORD.findall(page_text.lower()))) / len(words)


def support_pairs(text: str, valid: set[tuple[str, int]], page_text: PageText) -> list[PairSupport]:
    seen: set[tuple[str, str, int]] = set()
    pairs: list[PairSupport] = []
    for sentence, cites in cited_sentences(text, valid):
        key_text = re.sub(r"\s+", " ", sentence.lower())
        for cite in dict.fromkeys(cites):
            if (key_text, cite.doc_id, cite.page) in seen:
                continue
            seen.add((key_text, cite.doc_id, cite.page))
            page = page_text(cite.doc_id, cite.page) or ""
            pairs.append(
                PairSupport(
                    cite.doc_id,
                    cite.page,
                    numeric_containment(sentence, page),
                    content_overlap(sentence, page),
                )
            )
    return pairs
