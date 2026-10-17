"""Extractive backend: no LLM. Returns the retrieved sentences that best match the question.

Sentences are scored by the summed idf of the question words they contain (idf over the retrieved
context itself), divided by the square root of their length so that short, on-point sentences beat long
ones. Each selected sentence is quoted verbatim with a citation ``[doc_id p.N]``. A chunk can span
pages, so the page is found by locating the sentence in the page texts of the chunk's pages (falling
back to the chunk's first page). Sentences that look like instructions (``injection.patterns``) are never
quoted. This backend cannot decline by itself; refusal comes from the retrieval-score gate.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

from fincrime_rag.config import Config
from fincrime_rag.guardrails import InjectionDetector
from fincrime_rag.llm.base import GenerationRequest, LLMResponse

PageText = Callable[[str, int], str | None]
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(§\d])|\s*•\s*")
LOCATE_PREFIX_CHARS = 80  # leading characters of a sentence used to find its page


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENTENCE_BREAK.split(flat) if s and s.strip()]


class ExtractiveClient:
    backend = "extractive"

    def __init__(self, cfg: Config, page_text: PageText | None = None):
        self._n = cfg.generation.extractive_sentences
        self._min_words = cfg.generation.min_sentence_words
        self._detector = InjectionDetector(cfg.injection)
        self._page_text = page_text

    def _locate_page(self, chunk: dict, sentence: str) -> int:
        pages = chunk["pdf_pages"]
        if self._page_text and len(pages) > 1:
            needle = normalize(sentence)[:LOCATE_PREFIX_CHARS]
            for page in pages:
                text = self._page_text(chunk["doc_id"], page)
                if text and needle in normalize(text):
                    return page
        return pages[0]

    def generate(self, request: GenerationRequest) -> LLMResponse:
        candidates: list[tuple[int, str, dict]] = []  # (chunk rank, sentence, chunk)
        for rank, chunk in enumerate(request.chunks):
            for sentence in split_sentences(chunk["text"]):
                if len(sentence.split()) >= self._min_words and not self._detector.matches(sentence):
                    candidates.append((rank, sentence, chunk))
        q_words = set(_WORD.findall(request.question.lower()))
        df: dict[str, int] = {}
        for _, sentence, _ in candidates:
            for word in set(_WORD.findall(sentence.lower())):
                df[word] = df.get(word, 0) + 1
        total = max(len(candidates), 1)

        def score(item: tuple[int, str, dict]) -> tuple[float, int]:
            words = _WORD.findall(item[1].lower())
            overlap = sum(math.log(1 + total / df[w]) for w in set(words) & q_words)
            return (-overlap / math.sqrt(max(len(words), 1)), item[0])

        chosen = sorted(candidates, key=score)[: self._n]
        lines = [f"- {s} [{c['doc_id']} p.{self._locate_page(c, s)}]" for _, s, c in chosen]
        return LLMResponse(text="\n".join(lines), backend=self.backend)
