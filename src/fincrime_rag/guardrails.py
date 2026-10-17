"""Guardrails: injection handling and the retrieval-score refusal gate.

Injection (D-024)
* Instruction-like phrasings are matched with the generic regexes in ``injection.patterns``.
* The question is *sanitised* before retrieval: sentences that look like instructions are dropped (if
  something answerable remains), so an embedded instruction neither steers retrieval nor reaches the model.
* Retrieved chunks are delimited and the system prompt says they are data (see ``prompting``).
* Output check: a reply that merely echoes an injected sentence (from the question or a chunk) is rejected.

Refusal gate (D-023)
* Refuse when the top-1 dense cosine similarity is below a threshold derived, a priori, from the corpus
  itself: sample sentences from every document, use each as a query (with bge's query instruction), drop the
  chunks containing that sentence, and take a low percentile of the best remaining similarity. No eval
  question is used. ``python -m fincrime_rag.guardrails calibrate`` writes ``refusal.calibration_file``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from fincrime_rag.config import Config, InjectionConfig, load_config

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_SANITIZED_WORDS = 3  # a question shorter than this after sanitising keeps its original text
MAX_QUERY_WORDS = 40  # calibration queries longer than this are skipped (not question-like)
CALIBRATION_DEPTH = 8  # chunks examined per calibration query


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(re.sub(r"\s+", " ", text).strip()) if s.strip()]


class InjectionDetector:
    def __init__(self, cfg: InjectionConfig):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in cfg.patterns]
        self._echo_overlap = cfg.echo_overlap

    def matches(self, text: str) -> bool:
        return any(p.search(text) for p in self._patterns)

    def injected_sentences(self, text: str) -> list[str]:
        return [s for s in sentences(text) if self.matches(s)]

    def sanitize_question(self, question: str) -> tuple[str, bool]:
        """Drop instruction-like sentences. Returns (clean question, whether anything was dropped)."""
        kept = [s for s in sentences(question) if not self.matches(s)]
        clean = " ".join(kept)
        if len(clean.split()) < MIN_SANITIZED_WORDS:
            return question, self.matches(question)
        return clean, len(kept) != len(sentences(question))

    def context_injections(self, chunks: list[dict]) -> list[str]:
        found: list[str] = []
        for chunk in chunks:
            found.extend(self.injected_sentences(chunk["text"]))
        return found

    def is_echo(self, reply: str, injected: list[str]) -> bool:
        """True when (nearly) all of the reply's words come from one injected sentence."""
        body = re.sub(r"\[[^\]]*\]", " ", reply)  # citations are not content
        words = _WORD.findall(body.lower())
        if not words:
            return False
        for sentence in injected:
            vocabulary = set(_WORD.findall(sentence.lower()))
            if sum(w in vocabulary for w in words) / len(words) >= self._echo_overlap:
                return True
        return False


# --------------------------------------------------------------------------- refusal gate
@dataclass(frozen=True)
class RefusalGate:
    threshold: float

    def should_refuse(self, top_similarity: float | None) -> bool:
        return top_similarity is None or top_similarity < self.threshold


def is_insufficient(text: str, marker: str) -> bool:
    """The model said the context is insufficient (the marker alone, ignoring case and punctuation)."""
    return text.strip().strip(".").strip().upper() == marker.upper()


def load_gate(cfg: Config) -> RefusalGate:
    path = Path(cfg.refusal.calibration_file)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m fincrime_rag.guardrails calibrate`")
    return RefusalGate(float(json.loads(path.read_text(encoding="utf-8"))["threshold"]))


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (same convention as numpy's default)."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def sample_queries(chunks: list[dict], per_doc: int, min_words: int, seed: int) -> list[tuple[str, str]]:
    """(doc_id, sentence) pairs, ``per_doc`` per document at most, drawn with a fixed seed."""
    rng = random.Random(seed)
    by_doc: dict[str, list[str]] = {}
    for chunk in chunks:
        for s in sentences(chunk["text"]):
            if min_words <= len(s.split()) <= MAX_QUERY_WORDS:
                by_doc.setdefault(chunk["doc_id"], []).append(s)
    queries: list[tuple[str, str]] = []
    for doc_id in sorted(by_doc):
        pool = sorted(set(by_doc[doc_id]))
        queries.extend((doc_id, s) for s in rng.sample(pool, min(per_doc, len(pool))))
    return queries


def calibrate(cfg: Config, retriever) -> dict:
    """Derive the refusal threshold from the corpus alone (no eval question involved)."""
    chunks = list(retriever.chunks.values())
    n_docs = len({c["doc_id"] for c in chunks})
    per_doc = -(-cfg.refusal.calibration_queries // n_docs)
    queries = sample_queries(chunks, per_doc, cfg.refusal.min_query_words, cfg.seed)
    best: list[float] = []
    for _, sentence in queries:
        for cid, sim in retriever.dense_similarities(sentence, CALIBRATION_DEPTH):
            if sentence not in re.sub(r"\s+", " ", retriever.chunks[cid]["text"]):
                best.append(sim)  # first similarity from a chunk that does not contain the sentence
                break
    threshold = percentile(best, cfg.refusal.percentile)
    return {
        "method": "percentile of in-corpus top-1 dense cosine similarity (source chunks excluded)",
        "percentile": cfg.refusal.percentile,
        "threshold": threshold,
        "n_queries": len(best),
        "n_documents": n_docs,
        "observed_min": min(best),
        "observed_p25": percentile(best, 25),
        "observed_median": percentile(best, 50),
        "observed_p75": percentile(best, 75),
        "observed_max": max(best),
        "eval_set_used": False,
        "tuned": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.guardrails", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["calibrate"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    from fincrime_rag.embeddings import build_embedder
    from fincrime_rag.retrieval import Retriever

    result = calibrate(cfg, Retriever(cfg, build_embedder(cfg)))
    path = Path(cfg.refusal.calibration_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
