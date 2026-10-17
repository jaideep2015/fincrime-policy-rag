"""Phase 4: dense, BM25 and hybrid retrieval behind one ``retrieve(query, k)`` interface.

* ``dense``: embed the query (bge query instruction) and take the nearest chunks from Chroma.
* ``bm25``: lexical Okapi BM25 over the same chunk texts, read back from the Chroma collection so
  both rankers see exactly the same corpus.
* ``hybrid``: reciprocal rank fusion, ``score(d) = sum over rankers of 1 / (rrf_k + rank_d)``, over
  the top ``retrieval.candidate_pool`` of each ranker. RRF needs no score calibration, which is why
  it suits mixing cosine similarities with BM25 scores.

Every hit carries doc_id, pdf pages, printed-page guess and section so it can be cited.
Retrieval never calls an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fincrime_rag.config import Config, load_config
from fincrime_rag.embeddings import Embedder
from fincrime_rag.vectorstore import chunk_from_record

MODES = ("dense", "bm25", "hybrid")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'’][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; keeps section numbers like ``1005.11`` whole."""
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    rank: int  # 1-based
    score: float
    mode: str
    chunk_id: str
    doc_id: str
    pdf_pages: tuple[int, ...]
    printed_page_guess: str | None
    section: str | None
    text: str


def rrf_fuse(rankings: list[list[str]], rrf_k: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of ranked id lists. Ties break by first appearance (deterministic)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


class Retriever:
    def __init__(self, cfg: Config, embedder: Embedder, collection=None):
        from rank_bm25 import BM25Okapi

        self.cfg = cfg
        self.embedder = embedder
        if collection is None:
            from fincrime_rag.vectorstore import get_collection

            collection = get_collection(cfg)
        self.collection = collection
        stored = collection.get(include=["documents", "metadatas"])
        order = sorted(range(len(stored["ids"])), key=lambda i: stored["ids"][i])
        self.chunks: dict[str, dict] = {}
        for i in order:
            cid = stored["ids"][i]
            self.chunks[cid] = chunk_from_record(cid, stored["documents"][i], stored["metadatas"][i])
        self._ids = list(self.chunks)
        self._bm25 = BM25Okapi(
            [tokenize(self.chunks[i]["text"]) for i in self._ids],
            k1=cfg.retrieval.bm25_k1,
            b=cfg.retrieval.bm25_b,
        )

    # ---- rankers: each returns [(chunk_id, score)] best first, at most `depth` long
    def _dense(self, query: str, depth: int) -> list[tuple[str, float]]:
        vector = self.embedder.embed_query(query)
        result = self.collection.query(
            query_embeddings=[vector.tolist()], n_results=min(depth, len(self._ids)), include=["distances"]
        )
        return [(cid, -float(d)) for cid, d in zip(result["ids"][0], result["distances"][0], strict=True)]

    def _bm25_rank(self, query: str, depth: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self._ids)), key=lambda i: (-scores[i], self._ids[i]))[:depth]
        return [(self._ids[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def dense_similarities(self, query: str, depth: int) -> list[tuple[str, float]]:
        """(chunk_id, cosine similarity) best first. Used by the refusal gate, whatever the mode:
        RRF scores depend only on ranks, so they cannot say how close the best chunk really is."""
        if self.cfg.vectorstore.space != "cosine":
            raise ValueError("dense_similarities needs vectorstore.space = cosine")
        return [(cid, 1.0 + score) for cid, score in self._dense(query, depth)]

    def retrieve(self, query: str, k: int | None = None, mode: str | None = None) -> list[Hit]:
        mode = mode or self.cfg.retrieval.mode
        k = k or self.cfg.retrieval.top_k
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {MODES}")
        pool = max(self.cfg.retrieval.candidate_pool, k)
        if mode == "dense":
            ranked = self._dense(query, k)
        elif mode == "bm25":
            ranked = self._bm25_rank(query, k)
        else:
            fused = rrf_fuse(
                [
                    [cid for cid, _ in self._dense(query, pool)],
                    [cid for cid, _ in self._bm25_rank(query, pool)],
                ],
                self.cfg.retrieval.hybrid_rrf_k,
            )
            ranked = fused[:k]
        return [self._hit(rank, cid, score, mode) for rank, (cid, score) in enumerate(ranked, start=1)]

    def _hit(self, rank: int, chunk_id: str, score: float, mode: str) -> Hit:
        c = self.chunks[chunk_id]
        return Hit(
            rank=rank,
            score=score,
            mode=mode,
            chunk_id=chunk_id,
            doc_id=c["doc_id"],
            pdf_pages=tuple(c["pdf_pages"]),
            printed_page_guess=c["printed_page_guess"],
            section=c["section"],
            text=c["text"],
        )


_default: Retriever | None = None


def retrieve(query: str, k: int | None = None, mode: str | None = None) -> list[Hit]:
    """Module-level convenience: build the default retriever once, then query it."""
    global _default
    if _default is None:
        from fincrime_rag.embeddings import build_embedder

        cfg = load_config()
        _default = Retriever(cfg, build_embedder(cfg))
    return _default.retrieve(query, k, mode)
