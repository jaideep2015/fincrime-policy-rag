"""Glue for the demo app: load the pipeline once, answer a question, return everything the UI shows.

* The Ollama backend is used directly (no silent fallback): if the server is unreachable the caller is told
  and may choose the extractive backend explicitly.
* The Ollama reply cache is switched off here, and nothing about the question is written to disk.
* Retrieval is the pipeline default (``retrieval.mode``, ``generation.context_k``); the app shows these and
  does not let the user change them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from fincrime_rag.config import Config, load_config
from fincrime_rag.generation import Answer, RAGPipeline, load_page_text_lookup
from fincrime_rag.guardrails import RefusalGate, load_gate
from fincrime_rag.llm.base import LLMClient, LLMUnavailable
from fincrime_rag.llm.extractive import ExtractiveClient
from fincrime_rag.llm.ollama_client import OllamaClient
from fincrime_rag.retrieval import Retriever


@dataclass
class Resources:
    cfg: Config
    retriever: Retriever
    gate: RefusalGate
    page_text: object  # callable (doc_id, page) -> str | None


@dataclass
class Result:
    answer: Answer
    elapsed_seconds: float
    chunks: list[dict]  # retrieved chunks, best first
    passages: dict[tuple[str, int], str]  # valid citation -> text of the cited page


def index_available(cfg: Config) -> bool:
    return Path(cfg.paths.index_dir).exists() and Path(cfg.refusal.calibration_file).exists()


def load_resources(config_path: str = "config.yaml") -> Resources:
    from fincrime_rag.embeddings import SentenceTransformerEmbedder

    cfg = load_config(config_path)
    retriever = Retriever(cfg, SentenceTransformerEmbedder(cfg))
    return Resources(cfg, retriever, load_gate(cfg), load_page_text_lookup(cfg))


def ollama_status(cfg: Config) -> tuple[bool, str]:
    """(reachable with the model pulled, human-readable detail). Never raises."""
    try:
        digest = OllamaClient(cfg, use_cache=False).check_available()
    except LLMUnavailable as exc:
        return False, str(exc)
    return True, f"{cfg.llm.model} (digest {digest[:12]})"


def make_llm(res: Resources, backend: str) -> LLMClient:
    if backend == "extractive":
        return ExtractiveClient(res.cfg, res.page_text)
    if backend == "ollama":
        return OllamaClient(res.cfg, use_cache=False)  # direct client: no fallback to extractive
    raise ValueError(f"unknown backend {backend!r}")


def ask(res: Resources, question: str, backend: str) -> Result:
    """Run one question. Raises ``LLMUnavailable`` if the chosen backend cannot be reached."""
    pipeline = RAGPipeline(res.cfg, res.retriever, make_llm(res, backend), res.gate)
    start = time.perf_counter()
    answer = pipeline.answer(question)
    elapsed = time.perf_counter() - start
    chunks = [res.retriever.chunks[cid] for cid in answer.context_chunk_ids]  # chunks that were in the prompt
    passages = {
        (c.doc_id, c.page): res.page_text(c.doc_id, c.page) or "(page text unavailable)"
        for c in answer.valid_citations
    }
    return Result(answer, elapsed, chunks, passages)
