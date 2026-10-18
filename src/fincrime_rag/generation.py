"""Phase 5: retrieve, gate, generate, validate.

Flow for one question
1. Sanitise the question (instruction-like sentences dropped, D-024).
2. Retrieve the top ``generation.context_k`` chunks (hybrid by default).
3. Refusal gate: top-1 dense cosine similarity below the calibrated threshold -> refuse, no model call.
4. Build the delimited prompt, assert it fits ``llm.num_ctx``, call the backend.
5. Model said INSUFFICIENT_CONTEXT -> refuse.
6. Validate citations against the retrieved chunks; strip invalid ones. No valid citation -> ungrounded.
7. Output check: a reply that only echoes an injected sentence is rejected.

Statuses: ``answered`` (grounded), ``ungrounded`` (an answer without any valid citation), ``refused``
(``refusal_reason`` says why), ``rejected_injection``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fincrime_rag.citations import Citation, validate_citations
from fincrime_rag.config import Config
from fincrime_rag.guardrails import InjectionDetector, RefusalGate, is_insufficient
from fincrime_rag.llm.base import GenerationRequest, LLMClient
from fincrime_rag.prompting import assert_fits, build_user_prompt, system_prompt
from fincrime_rag.retrieval import Retriever

REFUSAL_TEXT = "The indexed documents do not contain enough information to answer this question."
INJECTION_TEXT = "The reply only repeated an embedded instruction, so it was withheld."


@dataclass
class Answer:
    status: str  # answered | ungrounded | refused | rejected_injection
    text: str
    refusal_reason: str | None = None  # low_retrieval_score | model_insufficient
    valid_citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[Citation] = field(default_factory=list)
    top_similarity: float | None = None
    retrieved: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)  # (doc_id, pdf_pages)
    question_sanitized: bool = False
    context_injection: bool = False
    backend: str | None = None
    model: str | None = None
    model_digest: str | None = None
    num_ctx: int | None = None
    fallback: bool = False
    fallback_reason: str | None = None
    prompt_tokens_estimate: int | None = None
    prompt_eval_count: int | None = None
    cached: bool = False
    raw_text: str | None = None  # the backend's reply before citation validation
    context_chunk_ids: list[str] = field(default_factory=list)  # chunks that were in the prompt, best first

    @property
    def refused(self) -> bool:
        return self.status == "refused"


class RAGPipeline:
    def __init__(self, cfg: Config, retriever: Retriever, llm: LLMClient, gate: RefusalGate):
        self.cfg = cfg
        self.retriever = retriever
        self.llm = llm
        self.gate = gate
        self.detector = InjectionDetector(cfg.injection)
        self._system = system_prompt(cfg)

    def answer(self, question: str, mode: str | None = None) -> Answer:
        clean, sanitized = self.detector.sanitize_question(question)
        hits = self.retriever.retrieve(clean, self.cfg.generation.context_k, mode)
        chunks = [self.retriever.chunks[h.chunk_id] for h in hits]
        sims = self.retriever.dense_similarities(clean, 1)
        top = sims[0][1] if sims else None
        base = {
            "top_similarity": top,
            "retrieved": [(h.doc_id, h.pdf_pages) for h in hits],
            "context_chunk_ids": [h.chunk_id for h in hits],
            "question_sanitized": sanitized,
            "context_injection": bool(self.detector.context_injections(chunks)),
        }
        if self.gate.should_refuse(top):
            return Answer("refused", REFUSAL_TEXT, "low_retrieval_score", **base)

        user = build_user_prompt(clean, chunks)
        tokens = assert_fits(self.cfg, self._system, user)
        response = self.llm.generate(GenerationRequest(self._system, user, clean, chunks))
        meta = {
            **base,
            "backend": response.backend,
            "model": response.model,
            "model_digest": response.model_digest,
            "num_ctx": response.num_ctx,
            "fallback": response.fallback,
            "fallback_reason": response.fallback_reason,
            "prompt_eval_count": response.prompt_eval_count,
            "cached": response.cached,
            "raw_text": response.text,
            "prompt_tokens_estimate": tokens,
        }
        if is_insufficient(response.text, self.cfg.generation.insufficient_marker):
            return Answer("refused", REFUSAL_TEXT, "model_insufficient", **meta)

        injected = self.detector.injected_sentences(question) + self.detector.context_injections(chunks)
        if self.detector.is_echo(response.text, injected):
            return Answer("rejected_injection", INJECTION_TEXT, **meta)

        check = validate_citations(response.text, chunks)
        status = "answered" if check.grounded else "ungrounded"
        return Answer(
            status, check.text, valid_citations=check.valid, invalid_citations=check.invalid, **meta
        )


# --------------------------------------------------------------------------- wiring
def load_page_text_lookup(cfg: Config):
    """Callable (doc_id, pdf_page) -> cleaned page text, read lazily from data/processed."""
    from fincrime_rag.extraction import read_pages

    cache: dict[str, dict[int, str]] = {}

    def lookup(doc_id: str, page: int) -> str | None:
        if doc_id not in cache:
            path = cfg.paths.processed_dir / f"{doc_id}.pages.jsonl"
            cache[doc_id] = {p["pdf_page"]: p["text"] for p in read_pages(path)} if path.exists() else {}
        return cache[doc_id].get(page)

    return lookup


def build_llm(
    cfg: Config, backend: str, page_text=None, require_count: bool = True, cache_only: bool = False
) -> LLMClient:
    """``extractive``: no LLM. ``ollama``: the configured model, falling back to extractive if unreachable."""
    from fincrime_rag.llm import FallbackClient
    from fincrime_rag.llm.extractive import ExtractiveClient
    from fincrime_rag.llm.ollama_client import OllamaClient

    extractive = ExtractiveClient(cfg, page_text)
    if backend == "extractive":
        return extractive
    if backend != "ollama":
        raise ValueError(f"unknown backend {backend!r}")
    ollama = OllamaClient(cfg, require_count=require_count, cache_only=cache_only)
    return FallbackClient(ollama, extractive) if cfg.llm.fallback_to_extractive else ollama


def main(argv: list[str] | None = None) -> int:
    import argparse

    from fincrime_rag.config import load_config
    from fincrime_rag.embeddings import build_embedder
    from fincrime_rag.guardrails import load_gate

    parser = argparse.ArgumentParser(prog="fincrime_rag.generation", description="Ask one question.")
    parser.add_argument("question")
    parser.add_argument("--backend", choices=["ollama", "extractive"], default="ollama")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    retriever = Retriever(cfg, build_embedder(cfg))
    llm = build_llm(cfg, args.backend, load_page_text_lookup(cfg))
    result = RAGPipeline(cfg, retriever, llm, load_gate(cfg)).answer(args.question, args.mode)
    print(result.text)
    print(
        f"\n[{result.status}"
        + (f": {result.refusal_reason}" if result.refusal_reason else "")
        + f" | backend={result.backend} fallback={result.fallback} top_sim={result.top_similarity}]"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
