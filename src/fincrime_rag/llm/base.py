"""LLMClient interface shared by every generation backend.

A backend receives a ``GenerationRequest`` (the assembled prompt for chat models, plus the question and
the retrieved chunks for backends that do not use a prompt, such as the extractive one) and returns a
``LLMResponse`` that always records which backend produced the text, so results are traceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMUnavailable(RuntimeError):
    """The backend cannot be reached (server down, model not pulled)."""


@dataclass(frozen=True)
class GenerationRequest:
    system: str
    user: str
    question: str
    chunks: list[dict]  # retrieved chunks, best first (chunk_id, doc_id, pdf_pages, text, ...)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    backend: str  # "ollama" | "extractive"
    model: str | None = None
    model_digest: str | None = None
    num_ctx: int | None = None
    fallback: bool = False  # True when this backend replaced an unreachable primary
    fallback_reason: str | None = None
    cached: bool = False
    prompt_eval_count: int | None = None  # tokens Ollama actually evaluated for the prompt


class LLMClient(Protocol):
    backend: str

    def generate(self, request: GenerationRequest) -> LLMResponse: ...
