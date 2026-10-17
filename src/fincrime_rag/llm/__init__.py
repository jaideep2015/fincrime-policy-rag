"""LLM backends behind the ``LLMClient`` interface."""

from __future__ import annotations

from fincrime_rag.llm.base import GenerationRequest, LLMClient, LLMResponse, LLMUnavailable


class FallbackClient:
    """Use ``primary``; if it is unreachable, use ``fallback`` and record that it happened."""

    def __init__(self, primary: LLMClient, fallback: LLMClient):
        self.primary = primary
        self.fallback = fallback
        self.backend = primary.backend

    def generate(self, request: GenerationRequest) -> LLMResponse:
        try:
            return self.primary.generate(request)
        except LLMUnavailable as exc:
            resp = self.fallback.generate(request)
            return LLMResponse(
                text=resp.text,
                backend=resp.backend,
                model=resp.model,
                model_digest=resp.model_digest,
                num_ctx=resp.num_ctx,
                fallback=True,
                fallback_reason=str(exc),
            )


__all__ = ["FallbackClient", "GenerationRequest", "LLMClient", "LLMResponse", "LLMUnavailable"]
