"""Ollama backend: a small local chat model, temperature 0, fixed seed, explicit context window.

Every call sends ``num_ctx`` in the Ollama options (the server default is much smaller and silently
truncates long prompts). The model digest is read from ``/api/tags`` (the same value ``ollama list``
shows) and returned with every response. Outputs are cached on disk keyed by a hash of the model,
digest, options and prompt, so a repeated evaluation makes no model calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

from fincrime_rag.config import Config
from fincrime_rag.llm.base import GenerationRequest, LLMResponse, LLMUnavailable

AVAILABILITY_TIMEOUT_SECONDS = 3


class OllamaClient:
    backend = "ollama"

    def __init__(self, cfg: Config, session: requests.Session | None = None):
        if not cfg.llm.model:
            raise ValueError("llm.model must name an Ollama model")
        self._cfg = cfg.llm
        self._seed = cfg.seed
        self._host = cfg.llm.ollama_host.rstrip("/")
        self._http = session or requests.Session()
        self._cache_dir = Path(cfg.paths.cache_dir) / "llm"
        self._digest: str | None = None
        self.model = cfg.llm.model

    # ---- availability -------------------------------------------------
    def check_available(self) -> str:
        """Return the model digest, or raise ``LLMUnavailable`` (server down or model not pulled)."""
        if self._digest is not None:
            return self._digest
        try:
            resp = self._http.get(f"{self._host}/api/tags", timeout=AVAILABILITY_TIMEOUT_SECONDS)
            resp.raise_for_status()
            models = resp.json().get("models", [])
        except (requests.RequestException, ValueError) as exc:
            raise LLMUnavailable(f"Ollama not reachable at {self._host}: {type(exc).__name__}") from exc
        for entry in models:
            if entry.get("name") == self.model or entry.get("model") == self.model:
                self._digest = str(entry.get("digest", ""))
                return self._digest
        raise LLMUnavailable(f"model {self.model} is not pulled (ollama pull {self.model})")

    # ---- generation ---------------------------------------------------
    def _options(self) -> dict:
        return {
            "temperature": self._cfg.temperature,
            "seed": self._seed,
            "num_ctx": self._cfg.num_ctx,
            "num_predict": self._cfg.max_output_tokens,
        }

    def _cache_path(self, digest: str, request: GenerationRequest) -> Path:
        payload = json.dumps(
            [self.model, digest, self._options(), request.system, request.user], sort_keys=True
        )
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self._cache_dir / key[:2] / f"{key}.json"

    def generate(self, request: GenerationRequest) -> LLMResponse:
        digest = self.check_available()
        path = self._cache_path(digest, request)
        if path.exists():
            text = json.loads(path.read_text(encoding="utf-8"))["text"]
            return self._response(text, digest, cached=True)
        body = {
            "model": self.model,
            "stream": False,
            "options": self._options(),
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        try:
            resp = self._http.post(
                f"{self._host}/api/chat", json=body, timeout=self._cfg.request_timeout_seconds
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMUnavailable(f"Ollama request failed: {type(exc).__name__}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text}), encoding="utf-8")
        return self._response(text, digest, cached=False)

    def _response(self, text: str, digest: str, cached: bool) -> LLMResponse:
        return LLMResponse(
            text=text,
            backend=self.backend,
            model=self.model,
            model_digest=digest,
            num_ctx=self._cfg.num_ctx,
            cached=cached,
        )
