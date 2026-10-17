"""Phase 3: embedder interface and an on-disk cache keyed by input hash.

``Embedder`` is the interface the rest of the pipeline uses. ``SentenceTransformerEmbedder`` runs
BAAI/bge-small-en-v1.5 (or any model named in ``embedding.model``) on CPU. ``CachedEmbedder`` wraps
any embedder: vectors are stored under ``paths.cache_dir/embeddings/`` keyed by
sha256(model + text), so a rebuild with unchanged chunks never re-encodes anything.

bge models want an instruction prefix on *queries* only (``embedding.query_instruction``); passages
are embedded as they are.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from fincrime_rag.config import Config


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, cfg: Config):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.manual_seed(cfg.seed)
        self.name = cfg.embedding.model
        self._cfg = cfg.embedding
        self._model = SentenceTransformer(cfg.embedding.model, device="cpu")
        self._model.max_seq_length = cfg.embedding.max_seq_length

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._cfg.batch_size,
            normalize_embeddings=self._cfg.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._cfg.query_instruction + text])[0]


def text_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class CachedEmbedder:
    """Disk cache for document embeddings (one .npy file per input hash, grouped by model)."""

    def __init__(self, inner: Embedder, cache_dir: Path):
        self.inner = inner
        self.name = inner.name
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", inner.name)
        self._dir = cache_dir / "embeddings" / slug
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.npy"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        keys = [text_key(self.name, t) for t in texts]
        vectors: list[np.ndarray | None] = []
        missing: list[int] = []
        for i, key in enumerate(keys):
            path = self._path(key)
            if path.exists():
                vectors.append(np.load(path))
                self.hits += 1
            else:
                vectors.append(None)
                missing.append(i)
        if missing:
            fresh = self.inner.embed_documents([texts[i] for i in missing])
            for i, vector in zip(missing, fresh, strict=True):
                path = self._path(keys[i])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, vector)
                vectors[i] = vector
            self.misses += len(missing)
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)


def build_embedder(cfg: Config) -> CachedEmbedder:
    return CachedEmbedder(SentenceTransformerEmbedder(cfg), cfg.paths.cache_dir)


