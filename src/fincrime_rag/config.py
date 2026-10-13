"""Load and validate config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path("config.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    manifest: Path
    raw_dir: Path
    extracted_dir: Path
    processed_dir: Path
    eval_set: Path
    index_dir: Path
    cache_dir: Path
    experiments_dir: Path


class ChunkingConfig(_Strict):
    strategy: Literal["recursive", "section_aware"] = "recursive"
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class EmbeddingConfig(_Strict):
    provider: Literal["local"] = "local"
    model: str
    batch_size: int = Field(gt=0)


class VectorStoreConfig(_Strict):
    backend: Literal["chroma"] = "chroma"
    collection: str


class RetrievalConfig(_Strict):
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = Field(gt=0)
    hybrid_rrf_k: int = Field(gt=0)


class LLMConfig(_Strict):
    provider: Literal["ollama", "anthropic"] = "ollama"
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0)
    ollama_host: str = "http://localhost:11434"


class Config(_Strict):
    paths: PathsConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    seed: int = 42


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML file and return a validated Config. Raises on unknown or invalid fields."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
