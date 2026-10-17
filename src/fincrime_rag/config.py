"""Load and validate config.yaml."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    reports_dir: Path


class IngestionConfig(_Strict):
    allowed_download_hosts: list[str]
    min_pdf_bytes: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    user_agent: str


class OCRConfig(_Strict):
    enabled: bool = True
    engine: Literal["rapidocr"] = "rapidocr"
    dpi: int = Field(gt=0)
    recover_box_thresh: float = Field(gt=0.0, le=1.0)
    recover_text_score: float = Field(ge=0.0, le=1.0)
    recover_unclip_ratio: float = Field(gt=0.0)
    coverage_threshold: float = Field(gt=0.0, le=1.0)
    coverage_min_ocr_words: int = Field(ge=0)
    merge_covered_fraction: float = Field(gt=0.0, le=1.0)
    merge_min_layer_words: int = Field(ge=0)
    merge_min_box_coverage: float = Field(ge=0.0, le=1.0)
    min_confidence: float = Field(ge=0.0, le=1.0)
    icon_gutter_fraction: float = Field(ge=0.0, le=1.0)
    icon_max_chars: int = Field(ge=0)
    icon_max_confidence: float = Field(ge=0.0, le=1.0)
    paragraph_gap_factor: float = Field(gt=0.0)


class PageTag(_Strict):
    doc_id: str
    first_page: int = Field(ge=1)
    last_page: int = Field(ge=1)
    content_type: str

    @model_validator(mode="after")
    def _ordered(self) -> PageTag:
        if self.last_page < self.first_page:
            raise ValueError("last_page must be >= first_page")
        return self


class IndexingConfig(_Strict):
    exclude_content_types: list[str] = []


class CleaningConfig(_Strict):
    header_footer_scan_lines: int = Field(gt=0)
    page_number_scan_lines: int = Field(gt=0)
    repeat_fraction: float = Field(gt=0.0, le=1.0)
    min_pages_for_detection: int = Field(gt=1)
    strip_edge_page_numbers: bool = True
    running_head_scan_lines: int = Field(gt=0)
    running_head_patterns: list[str] = []
    printed_page_head_patterns: list[str] = []
    page_tags: list[PageTag] = []

    @field_validator("running_head_patterns", "printed_page_head_patterns")
    @classmethod
    def _valid_regexes(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return patterns


class QAConfig(_Strict):
    empty_page_max_chars: int = Field(ge=0)
    flag_empty_page_share: float = Field(ge=0.0, le=1.0)
    flag_empty_page_min_count: int = Field(ge=1)
    flag_bad_char_ratio: float = Field(ge=0.0, le=1.0)
    flag_odd_token_ratio: float = Field(ge=0.0, le=1.0)
    odd_token_ignore_acronyms: bool = True
    flag_min_chars_per_page: int = Field(ge=0)


class EvalSetConfig(_Strict):
    max_questions_per_doc: int = Field(gt=0)
    min_key_facts: int = Field(ge=1)
    max_key_facts: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> EvalSetConfig:
        if self.max_key_facts < self.min_key_facts:
            raise ValueError("max_key_facts must be >= min_key_facts")
        return self


class ChunkingConfig(_Strict):
    strategy: Literal["recursive", "section_aware"] = "recursive"
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    special_tokens: int = Field(ge=0)
    separators: list[str]
    min_chunk_tokens: int = Field(ge=0)
    section_doc_prefixes: list[str] = []
    section_heading_patterns: list[str] = []

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size - self.special_tokens:
            raise ValueError("chunk_overlap must be smaller than chunk_size - special_tokens")
        return self

    @field_validator("section_heading_patterns")
    @classmethod
    def _valid_patterns(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
            if "section" not in compiled.groupindex:
                raise ValueError(f"pattern {pattern!r} needs a named group `section`")
        return patterns


class EmbeddingConfig(_Strict):
    provider: Literal["local"] = "local"
    model: str
    batch_size: int = Field(gt=0)
    max_seq_length: int = Field(gt=0)
    normalize: bool = True
    query_instruction: str = ""


class VectorStoreConfig(_Strict):
    backend: Literal["chroma"] = "chroma"
    collection: str
    space: Literal["cosine", "l2", "ip"] = "cosine"


class RetrievalConfig(_Strict):
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = Field(gt=0)
    hybrid_rrf_k: int = Field(gt=0)
    candidate_pool: int = Field(gt=0)
    bm25_k1: float = Field(gt=0.0)
    bm25_b: float = Field(ge=0.0, le=1.0)


class RetrievalEvalConfig(_Strict):
    retrieval_ks: list[int]
    splits: list[Literal["dev", "test"]]

    @field_validator("retrieval_ks")
    @classmethod
    def _positive_sorted(cls, ks: list[int]) -> list[int]:
        if not ks or any(k < 1 for k in ks) or ks != sorted(set(ks)):
            raise ValueError("retrieval_ks must be a sorted list of distinct positive integers")
        return ks


class LLMConfig(_Strict):
    provider: Literal["ollama", "anthropic"] = "ollama"
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0)
    ollama_host: str = "http://localhost:11434"


class Config(_Strict):
    paths: PathsConfig
    ingestion: IngestionConfig
    ocr: OCRConfig
    cleaning: CleaningConfig
    indexing: IndexingConfig
    qa: QAConfig
    evalset: EvalSetConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    retrieval: RetrievalConfig
    evaluation: RetrievalEvalConfig
    llm: LLMConfig
    seed: int = 42


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML file and return a validated Config. Raises on unknown or invalid fields."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
