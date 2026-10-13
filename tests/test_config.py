from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fincrime_rag.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def _raw() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_repo_config_loads():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.chunking.chunk_size == 512
    assert cfg.chunking.chunk_overlap == 64
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"
    assert cfg.vectorstore.backend == "chroma"


def test_default_llm_is_local_ollama():
    assert load_config(ROOT / "config.yaml").llm.provider == "ollama"


def test_overlap_must_be_smaller_than_size():
    raw = _raw()
    raw["chunking"]["chunk_overlap"] = raw["chunking"]["chunk_size"]
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_unknown_key_rejected():
    raw = _raw()
    raw["retrieval"]["tpo_k"] = 3
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
