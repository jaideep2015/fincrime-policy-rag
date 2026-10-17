"""Embedding cache and index build tests with a fake embedder (no model download, no network)."""

import numpy as np

from conftest import CHUNKS, FakeEmbedder
from fincrime_rag.embeddings import CachedEmbedder
from fincrime_rag.vectorstore import build_index, chunk_from_record, chunk_to_metadata, get_collection


def test_index_rebuild_is_idempotent(index_cfg):
    build_index(index_cfg, FakeEmbedder(), CHUNKS)
    build_index(index_cfg, FakeEmbedder(), CHUNKS)
    collection = get_collection(index_cfg)
    assert collection.count() == len(CHUNKS)
    assert sorted(collection.get()["ids"]) == sorted(c["chunk_id"] for c in CHUNKS)


def test_embedding_cache_skips_recompute(tmp_path):
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    first = cached.embed_documents(["alpha beta", "gamma"])
    second = cached.embed_documents(["alpha beta", "gamma"])
    assert inner.calls == 2 and cached.hits == 2
    assert np.allclose(first, second)


def test_metadata_round_trip_keeps_doc_and_pages():
    original = CHUNKS[1]
    back = chunk_from_record(original["chunk_id"], original["text"], chunk_to_metadata(original))
    assert (back["doc_id"], back["pdf_pages"], back["printed_page_guess"]) == ("a", [2, 3], "3")
    assert back["section"] is None
