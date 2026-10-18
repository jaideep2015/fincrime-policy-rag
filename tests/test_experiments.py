"""Experiment runner tests on a tiny fixture: fake embedder, word-count tokenizer, no model download."""

import csv

import numpy as np

from conftest import CHUNKS, FakeEmbedder
from fincrime_rag.config import ChunkSpec
from fincrime_rag.experiments import (
    ExactRetriever,
    build_configuration,
    conclusion_text,
    paired_difference,
    reg_e_share,
    run_grid,
    summarize_grid,
)
from fincrime_rag.extraction import write_pages
from fincrime_rag.ingestion import MANIFEST_FIELDS


def words(text: str) -> int:
    return len(text.split())


def exact(cfg, chunks=CHUNKS):
    emb = FakeEmbedder()
    return ExactRetriever(cfg, chunks, emb.embed_documents([c["text"] for c in chunks]), emb)


def test_exact_dense_is_deterministic_and_breaks_ties_by_chunk_id(cfg):
    twin_a = {**CHUNKS[0], "chunk_id": "z_twin", "text": "identical words here"}
    twin_b = {**CHUNKS[0], "chunk_id": "a_twin", "text": "identical words here"}
    r1 = exact(cfg, [twin_a, twin_b, *CHUNKS[1:]])
    r2 = exact(cfg, [twin_b, *CHUNKS[1:], twin_a])  # different input order
    top1 = [c["chunk_id"] for c in r1.retrieve("identical words here", 2, "dense")]
    top2 = [c["chunk_id"] for c in r2.retrieve("identical words here", 2, "dense")]
    assert top1 == top2 == ["a_twin", "z_twin"]


def test_modes_return_ranked_chunks_with_pages(cfg):
    r = exact(cfg)
    for mode in ("dense", "bm25", "hybrid"):
        hits = r.retrieve("elder financial exploitation red flags", 3, mode)
        assert hits[0]["doc_id"] == "a" and hits[0]["pdf_pages"] == [2, 3]


def test_hybrid_matches_pipeline_hybrid_when_dense_is_exact(index_cfg):
    from fincrime_rag.retrieval import Retriever
    from fincrime_rag.vectorstore import build_index

    emb = FakeEmbedder()
    build_index(index_cfg, emb, CHUNKS)
    pipeline = Retriever(index_cfg, emb)
    ex = exact(index_cfg)
    for q in ("money mule accounts", "suspicious activity reports thirty days", "error resolution 1005.11"):
        a = [h.chunk_id for h in pipeline.retrieve(q, 4, "hybrid")]
        b = [c["chunk_id"] for c in ex.retrieve(q, 4, "hybrid")]
        assert a == b  # tiny index: approximate and exact search agree


def _write_corpus(cfg):
    rows = [
        {"doc_id": "cfr_12_1005_reg_e", "source": "manual"},
        {"doc_id": "other", "source": "manual"},
    ]
    with open(cfg.paths.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    for doc in ("cfr_12_1005_reg_e", "other"):
        pages = [
            {
                "doc_id": doc,
                "pdf_page": p,
                "printed_page_guess": str(p),
                "section_hint": None,
                "content_type": "body",
                "text": " ".join(f"{doc[:3]}{p}w{i}" for i in range(60)),
            }
            for p in (1, 2, 3)
        ]
        write_pages(cfg.paths.processed_dir / f"{doc}.pages.jsonl", pages)


def test_build_configuration_is_idempotent_and_reports_reg_e_share(cfg, tmp_path):
    _write_corpus(cfg)
    emb = FakeEmbedder()
    spec = ChunkSpec(chunk_size=40, chunk_overlap=8)
    chunks, vectors = build_configuration(cfg, spec, emb, words, tmp_path / "work")
    assert len(chunks) == len(vectors) > 2 and all(c["pdf_pages"] for c in chunks)
    calls = emb.calls
    again, _ = build_configuration(cfg, spec, emb, words, tmp_path / "work")
    assert emb.calls == calls and [c["chunk_id"] for c in again] == [c["chunk_id"] for c in chunks]
    assert 0.0 < reg_e_share(chunks) < 1.0
    assert (tmp_path / "work" / spec.name / "vectors.npy").exists()


def test_grid_summary_paired_differences_and_conclusion(cfg, tmp_path):
    _write_corpus(cfg)
    small = cfg.model_copy(
        update={
            "experiments": cfg.experiments.model_copy(
                update={
                    "configs": [
                        ChunkSpec(chunk_size=40, chunk_overlap=8),
                        ChunkSpec(chunk_size=80, chunk_overlap=8),
                    ],
                    "baseline": ChunkSpec(chunk_size=80, chunk_overlap=8),
                    "ks": [1, 3, 5, 8, 10],
                }
            ),
            "generation_eval": cfg.generation_eval.model_copy(update={"bootstrap_samples": 200}),
        }
    )
    questions = [
        ("oth1w3 oth1w4", frozenset({("other", 1)})),
        ("cfr2w5 cfr2w6", frozenset({("cfr_12_1005_reg_e", 2)})),
    ]
    emb = FakeEmbedder()
    grid = run_grid(small, lambda cache: emb, words, questions, tmp_path / "work")
    again = run_grid(small, lambda cache: emb, words, questions, tmp_path / "work")
    assert grid["configs"].keys() == again["configs"].keys()
    assert grid["configs"]["c80_o8"]["rows"] == again["configs"]["c80_o8"]["rows"]  # deterministic
    summary = summarize_grid(small, grid)
    base_hybrid = summary["configs"]["c80_o8"]["modes"]["hybrid"]["paired_vs_baseline_hybrid"]
    assert all(v == (0.0, 0.0, 0.0) for v in base_hybrid.values())  # baseline against itself
    text = conclusion_text(small, summary)
    assert "default stays 80/8 hybrid" in text and "adopted" in text


def test_paired_difference_uses_the_same_questions():
    rows = [{"m": 1.0}, {"m": 0.0}, {"m": 1.0}]
    base = [{"m": 0.0}, {"m": 0.0}, {"m": 1.0}]
    est, lo, hi = paired_difference(rows, base, "m", 200, 0.95, 1)
    assert abs(est - 1 / 3) < 1e-9 and lo <= est <= hi
    assert np.isfinite([est, lo, hi]).all()
