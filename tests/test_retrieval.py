"""Retrieval and metric tests with a fake embedder (no model download, no network)."""

import pytest

from conftest import CHUNKS, FakeEmbedder
from fincrime_rag.evaluation.retrieval_metrics import RankedHit, question_metrics, summarize
from fincrime_rag.retrieval import Retriever, rrf_fuse, tokenize
from fincrime_rag.vectorstore import build_index


@pytest.fixture()
def retriever(index_cfg):
    build_index(index_cfg, FakeEmbedder(), CHUNKS)
    return Retriever(index_cfg, FakeEmbedder())


def test_tokenize_keeps_section_numbers():
    assert "1005.11" in tokenize("See § 1005.11, error resolution.")


def test_rrf_prefers_items_ranked_by_both():
    fused = rrf_fuse([["x", "y", "z"], ["z", "x", "w"]], rrf_k=60)
    assert [i for i, _ in fused][:2] == ["x", "z"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_modes_return_ranked_hits_with_metadata(retriever, mode):
    hits = retriever.retrieve("elder financial exploitation red flags", 3, mode)
    assert hits and [h.rank for h in hits] == list(range(1, len(hits) + 1))
    top = hits[0]
    assert (top.doc_id, top.pdf_pages) == ("a", (2, 3))
    assert top.printed_page_guess == "3" and top.mode == mode and top.text


def test_bm25_matches_exact_section_number(retriever):
    assert retriever.retrieve("1005.11", 1, "bm25")[0].section == "1005.11"


def test_unknown_mode_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("q", 3, "nope")


def test_default_mode_comes_from_config(retriever):
    assert retriever.retrieve("mule accounts")[0].mode == retriever.cfg.retrieval.mode


# ---------------------------------------------------------------- metrics
def test_question_metrics_pages_and_docs():
    gold = frozenset({("a", 2), ("a", 7), ("b", 1)})
    hits = [RankedHit("c", (1,)), RankedHit("a", (2, 3)), RankedHit("b", (9,)), RankedHit("a", (7,))]
    m = question_metrics(hits, gold, [1, 3, 5], hit_k=3)
    assert m["any@1"] == 0.0 and m["any@3"] == 1.0
    assert m["frac@3"] == pytest.approx(1 / 3) and m["frac@5"] == pytest.approx(2 / 3)
    assert m["mrr"] == 0.5
    assert m["hit"] == 1.0  # doc "a" within top 3


def test_question_metrics_miss_and_summary():
    gold = frozenset({("a", 1)})
    miss = question_metrics([RankedHit("a", (9,))], gold, [1], hit_k=1)
    assert miss["any@1"] == 0.0 and miss["mrr"] == 0.0 and miss["hit"] == 1.0
    ok = question_metrics([RankedHit("a", (1,))], gold, [1], hit_k=1)
    summary = summarize([("lookup", miss), ("lookup", ok), ("list", ok)])
    assert summary["overall"][0] == 3 and summary["lookup"][0] == 2
    assert summary["lookup"][1]["any@1"] == 0.5
    assert list(summary) == ["overall", "list", "lookup"]
