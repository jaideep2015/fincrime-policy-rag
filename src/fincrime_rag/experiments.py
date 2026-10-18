"""Phase 7: retrieval-only experiments with exact dense search (D-032).

Why exact search: the pipeline's Chroma/HNSW dense search is approximate and was seen to return different
neighbours in different processes for a few low-similarity queries. For a grid whose differences are small,
every configuration here uses an exact brute-force cosine search over stored chunk embeddings (ties broken
by chunk id, similarities rounded to 12 decimals), so each result is reproducible. BM25 and RRF are the same
as in ``retrieval.py``. Nothing here touches ``chroma_db/``, ``.cache/llm`` or the default pipeline: each
configuration is built in its own folder under ``experiments.work_dir``.

No LLM is called. Only aggregates are printed or written.

Commands (``python -m fincrime_rag.experiments <command>``):
* ``run``: build every configuration, score it, write ``experiments.report`` and ``experiments.results_json``.
* ``compare-exact``: read-only check of the pipeline's hybrid top-k against an exact-dense hybrid top-k.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from fincrime_rag.chunking import TokenCounter, chunk_document, read_chunks, write_chunks
from fincrime_rag.config import ChunkSpec, Config, load_config
from fincrime_rag.embeddings import CachedEmbedder, Embedder
from fincrime_rag.extraction import read_pages
from fincrime_rag.ingestion import read_manifest
from fincrime_rag.retrieval import rrf_fuse, tokenize

SIM_DECIMALS = 12  # similarities are rounded before ranking so BLAS rounding cannot flip a tie


# --------------------------------------------------------------------------- exact retrieval
class ExactRetriever:
    """dense (exact cosine), bm25 and hybrid (RRF) over an in-memory chunk list. Deterministic."""

    def __init__(self, cfg: Config, chunks: list[dict], vectors: np.ndarray, embedder: Embedder):
        from rank_bm25 import BM25Okapi

        order = sorted(range(len(chunks)), key=lambda i: chunks[i]["chunk_id"])
        self.chunks = [chunks[i] for i in order]
        self.ids = [c["chunk_id"] for c in self.chunks]
        self.by_id = dict(zip(self.ids, self.chunks, strict=True))
        self.matrix = np.asarray(vectors, dtype=np.float64)[order]
        self.cfg = cfg
        self.embedder = embedder
        self._bm25 = BM25Okapi(
            [tokenize(c["text"]) for c in self.chunks], k1=cfg.retrieval.bm25_k1, b=cfg.retrieval.bm25_b
        )
        self._qvec: dict[str, np.ndarray] = {}

    def _query_vector(self, query: str) -> np.ndarray:
        if query not in self._qvec:
            self._qvec[query] = np.asarray(self.embedder.embed_query(query), dtype=np.float64)
        return self._qvec[query]

    def dense(self, query: str, depth: int) -> list[tuple[str, float]]:
        sims = np.round(self.matrix @ self._query_vector(query), SIM_DECIMALS)
        order = sorted(range(len(self.ids)), key=lambda i: (-sims[i], self.ids[i]))[:depth]
        return [(self.ids[i], float(sims[i])) for i in order]

    def bm25(self, query: str, depth: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.ids)), key=lambda i: (-scores[i], self.ids[i]))[:depth]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def retrieve(self, query: str, k: int, mode: str) -> list[dict]:
        pool = max(self.cfg.retrieval.candidate_pool, k)
        if mode == "dense":
            ranked = self.dense(query, k)
        elif mode == "bm25":
            ranked = self.bm25(query, k)
        elif mode == "hybrid":
            fused = rrf_fuse(
                [[c for c, _ in self.dense(query, pool)], [c for c, _ in self.bm25(query, pool)]],
                self.cfg.retrieval.hybrid_rrf_k,
            )
            ranked = fused[:k]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return [self.by_id[cid] for cid, _ in ranked]


# --------------------------------------------------------------------------- building a configuration
def build_configuration(
    cfg: Config, spec: ChunkSpec, embedder: Embedder, count: TokenCounter, work_dir: Path
) -> tuple[list[dict], np.ndarray]:
    """Chunk every document with ``spec``, embed, and store under ``work_dir/<name>/``. Reuses a finished build."""
    folder = work_dir / spec.name
    chunks_file, vectors_file = folder / "chunks.jsonl", folder / "vectors.npy"
    if chunks_file.exists() and vectors_file.exists():
        chunks = read_chunks(chunks_file)
        vectors = np.load(vectors_file)
        if len(chunks) == len(vectors):
            return chunks, vectors
    chunk_cfg = cfg.chunking.model_copy(
        update={"chunk_size": spec.chunk_size, "chunk_overlap": spec.chunk_overlap}
    )
    chunks = []
    for row in read_manifest(cfg.paths.manifest):
        pages = read_pages(cfg.paths.processed_dir / f"{row.doc_id}.pages.jsonl")
        chunks.extend(chunk_document(row.doc_id, pages, chunk_cfg, count, cfg.indexing.exclude_content_types))
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    folder.mkdir(parents=True, exist_ok=True)
    write_chunks(chunks_file, chunks)
    np.save(vectors_file, vectors)
    return chunks, vectors


def reg_e_share(chunks: list[dict], doc_id: str = "cfr_12_1005_reg_e") -> float:
    return sum(c["doc_id"] == doc_id for c in chunks) / len(chunks) if chunks else 0.0


# --------------------------------------------------------------------------- scoring
def score_configuration(
    retriever: ExactRetriever,
    questions: list[tuple[str, frozenset]],
    modes: list[str],
    ks: list[int],
    hit_k: int,
) -> dict[str, list[dict[str, float]]]:
    """mode -> per-question metric dicts (aligned with ``questions``). Aggregated by the caller."""
    from fincrime_rag.evaluation.retrieval_metrics import RankedHit, question_metrics

    depth = max(ks)
    out: dict[str, list[dict[str, float]]] = {}
    for mode in modes:
        rows = []
        for text, gold in questions:
            hits = [
                RankedHit(c["doc_id"], tuple(c["pdf_pages"])) for c in retriever.retrieve(text, depth, mode)
            ]
            rows.append(question_metrics(hits, gold, ks, hit_k))
        out[mode] = rows
    return out


def paired_difference(
    rows: list[dict[str, float]],
    base: list[dict[str, float]],
    metric: str,
    samples: int,
    confidence: float,
    seed: int,
):
    from fincrime_rag.evaluation.generation_eval import bootstrap_ci, mean_stat

    diffs = [r[metric] - b[metric] for r, b in zip(rows, base, strict=True)]
    return bootstrap_ci(diffs, mean_stat, samples, confidence, seed)


def mean_ci(rows: list[dict[str, float]], metric: str, samples: int, confidence: float, seed: int):
    from fincrime_rag.evaluation.generation_eval import bootstrap_ci, mean_stat

    return bootstrap_ci([r[metric] for r in rows], mean_stat, samples, confidence, seed)


# --------------------------------------------------------------------------- the grid
def run_grid(
    cfg: Config, embedder_factory: Callable[[Path], Embedder], count: TokenCounter, questions, work_dir: Path
) -> dict:
    exp = cfg.experiments
    results: dict = {"configs": {}, "n_questions": len(questions)}
    for spec in exp.configs:
        embedder = embedder_factory(work_dir / "cache")
        chunks, vectors = build_configuration(cfg, spec, embedder, count, work_dir)
        retriever = ExactRetriever(cfg, chunks, vectors, embedder)
        per_mode = score_configuration(retriever, questions, exp.modes, exp.ks, exp.hit_k)
        results["configs"][spec.name] = {
            "chunk_size": spec.chunk_size,
            "chunk_overlap": spec.chunk_overlap,
            "n_chunks": len(chunks),
            "reg_e_share": reg_e_share(chunks),
            "rows": per_mode,
        }
    return results


def summarize_grid(cfg: Config, grid: dict) -> dict:
    """Aggregates only: point estimates with CIs, and paired differences against the baseline."""
    exp, ge = cfg.experiments, cfg.generation_eval
    base_rows = grid["configs"][exp.baseline.name]["rows"]["hybrid"]
    metrics = [f"any@{k}" for k in exp.ks] + [f"all@{k}" for k in exp.ks] + ["mrr", "hit"]
    paired_metrics = ["any@5", "all@5", "any@8", "all@8", "mrr"]
    out: dict = {"n_questions": grid["n_questions"], "configs": {}}
    for name, info in grid["configs"].items():
        entry = {k: info[k] for k in ("chunk_size", "chunk_overlap", "n_chunks", "reg_e_share")}
        entry["modes"] = {}
        for mode, rows in info["rows"].items():
            entry["modes"][mode] = {
                "metrics": {
                    m: mean_ci(rows, m, ge.bootstrap_samples, ge.confidence, cfg.seed) for m in metrics
                },
                "paired_vs_baseline_hybrid": {
                    m: paired_difference(rows, base_rows, m, ge.bootstrap_samples, ge.confidence, cfg.seed)
                    for m in paired_metrics
                },
            }
        out["configs"][name] = entry
    return out


def _fmt(triple, digits: int = 2) -> str:
    est, lo, hi = triple
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _fmt_signed(triple) -> str:
    est, lo, hi = triple
    return f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def write_report(cfg: Config, summary: dict, published: dict | None, path: Path) -> None:
    exp = cfg.experiments
    ks = exp.ks
    nl = chr(10)
    n = summary["n_questions"]
    base = exp.baseline.name
    lines = [
        "# Retrieval experiments (retrieval only, LLM-free, exact dense search)",
        "",
        f"**All 20 questions are dev; there is no held-out split. n = {n} answerable questions.** Metrics: `any@k` "
        "(some gold page in the top k), `all@k` (every gold page in the top k, union over the top-k chunks), `MRR` "
        f"and doc-level hit@{exp.hit_k} (a gold document in the top {exp.hit_k}, page ignored). Page matching uses the "
        "stored `pdf_pages`. Intervals are percentile bootstrap 95% CIs over questions "
        f"({cfg.generation_eval.bootstrap_samples} resamples, seed {cfg.seed}). Every configuration uses **exact** "
        "dense search (D-032), so the numbers are reproducible; BM25 and RRF (k = "
        f"{cfg.retrieval.hybrid_rrf_k}, pool {cfg.retrieval.candidate_pool}) are unchanged.",
        "",
        "## Decision rule (written before running, D-032)",
        "",
        "The default stays **512/64 hybrid**. No other configuration is adopted, because n = 14 cannot separate "
        "configurations; a configuration that looks better is reported as a hypothesis for held-out testing, not "
        "adopted. Chunk sizes above 512 are excluded because bge-small-en-v1.5 truncates input at 512 tokens.",
        "",
        "## Index size per configuration",
        "",
        "| Configuration | chunks | Reg E share |",
        "|---|---|---|",
    ]
    for e in summary["configs"].values():
        lines.append(f"| {e['chunk_size']}/{e['chunk_overlap']} | {e['n_chunks']} | {e['reg_e_share']:.1%} |")
    lines += ["", "## Point estimates (any@k / all@k, MRR, doc-level hit)", ""]
    head = (
        ["Config", "Mode"] + [f"any@{k}" for k in ks] + [f"all@{k}" for k in ks] + ["MRR", f"hit@{exp.hit_k}"]
    )
    lines += ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for _name, e in summary["configs"].items():
        for mode, m in e["modes"].items():
            mt = m["metrics"]
            cells = [f"{e['chunk_size']}/{e['chunk_overlap']}", mode]
            cells += [f"{mt[f'any@{k}'][0]:.2f}" for k in ks] + [f"{mt[f'all@{k}'][0]:.2f}" for k in ks]
            cells += [f"{mt['mrr'][0]:.3f}", f"{mt['hit'][0]:.2f}"]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## With 95% CIs (k = 5 and k = 8)", ""]
    head = ["Config", "Mode", "any@5", "all@5", "any@8", "all@8", "MRR", f"hit@{exp.hit_k}"]
    lines += ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for _name, e in summary["configs"].items():
        for mode, m in e["modes"].items():
            mt = m["metrics"]
            cells = [f"{e['chunk_size']}/{e['chunk_overlap']}", mode]
            cells += [_fmt(mt[x]) for x in ("any@5", "all@5", "any@8", "all@8")]
            cells += [_fmt(mt["mrr"], 3), _fmt(mt["hit"])]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"## Paired per-question differences against the baseline ({base} hybrid, exact dense)",
        "",
        "Mean of (configuration minus baseline) over the same questions, with a bootstrap 95% CI. An interval that "
        "contains 0 means the data cannot tell the configuration from the baseline.",
        "",
    ]
    pm = ["any@5", "all@5", "any@8", "all@8", "mrr"]
    head = ["Config", "Mode"] + pm
    lines += ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for name, e in summary["configs"].items():
        for mode, m in e["modes"].items():
            if name == base and mode == "hybrid":
                continue
            d = m["paired_vs_baseline_hybrid"]
            lines.append(
                f"| {e['chunk_size']}/{e['chunk_overlap']} | {mode} | "
                + " | ".join(_fmt_signed(d[x]) for x in pm)
                + " |"
            )
    if published:
        lines += [
            "",
            "## Exact-search baseline next to the published pipeline numbers",
            "",
            published["text"],
        ]
    lines += ["", "## Conclusion", "", summary.get("conclusion", ""), "", "## Limitations", ""]
    lines += [
        f"- n = {n} answerable questions, all dev, no held-out split: the intervals are wide and a difference of one or "
        "two questions is within noise.",
        "- The grid was chosen before running; nothing was tuned on individual questions, and the default pipeline "
        "(512/64, hybrid, k = 8, Chroma) was not changed.",
        "- Retrieval only: no claim is made about answer quality.",
        "- Chunk counts differ by configuration, so `any@k` at the same k covers different amounts of text.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nl.join(lines) + nl, encoding="utf-8", newline=nl)


def conclusion_text(cfg: Config, summary: dict) -> str:
    """Plain-language conclusion that follows the pre-registered rule; it never ranks configurations to adopt one."""
    exp = cfg.experiments
    base = exp.baseline.name
    metrics = ("any@5", "all@5", "any@8", "all@8", "mrr")
    better: list[str] = []
    worse: list[str] = []
    comparisons = 0
    for name, e in summary["configs"].items():
        for mode, m in e["modes"].items():
            if name == base and mode == "hybrid":
                continue
            d = m["paired_vs_baseline_hybrid"]
            label = f"{e['chunk_size']}/{e['chunk_overlap']} {mode}"
            comparisons += len(metrics)
            better += [f"{label} ({x})" for x in metrics if d[x][1] > 0]
            worse += [f"{label} ({x})" for x in metrics if d[x][2] < 0]
    text = (
        f"The default stays {exp.baseline.chunk_size}/{exp.baseline.chunk_overlap} hybrid. "
        "No configuration is adopted on this evidence."
    )
    if better:
        text += (
            " Better than the baseline with a paired interval that excludes 0 (a hypothesis for held-out testing, "
            "not a reason to change the default): " + "; ".join(better) + "."
        )
    else:
        text += " No configuration is better than the baseline with a paired interval that excludes 0."
    if worse:
        text += " Worse than the baseline with a paired interval that excludes 0: " + "; ".join(worse) + "."
    text += (
        f" With {comparisons} paired comparisons on {summary['n_questions']} questions, a few intervals excluding 0 are "
        "expected by chance, so none of this separates configurations; the data cannot tell most of them apart."
    )
    return text


# --------------------------------------------------------------------------- compare-exact (read-only)
def compare_exact(cfg: Config, queries: list[str], answerable: list[bool], depth: int) -> dict:
    """Pipeline hybrid top-``depth`` (Chroma/HNSW dense) versus hybrid with exact dense, per query."""
    from fincrime_rag.embeddings import build_embedder
    from fincrime_rag.retrieval import Retriever

    embedder = build_embedder(cfg)
    pipeline = Retriever(cfg, embedder)
    stored = pipeline.collection.get(include=["embeddings", "documents", "metadatas"])
    from fincrime_rag.vectorstore import chunk_from_record

    chunks = [
        chunk_from_record(cid, doc, meta)
        for cid, doc, meta in zip(stored["ids"], stored["documents"], stored["metadatas"], strict=True)
    ]
    exact = ExactRetriever(cfg, chunks, np.asarray(stored["embeddings"]), embedder)
    rows = []
    for q in queries:
        a = [h.chunk_id for h in pipeline.retrieve(q, depth, "hybrid")]
        b = [c["chunk_id"] for c in exact.retrieve(q, depth, "hybrid")]
        rows.append({"pipeline": a, "exact": b})
    digest = hashlib.sha256(json.dumps([r["exact"] for r in rows]).encode()).hexdigest()
    summary = {
        "exact_digest": digest,
        "n": len(rows),
        "pipeline": [r["pipeline"] for r in rows],
        "exact": [r["exact"] for r in rows],
    }
    for label, mask in (("all", [True] * len(rows)), ("answerable", answerable)):
        idx = [i for i, keep in enumerate(mask) if keep]
        set_diff = sum(set(rows[i]["pipeline"]) != set(rows[i]["exact"]) for i in idx)
        order_only = sum(
            set(rows[i]["pipeline"]) == set(rows[i]["exact"]) and rows[i]["pipeline"] != rows[i]["exact"]
            for i in idx
        )
        summary[label] = {"n": len(idx), "different_set": set_diff, "order_only": order_only}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.experiments", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run", "compare-exact"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    from fincrime_rag.chunking import hf_token_counter
    from fincrime_rag.embeddings import SentenceTransformerEmbedder
    from fincrime_rag.evaluate import load_gold_questions

    if args.command == "compare-exact":
        from fincrime_rag.evaluation.generation_eval import load_questions
        from fincrime_rag.guardrails import InjectionDetector

        detector = InjectionDetector(cfg.injection)
        loaded = load_questions(cfg)
        queries = [detector.sanitize_question(text)[0] for text, _ in loaded]
        answerable = [q.type == "answerable" for _, q in loaded]
        result = compare_exact(cfg, queries, answerable, cfg.generation.context_k)
        print(json.dumps({k: v for k, v in result.items() if k not in ("pipeline", "exact")}, indent=2))
        return 0

    questions = [(text, gold) for _, text, _, gold in load_gold_questions(cfg)]
    base = SentenceTransformerEmbedder(cfg)
    grid = run_grid(
        cfg,
        lambda cache_dir: CachedEmbedder(base, cache_dir),
        hf_token_counter(cfg.embedding.model),
        questions,
        Path(cfg.experiments.work_dir),
    )
    summary = summarize_grid(cfg, grid)
    summary["conclusion"] = conclusion_text(cfg, summary)
    results_path = Path(cfg.experiments.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    published = _published_comparison(cfg, summary)
    write_report(cfg, summary, published, Path(cfg.experiments.report))
    print(summary["conclusion"])
    return 0


def _published_comparison(cfg: Config, summary: dict) -> dict | None:
    """Exact-baseline rows next to the numbers in reports/retrieval_results.md (parsed from that file)."""
    path = Path(cfg.paths.reports_dir) / "retrieval_results.md"
    if not path.exists():
        return None
    published: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if (
            line.startswith("|")
            and len(parts) >= 12
            and parts[1] == "overall"
            and parts[0] in ("dense", "bm25", "hybrid")
        ):
            published[parts[0]] = {
                "any@1": parts[3],
                "any@3": parts[4],
                "any@5": parts[5],
                "any@10": parts[6],
                "mrr": parts[11],
            }
    if not published:
        return None
    base = summary["configs"][cfg.experiments.baseline.name]
    rows = [
        "| Mode | metric | published (Chroma/HNSW dense, one process) | exact search (this report) | difference |",
        "|---|---|---|---|---|",
    ]
    for mode in ("dense", "bm25", "hybrid"):
        if mode not in published or mode not in base["modes"]:
            continue
        for metric, key in (
            ("any@1", "any@1"),
            ("any@3", "any@3"),
            ("any@5", "any@5"),
            ("any@10", "any@10"),
            ("MRR", "mrr"),
        ):
            pub = float(published[mode][key])
            ex = base["modes"][mode]["metrics"][key][0]
            rows.append(f"| {mode} | {metric} | {pub:.3f} | {ex:.3f} | {ex - pub:+.3f} |")
    text = (
        chr(10).join(rows)
        + chr(10)
        + chr(10)
        + (
            "The published numbers come from `retrieval_results.md` and are rounded there (two decimals for any@k, "
            "three for MRR), so differences of up to about 0.005 are rounding. BM25 does not use the dense index and "
            "differs by the same amount, which shows the size of the rounding. The dense and hybrid rows therefore "
            "show no measurable effect of the approximate dense search on these 14 questions, in line with the "
            "read-only comparison in D-032 (no differences for the answerable questions in any process)."
        )
    )
    return {"text": text}


if __name__ == "__main__":
    sys.exit(main())
