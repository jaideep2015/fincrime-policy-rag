"""Evaluation entry point: ``python -m fincrime_rag.evaluate retrieval``.

Scores every retrieval mode (dense, bm25, hybrid) on the answerable questions of the splits listed in
``evaluation.splits`` (default: dev only; the held-out test split is scored once, at the end). No LLM
is called. Only aggregate numbers are printed or written: never question text, qids or per-question
results, so nothing here invites tuning against individual questions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fincrime_rag.config import Config, load_config
from fincrime_rag.embeddings import Embedder, build_embedder
from fincrime_rag.evalset import Report, load_records
from fincrime_rag.evaluation.retrieval_metrics import RankedHit, question_metrics, summarize
from fincrime_rag.retrieval import MODES, Retriever


def load_gold_questions(cfg: Config) -> list[tuple[str, str, frozenset]]:
    """(question, category, gold pairs) for answerable questions in the configured splits."""
    records = load_records(cfg.paths.eval_set, Report())
    selected = []
    for _, rec in records:
        if rec.get("split") not in cfg.evaluation.splits or not rec.get("gold"):
            continue
        gold = frozenset((g["doc_id"], int(p)) for g in rec["gold"] for p in g["pages"])
        selected.append((rec["question"], rec["category"], gold))
    return selected


def evaluate_retrieval(cfg: Config, retriever: Retriever, questions: list[tuple[str, str, frozenset]]):
    ks = cfg.evaluation.retrieval_ks
    depth = ks[-1]
    results = {}
    for mode in MODES:
        per_question = []
        for question, category, gold in questions:
            hits = [RankedHit(h.doc_id, h.pdf_pages) for h in retriever.retrieve(question, depth, mode)]
            per_question.append((category, question_metrics(hits, gold, ks, cfg.retrieval.top_k)))
        results[mode] = summarize(per_question)
    return results


def format_table(cfg: Config, results) -> str:
    ks = cfg.evaluation.retrieval_ks
    head = (
        ["Mode", "Slice", "n"]
        + [f"any@{k}" for k in ks]
        + [f"frac@{k}" for k in ks]
        + ["MRR", f"doc-hit@{cfg.retrieval.top_k}"]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for mode, summary in results.items():
        for name, (n, m) in summary.items():
            cells = [mode, name, str(n)]
            cells += [f"{m[f'any@{k}']:.2f}" for k in ks] + [f"{m[f'frac@{k}']:.2f}" for k in ks]
            cells += [f"{m['mrr']:.3f}", f"{m['hit']:.2f}"]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(cfg: Config, results, path: Path) -> None:
    n = next(iter(results.values()))["overall"][0]
    top_k = cfg.retrieval.top_k
    text = (
        "# Retrieval results (LLM-free)\n\n"
        f"Splits scored: {', '.join(cfg.evaluation.splits)}; answerable questions only; **n = {n}**. "
        f"Chunking {cfg.chunking.chunk_size}/{cfg.chunking.chunk_overlap} tokens, "
        f"embedding `{cfg.embedding.model}`, RRF k = {cfg.retrieval.hybrid_rrf_k}, "
        f"candidate pool {cfg.retrieval.candidate_pool}.\n\n"
        "`any@k`: a top-k hit covers at least one gold page. `frac@k`: share of gold pages covered "
        "in the top k. `MRR`: reciprocal rank of the first covering hit. "
        f"`doc-hit@{top_k}`: a gold document appears in the top {top_k} (page ignored). "
        "With this n, per-category rows rest on very few questions and differences between modes "
        "are not statistically established.\n\n" + format_table(cfg, results) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("target", choices=["retrieval"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-report", action="store_true", help="print only; do not write the report")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    questions = load_gold_questions(cfg)
    if not questions:
        print("no answerable questions in the configured splits")
        return 1
    embedder: Embedder = build_embedder(cfg)
    results = evaluate_retrieval(cfg, Retriever(cfg, embedder), questions)
    print(format_table(cfg, results))
    if not args.no_report:
        path = cfg.paths.reports_dir / "retrieval_results.md"
        write_report(cfg, results, path)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
