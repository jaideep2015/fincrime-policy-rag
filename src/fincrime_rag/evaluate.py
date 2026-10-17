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
from fincrime_rag.evaluation.retrieval_metrics import RankedHit, aggregate, question_metrics, summarize
from fincrime_rag.retrieval import MODES, Retriever


def load_gold_questions(cfg: Config) -> list[tuple[str, str, str, frozenset]]:
    """(qid, question, category, gold pairs) for answerable questions in the configured splits."""
    records = load_records(cfg.paths.eval_set, Report())
    selected = []
    for _, rec in records:
        if rec.get("split") not in cfg.evaluation.splits or not rec.get("gold"):
            continue
        gold = frozenset((g["doc_id"], int(p)) for g in rec["gold"] for p in g["pages"])
        selected.append((rec["qid"], rec["question"], rec["category"], gold))
    return selected


MISS_K = 5  # the k at which misses are listed (by qid only)


def evaluate_retrieval(cfg: Config, retriever: Retriever, questions: list[tuple[str, str, str, frozenset]]):
    """Returns (results, extras). ``extras`` holds the multi-page-gold subset and the qids missing at k=5."""
    ks = cfg.evaluation.retrieval_ks
    depth = ks[-1]
    results = {}
    multi_page: dict[str, tuple[int, dict[str, float]]] = {}
    misses: dict[str, list[str]] = {}
    for mode in MODES:
        per_question = []
        multi_rows = []
        misses[mode] = []
        for qid, question, category, gold in questions:
            hits = [RankedHit(h.doc_id, h.pdf_pages) for h in retriever.retrieve(question, depth, mode)]
            metrics = question_metrics(hits, gold, ks, cfg.retrieval.top_k)
            per_question.append((category, metrics))
            if len(gold) >= 2:
                multi_rows.append(metrics)
            if metrics[f"any@{MISS_K}"] == 0.0:
                misses[mode].append(qid)
        results[mode] = summarize(per_question)
        multi_page[mode] = (len(multi_rows), aggregate(multi_rows))
    return results, {"multi_page": multi_page, "misses": misses}


def format_table(cfg: Config, results) -> str:
    ks = cfg.evaluation.retrieval_ks
    head = (
        ["Mode", "Slice", "n"]
        + [f"any@{k}" for k in ks]
        + [f"frac@{k}" for k in ks]
        + ["MRR", f"doc-level hit rate@{cfg.retrieval.top_k}"]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for mode, summary in results.items():
        for name, (n, m) in summary.items():
            cells = [mode, name, str(n)]
            cells += [f"{m[f'any@{k}']:.2f}" for k in ks] + [f"{m[f'frac@{k}']:.2f}" for k in ks]
            cells += [f"{m['mrr']:.3f}", f"{m['hit']:.2f}"]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_multi_page(cfg: Config, multi_page) -> str:
    ks = cfg.evaluation.retrieval_ks
    head = ["Mode", "n"] + [f"any@{k}" for k in ks] + [f"all@{k}" for k in ks]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for mode, (n, m) in multi_page.items():
        cells = [mode, str(n)] + [f"{m[f'any@{k}']:.2f}" for k in ks] + [f"{m[f'all@{k}']:.2f}" for k in ks]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def chunk_share(retriever: Retriever) -> list[tuple[str, int, float]]:
    counts: dict[str, int] = {}
    for chunk in retriever.chunks.values():
        counts[chunk["doc_id"]] = counts.get(chunk["doc_id"], 0) + 1
    total = sum(counts.values())
    return [(doc, n, n / total) for doc, n in sorted(counts.items(), key=lambda kv: -kv[1])]


def write_report(cfg: Config, results, extras, shares, path: Path) -> None:
    n = next(iter(results.values()))["overall"][0]
    top_k = cfg.retrieval.top_k
    share_rows = "\n".join(f"| {doc} | {c} | {frac:.1%} |" for doc, c, frac in shares)
    miss_lines = "\n".join(
        f"- {mode}: {', '.join(ids) if ids else 'none'}" for mode, ids in extras["misses"].items()
    )
    text = (
        "# Retrieval results (LLM-free)\n\n"
        f"Splits scored: {', '.join(cfg.evaluation.splits)}; answerable questions only; **n = {n}**. "
        f"Chunking {cfg.chunking.chunk_size}/{cfg.chunking.chunk_overlap} tokens, "
        f"embedding `{cfg.embedding.model}`, RRF constant k = {cfg.retrieval.hybrid_rrf_k} "
        f"(score = 1/(k + rank)), candidate pool {cfg.retrieval.candidate_pool}.\n\n"
        "> **Note.** Default mode = hybrid was selected on dev aggregates (n=14); differences between "
        "modes are 1-2 questions; there is no held-out split. Dense beats hybrid on any@5 (0.93 vs 0.86).\n\n"
        "## Page identity\n\n"
        "Chunks store the **PDF page index** (field `pdf_pages`, a list; Chroma: `pdf_pages` as \"4,5\" plus "
        "`pdf_page_start` / `pdf_page_end`), taken from each page record's `pdf_page`. The printed page is a "
        "separate field (`printed_page_guess`) and is not used for matching. Gold pages in `qa_set.jsonl` "
        "are PDF indexes (`gold[].pages`); a hit covers a gold page when `doc_id` matches and the chunk's "
        "`pdf_pages` contains it.\n\n"
        "## Overall and by category\n\n"
        "`any@k`: a top-k hit covers at least one gold page. `frac@k`: share of gold pages covered in the "
        "top k. `MRR`: reciprocal rank of the first covering hit. "
        f"`doc-level hit rate@{top_k}`: the right document (any page) is in the top {top_k}; this is "
        f"not the page-level any@{top_k}. Per-category rows rest on very few questions.\n\n"
        + format_table(cfg, results)
        + "\n\n## Questions with 2+ gold pages: any@k vs all@k\n\n"
        "`all@k`: the top k hits cover every gold page.\n\n"
        + format_multi_page(cfg, extras["multi_page"])
        + "\n\n## Chunk share per document\n\n"
        "| doc_id | chunks | share of index |\n|---|---|---|\n" + share_rows + "\n\n"
        "Reg E dominates the index. Dense and BM25 rank chunks across the whole corpus, so a large document "
        "can crowd smaller ones out of the top k (for example the 4-page 2025 SAR FAQs, 7 chunks). "
        "This is a possible effect, not something measured here; the table above does not isolate it.\n\n"
        f"## Questions with no gold page in the top {MISS_K} (by question ID only)\n\n"
        + miss_lines
        + "\n\nReported for information; nothing was changed in response.\n"
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
    retriever = Retriever(cfg, embedder)
    results, extras = evaluate_retrieval(cfg, retriever, questions)
    print(format_table(cfg, results))
    if not args.no_report:
        path = cfg.paths.reports_dir / "retrieval_results.md"
        write_report(cfg, results, extras, chunk_share(retriever), path)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
