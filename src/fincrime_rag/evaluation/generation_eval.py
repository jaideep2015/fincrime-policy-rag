"""Phase 6: generation evaluation -> reports/generation_results.md. Aggregate numbers only.

Runs the full pipeline (retrieve, refusal gate, generate, citation check) over every question of the
evaluation set and reports, per backend: citation precision/recall (PDF-page level), hallucinated-citation
rate, refusal accuracy, injection pass rate, key-fact match (counts), and the newer-source check for
conflict questions. Each rate carries n and a bootstrap confidence interval (fixed seed).
Also reports the retrieval-only *context recall ceiling* at the fixed context k.

Nothing here prints or stores question text, qids or per-question results, and nothing in ``src/`` may
contain question-specific strings. All 20 questions are ``dev``; there is no held-out split.

Command: ``.venv\\Scripts\\python scripts\\eval_generation.py`` (or ``python -m fincrime_rag.evaluation.generation_eval``).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fincrime_rag.config import Config, load_config
from fincrime_rag.embeddings import build_embedder
from fincrime_rag.evalset import Report, load_records
from fincrime_rag.evaluation.retrieval_metrics import RankedHit, question_metrics
from fincrime_rag.generation import Answer, RAGPipeline, build_llm, load_page_text_lookup
from fincrime_rag.guardrails import InjectionDetector, load_gate
from fincrime_rag.ingestion import read_manifest
from fincrime_rag.llm.base import LLMUnavailable
from fincrime_rag.llm.ollama_client import OllamaClient
from fincrime_rag.retrieval import Retriever


# --------------------------------------------------------------------------- bootstrap
def bootstrap_ci(items: list, stat: Callable[[list], float], samples: int, confidence: float, seed: int):
    """(point estimate, low, high) by resampling ``items`` with replacement."""
    if not items:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(items)
    draws = sorted(stat([items[rng.randrange(n)] for _ in range(n)]) for _ in range(samples))
    alpha = (1.0 - confidence) / 2.0
    return stat(items), draws[int(alpha * samples)], draws[min(int((1.0 - alpha) * samples), samples - 1)]


def mean_stat(values: list[float]) -> float:
    return sum(values) / len(values)


def ratio_stat(pairs: list[tuple[int, int]]) -> float:
    den = sum(b for _, b in pairs)
    return sum(a for a, _ in pairs) / den if den else float("nan")


# --------------------------------------------------------------------------- per-question results
@dataclass(frozen=True)
class Q:
    type: str
    category: str
    gold: frozenset
    key_facts: list[str]


def load_questions(cfg: Config) -> list[tuple[str, Q]]:
    """(question text, metadata) for every record. Text is used only to run the pipeline."""
    out = []
    for _, rec in load_records(cfg.paths.eval_set, Report()):
        gold = frozenset((g["doc_id"], int(p)) for g in rec.get("gold", []) for p in g["pages"])
        out.append((rec["question"], Q(rec["type"], rec["category"], gold, list(rec.get("key_facts", [])))))
    return out


def word_coverage(fact: str, text: str) -> float:
    words = set(re.findall(r"[a-z0-9]+", fact.lower()))
    have = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(words & have) / len(words) if words else 0.0


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def newest_doc(cfg: Config, doc_ids: set[str]) -> str | None:
    dates = {r.doc_id: (r.version_date or r.date or "") for r in read_manifest(cfg.paths.manifest)}
    dated = [(dates.get(d, ""), d) for d in doc_ids if dates.get(d)]
    return max(dated)[1] if len(dated) == len(doc_ids) and dated else None


def summarize_backend(cfg: Config, results: list[tuple[Q, Answer]]) -> dict[str, tuple]:
    """metric name -> (display value, n, lo, hi) with a plain-count display where stated."""
    ge = cfg.generation_eval
    seed = cfg.seed

    def rate(name: str, flags: list[float]):
        est, lo, hi = bootstrap_ci(flags, mean_stat, ge.bootstrap_samples, ge.confidence, seed)
        return name, (est, len(flags), lo, hi, None)

    answerable = [(q, a) for q, a in results if q.type == "answerable"]
    answered = [(q, a) for q, a in answerable if not a.refused]
    out: dict[str, tuple] = {}

    def add(pair):
        out[pair[0]] = pair[1]

    add(rate("Answerable not refused", [0.0 if a.refused else 1.0 for _, a in answerable]))
    add(
        rate(
            "Answerable answered with a valid citation",
            [1.0 if a.valid_citations else 0.0 for _, a in answerable],
        )
    )
    precisions, recalls = [], []
    for q, a in answered:
        cited = {(c.doc_id, c.page) for c in a.valid_citations}
        recalls.append(len(cited & q.gold) / len(q.gold))
        if cited:
            precisions.append(len(cited & q.gold) / len(cited))
    add(rate("Citation precision (answered with >=1 valid citation)", precisions))
    add(rate("Citation recall (answered, not refused)", recalls))
    pairs = [
        (len(a.invalid_citations), len(a.valid_citations) + len(a.invalid_citations)) for _, a in results
    ]
    est, lo, hi = bootstrap_ci(pairs, ratio_stat, ge.bootstrap_samples, ge.confidence, seed)
    out["Hallucinated-citation rate (invalid / all citations, all questions)"] = (
        est,
        len(pairs),
        lo,
        hi,
        f"{sum(a for a, _ in pairs)}/{sum(b for _, b in pairs)} citations",
    )
    add(
        rate(
            "Unanswerable refused",
            [1.0 if a.refused else 0.0 for q, a in results if q.type == "unanswerable"],
        )
    )
    detector = InjectionDetector(cfg.injection)
    add(
        rate(
            "Injection pass rate (not complied/echoed; heuristic)",
            [
                1.0
                if (a.status in ("refused", "rejected_injection") or not detector.matches(a.text))
                else 0.0
                for q, a in results
                if q.type == "injection"
            ],
        )
    )
    facts = [all(norm(f) in norm(a.text) for f in q.key_facts) for q, a in answerable if q.key_facts]
    out["Key facts all present in answer (string match)"] = (
        sum(facts) / len(facts) if facts else float("nan"),
        len(facts),
        None,
        None,
        f"{sum(facts)} of {len(facts)}",
    )
    # Added after the first run showed the all-or-nothing count at 0: a graded view of the same check
    # (a generic metric change, logged in D-025, not a change to any prompt, threshold or k).
    share = [
        sum(norm(f) in norm(a.text) for f in q.key_facts) / len(q.key_facts)
        for q, a in answerable
        if q.key_facts
    ]
    add(rate("Key facts found (share of each question's key_facts strings)", share))
    # Key facts are paraphrased sentences (about 14 words), so verbatim matching is nearly always 0, even
    # against the full retrieved context. A word-overlap view is a coarse but graded proxy (D-025).
    coverage = [
        sum(word_coverage(f, a.text) for f in q.key_facts) / len(q.key_facts)
        for q, a in answerable
        if q.key_facts
    ]
    add(rate("Key-fact word coverage (mean share of key-fact words in the answer; coarse proxy)", coverage))
    conflict = [(q, a) for q, a in results if q.type == "answerable" and q.category == "conflict"]
    checks = []
    for q, a in conflict:
        newest = newest_doc(cfg, {d for d, _ in q.gold})
        if newest and len({d for d, _ in q.gold}) >= 2:
            checks.append(any(c.doc_id == newest for c in a.valid_citations))
    out["Conflict: cites the newest gold document"] = (
        sum(checks) / len(checks) if checks else float("nan"),
        len(checks),
        None,
        None,
        f"{sum(checks)} of {len(checks)}",
    )
    return out


def context_ceiling(cfg: Config, retriever: Retriever, questions: list[tuple[str, Q]], mode: str):
    k = cfg.generation.context_k
    any_k, all_k = [], []
    for text, q in questions:
        if q.type != "answerable":
            continue
        hits = [RankedHit(h.doc_id, h.pdf_pages) for h in retriever.retrieve(text, k, mode)]
        m = question_metrics(hits, q.gold, [k], k)
        any_k.append(m[f"any@{k}"])
        all_k.append(m[f"all@{k}"])
    ge = cfg.generation_eval
    return {
        "any": bootstrap_ci(any_k, mean_stat, ge.bootstrap_samples, ge.confidence, cfg.seed) + (len(any_k),),
        "all": bootstrap_ci(all_k, mean_stat, ge.bootstrap_samples, ge.confidence, cfg.seed) + (len(all_k),),
    }


def run_backend(cfg, retriever, backend: str, mode: str, questions) -> tuple[list[tuple[Q, Answer]], dict]:
    llm = build_llm(cfg, backend, load_page_text_lookup(cfg))
    pipeline = RAGPipeline(cfg, retriever, llm, load_gate(cfg))
    results = [(q, pipeline.answer(text, mode)) for text, q in questions]
    answers = [a for _, a in results]
    info = {
        "backend": next((a.backend for a in answers if a.backend), backend),
        "fallbacks": sum(a.fallback for a in answers),
        "statuses": {s: sum(a.status == s for a in answers) for s in sorted({a.status for a in answers})},
        "refusal_reasons": {
            r: sum(a.refusal_reason == r for a in answers)
            for r in sorted({a.refusal_reason for a in answers if a.refusal_reason})
        },
    }
    return results, info


# --------------------------------------------------------------------------- report
def fmt(entry: tuple) -> str:
    est, n, lo, hi, counts = entry
    if est != est:  # NaN
        return f"n/a (n={n})"
    if lo is None:  # a count, no CI
        return f"{counts} (n={n})"
    text = f"{est:.2f} [{lo:.2f}, {hi:.2f}]"
    return f"{text} ({counts}, n={n})" if counts else f"{text} (n={n})"


def build_report(cfg: Config, columns: dict[str, dict | None], ollama_note: str, ceilings, infos, thr) -> str:
    names = list(columns)
    metric_names = next(c for c in columns.values() if c)
    rows = []
    for metric in metric_names:
        cells = [fmt(columns[n][metric]) if columns[n] else "not evaluated" for n in names]
        rows.append(f"| {metric} | " + " | ".join(cells) + " |")
    ce = []
    for mode, c in ceilings.items():
        ce.append(
            f"| {mode} | {c['any'][0]:.2f} [{c['any'][1]:.2f}, {c['any'][2]:.2f}] | "
            f"{c['all'][0]:.2f} [{c['all'][1]:.2f}, {c['all'][2]:.2f}] | {c['any'][3]} |"
        )
    k = cfg.generation.context_k
    info_lines = "\n".join(f"- {n}: {i}" for n, i in infos.items())
    return f"""# Generation results

**All 20 questions are dev; there is no held-out split.** The default retrieval mode (hybrid) was selected on
these same dev questions (see `retrieval_results.md`), differences of 1-2 questions are within noise, and
n is small everywhere, so the confidence intervals are wide. Rates are means over questions; intervals are
percentile bootstrap 95% CIs ({cfg.generation_eval.bootstrap_samples} resamples, seed {cfg.seed}); counts are
given where a CI is not meaningful. Context k = {k} was fixed a priori (D-022) and not changed afterwards.

## Backends

- **Ollama ({cfg.llm.model}):** {ollama_note}
- **Extractive:** no LLM; returns the top {cfg.generation.extractive_sentences} best-matching sentences from the
  retrieved context with `[doc_id p.N]` citations, so it cannot hallucinate a citation and cannot say
  "insufficient context" itself (refusal comes only from the retrieval-score gate). It is a floor, not a
  competitor to a language model.
- Refusal gate: top-1 dense cosine similarity below {thr["threshold"]:.4f} (the {thr["percentile"]:g}th percentile of
  in-corpus top-1 similarities; observed range {thr["observed_min"]:.3f} to {thr["observed_max"]:.3f}, median
  {thr["observed_median"]:.3f}; {thr["n_queries"]} corpus sentences as queries; no eval question used, not tuned).

Run summary (status counts and refusal reasons per column):
{info_lines}

## Results

| Metric | {" | ".join(names)} |
|---|{"---|" * len(names)}
{chr(10).join(rows)}

Definitions. Citation precision/recall: cited (doc_id, PDF page) pairs that are valid (present in the retrieved
context) compared with the gold pages, per question then averaged; precision only over answers with at least one
valid citation. Hallucinated citation: a cited (doc_id, page) not in the retrieved context (always 0 for the
extractive backend by construction). Refusal accuracy is shown as the two rates above. Injection pass: the final
reply is a refusal, was rejected as an echo, or contains no instruction-like sentence. Key facts: every `key_facts`
string of the question appears (case-insensitive) in the answer text. Conflict: the answer's valid citations
include the most recent gold document by manifest date (questions whose gold has 2+ documents only).

## Context recall ceiling at k = {k} (retrieval only, no generation)

Union of the pages of the top-{k} chunks; stored `pdf_pages` are parsed back to integers before matching.
`any`: some gold page is in the context; `all`: every gold page is in the context.

| Mode | any gold page | all gold pages | n |
|---|---|---|---|
{chr(10).join(ce)}

No generation result can exceed these ceilings for the questions they cover, so a gap between the ceiling and a
citation metric points to generation, not retrieval.

## Known retrieval limitations (carried over, not fixed)

- One answerable question has no gold page in the top 5 in any mode; another is missed by BM25 and hybrid at k=5
  (see `retrieval_results.md`).
- `all@k` is well below `any@k`: multi-page answers are often only partly covered.
- Reg E is 58.5% of the index, so small documents can be crowded out.

## Not evaluated automatically

- Faithfulness and answer relevance (no judge model is available here).
- Whether prose is correct. Key facts are paraphrased sentences, so verbatim matching is almost always 0 (even
  against the full retrieved context); the word-coverage row is only a coarse proxy. Judging correctness needs
  manual labels or a judge model.
- Whether a conflict answer actually *states* the conflict (only the newest-source citation is checked); whether a conflict answer actually *states* the conflict.
- Injection compliance beyond the generic instruction-pattern heuristic and the echo check.
- The Ollama backend, unless the column above shows numbers.
- Any held-out generalisation: there is no held-out split.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_generation", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    questions = load_questions(cfg)
    retriever = Retriever(cfg, build_embedder(cfg))
    thr = json.loads(Path(cfg.refusal.calibration_file).read_text(encoding="utf-8"))
    mode = cfg.retrieval.mode

    columns: dict[str, dict | None] = {}
    infos: dict[str, dict] = {}
    try:
        digest = OllamaClient(cfg).check_available()
        ollama_note = (
            f"evaluated; digest {digest}, num_ctx {cfg.llm.num_ctx}, temperature 0, seed {cfg.seed}."
        )
        results, info = run_backend(cfg, retriever, "ollama", mode, questions)
        columns[f"Ollama ({mode})"] = summarize_backend(cfg, results)
        infos[f"Ollama ({mode})"] = info
    except LLMUnavailable as exc:
        ollama_note = f"**not evaluated: {exc}.** Nothing was installed or pulled. Column left empty."
        columns[f"Ollama ({mode})"] = None
    for label, m in ((f"Extractive ({mode})", mode), ("Extractive (dense, ablation)", "dense")):
        results, info = run_backend(cfg, retriever, "extractive", m, questions)
        columns[label] = summarize_backend(cfg, results)
        infos[label] = info
    ceilings = {m: context_ceiling(cfg, retriever, questions, m) for m in (mode, "dense")}
    text = build_report(cfg, columns, ollama_note, ceilings, infos, thr)
    path = cfg.paths.reports_dir / "generation_results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
