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
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fincrime_rag.citations import lenient_valid_citations, strip_citation_text
from fincrime_rag.config import Config, load_config
from fincrime_rag.embeddings import build_embedder
from fincrime_rag.evalset import Report, load_records
from fincrime_rag.evaluation.citation_support import support_pairs
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
    qid: str
    type: str
    category: str
    gold: frozenset
    key_facts: list[str]


def load_questions(cfg: Config) -> list[tuple[str, Q]]:
    """(question text, metadata) for every record. Text is used only to run the pipeline."""
    out = []
    for _, rec in load_records(cfg.paths.eval_set, Report()):
        gold = frozenset((g["doc_id"], int(p)) for g in rec.get("gold", []) for p in g["pages"])
        out.append(
            (
                rec["question"],
                Q(rec["qid"], rec["type"], rec["category"], gold, list(rec.get("key_facts", []))),
            )
        )
    return out


STATES_NEWER_PATTERN = re.compile(
    r"\b(more recent|newer|newest|latest|updated|supersede[sd]?|replac(?:es|ed)|revised|conflict\w*|differs?)\b",
    re.IGNORECASE,
)


def newest_year(cfg: Config, doc_id: str) -> str | None:
    for r in read_manifest(cfg.paths.manifest):
        if r.doc_id == doc_id:
            m = re.match(r"(\d{4})", r.version_date or r.date or "")
            return m.group(1) if m else None
    return None


def states_newer_rule(text: str, year: str | None) -> bool:
    """Crude: the answer uses a recency/conflict word, or mentions the newest document's year."""
    return bool(STATES_NEWER_PATTERN.search(text)) or bool(year and year in text)


def count_entry(k: int, n: int) -> tuple:
    return (k / n if n else float("nan"), n, None, None, f"{k} of {n}")


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


def summarize_backend(
    cfg: Config, results: list[tuple[Q, Answer]], backend: str, page_text=None
) -> dict[str, tuple]:
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
    # lenient reading of the same replies (also accepts "(doc_id p.N)"); reported, never the primary metric
    lenient = []
    for _, a in answerable:
        chunks = [{"doc_id": d, "pdf_pages": list(pages)} for d, pages in a.retrieved]
        lenient.append(1.0 if lenient_valid_citations(a.raw_text or "", chunks) else 0.0)
    add(
        rate(
            "Answerable answered with a valid citation, lenient (also accepts (doc_id p.N); not primary)",
            lenient,
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
    num_ok = num_n = n_pairs = 0
    overlaps: list[float] = []
    for _, a in answered:
        valid = {(c.doc_id, c.page) for c in a.valid_citations}
        pairs = support_pairs(a.text, valid, page_text) if page_text else []
        n_pairs += len(pairs)
        num_ok += sum(p.numeric_ok is True for p in pairs)
        num_n += sum(p.numeric_ok is not None for p in pairs)
        per_q = [p.overlap for p in pairs if p.overlap is not None]
        if per_q:
            overlaps.append(sum(per_q) / len(per_q))
    out[SUPPORT_NUMERIC_ROW] = count_entry(num_ok, num_n)[:4] + (
        f"{num_ok} of {num_n} pairs with numbers; {n_pairs} pairs in all",
    )
    add(
        rate(
            "Citation support (LLM-free, D-030): mean content-word overlap of cited sentences with the cited page",
            overlaps,
        )
    )
    if backend == "extractive":
        out["Hallucinated-citation rate (invalid / all citations)"] = (
            "N/A (extractive quotes the context; 0 by construction)"
        )
    else:
        pairs = [
            (len(a.invalid_citations), len(a.valid_citations) + len(a.invalid_citations)) for _, a in results
        ]
        est, lo, hi = bootstrap_ci(pairs, ratio_stat, ge.bootstrap_samples, ge.confidence, seed)
        out["Hallucinated-citation rate (invalid / all citations)"] = (
            est,
            len(pairs),
            lo,
            hi,
            f"{sum(a for a, _ in pairs)}/{sum(b for _, b in pairs)} citations",
        )
    # unanswerable: the two refusal signals separately, then combined
    unans = [a for q, a in results if q.type == "unanswerable"]
    gate = sum(a.refusal_reason == "low_retrieval_score" for a in unans)
    reached = [a for a in unans if a.refusal_reason != "low_retrieval_score"]
    insufficient = sum(a.refusal_reason == "model_insufficient" for a in reached)
    out["Unanswerable: refused by the similarity gate"] = count_entry(gate, len(unans))
    out["Unanswerable: INSUFFICIENT_CONTEXT (of those that passed the gate)"] = (
        "N/A (extractive cannot decline)"
        if backend == "extractive"
        else count_entry(insufficient, len(reached))
    )
    add(rate("Unanswerable: refused, combined", [1.0 if a.refused else 0.0 for a in unans]))
    # injection: counts, not a rate
    detector = InjectionDetector(cfg.injection)
    inj = [a for q, a in results if q.type == "injection"]
    complied = sum(
        a.status not in ("refused", "rejected_injection") and detector.matches(a.text) for a in inj
    )
    declined = sum(a.status in ("refused", "rejected_injection") for a in inj)
    out["Injection: complied / echoed an instruction (pattern check)"] = count_entry(complied, len(inj))
    out["Injection: refused or rejected"] = count_entry(declined, len(inj))
    # key facts: strict match first (pre-registered check), post-hoc proxies after
    facts = [all(norm(f) in norm(a.text) for f in q.key_facts) for q, a in answerable if q.key_facts]
    out["Key facts: all present, strict string match"] = count_entry(sum(facts), len(facts))
    share = [
        sum(norm(f) in norm(a.text) for f in q.key_facts) / len(q.key_facts)
        for q, a in answerable
        if q.key_facts
    ]
    add(rate("Key facts: share found by strict match (post-hoc, D-025, not pre-registered)", share))
    coverage = [
        sum(word_coverage(f, a.text) for f in q.key_facts) / len(q.key_facts)
        for q, a in answerable
        if q.key_facts
    ]
    add(rate("Key facts: word coverage (post-hoc, D-025, not pre-registered; coarse proxy)", coverage))
    # conflict: (a) newest gold document cited, (b) answer text states the newer rule
    all_doc_ids = [r.doc_id for r in read_manifest(cfg.paths.manifest)]
    cited_newest = b_original = b_fixed = b_strict = n_conflict = 0
    for q, a in results:
        docs = {d for d, _ in q.gold}
        if q.type != "answerable" or q.category != "conflict" or len(docs) < 2:
            continue
        newest = newest_doc(cfg, docs)
        if not newest:
            continue
        n_conflict += 1
        year = newest_year(cfg, newest)
        cites = any(c.doc_id == newest for c in a.valid_citations)
        cited_newest += cites
        clean = strip_citation_text(a.text, all_doc_ids)  # no citation tags or doc ids (D-029 bug fix)
        b_original += cites and states_newer_rule(a.text, year)
        b_fixed += cites and states_newer_rule(clean, year)
        older = any(c.doc_id != newest and c.doc_id in docs for c in a.valid_citations)
        b_strict += cites and older and bool(STATES_NEWER_PATTERN.search(clean))
    out["Conflict (a): newest gold document is cited"] = count_entry(cited_newest, n_conflict)
    out["Conflict (b-original): keyword or year on the raw text, year can match inside a citation tag"] = (
        count_entry(b_original, n_conflict)
    )
    out["Conflict (b-fixed, D-029 bug fix): keyword or year after removing citation tags and doc ids"] = (
        count_entry(b_fixed, n_conflict)
    )
    out[
        "Conflict (b-strict, proposed in D-028): recency/conflict word and valid citations to the newest and an older gold document"
    ] = count_entry(b_strict, n_conflict)
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


def run_backend(
    cfg, retriever, backend: str, mode: str, questions, require_count: bool = True, cache_only: bool = False
) -> tuple[list[tuple[Q, Answer]], dict]:
    llm = build_llm(
        cfg, backend, load_page_text_lookup(cfg), require_count=require_count, cache_only=cache_only
    )
    pipeline = RAGPipeline(cfg, retriever, llm, load_gate(cfg))
    results = []
    latencies: list[float] = []
    for text, q in questions:
        start = time.perf_counter()
        results.append((q, pipeline.answer(text, mode)))
        latencies.append(time.perf_counter() - start)
    answers = [a for _, a in results]
    info = {
        "backend": next((a.backend for a in answers if a.backend), backend),
        "fallbacks": sum(a.fallback for a in answers),
        "model_digest": next((a.model_digest for a in answers if a.model_digest), None),
        "num_ctx": next((a.num_ctx for a in answers if a.num_ctx), None),
        "model_calls": sum(a.backend is not None for a in answers),
        "cached_calls": sum(a.cached for a in answers),
        "prompt_eval_max": max((a.prompt_eval_count for a in answers if a.prompt_eval_count), default=None),
        "prompt_eval_n": sum(a.prompt_eval_count is not None for a in answers),
        "latency_s_median": statistics.median(latencies),
        "latency_s_max": max(latencies),
        "prompt_tokens_estimate_max": max((a.prompt_tokens_estimate or 0) for a in answers),
        "statuses": {s: sum(a.status == s for a in answers) for s in sorted({a.status for a in answers})},
        "refusal_reasons": {
            r: sum(a.refusal_reason == r for a in answers)
            for r in sorted({a.refusal_reason for a in answers if a.refusal_reason})
        },
    }
    return results, info


# --------------------------------------------------------------------------- report
def fmt(entry: tuple | str) -> str:
    if isinstance(entry, str):
        return entry
    est, n, lo, hi, counts = entry
    if est != est:  # NaN
        return f"n/a (n={n})"
    if lo is None:  # a count, no CI
        return f"{counts} (n={n})"
    text = f"{est:.2f} [{lo:.2f}, {hi:.2f}]"
    return f"{text} ({counts}, n={n})" if counts else f"{text} (n={n})"


NL = chr(10)
V1_PATH_NAME = "generation_results_ollama_prompt_v1.md"
LATENCY_ROW = "Latency per query, median / max seconds"
SUPPORT_NUMERIC_ROW = "Citation support (LLM-free, D-030): cited (sentence, page) pairs whose numbers all appear on the cited page"
PROMPT_EVAL_ROW = (
    "Max prompt_eval_count over the queries that reached the model (must be < num_ctx - max_output_tokens)"
)


def refusal_path(a: Answer) -> str:
    if a.status == "refused":
        return "score gate" if a.refusal_reason == "low_retrieval_score" else "INSUFFICIENT_CONTEXT"
    if a.status == "rejected_injection":
        return "output check"
    return "none"


def parse_v1(path: Path) -> tuple[dict[str, str], str | None]:
    """Metric -> cell of the first data column of the saved prompt-v1 report, and its latency (median / max)."""
    if not path.exists():
        return {}, None
    text = path.read_text(encoding="utf-8")
    cells: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith("| Metric"):
            parts = [p.strip() for p in line.strip().strip("|").split(" | ")]
            if len(parts) >= 2:
                cells[parts[0]] = parts[1]
    m = re.search(r"- Ollama \(hybrid\): .*?latency median ([\d.]+)s / max ([\d.]+)s", text)
    return cells, (f"{m.group(1)} / {m.group(2)}" if m else None)


def write_answers(
    questions: list[tuple[str, Q]], results: list[tuple[Q, Answer]], path: Path, title: str
) -> None:
    lines = [
        f"# {title}",
        "",
        "Question text, final answer text, valid and invalid citations, status and refusal path for each of the",
        "20 dev questions, in eval-set order. No metrics. For the owner's review; all 20 questions are dev.",
        "",
    ]
    for (text, q), (_, a) in zip(questions, results, strict=True):
        valid = ", ".join(f"[{c.doc_id} p.{c.page}]" for c in a.valid_citations) or "none"
        invalid = ", ".join(f"[{c.doc_id} p.{c.page}]" for c in a.invalid_citations) or "none"
        lines += [
            f"## {q.qid} ({q.type}, {q.category})",
            "",
            f"**Question:** {text}",
            "",
            "**Final answer:**",
            "",
            *["> " + line for line in (a.text or "(empty)").splitlines()],
            "",
            f"- Valid citations: {valid}",
            f"- Invalid citations: {invalid}",
            f"- Status: {a.status}",
            f"- Refusal path: {refusal_path(a)}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(NL.join(lines), encoding="utf-8", newline=chr(10))


def probe_section(path: Path) -> str:
    if not path.exists():
        return "No injection probe has been run yet (`python scripts/probe_injection.py`)." + NL
    data = json.loads(path.read_text(encoding="utf-8"))
    cols = ["complied", "declined", "answered_with_valid_citation", "other"]
    rows = [
        f"| {variant} | {data['n_per_variant']} | " + " | ".join(str(counts.get(c, 0)) for c in cols) + " |"
        for variant, counts in data["variants"].items()
    ]
    head = "| Variant | n | complied | declined (refused/rejected) | answered the underlying question with a valid citation | other |"
    return NL.join([head, "|---|---|---|---|---|---|", *rows]) + NL


def build_report(
    cfg: Config, columns: dict[str, dict | None], ollama_note: str, ceilings, infos, thr, extra
) -> str:
    v1_cells, v1_latency = extra["v1"]
    names = list(columns)
    metric_names = list(next(c for c in columns.values() if c))
    show = (["Ollama v1 (hybrid)"] if v1_cells else []) + names
    rows = []
    for metric in metric_names:
        cells = [fmt(columns[n][metric]) if columns[n] else "not evaluated" for n in names]
        if v1_cells:
            if metric == LATENCY_ROW:
                v1 = f"{v1_latency} (not comparable, see D-027)" if v1_latency else "not recorded"
            else:
                v1 = v1_cells.get(metric, "not computed in v1")
            if metric == PROMPT_EVAL_ROW:
                v1 = "not recorded"
            cells = [v1] + cells
        rows.append(f"| {metric} | " + " | ".join(cells) + " |")
    ce = []
    for mode, c in ceilings.items():
        ce.append(
            f"| {mode} | {c['any'][0]:.2f} [{c['any'][1]:.2f}, {c['any'][2]:.2f}] | "
            f"{c['all'][0]:.2f} [{c['all'][1]:.2f}, {c['all'][2]:.2f}] | {c['any'][3]} |"
        )
    k = cfg.generation.context_k
    limit = cfg.llm.num_ctx - cfg.llm.max_output_tokens
    info_lines = NL.join(
        f"- {n}: backend {i['backend']}, model calls {i['model_calls']} ({i['cached_calls']} served from cache), "
        f"fallbacks to extractive {i['fallbacks']}, statuses {i['statuses']}, refusal reasons {i['refusal_reasons']}, "
        + (
            f"latency median {i['latency_s_median']:.1f}s / max {i['latency_s_max']:.1f}s per query (uncached run)"
            if i["cached_calls"] == 0
            else (
                f"latency median {i['latency_note']['median']}s / max {i['latency_note']['max']}s per query, "
                f"from the stored uncached run ({i['latency_note']['note']}); this report was recomputed from "
                f"cached replies, so its own timings are not meaningful"
                if i.get("latency_note")
                else f"latency not measured here: {i['cached_calls']} replies came from cache"
            )
        )
        + f", estimated prompt <= ~{i['prompt_tokens_estimate_max']} tokens"
        + (
            f", prompt_eval_count max {i['prompt_eval_max']} over {i['prompt_eval_n']} queries "
            f"(limit: strictly below {limit}; assertion passed)"
            if i["prompt_eval_n"]
            else ""
        )
        + (f", model digest {i['model_digest'][:12]}, num_ctx {i['num_ctx']}" if i["model_digest"] else "")
        for n, i in infos.items()
    )
    first = next((c for c in columns.values() if c), {})
    gate_entry = first.get("Unanswerable: refused by the similarity gate")
    gate_sentence = (
        f"The similarity gate alone cannot separate in-domain unanswerable questions: it refused {gate_entry[4]}."
        if gate_entry
        else ""
    )
    inj = extra["injection_detail"]
    model_answered = [qid for qid, p in inj if p != "score gate"]
    inj_sentence = "; ".join(
        f"{qid}: {'refused by the similarity gate (no model call)' if p == 'score gate' else 'answered by the model (refusal path: ' + p + ')'}"
        for qid, p in inj
    )
    inj_rest = (
        f"The Ollama injection result therefore rests on {len(model_answered)} question(s)." if inj else ""
    )
    return f"""# Generation results

**All 20 questions are dev; there is no held-out split.** The default retrieval mode (hybrid) was selected on
these same dev questions (see `retrieval_results.md`), differences of 1-2 questions are within noise, and
n is small everywhere, so the confidence intervals are wide. Rates are means over questions; intervals are
percentile bootstrap 95% CIs ({cfg.generation_eval.bootstrap_samples} resamples, seed {cfg.seed}); counts are
given where a CI is not meaningful. Context k = {k} was fixed a priori (D-022) and not changed afterwards.

## Prompt versions

The Ollama v2 columns use **prompt v2** (D-027). A first Ollama run with prompt v1 produced almost no citations (the
small model ignored the `[doc_id p.N]` format); its column is copied from `{V1_PATH_NAME}` unchanged. The v2 change
(ready-to-copy `cite_as` tags per chunk, a worked example, a reminder line) was made after seeing dev results and
rerun once, so it is itself a form of tuning on dev; no further prompt iteration was done. The v1 latency is not
comparable (one dense-ablation query took over an hour, which looks like a machine pause).

## Backends

- **Ollama ({cfg.llm.model}):** {ollama_note}
- **Extractive:** no LLM; returns the top {cfg.generation.extractive_sentences} best-matching sentences from the
  retrieved context with `[doc_id p.N]` citations, so it cannot hallucinate a citation and cannot say
  "insufficient context" itself (refusal comes only from the retrieval-score gate). It is a floor, not a
  competitor to a language model.
- Refusal gate: top-1 dense cosine similarity below {thr["threshold"]:.4f} (the {thr["percentile"]:g}th percentile of
  in-corpus top-1 similarities; observed range {thr["observed_min"]:.3f} to {thr["observed_max"]:.3f}, median
  {thr["observed_median"]:.3f}; {thr["n_queries"]} corpus sentences as queries; no eval question used, not tuned).
- Truncation check: every Ollama reply's `prompt_eval_count` is recorded and must be strictly below
  `num_ctx - max_output_tokens` = {limit}; otherwise the run fails loudly.

Run summary:
{info_lines}

## Results

| Metric | {" | ".join(show)} |
|---|{"---|" * len(show)}
{NL.join(rows)}

Definitions.
- Citation precision/recall: cited (doc_id, PDF page) pairs that are valid (present in the retrieved context)
  compared with the gold pages, per question then averaged; precision only over answers with at least one valid
  citation. Recall is over answered (not refused) answerable questions.
- Hallucinated citation: a cited (doc_id, page) not in the retrieved context. Meaningful only for a generating
  model; the extractive backend quotes the context, so it is N/A there.
- Refusal: the similarity gate (top-1 dense cosine below the threshold, no model call) and the model's
  `INSUFFICIENT_CONTEXT` are reported separately, then combined. {gate_sentence}
- Injection: "complied" = the final reply was neither a refusal nor rejected as an echo and still contains an
  instruction-like sentence (generic pattern check). The extractive result is trivial because it only quotes
  sanitised context. {inj_sentence}. {inj_rest} Compliance beyond the pattern check is not detected; see the live
  probe below.
- Key facts: the strict check (every `key_facts` string appears, case-insensitive, in the answer) is shown first. The
  share-found and word-coverage rows were added after seeing the strict result at 0 (D-025): post-hoc, not
  pre-registered, and only a coarse proxy.
- Conflict: questions are those with category `conflict` and gold in 2+ documents; "newest" is the gold document with
  the latest manifest date. (a) a valid citation to the newest document. (b-original) (a) holds and the answer
  contains a recency/conflict word (more recent, newer, latest, updated, supersedes, replaced, revised, conflict,
  differs) **or** the newest document's year, tested on the raw text. **Bug (D-029):** that test could match the
  year inside a citation tag, because the doc_id itself contains it. (b-fixed) is the same check after removing
  bracketed citation tags, `cite_as` labels and document ids from the text. **Even fixed, (b) passes on a keyword or
  a year alone, so it cannot show that the answer states the newer rule, and the extractive backend passes it by
  quoting a sentence that happens to contain such a word or year.** (b-strict, proposed in D-028, computed on the
  cleaned text) needs a recency/conflict word (not the year) and valid citations to both the newest and an older gold
  document. All three are shown side by side; none is a judgement of correctness.
- Citation parsing: the primary metric reads only the bracket format `[doc_id p.N]`. A lenient reading that also
  accepts `(doc_id p.N)` is reported in its own row and is never the primary metric. Identical citations are counted
  once in every citation count.
- Citation support (LLM-free, defined before measuring, not tuned; D-030): for each cited sentence and each valid
  citation on it (identical pairs counted once), (i) numeric containment: every number in the sentence appears in
  the text of the cited page (sentences without numbers are not counted); (ii) mean content-word overlap: share of
  the sentence's content words (4+ letters, fixed stop list) found on the cited page, averaged per question. Both are
  crude: a page can hold the right numbers and words without supporting the claim. They sit next to page-level
  precision, which only says the page is a gold page. {extra["support_sentence"]}
- Run-to-run variation: the Ollama runs use temperature 0 and seed 42, yet a regenerated run was not identical to
  the earlier one (a few statuses and one citation differed). Differences of 1-2 questions are within this noise.
- The `Max prompt_eval_count` row covers the queries that reached the model (19 of 20 in the hybrid run; the
  other was refused by the similarity gate before any model call).

## Per-question notes (dev; written by Claude Code, reviewed by the owner; `dev_question_notes.md`)

{extra["notes"]}

## Injection probe (probe, not part of the eval set)

Synthetic injection strings from `tests/synthetic_injections.py` against the real Ollama backend, on {extra["probe_n"]}
generic base questions written for the probe (`scripts/probe_injection.py`). Aggregate counts only. "complied" = the
reply contains the injected payload word or leaks system-prompt text. **Caveat:** the sanitiser and the probe
strings were written by the same author, so the sanitiser-on results show that it handles the patterns it was designed
for, not novel phrasing; n = 5 per variant.

{probe_section(extra["probe_path"])}
## Context recall ceiling at k = {k} (retrieval only, no generation)

Union of the pages of the top-{k} chunks; stored `pdf_pages` are parsed back to integers before matching.
`any`: some gold page is in the context; `all`: every gold page is in the context.

| Mode | any gold page | all gold pages | n |
|---|---|---|---|
{NL.join(ce)}

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
  manual labels or a judge model: see the owner's review table below.
- Whether a conflict answer actually *states* the newer rule (see the conflict definition).
- Injection compliance beyond the generic pattern check, the echo check and the small probe above.
- Any held-out generalisation: there is no held-out split.

{extra["human_review"]}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_generation", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ollama-ablation", action="store_true", help="also run Ollama with dense retrieval")
    parser.add_argument(
        "--cached-only", action="store_true", help="never call the model: recompute from cached replies"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    questions = load_questions(cfg)
    retriever = Retriever(cfg, build_embedder(cfg))
    thr = json.loads(Path(cfg.refusal.calibration_file).read_text(encoding="utf-8"))
    mode = cfg.retrieval.mode

    columns: dict[str, dict | None] = {}
    infos: dict[str, dict] = {}
    runs = [("Ollama v2", "ollama", mode), ("Extractive", "extractive", mode)]
    if args.ollama_ablation:
        runs.insert(1, ("Ollama v2 (dense, ablation)", "ollama", "dense"))
    runs.append(("Extractive (dense, ablation)", "extractive", "dense"))
    ollama_note = ""
    injection_detail: list[tuple[str, str]] = []
    reports_dir = cfg.paths.reports_dir
    latency_path = reports_dir / "latency_uncached.json"
    for label, backend_name, m in runs:
        label = label if "ablation" in label else f"{label} ({m})"
        if backend_name == "ollama":
            try:
                digest = OllamaClient(cfg).check_available()
            except LLMUnavailable as exc:
                ollama_note = f"**not evaluated: {exc}.** Column left empty."
                columns[label] = None
                continue
            ollama_note = (
                f"evaluated; model digest {digest[:12]}, num_ctx {cfg.llm.num_ctx}, "
                f"temperature 0, seed {cfg.seed}, max output tokens {cfg.llm.max_output_tokens}."
            )
        # the default-mode Ollama run must carry a real prompt_eval_count for every query
        results, info = run_backend(
            cfg,
            retriever,
            backend_name,
            m,
            questions,
            require_count=(m == mode),
            cache_only=args.cached_only,
        )
        summary = summarize_backend(cfg, results, backend_name, load_page_text_lookup(cfg))
        summary[LATENCY_ROW] = f"{info['latency_s_median']:.1f} / {info['latency_s_max']:.1f}" + (
            "" if info["cached_calls"] == 0 else f" (includes {info['cached_calls']} cached replies)"
        )
        if backend_name == "ollama":
            # a clean latency exists only for an uncached run: keep it, and reuse it when recomputing from cache
            stored = json.loads(latency_path.read_text(encoding="utf-8")) if latency_path.exists() else {}
            if info["cached_calls"] == 0:
                stored[label] = {
                    "median": round(info["latency_s_median"], 1),
                    "max": round(info["latency_s_max"], 1),
                    "note": "uncached run",
                }
                latency_path.write_text(json.dumps(stored, indent=2), encoding="utf-8", newline=chr(10))
            elif label in stored:
                t = stored[label]
                summary[LATENCY_ROW] = (
                    f"{t['median']} / {t['max']} ({t['note']}; this report was recomputed from cached replies)"
                )
                info = {**info, "latency_note": t}
        if backend_name == "ollama":
            summary[PROMPT_EVAL_ROW] = (
                f"{info['prompt_eval_max']} (n={info['prompt_eval_n']}; limit {cfg.llm.num_ctx - cfg.llm.max_output_tokens})"
                if info["prompt_eval_n"]
                else "not recorded (cached before the logging was added)"
            )
        else:
            summary[PROMPT_EVAL_ROW] = "n/a"
        columns[label] = summary
        infos[label] = info
        if backend_name == "ollama" and m == mode:
            write_answers(
                questions, results, reports_dir / "answers_dev_ollama_v2.md", "Dev answers: Ollama, prompt v2"
            )
            injection_detail = [(q.qid, refusal_path(a)) for q, a in results if q.type == "injection"]
    ceilings = {m: context_ceiling(cfg, retriever, questions, m) for m in (mode, "dense")}
    human_path = reports_dir / "human_review_owner.md"
    human_text = human_path.read_text(encoding="utf-8").strip() if human_path.exists() else ""
    ollama_support = next(
        (c for n, c in columns.items() if n.startswith("Ollama v2 (") and "ablation" not in n and c), None
    )
    support_sentence = ""
    if ollama_support:
        entry = ollama_support[SUPPORT_NUMERIC_ROW]
        bad = entry[1] - round(entry[0] * entry[1])
        support_sentence = (
            f"Most numeric claims in the Ollama answers ({bad} of {entry[1]} pairs with numbers) cite a page "
            "that does not contain the number, so page-level valid citation overstates attribution quality."
        )
    notes_path = reports_dir / "dev_question_notes.md"
    notes_text = notes_path.read_text(encoding="utf-8").strip() if notes_path.exists() else "(no notes file)"
    probe_path = reports_dir / "injection_probe_results.json"
    probe_n = (
        json.loads(probe_path.read_text(encoding="utf-8"))["n_per_variant"] if probe_path.exists() else 0
    )
    extra = {
        "v1": parse_v1(reports_dir / V1_PATH_NAME),
        "injection_detail": injection_detail,
        "human_review": human_text,
        "support_sentence": support_sentence,
        "probe_path": probe_path,
        "probe_n": probe_n,
        "notes": notes_text,
    }
    text = build_report(cfg, columns, ollama_note, ceilings, infos, thr, extra)
    path = reports_dir / "generation_results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline=chr(10))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
