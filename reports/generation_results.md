# Generation results

**All 20 questions are dev; there is no held-out split.** The default retrieval mode (hybrid) was selected on
these same dev questions (see `retrieval_results.md`), differences of 1-2 questions are within noise, and
n is small everywhere, so the confidence intervals are wide. Rates are means over questions; intervals are
percentile bootstrap 95% CIs (2000 resamples, seed 42); counts are
given where a CI is not meaningful. Context k = 8 was fixed a priori (D-022) and not changed afterwards.

## Backends

- **Ollama (qwen2.5:3b-instruct):** **not evaluated: Ollama not reachable at http://localhost:11434: ConnectionError.** Nothing was installed or pulled. Column left empty.
- **Extractive:** no LLM; returns the top 3 best-matching sentences from the
  retrieved context with `[doc_id p.N]` citations, so it cannot hallucinate a citation and cannot say
  "insufficient context" itself (refusal comes only from the retrieval-score gate). It is a floor, not a
  competitor to a language model.
- Refusal gate: top-1 dense cosine similarity below 0.6247 (the 5th percentile of
  in-corpus top-1 similarities; observed range 0.499 to 0.943, median
  0.744; 300 corpus sentences as queries; no eval question used, not tuned).

Run summary (status counts and refusal reasons per column):
- Extractive (hybrid): {'backend': 'extractive', 'fallbacks': 0, 'statuses': {'answered': 19, 'refused': 1}, 'refusal_reasons': {'low_retrieval_score': 1}}
- Extractive (dense, ablation): {'backend': 'extractive', 'fallbacks': 0, 'statuses': {'answered': 19, 'refused': 1}, 'refusal_reasons': {'low_retrieval_score': 1}}

## Results

| Metric | Ollama (hybrid) | Extractive (hybrid) | Extractive (dense, ablation) |
|---|---|---|---|
| Answerable not refused | not evaluated | 1.00 [1.00, 1.00] (n=14) | 1.00 [1.00, 1.00] (n=14) |
| Answerable answered with a valid citation | not evaluated | 1.00 [1.00, 1.00] (n=14) | 1.00 [1.00, 1.00] (n=14) |
| Citation precision (answered with >=1 valid citation) | not evaluated | 0.50 [0.31, 0.68] (n=14) | 0.54 [0.36, 0.70] (n=14) |
| Citation recall (answered, not refused) | not evaluated | 0.45 [0.29, 0.61] (n=14) | 0.48 [0.33, 0.62] (n=14) |
| Hallucinated-citation rate (invalid / all citations, all questions) | not evaluated | 0.00 [0.00, 0.00] (0/57 citations, n=20) | 0.00 [0.00, 0.00] (0/57 citations, n=20) |
| Unanswerable refused | not evaluated | 0.00 [0.00, 0.00] (n=4) | 0.00 [0.00, 0.00] (n=4) |
| Injection pass rate (not complied/echoed; heuristic) | not evaluated | 1.00 [1.00, 1.00] (n=2) | 1.00 [1.00, 1.00] (n=2) |
| Key facts all present in answer (string match) | not evaluated | 0 of 14 (n=14) | 0 of 14 (n=14) |
| Key facts found (share of each question's key_facts strings) | not evaluated | 0.00 [0.00, 0.00] (n=14) | 0.00 [0.00, 0.00] (n=14) |
| Key-fact word coverage (mean share of key-fact words in the answer; coarse proxy) | not evaluated | 0.48 [0.36, 0.60] (n=14) | 0.48 [0.37, 0.61] (n=14) |
| Conflict: cites the newest gold document | not evaluated | 2 of 2 (n=2) | 2 of 2 (n=2) |

Definitions. Citation precision/recall: cited (doc_id, PDF page) pairs that are valid (present in the retrieved
context) compared with the gold pages, per question then averaged; precision only over answers with at least one
valid citation. Hallucinated citation: a cited (doc_id, page) not in the retrieved context (always 0 for the
extractive backend by construction). Refusal accuracy is shown as the two rates above. Injection pass: the final
reply is a refusal, was rejected as an echo, or contains no instruction-like sentence. Key facts: every `key_facts`
string of the question appears (case-insensitive) in the answer text. Conflict: the answer's valid citations
include the most recent gold document by manifest date (questions whose gold has 2+ documents only).

## Context recall ceiling at k = 8 (retrieval only, no generation)

Union of the pages of the top-8 chunks; stored `pdf_pages` are parsed back to integers before matching.
`any`: some gold page is in the context; `all`: every gold page is in the context.

| Mode | any gold page | all gold pages | n |
|---|---|---|---|
| hybrid | 1.00 [1.00, 1.00] | 0.86 [0.64, 1.00] | 14 |
| dense | 1.00 [1.00, 1.00] | 0.64 [0.36, 0.86] | 14 |

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
