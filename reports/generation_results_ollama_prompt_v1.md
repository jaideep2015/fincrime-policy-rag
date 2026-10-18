# Generation results

**All 20 questions are dev; there is no held-out split.** The default retrieval mode (hybrid) was selected on
these same dev questions (see `retrieval_results.md`), differences of 1-2 questions are within noise, and
n is small everywhere, so the confidence intervals are wide. Rates are means over questions; intervals are
percentile bootstrap 95% CIs (2000 resamples, seed 42); counts are
given where a CI is not meaningful. Context k = 8 was fixed a priori (D-022) and not changed afterwards.

## Backends

- **Ollama (qwen2.5:3b-instruct):** evaluated; model digest 357c53fb659c, num_ctx 8192, temperature 0, seed 42.
- **Extractive:** no LLM; returns the top 3 best-matching sentences from the
  retrieved context with `[doc_id p.N]` citations, so it cannot hallucinate a citation and cannot say
  "insufficient context" itself (refusal comes only from the retrieval-score gate). It is a floor, not a
  competitor to a language model.
- Refusal gate: top-1 dense cosine similarity below 0.6247 (the 5th percentile of
  in-corpus top-1 similarities; observed range 0.499 to 0.943, median
  0.744; 300 corpus sentences as queries; no eval question used, not tuned).

Run summary (status counts and refusal reasons per column):
- Ollama (hybrid): backend ollama, model calls 19, fallbacks to extractive 0, statuses {'refused': 5, 'ungrounded': 15}, refusal reasons {'low_retrieval_score': 1, 'model_insufficient': 4}, latency median 6.4s / max 16.8s per query (includes cached replies), prompt <= ~7284 tokens of num_ctx 8192, model digest 357c53fb659c
- Ollama (dense, ablation): backend ollama, model calls 19, fallbacks to extractive 0, statuses {'refused': 5, 'ungrounded': 15}, refusal reasons {'low_retrieval_score': 1, 'model_insufficient': 4}, latency median 43.2s / max 4570.1s per query (includes cached replies), prompt <= ~7475 tokens of num_ctx 8192, model digest 357c53fb659c
- Extractive (hybrid): backend extractive, model calls 19, fallbacks to extractive 0, statuses {'answered': 19, 'refused': 1}, refusal reasons {'low_retrieval_score': 1}, latency median 0.1s / max 3.7s per query (includes cached replies), prompt <= ~7284 tokens of num_ctx 8192
- Extractive (dense, ablation): backend extractive, model calls 19, fallbacks to extractive 0, statuses {'answered': 19, 'refused': 1}, refusal reasons {'low_retrieval_score': 1}, latency median 0.1s / max 0.2s per query (includes cached replies), prompt <= ~7475 tokens of num_ctx 8192

## Results

| Metric | Ollama (hybrid) | Ollama (dense, ablation) | Extractive (hybrid) | Extractive (dense, ablation) |
|---|---|---|---|---|
| Answerable not refused | 0.93 [0.79, 1.00] (n=14) | 0.93 [0.79, 1.00] (n=14) | 1.00 [1.00, 1.00] (n=14) | 1.00 [1.00, 1.00] (n=14) |
| Answerable answered with a valid citation | 0.00 [0.00, 0.00] (n=14) | 0.00 [0.00, 0.00] (n=14) | 1.00 [1.00, 1.00] (n=14) | 1.00 [1.00, 1.00] (n=14) |
| Citation precision (answered with >=1 valid citation) | n/a (n=0) | n/a (n=0) | 0.50 [0.31, 0.68] (n=14) | 0.54 [0.36, 0.70] (n=14) |
| Citation recall (answered, not refused) | 0.00 [0.00, 0.00] (n=13) | 0.00 [0.00, 0.00] (n=13) | 0.45 [0.29, 0.61] (n=14) | 0.48 [0.33, 0.62] (n=14) |
| Hallucinated-citation rate (invalid / all citations) | 1.00 [1.00, 1.00] (2/2 citations, n=20) | n/a (n=20) | N/A (extractive quotes the context; 0 by construction) | N/A (extractive quotes the context; 0 by construction) |
| Unanswerable: refused by the similarity gate | 0 of 4 (n=4) | 0 of 4 (n=4) | 0 of 4 (n=4) | 0 of 4 (n=4) |
| Unanswerable: INSUFFICIENT_CONTEXT (of those that passed the gate) | 3 of 4 (n=4) | 3 of 4 (n=4) | N/A (extractive cannot decline) | N/A (extractive cannot decline) |
| Unanswerable: refused, combined | 0.75 [0.25, 1.00] (n=4) | 0.75 [0.25, 1.00] (n=4) | 0.00 [0.00, 0.00] (n=4) | 0.00 [0.00, 0.00] (n=4) |
| Injection: complied / echoed an instruction (pattern check) | 0 of 2 (n=2) | 0 of 2 (n=2) | 0 of 2 (n=2) | 0 of 2 (n=2) |
| Injection: refused or rejected | 1 of 2 (n=2) | 1 of 2 (n=2) | 1 of 2 (n=2) | 1 of 2 (n=2) |
| Key facts: all present, strict string match | 0 of 14 (n=14) | 0 of 14 (n=14) | 0 of 14 (n=14) | 0 of 14 (n=14) |
| Key facts: share found by strict match (post-hoc, D-025, not pre-registered) | 0.00 [0.00, 0.00] (n=14) | 0.00 [0.00, 0.00] (n=14) | 0.00 [0.00, 0.00] (n=14) | 0.00 [0.00, 0.00] (n=14) |
| Key facts: word coverage (post-hoc, D-025, not pre-registered; coarse proxy) | 0.48 [0.37, 0.59] (n=14) | 0.58 [0.47, 0.68] (n=14) | 0.48 [0.36, 0.60] (n=14) | 0.48 [0.37, 0.61] (n=14) |
| Conflict (a): newest gold document is cited | 0 of 2 (n=2) | 0 of 2 (n=2) | 2 of 2 (n=2) | 2 of 2 (n=2) |
| Conflict (b): answer states the newer rule (pattern check, see definitions) | 0 of 2 (n=2) | 0 of 2 (n=2) | 2 of 2 (n=2) | 2 of 2 (n=2) |

Definitions.
- Citation precision/recall: cited (doc_id, PDF page) pairs that are valid (present in the retrieved context)
  compared with the gold pages, per question then averaged; precision only over answers with at least one valid
  citation. Recall is over answered (not refused) answerable questions.
- Hallucinated citation: a cited (doc_id, page) not in the retrieved context. Meaningful only for a generating
  model; the extractive backend quotes the context, so it is N/A there.
- Refusal: the similarity gate (top-1 dense cosine below the threshold, no model call) and the model's
  `INSUFFICIENT_CONTEXT` are reported separately, then combined. The similarity gate alone cannot separate in-domain unanswerable questions: it refused 0 of 4.
- Injection: "complied" = the final reply was neither a refusal nor rejected as an echo and still contains an
  instruction-like sentence (generic pattern check). The extractive result is trivial because it only quotes
  sanitised context; the Ollama result is the one that counts. n is tiny and compliance beyond the pattern check is
  not detected.
- Key facts: the strict check (every `key_facts` string appears, case-insensitive, in the answer) is shown first. The
  share-found and word-coverage rows were added after seeing the strict result at 0 (D-025): post-hoc, not
  pre-registered, and only a coarse proxy.
- Conflict: questions are those with category `conflict` and gold in 2+ documents; "newest" is the gold document with
  the latest manifest date. (a) a valid citation to the newest document; (b) (a) holds and the answer contains a
  recency/conflict word (more recent, newer, latest, updated, supersedes, replaced, revised, conflict, differs) or the
  newest document's year. (b) is a crude pattern check, not a judgement of correctness.

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
- Whether a conflict answer actually *states* the conflict (only the newest-source citation is checked)
- Injection compliance beyond the generic instruction-pattern heuristic and the echo check.
- The Ollama backend, unless the column above shows numbers.
- Any held-out generalisation: there is no held-out split.
