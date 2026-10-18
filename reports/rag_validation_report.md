# Validation report: fraud and financial-crime policy assistant (RAG)

**Read this first.** All 20 evaluation questions are dev questions, and there is no held-out split. The default
retrieval mode (hybrid) was chosen on these same questions, one prompt change was made after seeing dev results,
and the sample is small (14 answerable questions, 4 unanswerable, 2 injection). Differences of one or two
questions are within noise. The numbers below describe this system on these questions; they do not show how it
would perform on new ones. Every number comes from a file listed in section 9.

## 1. Scope and corpus

The system answers questions about fraud and anti-money-laundering policy from 15 public documents (FinCEN
advisories and FAQs, 31 CFR 1020.320, FFIEC examination-manual sections, FATF reports, and the CFPB Regulation E
text), 410 pages in total (`data/manifest.csv`, `reports/extraction_qa.md`). Pages 61-79 of the Regulation E document are model
forms (images with poor OCR) and are excluded from the index (D-016). The index holds 696 chunks; Regulation E is
58.5% of them. Everything runs locally on CPU.

## 2. Pipeline and decisions

Page-aware recursive chunks of 512 tokens with 64 overlap (D-019); BAAI/bge-small-en-v1.5 embeddings in a Chroma
index (D-020); dense, BM25 and hybrid retrieval with reciprocal-rank fusion, k = 60 (D-021); generation over the top
8 chunks (D-022) with `qwen2.5:3b-instruct` via Ollama (temperature 0, seed 42, context window 8192) or an extractive
backend that quotes the best sentences (D-025); a refusal gate on top-1 dense similarity, threshold 0.6247, set from
the corpus alone (D-023); input sanitising, delimited prompts and an echo check for injections (D-024); a citation
format change in the prompt (v2) made after a failed first run (D-027). The exact-search grid is in D-032.

## 3. Evaluation-set provenance

The 20 questions (14 answerable, 4 unanswerable, 2 injection) were drafted with an AI assistant outside Claude Code
from the PDFs only, then reviewed and edited by the owner, who owns them; Claude Code did not write or edit them (D-007).
All 20 are dev; there is no held-out split. Labels in the human review (section 4.1) were drafted by the reviewing
assistant from the source PDFs and verified by the owner. Nothing in the pipeline was tuned on individual questions.

## 4. Results

### 4.1 Human review (owner-verified), the main result

Single reviewer, n = 14, no inter-annotator agreement. The block below is copied unchanged from
`reports/human_review_owner.md`; it was not recomputed.

Labels were drafted by the reviewing assistant from the source PDFs and then verified by
the owner. Rubric: correct = all main facts present and nothing material wrong; partial =
a key fact missing or one clear error; incorrect = main claim wrong or answers something
else. "Supported" is "no" if any material claim is cited to a page that does not contain
it. Single reviewer, n=14, no inter-annotator agreement.

Tally: 9 correct, 4 partial, 1 incorrect; cited text supports the claim in 9 of 14.

| qid | correct | partial | incorrect | supported by cited text |
|---|---|---|---|---|
| q01 | x | | | no |
| q02 | | x | | yes |
| q03 | | x | | yes |
| q04 | x | | | yes |
| q05 | x | | | no |
| q06 | x | | | yes |
| q07 | x | | | no |
| q08 | x | | | yes |
| q09 | x | | | yes |
| q10 | | x | | yes |
| q11 | | x | | yes |
| q12 | | | x | no |
| q13 | x | | | yes |
| q14 | x | | | no |

Four of the nine "correct" answers are not supported by the text they cite, so "correct" and "cited correctly" are
different things here.

### 4.2 Retrieval (LLM-free; exact dense search; 14 answerable questions)

`any@k`: some gold page is in the top k. `all@k`: every gold page is in the top k (union over the top-k chunks).
Intervals are percentile bootstrap 95% CIs. Source: `reports/retrieval_experiments.md`, 512/64 rows.

| Mode | any@5 | all@5 | any@8 | all@8 | MRR |
|---|---|---|---|---|---|
| hybrid (default) | 0.86 [0.71, 1.00] | 0.50 [0.29, 0.79] | 1.00 [1.00, 1.00] | 0.86 [0.64, 1.00] | 0.796 [0.626, 0.952] |
| dense | 0.93 [0.79, 1.00] | 0.57 [0.29, 0.86] | 1.00 [1.00, 1.00] | 0.64 [0.36, 0.86] | 0.756 [0.583, 0.917] |
| BM25 | 0.86 [0.71, 1.00] | 0.57 [0.29, 0.86] | 0.86 [0.71, 1.00] | 0.71 [0.50, 0.93] | 0.693 [0.493, 0.893] |

Finding some gold page is much easier than finding all of them: at k = 5 `all@5` is 0.50 to 0.57 against `any@5` of
0.86 to 0.93. The pipeline's published numbers (`reports/retrieval_results.md`) and the exact-search numbers agree to
within rounding. Chunk sizes 256/32 and 384/48 were also tried (retrieval only): no configuration was better than the
baseline with a paired interval excluding 0, three BM25 configurations were worse on MRR, and the default was not
changed (D-032). Hybrid was selected on these dev questions, and the gap to dense is one or two questions.

### 4.3 Generation (Ollama prompt v2 versus extractive; prompt v1 for reference)

Source: `reports/generation_results.md`. n is the number of questions in each row; intervals are bootstrap 95% CIs.

| Metric | Ollama v1 | Ollama v2 | Extractive |
|---|---|---|---|
| Answerable answered with a valid citation (strict) | 0.00 (n=14) | 0.86 [0.64, 1.00] (n=14) | 1.00 (n=14) |
| Citation precision (page level) | n/a | 0.81 [0.58, 1.00] (n=12) | 0.50 [0.31, 0.68] (n=14) |
| Citation recall (page level) | 0.00 (n=13) | 0.40 [0.24, 0.58] (n=14) | 0.45 [0.29, 0.61] (n=14) |
| Invalid citations / all citations | 2/2 | 1/17 | N/A (quotes the context) |
| Key facts, strict string match | 0 of 14 | 0 of 14 | 0 of 14 |

Prompt v2 changed how citations are presented after prompt v1 produced almost none. That change was made after seeing
dev results, so the v2 numbers are not clean. The context-recall ceiling at k = 8 is 1.00 for any gold page and 0.86
[0.64, 1.00] for all gold pages (hybrid): for some questions not every gold page was in the context, so retrieval
limits part of what generation can do, but any gold page was always there. The strict key-fact
match is not informative here: the key facts are paraphrased sentences.

### 4.4 Citation attribution: page level versus support check

A "valid citation" only means the cited page was among the retrieved pages. Among the Ollama v2 answers to the 14
answerable questions, 2 of 10 numeric citation pairs (a sentence containing numbers and one page cited for it) cite a
page that contains all those numbers: 8 of 10 numeric citation pairs cite a page without the number. At sentence level,
2 of 8 sentences with numbers have at least one cited page that contains all their numbers. The one unanswerable
question that was answered instead of refused is not among the 14 reviewed answers; it adds one more numeric pair that
fails (2 of 11 pairs, 2 of 9 sentences when it is included). Mean content-word overlap of cited sentences with the cited page is 0.68 [0.51, 0.85] (n = 11
questions). Both checks are crude (some "numbers" are fragments of document identifiers or list numbering) and say
nothing about correctness. The owner's review reached the same conclusion: 5 of 14 answers are not supported by the
cited text (section 4.1). Page-level "valid citation" overstates attribution quality.

### 4.5 Refusal

Unanswerable questions (n = 4): the similarity gate refused 0 of 4; the model's `INSUFFICIENT_CONTEXT` refused 3 of 4
of those that passed the gate; combined 0.75 [0.25, 1.00]. The gate (5th percentile of in-corpus similarities) cannot
separate in-domain unanswerable questions. One unanswerable question was answered, without inventing names, instead of
refused. The extractive backend cannot decline and refused 0 of 4.

### 4.6 Injection

Injection questions (n = 2): one was handled by the similarity gate before any model call; the other was answered with a
valid citation but did not answer the question (a failure of responsiveness; the model quoted a passage from rank 8 of the
retrieved context, and the injected phrase was still in the prompt). So the Ollama evidence on dev rests on one question.
A separate live probe with synthetic strings (5 per variant, not part of the eval set): with the sanitiser on, 5 of 5,
5 of 5 and 5 of 5 replies answered the underlying question with a valid citation (instruction in the question, a "reveal
the prompt" request, and an instruction inside a retrieved chunk); with the detector switched off, 2 of 5 complied, 2 of
5 answered and 1 was ungrounded. Circularity caveat: the sanitiser and the probe strings were written by the same author,
so the sanitiser-on results show that it handles the patterns it was designed for, not novel phrasing.

### 4.7 Conflict handling

Conflict questions (n = 2): the newest gold document is cited in 2 of 2 answers. Both answers give the newer (2025) position and cite that source, but neither mentions the conflict or the older guidance (0 of 2 on the recency-word check). The (b) checks test whether the
answer acknowledges the conflict: (b-original) passed 2 of 2 only because of a bug (the year matched inside the citation
tag); after the fix (D-029) (b-fixed) passes 0 of 2, and a stricter variant (b-strict) also gives 0 of 2. No claim is made
that the system handles conflicting sources.

## 5. Failure analysis by category (aggregate)

Retrieval by category (`reports/retrieval_results.md`, published pipeline run): the one `list` question had no gold page
in the top 5 in any mode (dense finds a gold page only between ranks 6 and 10, MRR 0.167; BM25 not within the top 10); `paraphrase` is the weakest group for BM25
(any@5 0.75, n = 4); `conflict`, `lookup`, `multi_doc` and `false_premise` have a gold page at rank 1.
Generation by category (counted from the labels in section 4.1 and the category in each heading of
`reports/answers_dev_ollama_v2.md`): `paraphrase` 3 correct and 1 partial of 4, with 1 not supported by its citation;
`multi_page` 2 partial of 2; `lookup` 1 partial; `conflict` 2 correct of 2 with 1 not supported; `multi_doc` 1 correct,
not supported; `numeric` 2 correct with 1 not supported; `list` 1 incorrect, not supported; `false_premise` 1 correct. The
groups have one to four questions each, so these are descriptions, not rates.

## 6. Claims this evidence does NOT support

- Generalisation to other questions, other documents or other regulators: all questions are dev and were written
  by one person for this corpus.
- Held-out performance: there is no held-out split, and the default mode and one prompt change were chosen on dev.
- Answer correctness beyond the 14 reviewed answers, or any accuracy figure for the extractive backend (not reviewed).
- Robustness to novel injections: the live probe used the author's own strings against the author's own sanitiser; the
  dev evidence rests on one question.
- Speed claims beyond the transcribed latency (median 53.5 s, max 70.7 s per query on CPU, copied by hand from a run
  summary; not independently reproducible).
- That cited pages support the claims they are attached to: 5 of 14 reviewed answers were not supported, and 8 of 10
  numeric citation pairs cite a page without the number.
- That one configuration of chunk size, overlap or retrieval mode is better than another (section 4.2).

## 7. Limitations

- A 3B-parameter model on CPU; answers are slow (about a minute per question) and sometimes quote chunks instead of answering.
- Prompt v2 was a post-hoc change after a failed first run (D-027); hybrid retrieval was selected on dev (D-021).
- Run-to-run variation: a regenerated Ollama run differed slightly from the earlier one despite temperature 0 and seed 42
  (for example 14 versus 15 answered, 1 versus 0 invalid citations; D-028).
- Latency of the main run was transcribed by hand from a printed run summary (D-031).
- The default retrieval pipeline is not fully deterministic across processes: 2 of 20 queries differed in 2 processes, both
  non-answerable (one at rank 8, one at ranks 2-8). The likely cause is the approximate dense search in Chroma (not
  verified); two processes is a small sample. A read-only comparison with exact dense search across 4 pipeline processes
  found no difference of any kind for the 14 answerable questions, and the exact search gave identical lists in 2 processes
  (D-032). As a consequence, `--cached-only` recomputation can fail rarely on a cache miss.
- Regulation E is 58.5% of chunks, so small documents can be crowded out of the top k.
- OCR is used for pages without a usable text layer and for flagged pages (D-014); extraction errors are possible.
- Small samples: 4 unanswerable and 2 injection questions; intervals are wide and several are artefacts of n.

## 8. Reproducibility

Python 3.14.8; `requirements.txt` pins the packages (torch 2.14.1, chromadb 1.5.9, sentence-transformers 6.1.0,
streamlit 1.66.0). Ollama model `qwen2.5:3b-instruct`, digest 357c53fb659c. Commands (no `make`; use the venv Python):

```
python -m fincrime_rag.ingestion verify
python -m fincrime_rag.extraction && python -m fincrime_rag.cleaning && python -m fincrime_rag.qa
python -m fincrime_rag.chunking && python -m fincrime_rag.vectorstore build
python -m fincrime_rag.guardrails calibrate
python -m fincrime_rag.evaluate retrieval
python -m fincrime_rag.experiments run
python scripts/eval_generation.py --ollama-ablation        # needs Ollama; add --cached-only to reuse cached replies
streamlit run app/streamlit_app.py
```

Cached model replies live in `.cache/llm`; `chroma_db/` and `data/` derived files are git-ignored and rebuilt by the commands above.

## 9. Evidence: where each headline statement comes from

| Statement | File |
|---|---|
| 15 documents, 410 pages (the sum of the pages column of the per-document table); Reg E pages 61-79 excluded | `data/manifest.csv`, `reports/extraction_qa.md`, DECISIONS.md D-016 |
| 696 chunks; Reg E 58.5% | `reports/retrieval_experiments.md`, `reports/retrieval_results.md` |
| Human review tally and table | `reports/human_review_owner.md` |
| Retrieval any@k, all@k, MRR with CIs | `reports/retrieval_experiments.md` |
| Published pipeline retrieval numbers; category rows | `reports/retrieval_results.md` |
| Grid result and decision rule | `reports/retrieval_experiments.md`, DECISIONS.md D-032 |
| Generation metrics, v1 versus v2, extractive | `reports/generation_results.md`, `reports/generation_results_ollama_prompt_v1.md` |
| Context-recall ceiling | `reports/generation_results.md` |
| 2 of 10 numeric citation pairs supported (2 of 8 sentences at sentence level; q18 adds one failing pair) | `reports/generation_results.md` (citation support), DECISIONS.md D-031 and D-034 |
| Refusal: gate 0 of 4, model 3 of 4, combined 0.75 | `reports/generation_results.md` |
| Injection: one question gate, one failed; probe counts | `reports/dev_question_notes.md`, `reports/injection_probe_results.json` |
| Conflict 0 of 2 after the fix | `reports/generation_results.md`, DECISIONS.md D-029 |
| Latency 53.5 / 70.7 s, hand-transcribed | `reports/latency_uncached.json`, DECISIONS.md D-031 |
| Non-determinism, exact-versus-pipeline result | DECISIONS.md D-032 |
| Model digest 357c53fb659c | `reports/generation_results.md` |
| Evaluation-set provenance | DECISIONS.md D-007 |
