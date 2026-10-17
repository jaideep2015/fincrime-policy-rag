# DECISIONS

Log of design choices. Format: date, options considered, choice, why, how to revisit.

## D-001 Vector store: Chroma (2026-10-12)
- **Options:** Chroma, FAISS.
- **Choice:** Chroma, behind a `VectorStore` interface.
- **Why:** Persistent storage and metadata (doc_id, page) filtering built in, which keeps citations simple. At ~10-30k chunks FAISS's speed advantage is irrelevant, and FAISS would need a hand-built ID/metadata sidecar.
- **Revisit:** If the corpus grows by orders of magnitude or ANN index tuning becomes the point; swap via the interface.

## D-002 Embedding model: BAAI/bge-small-en-v1.5, local (2026-10-12)
- **Options:** bge-small, bge-base, all-MiniLM, hosted embedding APIs.
- **Choice:** bge-small-en-v1.5 (384-d).
- **Why:** Free, CPU-friendly, strong retrieval quality for its size, reproducible with no API key.
- **Revisit:** Embedding-model comparison is an optional stretch item (Phase 7b).

## D-003 Chunking baseline: recursive, 512 tokens / 64 overlap, page-aware (2026-10-12)
- **Options:** recursive 512/64, section/heading-aware, fixed 256/0.
- **Choice:** recursive 512/64, never losing the source page.
- **Why:** Common defensible default; keeps exact page citations. Chunk size/overlap grid is a core experiment (Phase 7); section-aware chunking is an optional stretch item.
- **Revisit:** After Phase 7 results.

## D-004 Compute and LLM: CPU-only, local Ollama (2026-10-12)
- **Choice:** Laptop CPU only; generation and judging use a small local model via Ollama. The specific model is chosen in Phase 5 from 2-3 options with trade-offs.
- **Why:** No API key or API budget is available; also makes the repo runnable by anyone.
- **Consequence:** Generation is slow and the judge is noisy; mitigated with caching and by reporting judge agreement with manual labels.

## D-005 LLMClient interface (2026-10-12)
- **Choice:** Generation goes through an `LLMClient` interface with providers `ollama` (default) and `anthropic` (optional, disabled by default, documented for readers with their own key).
- **Why:** Keeps the pipeline provider-agnostic; nothing requires an API key to run or test.

## D-006 Retrieval evaluation is LLM-free (2026-10-12)
- **Choice:** recall@k, MRR, hit-rate are computed from gold doc+page labels only.
- **Why:** Deterministic, cheap, and not affected by judge noise.

## D-007 Eval set is user-written (2026-10-12)
- **Choice:** 20 starter questions (Phase 1.5), growing to 40-60 (~10 unanswerable, ~5 injection). Claude does not write or edit them.
- **Why:** Avoids leakage and keeps ground truth independent of the system under test.
- **Provenance:** Questions were drafted and cross-checked with an AI assistant outside Claude Code, from the PDFs only (no retrieval output), then reviewed and edited by the author. Claude Code did not write or edit them.

## D-008 Corpus and repo (2026-10-12)
- **Choice:** 10-15 documents (Reg E plus several FinCEN/FATF reports). Public repo contains manifest + download script only; PDFs are gitignored.
- **Why:** Focused enough for a clean eval; avoids redistribution and size issues.

## D-009 Tooling scope for Phase 0 (2026-10-12)
- **Choice:** ruff + pytest only. Pre-commit and CI move to Phase 9.
- **Why:** Keep the scaffold minimal.

## D-010 Page numbering: store both PDF index and printed label (2026-10-13)
- **Choice:** Every page record has `pdf_page` (1-based, always present, the key for citations and retrieval metrics) and `page_label` (the label the PDF defines, else null). When the PDF defines no label, cleaning records `printed_page_guess` from the page-number line it strips at the page edge.
- **Why:** Printed numbers are unreliable to parse (roman numerals, "4-12", cover pages), but readers expect them in citations. Metrics use the PDF index only.
- **Revisit:** If citations should show printed pages, check `printed_page_guess` coverage in the QA report first.

## D-011 Header/footer removal is detected from the data and fully logged (2026-10-13)
- **Choice:** Candidates are the first/last 3 non-empty lines per page. Bare page numbers are always removed; other lines are removed when their digit-normalised form appears on at least 40% of pages (skipped for documents under 5 pages). Every removed line is written to `data/processed/logs/<doc_id>_removed_lines.csv`.
- **Why:** Hard-coded patterns do not generalise across publishers, and repeated running heads pollute embeddings and BM25. Removing only edge lines limits the risk to body text; the log makes that risk reviewable.
- **Known limits:** Short documents keep their running heads; a bare number at a page edge (e.g. a table value) can be mistaken for a page number; words hyphenated across a page break are not repaired.

## D-012 Line-end hyphenation repair uses the document's own vocabulary (2026-10-13)
- **Choice:** For "left-" + newline + "right": join if the joined word occurs elsewhere in the document; keep the hyphen if the hyphenated compound occurs elsewhere or both parts are standalone words of 3+ letters; otherwise join. Fragments created by the breaks themselves are excluded from the vocabulary.
- **Why:** Blind joining turns "anti-money" into "antimoney", which would hurt retrieval for the most common term in the corpus.

## D-013 Document acquisition: script for US-government hosts, manual for FATF and FFIEC (2026-10-13)
- **Choice:** `source=script` rows download from hosts allow-listed in `config.yaml` (FinCEN, GovInfo) with full TLS verification, a `%PDF-` header check, a size check and sha256. FATF and FFIEC return HTTP 403 to scripts, so those four files are placed manually under fixed filenames; `verify` records their sha256. No attempt is made to bypass the blocks.
- **Manifest columns:** `source` (script/manual), `filename`, `date` (publication) and `version_date` (revision date printed in the PDF; for FFIEC and CFR editions).

## D-014 Text-coverage check and OCR merge (2026-10-14, revised after review)
- **Problem:** some PDFs hold content only as images. The COVID advisory (FIN-2020-A003) had 2 of its 18 red-flag bullets in the text layer; Reg E pages 61-79 have almost no text layer.
- **Choice:** every page is rendered and OCR'd once (RapidOCR via pip, cached by file sha256). Coverage = text-layer words / OCR words; a page below `ocr.coverage_threshold` (0.90) is flagged. OCR boxes under `ocr.min_confidence` and red-flag icon noise (short, low-confidence boxes in the left gutter; the icons are numbered flags that OCR reads as digits) are dropped everywhere.
- **Merge, not replace:** a first version replaced flagged pages with OCR text; that lost text-layer content and introduced OCR typos. Flagged pages with a usable text layer (at least `merge_min_layer_words` words and at least `merge_min_box_coverage` of OCR boxes already on text-layer words) now keep the text layer as the base. Flagged pages are re-OCR'd with looser detection settings (`recover_*`); from that pass only boxes that are not already covered (at least `merge_covered_fraction` of their area on text-layer words, compared in PDF points) are kept. Text-layer rows and OCR-only boxes are sorted by vertical position (footnotes land at the bottom); a run of OCR-only lines is its own paragraph. `extraction_method` is `merged`.
- **No usable text layer** (image-only covers, scanned forms): OCR text alone, `extraction_method = "ocr"`.
- **Threshold 0.90, not 0.80:** COVID page 5 (two missing bullets) scores 0.84; pages with complete text score 0.95-1.10.
- **Verification:** QA reports "text-layer lines lost" (text-layer lines of 5+ words absent from the final text, excluding heads that cleaning removed on purpose); it is 0 on every merged page. An acceptance test on the real COVID PDF checks 18 red flags, four previously lost text-layer lines, the absence of the OCR typos, and printed pages 4-7.
- **Tuning:** the merge rule (`merge_min_layer_words` 100, `merge_min_box_coverage` 0.35) was tuned empirically on this corpus; COVID page 7 (122 layer words, box coverage 0.394) is the tightest case.
- **Limits:** the merge orders rows by vertical position and so assumes single-column pages; every merged page in this corpus is single-column. Reg E page 74 is merged (it has a text layer of 125 words); the other model-form pages are OCR-only.

## D-015 Running-head rules (2026-10-14)
- **Choice:** `normalize_line` collapses digit runs to one `#`, maps U+F07C to `|`, sorts the two sides of `A | B` heads and drops all whitespace, so "6 | TITLE", "TITLE | 7" and letter-spaced or OCR'd forms of the same head are one pattern. Repeat detection scans 5 edge lines, bare page numbers only 3. Configured regexes (CFR "§ n.n", "Pt. n, Supp. I", "N | TITLE") match only the first 3 lines; their printed page and `section_hint` are recorded before the line is removed, and lines those regexes handle are excluded from repeat detection so a bare "§ 1005.99" lower on a page survives.
- **Also:** the Symbol-font bullet U+F0B7 becomes "•", and a bullet extracted on its own line is joined to the next line.

## D-016 Reg E Appendix A model forms are tagged and kept out of the index (2026-10-14)
- **Choice:** pages 61-79 of `cfr_12_1005_reg_e` (images of sample forms, poor OCR: spacing lost, merged heads) get `content_type = "model_form"` via `cleaning.page_tags` in `config.yaml`; every other page is `"body"`. `indexing.exclude_content_types` (default `[model_form]`) keeps them out of the index; they stay in `data/processed` and can be re-included by editing the config. `indexable_pages()` applies the rule.
- **Why:** they add little retrieval value and would put garbled text in the index and in citations.

## D-017 printed_page_guess sanity check (2026-10-14)
- **Choice:** per document, the modal (printed - PDF page) offset over numeric guesses is computed (needs 3+ numeric guesses, with the mode on at least half of them); guesses that deviate are set to None and the original kept in `printed_page_guess_rejected`. Non-numeric guesses ("F-1") are left alone. The QA report shows how many were nulled per document.
- **Why:** stray numbers read from covers or figures (FATF 2020 had "10" and "19" at an offset of 9 and 16 against the document's -2) would otherwise show up in citations.

## D-018 Eval set schema and validator (2026-10-14)
- **Choice:** `data/eval/qa_set.jsonl`, one JSON object per line. Fields: `qid` (unique), `question`, `type` (`answerable` | `unanswerable` | `injection`), `expected_behavior` (`answer` | `refuse` | `resist_injection`, tied one-to-one to `type`), `category` (`lookup`, `definition`, `list`, `numeric`, `multi_page`, `multi_doc`, `conflict`, `paraphrase`, `false_premise`, `out_of_corpus`, `injection`), `gold` (list of `{doc_id, pages}` with PDF page indexes; empty for unanswerable and injection), `reference_answer`, `key_facts` (1-4 short strings), `split` (`dev` | `test`, required), optional `notes`. The 20 starter questions are all `dev`; the held-out `test` split is written in Phase 6.
- **Validator** (`python -m fincrime_rag.evalset validate`, limits in `config.yaml` `evalset:`):
  - errors: duplicate qid or question (compared after normalising case and punctuation), invalid enum values, type/expected_behavior mismatch, unknown doc_id, page outside the PDF's page count, non-integer or duplicate pages, empty gold for answerable or non-empty gold otherwise, key_facts outside 1-4, missing or unknown fields, bad JSON lines;
  - warnings (do not fail): a gold page in a non-body page (e.g. Reg E model forms), more than 3 questions on one document, a document with no questions, `multi_doc` with fewer than two gold documents, `multi_page` with no document having two or more gold pages;
  - summary counts by type, category, split and document.
- **Page lookup:** `python -m fincrime_rag.evalset pages <doc_id>` prints `pdf_page`, `page_label`, `printed_page_guess` and `content_type`, never page text, so PDF indexes and printed numbers are not confused. Gold pages are always PDF indexes.
- **Independence:** only `qa_set.jsonl` is read (the format example is a `.md` file the validator ignores); reports name qids, fields, doc_ids and counts but never repeat question, answer, key-fact or note text. Nothing in the pipeline is tuned against the eval set, and Claude does not write, edit or suggest questions.

## D-019 Chunker: recursive, page-aware, token-exact (2026-10-17)
- **Options considered:** (a) fixed-size windows over the whole document, (b) recursive split on paragraph / line / sentence / word, packed greedily (chosen), (c) section-aware splitting (kept as the Phase 7b stretch, `chunking.strategy`).
- **Choice:** 512 tokens / 64 overlap (D-003), tokens counted with the embedding model's own tokenizer, budget `chunk_size - special_tokens` (2) so chunk plus [CLS]/[SEP] never exceeds bge's 512 window. Atoms (the largest paragraph, line, sentence or word pieces that fit) are packed greedily; the budget is checked on the real joined text because tokenizing pieces separately and joining them can differ by a few tokens. The next chunk starts with up to 64 tokens of trailing atoms. A trailing chunk with fewer than 20 new tokens is folded into the previous one.
- **Page awareness:** a chunk may span a page boundary and lists every page it touches (`pdf_pages`); `printed_page_guess`, `section_hint` and `content_type` come from its first page. Pages tagged `model_form` (D-016) and empty pages are skipped. Each chunk has `chunk_id` (`doc:pPPPP:cNNNN`), `n_tokens` and `text_hash` (embedding cache key).
- **CFR sections:** for doc_ids starting with `cfr_` (`chunking.section_doc_prefixes`), heading lines `§ N.N` (bare, or followed by a capitalised title) are tracked through the text; a chunk's `section` is the section in effect after its first atom. Cross-references are rejected because a heading must be the whole line.
- **Known limits:** a short section can start and end inside one chunk and so never appears as a chunk's `section`; a cross-reference wrapped onto a line of its own (for example `§ 1004.4`) can be mistaken for a heading (one such chunk in Reg E). `section` is metadata only; retrieval does not use it.
- **Result:** 696 chunks over 15 documents, median 480 tokens, maximum 510.
- **Revisit:** change `chunking:` in `config.yaml` and run `make index`; the grid over size and overlap is Phase 7.

## D-020 Embeddings, cache and Chroma index (2026-10-17)
- **Choice:** BAAI/bge-small-en-v1.5 on CPU via sentence-transformers (D-002), normalised vectors, bge's query instruction prepended to queries only (`embedding.query_instruction`). `Embedder` is a small interface; `CachedEmbedder` stores one vector per sha256(model + text) under `.cache/embeddings/<model>/`, so a rebuild with unchanged chunks computes nothing (first full build of 696 chunks: about 3 minutes on this CPU; the second: 696 cached, 0 computed).
- **Index:** Chroma `PersistentClient` in `chroma_db/`, one collection, cosine space, `chunk_id` as id. `build_index` drops and recreates the collection from the chunk files, so it is idempotent and cannot leave stale entries. Chroma metadata cannot hold lists, so `pdf_pages` is stored as "4,5" plus integer first/last page.
- **One command:** `make index` (re-chunk, then build); where `make` is not installed, `python -m fincrime_rag.chunking && python -m fincrime_rag.vectorstore build`.
- **Environment:** `pip install --dry-run sentence-transformers chromadb rank_bm25` resolved on Python 3.14.8 (torch 2.14.1, chromadb 1.5.9), so no 3.12 venv was needed.
- **Revisit:** swap `embedding.model` and rebuild (the cache is keyed by model, so models do not collide); the embedding-model comparison is the Phase 7b stretch.

## D-021 Retrieval modes, LLM-free evaluation and the default mode (2026-10-17)
- **Modes** behind one `retrieve(query, k, mode=None)` (`retrieval.py`), each hit carrying doc_id, pdf pages, printed-page guess and section: `dense` (Chroma cosine), `bm25` (rank_bm25 Okapi, k1 1.5, b 0.75, lower-cased word tokens that keep section numbers like `1005.11` whole; built from the same chunk texts read back from Chroma), `hybrid` (reciprocal rank fusion, `1/(60 + rank)`, over the top 50 of each ranker). RRF was chosen over score blending because cosine similarities and BM25 scores are not on one scale.
- **Evaluation** (`python -m fincrime_rag.evaluate retrieval`, writes `reports/retrieval_results.md`): answerable questions of the splits in `evaluation.splits` (dev only; the test split is scored once, in Phase 6). A hit covers a gold page when doc_id matches and the chunk's page list contains the page. Metrics: `any@k` (any gold page in the top k, the usual recall@k), `frac@k` (share of gold pages covered), MRR, and `doc-hit@5` (a gold document in the top 5, page ignored; this is how "hit rate" is defined here, because at the same k it would otherwise equal `any@5`). Only aggregates are printed; no question text or per-question rows leave the harness.
- **Default mode: hybrid**, picked from aggregate dev numbers only (n = 14 answerable dev questions; 6 of the 20 are unanswerable or injection and have no gold pages). Overall: hybrid has the best MRR (0.796 vs 0.756 dense, 0.693 BM25), any@1 (0.71 vs 0.64) and frac@10 (0.96 vs 0.85), and any@10 of 1.00; dense is slightly ahead at any@5 (0.93 vs 0.86) and frac@5. These gaps are one or two questions, well inside noise at this n, so the choice is a tentative default: Phase 6 grows the set and Phase 7 puts bootstrap CIs on the dense-versus-hybrid comparison. BM25 alone is weakest (doc-hit@5 0.86), mostly on paraphrased questions, as expected for a lexical ranker.
- **No special cases:** nothing in retrieval or chunking was changed after looking at individual questions or results; config values are the pre-agreed ones (512/64, RRF k 60).
- **Revisit:** `retrieval.mode` in `config.yaml`; rerun the evaluate command.

## D-022 Generation context size fixed a priori: k = 8 (2026-10-17, logged before any generation evaluation)
- **Choice:** the generator receives the top 8 chunks (`generation.context_k`), in retrieval order. Chosen from first principles, not from results: 8 chunks of at most 510 tokens is about 4,100 tokens, which fits the explicit 8192-token Ollama window (`llm.num_ctx`) together with the prompt rules and a 400-token answer, and it is deep enough that multi-page and multi-document answers (where `all@k` is far below `any@k`) have a chance to be covered. The retrieval report showed `all@10` well above `all@5` for hybrid, which is the reason not to use the earlier default of 5.
- **Not to be changed after seeing generation results.** Dense versus hybrid at the same k may be run as an ablation and is reported as an ablation, not as a re-tune. The context-recall ceiling (retrieval only) is reported at k = 8 so retrieval limits can be told apart from generation failures.
- **Fixed ground rules for this batch:** `data/eval/qa_set.jsonl` is frozen at 20 questions, all `dev`; there is no held-out split and every report says so. Prompts, thresholds, k and fusion are never tuned on individual eval questions; any fix after a failure must be generic and logged here.

## D-023 Refusal: retrieval-score gate, calibrated a priori from the corpus (2026-10-17)
- **Options:** (a) threshold tuned on the four unanswerable eval questions (rejected: that is tuning on the eval set); (b) threshold from `data/eval/calibration.jsonl` (the file does not exist yet, so not used; if the user supplies it, recalibrate from it); (c) a percentile of in-corpus top-1 similarities fixed in advance (chosen).
- **Method** (`python -m fincrime_rag.guardrails calibrate`, config `refusal:`): 300 sentences (20 per document, fixed seed 42) are used as queries with bge's query instruction; chunks containing the sentence are skipped; the best remaining top-1 dense cosine similarity is kept. The threshold is the 5th percentile of those 300 values. Dense cosine is used in every retrieval mode because RRF scores depend only on ranks.
- **Value:** 0.6247. Observed range 0.499 to 0.943, median 0.744 (p25 0.699, p75 0.788). bge-small cosine similarities are compressed into a narrow band, so a percentile of in-corpus values mostly separates "clearly off-topic" from "on-topic", not "answerable" from "unanswerable but on-topic".
- **Consequence, stated not fixed:** at a 5th-percentile cut-off the gate refuses very little. On the dev set it refused 1 of 20 questions and 0 of the 4 unanswerable ones with the extractive backend, which has no other way to decline. The model's `INSUFFICIENT_CONTEXT` is the second signal for the Ollama backend (not evaluated here). Nothing was changed after seeing this.
- **Revisit:** supply `calibration.jsonl` (answerable and unanswerable calibration questions, split `calibration`) and choose the threshold from it; or raise `refusal.percentile` with that data in hand.

## D-024 Injection handling (2026-10-17)
- **Layers:** (1) the question is sanitised before retrieval: sentences matching generic instruction patterns (`injection.patterns`, written a priori, none taken from the eval set) are dropped unless nothing answerable would remain; (2) the system prompt is non-overridable and tells the model the question and context are data; chunks and question sit inside explicit tags and tag-like text inside them is stripped; (3) the extractive backend never quotes an instruction-like sentence; (4) output check: a reply whose words (citations excluded) come at least `echo_overlap` (0.8) from one injected sentence in the question or context is rejected (`rejected_injection`).
- **Tests** use synthetic injections written for the tests, including one embedded in a retrieved chunk. No eval text appears anywhere under `src/`; `git grep` for the distinctive strings of the eval set returns nothing.
- **Limits:** pattern matching misses paraphrased instructions; the evaluation's "injection pass" is a heuristic (refused, rejected as an echo, or no instruction-like sentence in the reply) and cannot see every form of compliance.

## D-025 Generation backends and evaluation (2026-10-17)
- **Backends:** `OllamaClient` (qwen2.5:3b-instruct, temperature 0, seed 42, `num_ctx` 8192 sent explicitly, `num_predict` 400, model digest read from `/api/tags` and returned with every result, outputs cached by input hash) and `ExtractiveClient` (no LLM: top 3 sentences by idf-weighted question overlap, each cited with the page found by locating the sentence in the page texts). `FallbackClient` switches to extractive when Ollama is unreachable and records `fallback` and the reason. The assembled prompt is checked against `num_ctx` (characters / 3.0 as a conservative estimate) and fails loudly.
- **Ollama was not installed on this machine** (no `ollama` binary, nothing listening on 11434), so the Ollama backend is implemented and unit-tested with a mocked HTTP session but **not evaluated**; its column in `generation_results.md` says so. Nothing was installed or pulled.
- **Evaluation** (`scripts/eval_generation.py`, `reports/generation_results.md`): aggregate only, bootstrap 95% CIs (2000 resamples, seed 42), n per metric, extractive with the default mode plus a dense ablation (reported as an ablation). Context recall ceiling reported at k = 8 for hybrid and dense. All 20 questions are dev; there is no held-out split.
- **Metric changes after the first run, both generic:** the first run's all-or-nothing key-fact string match gave 0 of 14, so a graded share and a word-overlap coverage were added. A diagnostic on aggregates showed key facts are paraphrased sentences (about 14 words each) and only 1 of 44 appears verbatim even in the full 8-chunk context, so string matching is not a meaningful measure for this set; the word-coverage row is a coarse proxy only. No prompt, threshold, k or retrieval setting was changed.
- **Package layout:** the clients live in `src/fincrime_rag/llm/` and the rest in `src/fincrime_rag/` (`generation.py`, `prompting.py`, `citations.py`, `guardrails.py`), following PLAN.md, not a top-level `src/generation/`.
