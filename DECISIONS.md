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
