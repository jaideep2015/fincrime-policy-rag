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

## D-008 Corpus and repo (2026-10-12)
- **Choice:** 10-15 documents (Reg E plus several FinCEN/FATF reports). Public repo contains manifest + download script only; PDFs are gitignored.
- **Why:** Focused enough for a clean eval; avoids redistribution and size issues.

## D-009 Tooling scope for Phase 0 (2026-10-12)
- **Choice:** ruff + pytest only. Pre-commit and CI move to Phase 9.
- **Why:** Keep the scaffold minimal.
