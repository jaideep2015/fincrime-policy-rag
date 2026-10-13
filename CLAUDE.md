# CLAUDE.md: Fraud & Financial Crime Policy Assistant

## Purpose
Portfolio-grade RAG over public AML/fraud documents. The owner must be able to explain every design choice in an interview. See PLAN.md for phases and DECISIONS.md for logged choices.

## Working rules
1. Plan first. Do not start a phase without the user's go-ahead.
2. One phase at a time. Stop at the phase's definition of done and wait for review. Never begin the next phase.
3. Key decisions (embedding model, chunking, vector store, reranker, generation/judge model) belong to the user: present 2-3 options with trade-offs, recommend one, ask, then log it in DECISIONS.md (date, options, choice, why, how to revisit).
4. Never write, edit, or generate the user's eval set (`data/eval/qa_set.jsonl`). Schema validators and harness code that read it are fine. Do not tune prompts, chunking, or thresholds by inspecting held-out questions.
5. Explain before coding: a short plain-language rationale for each module or file.

## Engineering conventions
- Python, `src/` layout, type hints, ruff + pytest. One module per pipeline stage; stages exchange data via files and interfaces, not hidden globals.
- All tunables live in `config.yaml`, validated by pydantic. No magic numbers in code.
- Everything runs locally on CPU. The default LLM is a small Ollama model behind the `LLMClient` interface. An `anthropic` provider may exist but is optional and disabled by default; no code, test, or evaluation may require an API key.
- `.env` is gitignored; `.env.example` lists variable names only. Never print or log secrets.
- Every chunk and every answer carries doc_id + page; citations are validated against the retrieved chunks.
- Determinism: fixed seeds, temperature 0 for evaluation, on-disk caching of embeddings and LLM outputs keyed by input hash.
- Tests mock the LLM; no network or model calls in unit tests.
- Do not commit `data/raw`, `data/extracted`, `data/processed`, vector indexes, caches, or `.env`.

## Evaluation integrity
- Retrieval metrics (recall@k, MRR, hit-rate) are computed without any LLM.
- Split the Q&A set into dev (tuning) and held-out test (scored once at the end).
- Report metrics with sample size and bootstrap CIs; change one factor per experiment.
- The judge is a small local model and is noisy: fix and version its rubric, and report its agreement with the user's manual labels.

## Definition of done (every phase)
Tests pass, ruff clean, DECISIONS.md updated, README updated where relevant, and a short summary of what changed and what to review.
