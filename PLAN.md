# Plan: Fraud & Financial Crime Policy Assistant (RAG)

End-to-end RAG over public fraud/AML documents (FinCEN, FATF, FFIEC, CFPB Reg E) that answers with doc + page citations, refuses when evidence is weak, and ships with a quantitative evaluation and validation layer. Everything runs locally on CPU; no API key is needed to run, test, or evaluate.

## Decisions already made
See [DECISIONS.md](DECISIONS.md): Chroma, bge-small-en-v1.5, recursive 512/64 page-aware chunking, CPU-only, local LLM via Ollama behind an `LLMClient` interface (`anthropic` provider optional and disabled by default).

## Repo structure
```
├── CLAUDE.md  PLAN.md  DECISIONS.md  README.md  .gitignore  .env.example
├── config.yaml  pyproject.toml
├── data/  manifest.csv, raw/ extracted/ processed/ (gitignored), eval/qa_set.jsonl (user-written)
├── src/fincrime_rag/
│   config.py ingestion.py extraction.py cleaning.py chunking.py embeddings.py
│   vectorstore.py retrieval.py prompting.py guardrails.py generation.py
│   llm/ (base.py, ollama_client.py, anthropic_client.py)
│   evaluation/ (retrieval_metrics.py, generation_metrics.py, judge.py, runner.py)
│   api/
├── app/streamlit_app.py
├── experiments/  configs/, results/<run_id>/
├── reports/rag_validation_report.md
└── tests/
```
Dockerfile, pre-commit and GitHub Actions arrive in Phase 9.

## Phases (pause for review after each)
| Phase | Scope | Definition of done |
|---|---|---|
| 0 Scaffold | Layout, pyproject (ruff + pytest only), config loader, .env.example, .gitignore, PLAN/CLAUDE/DECISIONS, README stub | `ruff check` and `pytest` pass; config.yaml loads and validates; no secrets in repo; files explained |
| 1 Ingestion, extraction, cleaning | Manifest download with sha256; PyMuPDF per-page text; cleaning; extraction QA stats | Per-page text with page numbers preserved; QA table reviewed; 5 pages spot-checked; tests on a fixture PDF |
| 1.5 Starter eval set | **User writes** 20 questions (gold doc + page) | File passes schema validation (validator code only; content is the user's) |
| 2 Chunking | Recursive page-aware chunker; chunk metadata | Deterministic; every chunk maps to doc + page; user confirms baseline |
| 3 Embeddings + index | Embedder interface + cache; Chroma build | One-command idempotent build; sanity queries plausible; decision logged |
| 4 Retrieval | Dense, BM25, hybrid (RRF); common `retrieve()` | LLM-free recall@k, MRR, hit-rate on the 20-question set |
| 5 Generation + guardrails | `LLMClient` (ollama default, anthropic optional/off), prompt, citation parsing/validation, refusal on weak retrieval, injection handling | User picks the Ollama model from 2-3 options; answers cite or refuse; injection cases pass |
| 6 Evaluation harness | LLM-free retrieval metrics; faithfulness and answer relevance via local judge; refusal accuracy | Deterministic and cached; judge agreement vs user's manual labels reported; eval set grown to 40-60 |
| 7 Experiments (core) | Chunk size/overlap grid; dense vs hybrid | Results in experiments/results/<run_id>/ with config snapshot; bootstrap CIs |
| 7b Stretch (optional, only after core works end to end) | Cross-encoder reranker; section-aware chunking; embedding-model comparison | Each reported separately; core results unaffected |
| 8 Validation report | reports/rag_validation_report.md with failure analysis and limitations | Numbers traceable to results files; states that a small local judge is noisy and reports its agreement with manual labels |
| 9 App, packaging, CI | FastAPI, Streamlit, Dockerfile, pre-commit, GitHub Actions (ruff, pytest, docker build; no network/LLM) | `docker run` serves API; CI green |

## Documents to download (verify URLs at download time)
- FinCEN advisories and SAR guidance: fincen.gov/resources/advisoriesbulletinsfact-sheets; 31 CFR 1020.320 via ecfr.gov
- FATF red-flag / typology reports: fatf-gafi.org > Publications
- FFIEC BSA/AML Examination Manual sections: bsaaml.ffiec.gov
- CFPB Regulation E (12 CFR 1005) and interpretations: consumerfinance.gov/rules-policy/regulations/1005/

Target 10-15 documents. Public repo ships `data/manifest.csv` and a download script only; PDFs are gitignored.

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Bad PDF extraction | Per-page QA stats, spot checks, fallback extractor only for flagged docs |
| Wrong page citations (PDF index vs printed label) | Store both; test on fixture |
| Eval leakage | Dev vs held-out split; no tuning on held-out; user writes all questions |
| Small eval set | Bootstrap CIs; multi-page gold; report n |
| Noisy small local judge | Fixed rubric, temperature 0, agreement with manual labels, stated as a limitation |
| Slow CPU generation | Small model, on-disk cache, limited eval runs |
| Prompt injection | Delimit untrusted context, ignore-instructions rule, citation validation, adversarial tests |
| Stale or hallucinated regulatory content | Refuse on weak retrieval, always cite, dates in manifest, disclaimer |
| Secrets/data committed | .gitignore, .env.example only |
| Scope creep | Phase gates; stretch items marked optional |
