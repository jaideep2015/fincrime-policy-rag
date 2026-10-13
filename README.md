# Fraud & Financial Crime Policy Assistant

A RAG pipeline over public fraud/AML documents (FinCEN, FATF, FFIEC BSA/AML manual, CFPB Reg E) that answers with document + page citations, refuses when retrieved context is insufficient, and includes a quantitative evaluation and validation layer.

Everything runs locally on CPU; no API key is required.

**Status:** Phase 0 (scaffold). See [PLAN.md](PLAN.md) for the phased plan and [DECISIONS.md](DECISIONS.md) for design choices.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev]"
ruff check .
pytest
```
