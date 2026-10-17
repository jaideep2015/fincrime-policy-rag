# `make index` rebuilds chunks and the Chroma index from data/processed (idempotent).
PYTHON ?= python

.PHONY: index chunks test lint

chunks:
	$(PYTHON) -m fincrime_rag.chunking

index: chunks
	$(PYTHON) -m fincrime_rag.vectorstore build

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
