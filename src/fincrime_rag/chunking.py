"""Phase 2: recursive, page-aware chunking of the cleaned pages.

Reads ``data/processed/<doc_id>.pages.jsonl`` and writes ``data/processed/<doc_id>.chunks.jsonl``.

How it works
1. Pages whose content type is listed in ``indexing.exclude_content_types`` (Reg E model forms) and
   empty pages are skipped.
2. Each page's text is split recursively (paragraph, line, sentence, word: ``chunking.separators``)
   into *atoms*, each no larger than the token budget. A coarser separator is only abandoned when a
   piece is still too large, so atoms are as natural as possible. Every atom remembers its page.
3. Atoms of a whole document are packed greedily into chunks of at most
   ``chunk_size - special_tokens`` tokens. A chunk may span a page boundary; it then lists every page
   it touches. The next chunk starts by re-using trailing atoms of the previous one, up to
   ``chunk_overlap`` tokens.
4. For documents matching ``chunking.section_doc_prefixes`` (the CFR texts) heading lines such as
   ``§ 1005.10`` are tracked while walking the text, so each chunk gets a ``section`` field.

Tokens are counted with the embedding model's own tokenizer so a chunk never exceeds the embedder's
window. Unit tests pass a word counter instead (no model download).

Command: ``python -m fincrime_rag.chunking [--doc DOC_ID]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fincrime_rag.cleaning import indexable_pages
from fincrime_rag.config import ChunkingConfig, Config, IndexingConfig, load_config
from fincrime_rag.extraction import read_pages
from fincrime_rag.ingestion import read_manifest

TokenCounter = Callable[[str], int]


# --------------------------------------------------------------------------- token counting
def hf_token_counter(model_name: str) -> TokenCounter:
    """Token counter backed by the embedding model's tokenizer (special tokens not counted)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False))

    return count


# --------------------------------------------------------------------------- atoms
@dataclass(frozen=True)
class Atom:
    text: str  # includes its trailing separator so that "".join(atoms) restores the original text
    tokens: int
    page: int
    section: str | None  # section in effect after this atom (CFR docs)


def _split_keep(text: str, sep: str) -> list[str]:
    """Split on ``sep`` keeping the separator at the end of each piece."""
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]]
    if parts[-1]:
        pieces.append(parts[-1])
    return pieces


def split_to_atoms(text: str, limit: int, separators: list[str], count: TokenCounter) -> list[str]:
    """Recursively split ``text`` into pieces of at most ``limit`` tokens."""
    if not text.strip():
        return []
    if count(text) <= limit:
        return [text]
    if not separators:
        return _split_by_tokens(text, limit, count)
    sep, rest = separators[0], separators[1:]
    pieces = _split_keep(text, sep)
    if len(pieces) == 1:
        return split_to_atoms(text, limit, rest, count)
    atoms: list[str] = []
    for piece in pieces:
        atoms.extend(split_to_atoms(piece, limit, rest, count))
    return atoms


def _split_by_tokens(text: str, limit: int, count: TokenCounter) -> list[str]:
    """Last resort for an unbroken string longer than ``limit``: cut on character halves."""
    if count(text) <= limit or len(text) < 2:
        return [text]
    mid = len(text) // 2
    return _split_by_tokens(text[:mid], limit, count) + _split_by_tokens(text[mid:], limit, count)


class SectionTracker:
    """Follows ``§ N.N`` headings through a document, one line at a time."""

    def __init__(self, patterns: list[str]):
        self._patterns = [re.compile(p) for p in patterns]
        self.current: str | None = None

    def feed(self, text: str) -> str | None:
        for line in text.split("\n"):
            stripped = line.strip()
            for pattern in self._patterns:
                match = pattern.match(stripped)
                if match:
                    self.current = match.group("section")
                    break
        return self.current


# --------------------------------------------------------------------------- packing
def pack_atoms(
    atoms: list[Atom], budget: int, overlap: int, min_tokens: int, measure: Callable[[list[Atom]], int]
) -> list[list[Atom]]:
    """Greedy packing of ``atoms`` into groups of at most ``budget`` tokens with trailing overlap.

    Each group after the first starts with up to ``overlap`` tokens of atoms carried over from the
    previous group. A final group whose new content is under ``min_tokens`` is folded into its
    predecessor when that still fits the budget. ``measure`` counts tokens of the joined text of a
    group: tokenizing pieces separately and joining them can differ by a few tokens, so the budget is
    checked on the real joined text.
    """
    groups: list[list[Atom]] = []
    current: list[Atom] = []
    carried = 0  # leading atoms of `current` that were carried over as overlap
    for atom in atoms:
        if current[carried:] and measure(current + [atom]) > budget:
            groups.append(current)
            current = _overlap_tail(current, overlap)
            while current and measure(current + [atom]) > budget:  # no room for the next atom: shrink overlap
                current.pop(0)
            carried = len(current)
        current.append(atom)
    if current[carried:]:
        new_tokens = sum(a.tokens for a in current[carried:])
        if groups and new_tokens < min_tokens and measure(groups[-1] + current[carried:]) <= budget:
            groups[-1] = groups[-1] + current[carried:]
        else:
            groups.append(current)
    return groups


def _overlap_tail(group: list[Atom], overlap: int) -> list[Atom]:
    tail: list[Atom] = []
    total = 0
    for atom in reversed(group[1:]):  # never carry the whole chunk over
        if total + atom.tokens > overlap:
            break
        tail.insert(0, atom)
        total += atom.tokens
    return tail


# --------------------------------------------------------------------------- chunking a document
def chunk_document(
    doc_id: str,
    pages: list[dict],
    cfg: ChunkingConfig,
    count: TokenCounter,
    indexing_exclude: list[str] | None = None,
) -> list[dict]:
    """Chunk the pages of one document. Deterministic: same input, same chunks."""
    keep = indexable_pages(pages, IndexingConfig(exclude_content_types=indexing_exclude or []))
    budget = cfg.chunk_size - cfg.special_tokens
    track = any(doc_id.startswith(prefix) for prefix in cfg.section_doc_prefixes)
    tracker = SectionTracker(cfg.section_heading_patterns) if track else None

    page_info: dict[int, dict] = {}
    atoms: list[Atom] = []
    for page in sorted(keep, key=lambda p: p["pdf_page"]):
        text = page["text"]
        if not text.strip():
            continue
        page_info[page["pdf_page"]] = page
        for piece in split_to_atoms(text, budget, cfg.separators, count):
            section = tracker.feed(piece) if tracker else None
            atoms.append(Atom(piece, count(piece), page["pdf_page"], section))

    chunks: list[dict] = []
    def measure(group: list[Atom]) -> int:
        return count("".join(a.text for a in group).strip())

    groups = pack_atoms(atoms, budget, cfg.chunk_overlap, cfg.min_chunk_tokens, measure)
    for index, group in enumerate(groups):
        text = "".join(a.text for a in group).strip()
        pages_in_chunk = sorted({a.page for a in group})
        first = page_info[pages_in_chunk[0]]
        chunks.append(
            {
                "chunk_id": f"{doc_id}:p{pages_in_chunk[0]:04d}:c{index:04d}",
                "doc_id": doc_id,
                "pdf_pages": pages_in_chunk,
                "printed_page_guess": first.get("printed_page_guess"),
                "section": group[0].section,
                "section_hint": first.get("section_hint"),
                "content_type": first.get("content_type", "body"),
                "n_tokens": count(text),
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
        )
    return chunks


# --------------------------------------------------------------------------- files
def chunks_path(cfg: Config, doc_id: str) -> Path:
    return cfg.paths.processed_dir / f"{doc_id}.chunks.jsonl"


def write_chunks(path: Path, chunks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_chunks(cfg: Config) -> list[dict]:
    """All chunks of all manifest documents, in manifest order."""
    chunks: list[dict] = []
    for row in read_manifest(cfg.paths.manifest):
        path = chunks_path(cfg, row.doc_id)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m fincrime_rag.chunking` first")
        chunks.extend(read_chunks(path))
    return chunks


def chunk_all(cfg: Config, count: TokenCounter, only: str | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in read_manifest(cfg.paths.manifest):
        if only and row.doc_id != only:
            continue
        pages = read_pages(cfg.paths.processed_dir / f"{row.doc_id}.pages.jsonl")
        chunks = chunk_document(row.doc_id, pages, cfg.chunking, count, cfg.indexing.exclude_content_types)
        write_chunks(chunks_path(cfg, row.doc_id), chunks)
        counts[row.doc_id] = len(chunks)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.chunking", description=__doc__.splitlines()[0])
    parser.add_argument("--doc", help="chunk only this doc_id")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    counts = chunk_all(cfg, hf_token_counter(cfg.embedding.model), only=args.doc)
    if not counts:
        print(f"unknown doc_id: {args.doc}")
        return 1
    for doc_id, n in counts.items():
        print(f"{doc_id}: {n} chunks")
    print(f"total: {sum(counts.values())} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
