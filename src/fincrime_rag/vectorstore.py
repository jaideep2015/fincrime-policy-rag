"""Phase 3: Chroma persistent index over the chunks.

``build_index`` is idempotent: it drops the collection and recreates it from the chunk files, with
chunk_id as the Chroma id, so running it twice yields the identical index (embeddings come from the
on-disk cache). Chroma metadata cannot hold lists, so ``pdf_pages`` is stored as "4,5" and the first
and last page as integers; ``chunk_from_record`` turns a stored record back into a chunk dict.

Command: ``python -m fincrime_rag.vectorstore build`` (run ``python -m fincrime_rag.chunking`` first).
"""

from __future__ import annotations

import argparse
import sys

from fincrime_rag.chunking import load_all_chunks
from fincrime_rag.config import Config, load_config
from fincrime_rag.embeddings import Embedder, build_embedder

ADD_BATCH = 256


def open_client(cfg: Config):
    import chromadb

    cfg.paths.index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(cfg.paths.index_dir))


def get_collection(cfg: Config):
    return open_client(cfg).get_collection(cfg.vectorstore.collection)


def chunk_to_metadata(chunk: dict) -> dict:
    pages = chunk["pdf_pages"]
    return {
        "doc_id": chunk["doc_id"],
        "pdf_pages": ",".join(str(p) for p in pages),
        "pdf_page_start": pages[0],
        "pdf_page_end": pages[-1],
        "printed_page_guess": chunk.get("printed_page_guess") or "",
        "section": chunk.get("section") or "",
        "section_hint": chunk.get("section_hint") or "",
        "content_type": chunk.get("content_type", "body"),
        "n_tokens": chunk["n_tokens"],
        "text_hash": chunk["text_hash"],
    }


def chunk_from_record(chunk_id: str, text: str, meta: dict) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": meta["doc_id"],
        "pdf_pages": [int(p) for p in meta["pdf_pages"].split(",")],
        "printed_page_guess": meta.get("printed_page_guess") or None,
        "section": meta.get("section") or None,
        "section_hint": meta.get("section_hint") or None,
        "content_type": meta.get("content_type", "body"),
        "text": text,
    }


def build_index(cfg: Config, embedder: Embedder, chunks: list[dict] | None = None) -> int:
    """Drop and rebuild the collection. Returns the number of chunks indexed."""
    chunks = load_all_chunks(cfg) if chunks is None else chunks
    client = open_client(cfg)
    try:
        client.delete_collection(cfg.vectorstore.collection)
    except Exception:  # noqa: BLE001  (collection did not exist yet; Chroma's exception type varies)
        pass
    collection = client.create_collection(
        cfg.vectorstore.collection, metadata={"hnsw:space": cfg.vectorstore.space}
    )
    for start in range(0, len(chunks), ADD_BATCH):
        batch = chunks[start : start + ADD_BATCH]
        vectors = embedder.embed_documents([c["text"] for c in batch])
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors.tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[chunk_to_metadata(c) for c in batch],
        )
    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.vectorstore", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "info"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "info":
        print(f"{get_collection(cfg).count()} chunks in {cfg.paths.index_dir}/{cfg.vectorstore.collection}")
        return 0
    embedder = build_embedder(cfg)
    n = build_index(cfg, embedder)
    print(f"indexed {n} chunks (embeddings: {embedder.hits} cached, {embedder.misses} computed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
