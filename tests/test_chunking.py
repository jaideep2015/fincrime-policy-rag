"""Chunker tests: word-count tokenizer, synthetic pages (no model download)."""

from fincrime_rag.chunking import SectionTracker, chunk_document, split_to_atoms
from fincrime_rag.config import ChunkingConfig


def words(text: str) -> int:
    return len(text.split())


def make_cfg(size: int = 30, overlap: int = 8, **over) -> ChunkingConfig:
    base = dict(
        chunk_size=size,
        chunk_overlap=overlap,
        special_tokens=0,
        separators=["\n\n", "\n", ". ", " "],
        min_chunk_tokens=3,
        section_doc_prefixes=["cfr_"],
        section_heading_patterns=[r"^§\s*(?P<section>\d{1,4}\.\d+)\s*$"],
    )
    base.update(over)
    return ChunkingConfig(**base)


def page(n: int, text: str, **extra) -> dict:
    return {
        "doc_id": "d",
        "pdf_page": n,
        "printed_page_guess": str(n + 1),
        "section_hint": None,
        "content_type": "body",
        "text": text,
        **extra,
    }


def para(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_split_respects_limit_and_preserves_text():
    text = para("a", 40) + "\n\n" + para("b", 10) + ". " + para("c", 50)
    atoms = split_to_atoms(text, 25, ["\n\n", "\n", ". ", " "], words)
    assert all(words(a) <= 25 for a in atoms)
    assert "".join(atoms).split() == text.split()


def test_chunks_respect_budget_and_carry_metadata():
    pages = [page(i, para(f"p{i}w", 20) + "\n\n" + para(f"p{i}x", 15)) for i in range(1, 6)]
    chunks = chunk_document("d", pages, make_cfg(), words)
    assert chunks
    for c in chunks:
        assert c["n_tokens"] <= 30
        assert c["pdf_pages"] == sorted(c["pdf_pages"])
        assert c["doc_id"] == "d" and c["chunk_id"].startswith("d:p")
        assert c["printed_page_guess"] == str(c["pdf_pages"][0] + 1)
        assert c["content_type"] == "body" and len(c["text_hash"]) == 64
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert {p for c in chunks for p in c["pdf_pages"]} == {1, 2, 3, 4, 5}


def test_chunk_can_span_pages():
    pages = [page(1, para("a", 12)), page(2, para("b", 12))]
    chunks = chunk_document("d", pages, make_cfg(size=40), words)
    assert len(chunks) == 1 and chunks[0]["pdf_pages"] == [1, 2]


def test_overlap_repeats_trailing_content():
    pages = [page(1, "\n".join(para(f"l{i}_", 6) for i in range(12)))]
    chunks = chunk_document("d", pages, make_cfg(size=20, overlap=8), words)
    assert len(chunks) > 2
    first_lines = chunks[0]["text"].split("\n")
    assert chunks[1]["text"].split("\n")[0] in first_lines  # carried over from the previous chunk


def test_deterministic():
    pages = [page(i, para(f"w{i}_", 70)) for i in range(1, 4)]
    assert chunk_document("d", pages, make_cfg(), words) == chunk_document("d", pages, make_cfg(), words)


def test_model_form_and_empty_pages_skipped():
    pages = [page(1, para("a", 10)), page(2, para("form", 10), content_type="model_form"), page(3, "  \n")]
    chunks = chunk_document("d", pages, make_cfg(), words, indexing_exclude=["model_form"])
    assert [c["pdf_pages"] for c in chunks] == [[1]]
    assert "form0" not in chunks[0]["text"]


def test_section_tracking_only_for_configured_docs():
    pages = [
        page(1, "§ 1005.9\nReceipts.\n" + para("a", 10)),
        page(2, para("b", 10) + "\n§ 1005.10\nPreauthorized.\n" + para("c", 10)),
    ]
    cfr = chunk_document("cfr_x", pages, make_cfg(size=14, overlap=0), words)
    sections = [c["section"] for c in cfr]
    assert sections[0] == "1005.9" and sections[-1] == "1005.10"
    other = chunk_document("fincen_x", pages, make_cfg(size=14, overlap=0), words)
    assert all(c["section"] is None for c in other)


def test_heading_patterns_from_config_reject_references():
    from fincrime_rag.config import load_config

    patterns = load_config("config.yaml").chunking.section_heading_patterns
    tracker = SectionTracker(patterns)
    assert tracker.feed("with § 1005.11.") is None
    assert tracker.feed("§ 1005.11(b) applies") is None
    assert tracker.feed("§ 1005.10") == "1005.10"
    assert tracker.feed("§ 1005.3 Coverage.") == "1005.3"
    assert tracker.feed("body text without heading") == "1005.3"  # state persists until the next heading
