from tests.conftest import N_PAGES, ROOT, make_pdf

from fincrime_rag.cleaning import (
    clean_document,
    detect_repeated_patterns,
    is_page_number,
    normalize_line,
)
from fincrime_rag.config import load_config
from fincrime_rag.extraction import extract_pdf, read_pages, write_pages
from fincrime_rag.qa import bad_char_count, doc_stats, page_stats, token_counts


def _cfg():
    return load_config(ROOT / "config.yaml")


# ----------------------------------------------------------------------------- extraction
def test_extract_pdf_pages_labels_and_empty_page(fixture_pdf):
    pages = extract_pdf(fixture_pdf, "fx")
    assert [p["pdf_page"] for p in pages] == list(range(1, N_PAGES + 1))
    assert pages[0]["page_label"] == "A-1" and pages[7]["page_label"] == "A-8"
    assert "Money mule accounts" in pages[0]["text"]
    assert pages[3]["text"].strip() == ""  # blank page


def test_jsonl_roundtrip(tmp_path, fixture_pdf):
    pages = extract_pdf(fixture_pdf, "fx")
    write_pages(tmp_path / "x" / "fx.jsonl", pages)
    assert read_pages(tmp_path / "x" / "fx.jsonl") == pages


# ----------------------------------------------------------------------------- cleaning
def test_page_number_detection():
    for yes in ["12", "iv", "XIV", "Page 3", "Page 3 of 10", "4-12", "F-3"]:
        assert is_page_number(yes), yes
    for no in ["Civil", "Mixed Case", "2022 report", "Section 5", "did you know"]:
        assert not is_page_number(no), no


def test_normalize_line_collapses_digits():
    assert normalize_line("Page 12 ") == normalize_line("page  13") == "page#"


def test_repeated_header_detected_and_removed_and_logged(fixture_pdf):
    cfg = _cfg().cleaning
    pages = extract_pdf(fixture_pdf, "fx")
    assert normalize_line("ACME Advisory") in detect_repeated_patterns(pages, cfg)

    cleaned, log = clean_document(pages, cfg)
    assert all("ACME Advisory" not in p["text"] for p in cleaned)
    assert all(not p["text"].endswith(f"Page {p['pdf_page']}") for p in cleaned)
    reasons = {r["reason"].split("_on_")[0] for r in log}
    assert reasons == {"page_number", "repeated"}
    assert {r["position"] for r in log} <= {"header", "footer"}
    # blank page has no lines to remove, every other page loses its header and its page number
    assert sum(1 for r in log if r["reason"] == "page_number") == N_PAGES - 1


def test_printed_page_guess_only_without_pdf_label(fixture_pdf):
    cfg = _cfg().cleaning
    pages = extract_pdf(fixture_pdf, "fx")
    cleaned, _ = clean_document(pages, cfg)
    assert all(p["printed_page_guess"] is None for p in cleaned)  # fixture PDF has labels

    for p in pages:
        p["page_label"] = None
    cleaned, _ = clean_document(pages, cfg)
    assert cleaned[0]["printed_page_guess"] == "1"
    assert cleaned[6]["printed_page_guess"] == "7"


def test_hyphen_repair_joins_words_but_keeps_real_compounds(fixture_pdf):
    cleaned, _ = clean_document(extract_pdf(fixture_pdf, "fx"), _cfg().cleaning)
    text = " ".join(p["text"] for p in cleaned)
    assert "continued activity" in text  # contin-/ued -> joined
    assert "anti-money laundering alerts" in text  # anti-/money -> compound kept
    assert "antimoney" not in text


def test_short_documents_skip_repeat_detection_but_strip_page_numbers(tmp_path):
    pdf = make_pdf(tmp_path / "short.pdf", n_pages=3)
    pages = extract_pdf(pdf, "short")
    cfg = _cfg().cleaning
    assert detect_repeated_patterns(pages, cfg) == {}
    cleaned, log = clean_document(pages, cfg)
    assert all("ACME Advisory" in p["text"] for p in cleaned)  # header kept: too few pages
    assert {r["reason"] for r in log} == {"page_number"}


def test_tidy_ligatures_dot_leaders_whitespace():
    ligature, nbsp, soft_hyphen = chr(0xFB01), chr(0xA0), chr(0xAD)
    nl = chr(10)
    parts = ["of", ligature, "ce", nbsp, " hours", nl * 4, "Intro ........ 4", nl]
    text_in = "".join(parts + ["soft", soft_hyphen, "hyphen"])
    pages = [{"doc_id": "d", "pdf_page": 1, "page_label": None, "text": text_in}]
    cleaned, _ = clean_document(pages, _cfg().cleaning)
    text = cleaned[0]["text"]
    assert "office hours" in text and "softhyphen" in text
    bullet_page = [{"doc_id": "d", "pdf_page": 1, "page_label": None, "text": chr(0xF0B7) + " first item"}]
    assert clean_document(bullet_page, _cfg().cleaning)[0][0]["text"] == chr(0x2022) + " first item"
    split_text = chr(0xF0B7) + " " + chr(10) + "item text"
    alone = [{"doc_id": "d", "pdf_page": 1, "page_label": None, "text": split_text}]
    assert clean_document(alone, _cfg().cleaning)[0][0]["text"] == chr(0x2022) + " item text"
    assert "........" not in text and "\n\n\n" not in text


# ----------------------------------------------------------------------------- QA stats
def test_bad_char_and_odd_token_counts():
    assert bad_char_count("ok � text \x01 ") == 3
    assert bad_char_count("clean\ntext\twith tabs") == 0
    total, odd = token_counts("The bank reviewed xqzvbnm and https://x.gov/pqrst 2024 SARs")
    assert odd == 1 and total >= 5  # only 'xqzvbnm'; URL and number are skipped, 'SARs' is short


def test_page_and_doc_stats_flag_empty_and_garbled(fixture_pdf):
    cfg = _cfg()
    pages = extract_pdf(fixture_pdf, "fx")
    stats = [page_stats(p, cfg.qa) for p in pages]
    assert [s["is_empty"] for s in stats].count(True) == 1 and stats[3]["is_empty"]

    from fincrime_rag.ingestion import ManifestRow

    row = ManifestRow("fx", "t", "p", "script", "fx.pdf", "https://x", "", "", "", "")
    doc = doc_stats(row, stats, [], [], cfg.qa)
    assert doc["pages"] == N_PAGES and doc["empty_pages"] == [4]
    assert "many empty pages" not in doc["flags"]  # a single empty page is not flagged
    assert "low text density" in doc["flags"]  # tiny fixture pages fall under the threshold

    garbled = [dict(s, bad_chars=50, chars=100) for s in stats]
    assert "bad characters" in doc_stats(row, garbled, [], [], cfg.qa)["flags"]
