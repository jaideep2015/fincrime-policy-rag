"""OCR coverage check / fallback (with a fake engine) and the running-head rules in cleaning."""

import re
from pathlib import Path

import pymupdf
import pytest
from tests.conftest import ROOT, extract_fixture_pages

from fincrime_rag.cleaning import (
    clean_document,
    count_residual_edge_lines,
    is_page_number,
    normalize_line,
)
from fincrime_rag.config import Config, load_config
from fincrime_rag.extraction import extract_pdf
from fincrime_rag.ocr import (
    OCRBox,
    PageOCR,
    assess_page,
    boxes_to_text,
    filter_icon_noise,
)
from fincrime_rag.qa import token_counts


def _cfg() -> Config:
    return load_config(ROOT / "config.yaml")


def box(x0, y0, text, conf=0.93, width=600, height=40):
    return OCRBox(x0, y0, x0 + width, y0 + height, text, conf)


# ----------------------------------------------------------------------------- OCR helpers
def test_icon_noise_is_dropped_only_when_short_low_confidence_and_in_gutter():
    cfg = _cfg().ocr
    page = PageOCR(1, 1700, 2200, [
        box(102, 500, "13", conf=0.62, width=40),  # red-flag icon read as a number: noise
        box(172, 500, "The customer opens a new account.", conf=0.93),
        box(102, 700, "6", conf=0.97, width=20),  # confident short gutter text: kept
        box(600, 900, "7", conf=0.7, width=20),  # not in the gutter: kept
        box(172, 1300, "aaiaaaon s o saide n", conf=0.35),  # garbage reading below min_confidence
        box(100, 1100, "Money mule schemes", conf=0.6),  # gutter but long: kept
    ])
    kept, dropped = filter_icon_noise(page, cfg)
    assert [b.text for b in dropped] == ["13", "aaiaaaon s o saide n"]
    assert {b.text for b in kept} == {"The customer opens a new account.", "6", "7", "Money mule schemes"}


def test_boxes_to_text_reading_order_rows_and_paragraph_gaps():
    cfg = _cfg().ocr
    boxes = [
        box(800, 105, "ADVISORY", width=100),  # same row as FINCEN, listed first on purpose
        box(586, 108, "FINCEN", width=100),
        box(100, 200, "line one"),
        box(100, 245, "line two"),
        box(100, 290, "line three"),
        box(100, 335, "line four"),
        box(100, 420, "new paragraph"),  # gap of 85 vs pitch 45 -> blank line
    ]
    expected = "FINCEN ADVISORY\n\nline one\nline two\nline three\nline four\n\nnew paragraph"
    assert boxes_to_text(boxes, cfg) == expected
    assert boxes_to_text([], cfg) == ""


def test_assess_page_coverage_flagging_and_minimum_words():
    cfg = _cfg().ocr
    ocr = PageOCR(1, 1700, 2200, [box(100, 100 + 60 * i, "word " * 10) for i in range(10)])  # 100 words
    threshold_words = round(100 * cfg.coverage_threshold)
    assert assess_page("w " * 100, ocr, cfg).flagged is False
    low = assess_page("w " * (threshold_words - 1), ocr, cfg)
    assert low.coverage == (threshold_words - 1) / 100 and low.flagged is True
    assert assess_page("w " * threshold_words, ocr, cfg).flagged is False  # at the threshold: kept
    tiny = PageOCR(1, 1700, 2200, [box(100, 100, "just five words on page")])
    assessed = assess_page("", tiny, cfg)
    assert assessed.coverage is None and assessed.flagged is False  # too few OCR words to judge


# ----------------------------------------------------------------------------- regression: COVID-style
FLAGS = [f"Red flag number {i} describes a customer moving money oddly." for i in range(1, 19)]


def test_regression_missing_red_flags_recovered_by_ocr_fallback(tmp_path):
    """Text layer holds 2 of 18 red flags (the rest are images); OCR must recover all 18."""
    pdf = tmp_path / "covid_like.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 40), "FINCEN ADVISORY", fontsize=9)
    page.insert_text((72, 120), "Financial red flag indicators may include:", fontsize=11)
    page.insert_text((72, 140), FLAGS[0], fontsize=11)
    page.insert_text((72, 160), FLAGS[4], fontsize=11)
    doc.save(pdf)
    doc.close()

    boxes = [box(172, 100, "Financial red flag indicators may include:", width=800)]
    for i, flag in enumerate(FLAGS):
        y = 200 + 90 * i
        boxes.append(box(172, y, flag, width=900))
        boxes.append(box(102, y + 8, str(13 + i % 6), conf=0.6, width=40))  # icon read as digits
    ocr_pages = {1: PageOCR(1, 1700, 2200, boxes)}

    text_only = extract_pdf(pdf, "covid_like")[0]
    assert sum(f in text_only["text"] for f in FLAGS) == 2
    assert text_only["extraction_method"] == "text"

    record = extract_pdf(pdf, "covid_like", ocr_pages, _cfg().ocr)[0]
    assert record["extraction_method"] == "ocr" and record["coverage"] < 0.8
    assert sum(f in record["text"] for f in FLAGS) == 18
    assert not re.search(r"^\d{1,2}$", record["text"], re.MULTILINE)  # icon noise stripped

    cleaned, _ = clean_document([record], _cfg().cleaning)
    assert cleaned[0]["extraction_method"] == "ocr"
    assert sum(f in cleaned[0]["text"] for f in FLAGS) == 18


def test_good_text_layer_is_kept_when_coverage_is_high(tmp_path, fixture_pdf):
    ocr_pages = {}
    for page in extract_pdf(fixture_pdf, "fx"):
        words = page["text"].split()
        if len(words) >= 10:
            ocr_pages[page["pdf_page"]] = PageOCR(
                page["pdf_page"], 1700, 2200, [box(100, 100, " ".join(words))]
            )
    records = extract_pdf(fixture_pdf, "fx", ocr_pages, _cfg().ocr)
    assert all(r["extraction_method"] == "text" for r in records)


@pytest.mark.skipif(
    not (ROOT / "data/raw/fincen_2020_a003_covid_mule.pdf").exists(), reason="needs the downloaded PDF"
)
def test_real_covid_advisory_merge_keeps_text_layer_and_has_all_18_red_flags():
    pytest.importorskip("rapidocr_onnxruntime")
    from fincrime_rag.ocr import make_engine, ocr_document

    cfg = _cfg()
    pdf = ROOT / "data/raw/fincen_2020_a003_covid_mule.pdf"
    cache = ROOT / cfg.paths.cache_dir
    engine = make_engine(cfg.ocr)

    def recover(numbers):
        return ocr_document(pdf, engine, cfg.ocr, cache, profile="recover", only_pages=numbers)

    pages = extract_pdf(pdf, "covid", ocr_document(pdf, engine, cfg.ocr, cache), cfg.ocr, recover)
    cleaned, _ = clean_document(pages, cfg.cleaning)
    by_page = {p["pdf_page"]: p for p in cleaned}
    assert all(by_page[n]["extraction_method"] == "merged" for n in (4, 5, 6, 7))
    assert [by_page[n]["printed_page_guess"] for n in (4, 5, 6, 7)] == ["4", "5", "6", "7"]

    def norm(text):  # straight quotes, single spaces
        table = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
        return " ".join(text.translate(table).split())

    p4, p5 = norm(by_page[4]["text"]), norm(by_page[5]["text"])
    assert 'to defraud the public. These operations may include words like "relief," "fund," "donation," and' in p4
    assert 'in domain extensions like ".com," ".org," and ".us"). Examples of U.S. government-identified' in p4
    assert 'her money movement activity." The individual is motivated by financial gain or an unwillingness' in p5
    assert 'of Their Assets," (December 4, 2019); and DOJ,' in p5
    everything = " ".join(p["text"] for p in cleaned)
    # OCR typos must not enter ("illusion" itself contains "llusion", hence the word boundary)
    assert not re.search(r"\bllusion", everything) and "Financiai" not in everything
    assert "give the illusion that they are a legitimate organization" in p4

    # Red-flag items are the paragraphs that follow each "Financial red flag indicators ... may include:"
    # intro, up to the next heading, footnote or lettered sub-list.
    flags, collecting = [], False
    for page in cleaned:
        for para in (p.strip() for p in page["text"].split("\n\n")):
            flat = " ".join(para.split())
            if re.search(r"red flag indicators.*may include:", flat):
                collecting = True
            elif collecting and flat.startswith(("Money Mule Schemes", "Information on Reporting")):
                collecting = False
            elif collecting and flat and not re.match(r"^(\d{1,2}\. |[abc]\. )", flat):
                flags.append(flat)
    assert len(flags) == 18


# ----------------------------------------------------------------------------- normalize / heads
def test_normalize_line_collapses_digit_runs_pipe_and_spacing():
    assert normalize_line("Page 9") == normalize_line("Page 1042") == "page#"
    assert normalize_line("6  TITLE") == normalize_line("TITLE | 7") == normalize_line("12 | TITLE")
    assert normalize_line("F I N C E N  A D V I S O R Y") == normalize_line("FINCEN ADVISORY")


def test_page_number_with_en_dash_appendix_style():
    assert is_page_number("F–1") and is_page_number("F-12") and is_page_number("4–12")


WORDS = ["alpha", "bravo", "cobalt", "delta", "ember", "falcon", "garnet", "harbor", "indigo", "jasper",
         "kestrel", "lantern", "meadow", "nectar", "onyx", "pebble", "quartz", "raven"]


def _doc(heads: list[list[str]]) -> list[dict]:
    """Pages with the given head lines and three distinct, digit-free body lines each."""
    pages = []
    for i, head in enumerate(heads):
        body = [f"{WORDS[i % 18]} story about {WORDS[(i * 5 + j) % 18]} mules" for j in range(3)]
        pages.append({"doc_id": "d", "pdf_page": i + 1, "page_label": None, "text": "\n".join(head + body)})
    return pages


def test_printed_page_from_n_pipe_title_heads_is_recorded_and_removed():
    title = "ILLICIT FLOWS REPORT"
    heads = [[f"{n} | {title}"] if n % 2 == 0 else [f"{title} | {n}"] for n in range(6, 18)]
    heads[0] = [f"6  {title}"]
    cleaned, log = clean_document(_doc(heads), _cfg().cleaning)
    assert [p["printed_page_guess"] for p in cleaned] == [str(n) for n in range(6, 18)]
    assert all(title not in p["text"] for p in cleaned)
    assert {r["reason"] for r in log} == {"printed_page_head"}


def test_cfr_running_heads_set_section_hint_and_only_match_first_three_lines():
    heads = [["204", "12 CFR Ch. X (1–1–25 Edition)", f"§ 1005.{n}"] for n in range(10, 22)]
    heads += [["265", "Consumer Financial Protection Bureau", "Pt. 1005, Supp. I"]] * 3
    pages = _doc(heads)
    pages[0]["text"] += "\n§ 1005.99"  # last line of the page, not a head: must stay
    cleaned, log = clean_document(pages, _cfg().cleaning)
    assert [p["section_hint"] for p in cleaned[:3]] == ["1005.10", "1005.11", "1005.12"]
    assert all(p["section_hint"] is None for p in cleaned[12:])  # Supp. I head has no section
    assert not any("Pt. 1005, Supp. I" in p["text"] for p in cleaned)
    assert "§ 1005.99" in cleaned[0]["text"]
    assert {"running_head", "page_number"} <= {r["reason"] for r in log}


def test_repeated_pattern_scan_covers_five_lines_but_page_numbers_only_three():
    heads = [["Alpha Review", "Beta Subtitle", "Gamma Subtitle", "Delta Subtitle", "Epsilon Stamp", "7"]] * 10
    pages = _doc(heads)
    pages[0]["text"] = "\n".join(["Alpha Review", "a", "b", "42", "c", "d", "e", "f", "g", "h"])
    cleaned, _ = clean_document(pages, _cfg().cleaning)
    assert "Epsilon Stamp" not in cleaned[1]["text"]  # 5th line: removed as a repeated head
    assert "\n42\n" in "\n" + cleaned[0]["text"] + "\n"  # a bare number on line 4 is not a page number


def test_residual_edge_lines_drop_to_zero_after_cleaning(fixture_pdf):
    cfg = _cfg().cleaning
    raw = extract_fixture_pages(fixture_pdf)
    assert count_residual_edge_lines(raw, cfg)["total"] > 0
    cleaned, _ = clean_document(raw, cfg)
    assert count_residual_edge_lines(cleaned, cfg)["total"] == 0


# ----------------------------------------------------------------------------- QA odd tokens
def test_odd_tokens_ignore_all_caps_acronyms_when_configured():
    text = "CMLN NMLRA TCOs CMLNs and xqzvbnm plus normal words here"
    assert token_counts(text, ignore_acronyms=False)[1] == 2  # vowel-less, 5+ letters: CMLNs, xqzvbnm
    total, odd = token_counts(text, ignore_acronyms=True)
    assert odd == 1 and total == 6  # only xqzvbnm is odd; the four acronyms are not counted at all


def test_config_rejects_bad_regex(tmp_path: Path):
    raw = load_config(ROOT / "config.yaml").model_dump()
    raw["cleaning"]["running_head_patterns"] = ["(unclosed"]
    with pytest.raises(ValueError):
        Config.model_validate(raw)


def test_recovery_pass_supplies_lines_the_scan_pass_missed(tmp_path):
    pdf = tmp_path / "scan_vs_recover.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Only a short caption is in the text layer.", fontsize=11)
    doc.save(pdf)
    doc.close()

    lines = [f"Line {chr(97 + i)} of the missing body text describes unusual activity." for i in range(12)]
    def page_of(texts):
        boxes = [box(172, 100 + 60 * i, t, width=900) for i, t in enumerate(texts)]
        return {1: PageOCR(1, 1700, 2200, boxes)}

    scan, full = page_of(lines[:-1]), page_of(lines)
    asked = []

    def recover(numbers):
        asked.append(numbers)
        return full

    record = extract_pdf(pdf, "d", scan, _cfg().ocr, recover)[0]
    assert asked == [[1]]  # only the flagged page is re-OCR'd
    assert record["extraction_method"] == "ocr"
    assert lines[-1] in record["text"]  # the line the scan pass missed is back

    unflagged = extract_pdf(pdf, "d", {}, _cfg().ocr, recover)[0]
    assert unflagged["extraction_method"] == "text" and asked == [[1]]  # no OCR data, no recovery


# ----------------------------------------------------------------------------- merge (text layer + OCR-only)
def _layer_pdf_with_gap(path: Path) -> list[str]:
    """One page with 14 text-layer lines and a vertical gap where OCR-only lines will go."""
    texts = [f"Layer line {i} keeps the customer account activity review process documented here today" for i in range(14)]
    doc = pymupdf.open()
    page = doc.new_page()
    for i, text in enumerate(texts):
        y = 100 + 20 * i if i < 7 else 400 + 20 * (i - 7)
        page.insert_text((72, y), text, fontsize=10)
    doc.save(path)
    doc.close()
    return texts


def _merge_ocr_page(pdf: Path, dpi: int, only_texts: list[str]) -> PageOCR:
    from fincrime_rag.ocr import layer_lines

    k = dpi / 72
    with pymupdf.open(pdf) as doc:
        lines = layer_lines(doc[0])
    boxes = [  # OCR re-reads every layer line (with a typo) at the same place: all already covered
        OCRBox(ln.x0 * k, ln.y0 * k, ln.x1 * k, ln.y1 * k, ln.text.replace("customer", "custorner"), 0.95)
        for ln in lines
    ]
    for j, text in enumerate(only_texts):  # lines the text layer lacks, in the vertical gap
        y = 262 + 22 * j
        boxes.append(OCRBox(72 * k, (y - 9) * k, 480 * k, (y + 3) * k, text, 0.9))
    boxes.append(OCRBox(36 * k, 300 * k, 51 * k, 312 * k, "7", 0.6))  # icon noise in the gutter
    boxes.append(OCRBox(72 * k, 380 * k, 300 * k, 392 * k, "aaiaaa sooo saide", 0.4))  # garbage
    return PageOCR(1, round(612 * k), round(792 * k), boxes)


def test_merge_keeps_text_layer_and_adds_only_uncovered_ocr_lines(tmp_path):
    cfg = _cfg().ocr
    pdf = tmp_path / "gap.pdf"
    layer = _layer_pdf_with_gap(pdf)
    only = [f"Missing red flag {j} describes a customer moving funds through unrelated accounts abroad" for j in range(6)]
    record = extract_pdf(pdf, "d", {1: _merge_ocr_page(pdf, cfg.dpi, only)}, cfg)[0]

    assert record["extraction_method"] == "merged"
    assert record["box_coverage"] >= cfg.merge_min_box_coverage
    text = record["text"]
    assert all(line in text for line in layer)  # every text-layer line, verbatim
    assert "custorner" not in text  # OCR typos never replace the text layer
    assert all(line in text for line in only)  # the lines the text layer lacks are added
    assert text.count("Layer line 3 ") == 1  # covered OCR boxes are not duplicated
    assert "aaiaaa" not in text and not re.search(r"^7$", text, re.MULTILINE)  # garbage and icon noise
    order = [text.index(s) for s in (layer[6], only[0], only[5], layer[7])]
    assert order == sorted(order)  # vertical order: layer, OCR-only run, layer
    assert len(text.split("\n\n")) == 3  # top lines | OCR-only run | bottom lines


def test_merge_is_not_used_without_a_usable_text_layer(tmp_path):
    cfg = _cfg().ocr
    pdf = tmp_path / "thin.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "A short caption only is in the text layer here.", fontsize=10)
    doc.save(pdf)
    doc.close()
    only = [f"Scanned form line {j} lists the fields a customer must complete and sign" for j in range(10)]
    page = PageOCR(1, 1700, 2200, [box(200, 100 + 60 * j, t, width=1000) for j, t in enumerate(only)])
    record = extract_pdf(pdf, "d", {1: page}, cfg)[0]
    assert record["extraction_method"] == "ocr"
    assert "short caption" not in record["text"]  # OCR text alone, as the page has no usable layer


# ----------------------------------------------------------------------------- QA: text-layer lines lost
def test_text_layer_lines_lost_ignores_removed_heads_and_counts_real_losses(fixture_pdf):
    from fincrime_rag.qa import text_layer_lines_lost

    cfg = _cfg().cleaning
    cleaned, removed = clean_document(extract_fixture_pages(fixture_pdf), cfg)
    assert text_layer_lines_lost(fixture_pdf, cleaned, removed) == {"all": [], "merged": []}

    cleaned[0]["text"] = cleaned[0]["text"].replace("Money mule accounts move funds for criminals.", "")
    assert text_layer_lines_lost(fixture_pdf, cleaned, removed) == {"all": [1], "merged": []}
    cleaned[0]["extraction_method"] = "merged"
    assert text_layer_lines_lost(fixture_pdf, cleaned, removed) == {"all": [1], "merged": [1]}


# ----------------------------------------------------------------------------- printed page sanity
def test_inconsistent_printed_page_guesses_are_nulled_against_the_modal_offset():
    from fincrime_rag.cleaning import reject_inconsistent_page_guesses

    pages = [
        {"pdf_page": n, "printed_page_guess": str(n - 2) if n >= 3 else None, "printed_page_guess_rejected": None}
        for n in range(1, 13)
    ]
    pages[0]["printed_page_guess"] = "10"  # stray number read on the cover
    pages[1]["printed_page_guess"] = "19"
    pages[5]["printed_page_guess"] = "F-1"  # non-numeric guesses are left alone
    assert reject_inconsistent_page_guesses(pages) == 2
    assert pages[0]["printed_page_guess"] is None and pages[0]["printed_page_guess_rejected"] == "10"
    assert pages[1]["printed_page_guess"] is None and pages[1]["printed_page_guess_rejected"] == "19"
    assert pages[5]["printed_page_guess"] == "F-1"
    assert pages[2]["printed_page_guess"] == "1" and pages[11]["printed_page_guess"] == "10"


def test_page_guess_check_needs_enough_consistent_evidence():
    from fincrime_rag.cleaning import reject_inconsistent_page_guesses

    scattered = [{"pdf_page": n, "printed_page_guess": str(n * 3), "printed_page_guess_rejected": None} for n in range(1, 7)]
    assert reject_inconsistent_page_guesses(scattered) == 0  # no modal offset: nothing is rejected
    few = [{"pdf_page": n, "printed_page_guess": str(n + 5), "printed_page_guess_rejected": None} for n in (1, 2)]
    assert reject_inconsistent_page_guesses(few) == 0


# ----------------------------------------------------------------------------- content_type / indexing
def test_page_tags_set_content_type_and_indexing_excludes_it():
    from fincrime_rag.cleaning import indexable_pages
    from fincrime_rag.config import IndexingConfig, PageTag

    cfg = _cfg().cleaning.model_copy(
        update={"page_tags": [PageTag(doc_id="d", first_page=2, last_page=3, content_type="model_form")]}
    )
    cleaned, _ = clean_document(_doc([["Head"]] * 6), cfg)
    assert [p["content_type"] for p in cleaned] == ["body", "model_form", "model_form", "body", "body", "body"]
    kept = indexable_pages(cleaned, IndexingConfig(exclude_content_types=["model_form"]))
    assert [p["pdf_page"] for p in kept] == [1, 4, 5, 6]
    assert len(indexable_pages(cleaned, IndexingConfig())) == 6  # nothing excluded unless configured


def test_repo_config_tags_reg_e_model_forms_and_excludes_them():
    full = _cfg()
    tags = [t for t in full.cleaning.page_tags if t.doc_id == "cfr_12_1005_reg_e"]
    assert [(t.first_page, t.last_page, t.content_type) for t in tags] == [(61, 79, "model_form")]
    assert "model_form" in full.indexing.exclude_content_types


def test_numbered_list_markers_at_page_bottom_are_not_repeated_heads():
    heads = [["Alpha Review"]] * 6
    pages = _doc(heads)
    for i, page in enumerate(pages):
        page["text"] += chr(10) + f"{i + 3}."  # a list number left alone at the bottom of every page
    cleaned, log = clean_document(pages, _cfg().cleaning)
    assert all(f"{i + 3}." in p["text"] for i, p in enumerate(cleaned))
    assert not any(r["line"].rstrip(".").isdigit() and r["reason"].startswith("repeated") for r in log)
