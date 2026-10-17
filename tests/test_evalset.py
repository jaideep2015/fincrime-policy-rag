"""Eval-set validator: schema rules, warnings, no content leakage, page helper. Uses synthetic records only."""

import json

import pytest
import yaml
from pydantic import ValidationError
from tests.conftest import ROOT, make_pdf

from fincrime_rag.config import EvalSetConfig, load_config
from fincrime_rag.evalset import (
    PageIndex,
    PageInfo,
    Report,
    build_page_index,
    format_pages,
    format_report,
    load_records,
    main,
    validate_file,
    validate_records,
)
from fincrime_rag.ingestion import ManifestRow, write_manifest

LIMITS = EvalSetConfig(max_questions_per_doc=3, min_key_facts=1, max_key_facts=4)


def make_index() -> PageIndex:
    body = PageInfo(None, "1", "body")
    return PageIndex(
        {
            "doc_a": {1: body, 2: body, 3: PageInfo(None, None, "model_form")},
            "doc_b": {1: body, 2: body},
            "doc_c": {1: body},
        }
    )


def rec(qid="q01", **over):
    base = {
        "qid": qid,
        "question": f"placeholder question number {qid}",
        "type": "answerable",
        "expected_behavior": "answer",
        "category": "lookup",
        "gold": [{"doc_id": "doc_a", "pages": [1]}],
        "reference_answer": "placeholder answer",
        "key_facts": ["placeholder fact"],
        "split": "dev",
    }
    base.update(over)
    return {k: v for k, v in base.items() if v is not None or k not in over}


def run(*records) -> Report:
    return validate_records([(i + 1, r) for i, r in enumerate(records)], make_index(), LIMITS)


def messages(report: Report, level="errors") -> str:
    return " | ".join(f"{i.where}: {i.message}" for i in getattr(report, level))


def unanswerable(qid="q02", **over):
    defaults = {"type": "unanswerable", "expected_behavior": "refuse", "category": "out_of_corpus", "gold": []}
    return rec(qid, **{**defaults, **over})


def injection(qid="q03", **over):
    defaults = {"type": "injection", "expected_behavior": "resist_injection", "category": "injection", "gold": []}
    return rec(qid, **{**defaults, **over})


# ----------------------------------------------------------------------------- happy path + summary
def test_valid_set_passes_and_summarises_counts():
    report = run(rec("q01"), unanswerable("q02"), injection("q03"), rec("q04", gold=[{"doc_id": "doc_b", "pages": [1, 2]}]))
    assert report.ok, messages(report)
    assert report.n_questions == 4
    assert report.summary["type"] == {"answerable": 2, "injection": 1, "unanswerable": 1}
    assert report.summary["category"] == {"injection": 1, "lookup": 2, "out_of_corpus": 1}
    assert report.summary["split"] == {"dev": 4}
    assert report.summary["document"] == {"doc_a": 1, "doc_b": 1}
    text = format_report(report)
    assert "0 error(s)" in text and "by type: answerable=2" in text


# ----------------------------------------------------------------------------- errors
@pytest.mark.parametrize(
    "record, expected",
    [
        (rec(type="maybe"), "type: 'maybe'"),
        (rec(category="trivia"), "category: 'trivia'"),
        (rec(expected_behavior="shrug"), "expected_behavior: 'shrug'"),
        (rec(split="train"), "split: 'train'"),
        (rec(expected_behavior="refuse"), "does not fit type 'answerable'"),
        (unanswerable(expected_behavior="answer"), "does not fit type 'unanswerable'"),
        (rec(gold=[]), "needs at least one"),
        (unanswerable(gold=[{"doc_id": "doc_a", "pages": [1]}]), "must be empty for type 'unanswerable'"),
        (injection(gold=[{"doc_id": "doc_a", "pages": [1]}]), "must be empty for type 'injection'"),
        (rec(gold=[{"doc_id": "nope", "pages": [1]}]), "doc_id 'nope' is not in data/manifest.csv"),
        (rec(gold=[{"doc_id": "doc_a", "pages": [4]}]), "page 4 is outside doc_a (PDF pages 1-3)"),
        (rec(gold=[{"doc_id": "doc_a", "pages": [0]}]), "page 0 is outside doc_a"),
        (rec(gold=[{"doc_id": "doc_a", "pages": ["2"]}]), "pages must be integers"),
        (rec(gold=[{"doc_id": "doc_a", "pages": [True]}]), "pages must be integers"),
        (rec(gold=[{"doc_id": "doc_a", "pages": []}]), "pages must be a non-empty list"),
        (rec(gold=[{"doc_id": "doc_a", "pages": [1, 1]}]), "duplicate page numbers"),
        (rec(gold=[{"doc_id": "doc_a", "pages": [1]}, {"doc_id": "doc_a", "pages": [2]}]), "appears more than once"),
        (rec(gold=[{"doc_id": "doc_a", "pages": [1], "extra": 1}]), "exactly the keys doc_id and pages"),
        (rec(gold="doc_a"), "gold: must be a list"),
        (rec(key_facts=[]), "key_facts: has 0 item(s); allowed 1-4"),
        (rec(key_facts=["a"] * 5), "key_facts: has 5 item(s); allowed 1-4"),
        (rec(key_facts=["ok", ""]), "key_facts: must be a list of non-empty strings"),
        (rec(key_facts="one fact"), "key_facts: must be a list"),
        (rec(reference_answer="  "), "reference_answer: must be a non-empty string"),
        (rec(question=""), "question: must be a non-empty string"),
        (rec(notes=5), "notes: must be a string"),
        (rec(extra_field="x"), "unknown field(s): extra_field"),
        (rec(qid=""), "qid: must be a non-empty string"),
    ],
)
def test_schema_violations_are_errors(record, expected):
    report = run(record)
    assert not report.ok
    assert expected in messages(report), messages(report)


@pytest.mark.parametrize("field_name", ["qid", "question", "type", "expected_behavior", "category", "gold",
                                        "reference_answer", "key_facts", "split"])
def test_each_required_field_is_required(field_name):
    record = rec()
    del record[field_name]
    assert f"missing required field: {field_name}" in messages(run(record))


def test_split_is_required_and_notes_optional():
    record = rec()
    assert "notes" not in record and run(record).ok
    del record["split"]
    assert not run(record).ok


def test_duplicate_qid_and_duplicate_question_are_errors():
    report = run(rec("q01"), rec("q01", question="a different placeholder"))
    assert "qid: duplicate (first used on line 1)" in messages(report)
    same = run(rec("q01", question="What is X?"), rec("q02", question="  what is   x ?? "))
    assert "q02: question: duplicates q01" in messages(same)


# ----------------------------------------------------------------------------- warnings
def test_warning_for_gold_page_tagged_non_body_does_not_fail():
    report = run(rec(gold=[{"doc_id": "doc_a", "pages": [3]}]))
    assert report.ok
    assert "content_type 'model_form'" in messages(report, "warnings")


def test_warning_when_a_document_has_more_than_three_questions():
    report = run(*[rec(f"q{i}") for i in range(4)])
    assert report.ok
    assert "doc_a: 4 questions use this document (limit 3)" in messages(report, "warnings")
    assert "3 questions use" not in messages(run(*[rec(f"q{i}") for i in range(3)]), "warnings")


def test_warning_for_documents_with_no_questions_only_when_there_are_questions():
    report = run(rec("q01"))
    warned = messages(report, "warnings")
    assert "doc_b: no question has this document" in warned and "doc_c: no question" in warned
    assert "doc_a: no question" not in warned
    assert run().warnings == []  # an empty file has nothing to warn about


def test_category_warnings_for_multi_doc_and_multi_page():
    warned = messages(run(rec("q01", category="multi_doc"), rec("q02", category="multi_page")), "warnings")
    assert "q01: category multi_doc but gold names fewer than two documents" in warned
    assert "q02: category multi_page but no document has two or more gold pages" in warned
    good = run(
        rec("q01", category="multi_doc", gold=[{"doc_id": "doc_a", "pages": [1]}, {"doc_id": "doc_b", "pages": [1]}]),
        rec("q02", category="multi_page", gold=[{"doc_id": "doc_a", "pages": [1, 2]}]),
    )
    assert "category" not in messages(good, "warnings")


# ----------------------------------------------------------------------------- loading
def test_load_records_reports_bad_lines_without_echoing_them(tmp_path):
    path = tmp_path / "qa_set.jsonl"
    good = json.dumps(rec("q01"))
    path.write_text(good + "\n\n{broken json ZZSECRETLINE\n[1, 2]\n" + good + "\n", encoding="utf-8")
    report = Report()
    records = load_records(path, report)
    assert [n for n, _ in records] == [1, 5]  # blank line skipped
    text = messages(report)
    assert "line 3: not valid JSON" in text and "line 4: each line must be a JSON object" in text
    assert "ZZSECRETLINE" not in text


# ----------------------------------------------------------------------------- no content leakage
def test_report_never_repeats_question_answer_fact_or_note_text():
    secrets = ("ZZSECRETQ", "ZZSECRETA", "ZZSECRETF", "ZZSECRETN", "ZZSECRETTYPE")
    bad = rec(
        question="ZZSECRETQ?",
        reference_answer="ZZSECRETA",
        key_facts=["ZZSECRETF"] * 6,
        notes="ZZSECRETN",
        type="ZZSECRETTYPE" * 10,
        category=["ZZSECRETQ"],
        gold=[{"doc_id": "bad_doc_id", "pages": [1]}],
    )
    dup = rec("q02", question="ZZSECRETQ?!")
    report = run(bad, dup)
    assert not report.ok
    text = format_report(report)
    assert not any(s in text for s in secrets[:4]), text
    assert "ZZSECRETTYPE" not in text  # over-long enum values are replaced by their type name
    assert "doc_id 'bad_doc_id' is not in data/manifest.csv" in text  # identifiers are shown, so typos are fixable


# ----------------------------------------------------------------------------- files and CLI
def write_project(cfg, tmp_path):
    """Manifest + processed pages for doc_a (3 pages, one model_form) and doc_b (PDF only)."""
    rows = [
        ManifestRow("doc_a", "A", "P", "manual", "doc_a.pdf", "https://x", "", "", "", ""),
        ManifestRow("doc_b", "B", "P", "manual", "doc_b.pdf", "https://x", "", "", "", ""),
    ]
    write_manifest(cfg.paths.manifest, rows)
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    pages = [
        {"doc_id": "doc_a", "pdf_page": 1, "page_label": "i", "printed_page_guess": None, "content_type": "body",
         "text": "SECRET PAGE TEXT one"},
        {"doc_id": "doc_a", "pdf_page": 2, "page_label": None, "printed_page_guess": "5", "content_type": "body",
         "text": "SECRET PAGE TEXT two"},
        {"doc_id": "doc_a", "pdf_page": 3, "page_label": None, "printed_page_guess": None,
         "content_type": "model_form", "text": "SECRET PAGE TEXT three"},
    ]
    (cfg.paths.processed_dir / "doc_a.pages.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pages) + "\n", encoding="utf-8"
    )
    make_pdf(cfg.paths.raw_dir / "doc_b.pdf", n_pages=4)


def tmp_config(cfg, tmp_path, eval_path):
    data = cfg.model_dump(mode="json")
    data["paths"]["eval_set"] = str(eval_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_build_page_index_uses_processed_pages_then_falls_back_to_the_pdf(cfg, tmp_path):
    write_project(cfg, tmp_path)
    index = build_page_index(cfg)
    assert index.page_count("doc_a") == 3 and index.page_count("doc_b") == 4
    assert index.info("doc_a", 3).content_type == "model_form"
    assert index.info("doc_a", 2).printed_page_guess == "5"
    assert index.info("doc_b", 1) == PageInfo(None, None, "body")


def test_validate_file_ignores_other_files_and_reports_missing_or_wrong_extension(cfg, tmp_path):
    write_project(cfg, tmp_path)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "EXAMPLE_FORMAT.md").write_text("{not valid json at all", encoding="utf-8")
    (eval_dir / "qa_set.jsonl").write_text(json.dumps(rec("q01", gold=[{"doc_id": "doc_a", "pages": [2]}])) + "\n")
    report = validate_file(eval_dir / "qa_set.jsonl", cfg)
    assert report.ok and report.n_questions == 1

    missing = validate_file(eval_dir / "nope.jsonl", cfg)
    assert "not found" in messages(missing) and "EXAMPLE_FORMAT.md" in messages(missing)
    assert "is not a .jsonl file" in messages(validate_file(eval_dir / "EXAMPLE_FORMAT.md", cfg))


def test_validate_cli_exit_codes_and_output(cfg, tmp_path, capsys):
    write_project(cfg, tmp_path)
    path = tmp_path / "qa_set.jsonl"
    config_path = tmp_config(cfg, tmp_path, path)

    path.write_text(json.dumps(rec("q01", gold=[{"doc_id": "doc_b", "pages": [4]}])) + "\n", encoding="utf-8")
    assert main(["--config", str(config_path), "validate"]) == 0
    assert "0 error(s)" in capsys.readouterr().out

    path.write_text(json.dumps(rec("q01", gold=[{"doc_id": "doc_b", "pages": [5]}])) + "\n", encoding="utf-8")
    assert main(["--config", str(config_path), "validate"]) == 1
    assert "outside doc_b (PDF pages 1-4)" in capsys.readouterr().out

    path.unlink()
    assert main(["--config", str(config_path), "validate"]) == 1
    assert "not found" in capsys.readouterr().out


def test_pages_helper_prints_page_numbers_but_never_page_text(cfg, tmp_path, capsys):
    write_project(cfg, tmp_path)
    config_path = tmp_config(cfg, tmp_path, tmp_path / "qa_set.jsonl")
    assert main(["--config", str(config_path), "pages", "doc_a"]) == 0
    out = capsys.readouterr().out
    assert "SECRET" not in out
    rows = [line.split() for line in out.splitlines()[1:4]]
    assert rows[0][:3] == ["1", "i", "-"] and rows[1][:3] == ["2", "-", "5"] and rows[2][3] == "model_form"
    assert "pdf_page" in out and "printed_page_guess" in out and "Gold pages use the pdf_page" in out
    assert "SECRET" not in format_pages("doc_a", build_page_index(cfg))

    assert main(["--config", str(config_path), "pages", "nope"]) == 2
    assert "unknown doc_id" in capsys.readouterr().out


# ----------------------------------------------------------------------------- repo wiring
def test_repo_wiring_example_file_exists_and_config_validates():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.paths.eval_set.name == "qa_set.jsonl"
    example = ROOT / "data" / "eval" / "EXAMPLE_FORMAT.md"
    assert example.exists() and example.suffix == ".md"
    assert cfg.evalset.max_questions_per_doc == 3 and (cfg.evalset.min_key_facts, cfg.evalset.max_key_facts) == (1, 4)
    with pytest.raises(ValidationError):
        EvalSetConfig(max_questions_per_doc=3, min_key_facts=3, max_key_facts=2)
