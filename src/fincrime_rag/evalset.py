"""Validator for the hand-written evaluation set (``data/eval/qa_set.jsonl``) and a page-lookup helper.

The questions belong to the user: this module only *checks* them. Its output never repeats question,
answer, key-fact or note text: problems are reported by qid, field and (for gold) document and page.

Commands (``python -m fincrime_rag.evalset <command>``):

* ``validate``: check every record against the schema (D-018) and print a summary of counts.
  Exit code 1 if there are errors. Warnings do not fail.
* ``pages <doc_id>``: print ``pdf_page``, ``page_label``, ``printed_page_guess`` and ``content_type``
  for each page of a document, so PDF indexes and printed numbers are not mixed up. Read-only; it
  never prints page text.

Schema, one JSON object per line::

    qid, question, type, expected_behavior, category, gold, reference_answer, key_facts, split, [notes]

``gold`` is a list of ``{"doc_id": ..., "pages": [pdf_page, ...]}`` (empty for unanswerable and
injection questions). Pages are PDF page indexes (1-based), not printed numbers. Only
``data/eval/qa_set.jsonl`` is read; other files in that folder (for example ``EXAMPLE_FORMAT.md``) are
ignored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from fincrime_rag.config import Config, EvalSetConfig, load_config
from fincrime_rag.extraction import read_pages
from fincrime_rag.ingestion import read_manifest

TYPES = ("answerable", "unanswerable", "injection")
BEHAVIOR_FOR_TYPE = {"answerable": "answer", "unanswerable": "refuse", "injection": "resist_injection"}
BEHAVIORS = tuple(BEHAVIOR_FOR_TYPE.values())
CATEGORIES = (
    "lookup",
    "definition",
    "list",
    "numeric",
    "multi_page",
    "multi_doc",
    "conflict",
    "paraphrase",
    "false_premise",
    "out_of_corpus",
    "injection",
)
SPLITS = ("dev", "test")
REQUIRED_FIELDS = (
    "qid",
    "question",
    "type",
    "expected_behavior",
    "category",
    "gold",
    "reference_answer",
    "key_facts",
    "split",
)
OPTIONAL_FIELDS = ("notes",)


# --------------------------------------------------------------------------- page index
@dataclass(frozen=True)
class PageInfo:
    page_label: str | None
    printed_page_guess: str | None
    content_type: str


class PageIndex:
    """doc_id -> {pdf_page: PageInfo}. Holds metadata only, never page text."""

    def __init__(self, docs: dict[str, dict[int, PageInfo]]):
        self.docs = docs

    def has_doc(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def page_count(self, doc_id: str) -> int:
        return max(self.docs[doc_id], default=0)

    def info(self, doc_id: str, pdf_page: int) -> PageInfo | None:
        return self.docs.get(doc_id, {}).get(pdf_page)


def build_page_index(cfg: Config) -> PageIndex:
    """Page metadata for every manifest document, from the processed pages (else the raw PDF)."""
    docs: dict[str, dict[int, PageInfo]] = {}
    for row in read_manifest(cfg.paths.manifest):
        processed = cfg.paths.processed_dir / f"{row.doc_id}.pages.jsonl"
        raw = cfg.paths.raw_dir / row.filename
        pages: dict[int, PageInfo] = {}
        if processed.exists():
            for p in read_pages(processed):
                pages[p["pdf_page"]] = PageInfo(
                    p.get("page_label"), p.get("printed_page_guess"), p.get("content_type", "body")
                )
        elif raw.exists():
            with pymupdf.open(raw) as doc:
                pages = {n: PageInfo(None, None, "body") for n in range(1, doc.page_count + 1)}
        docs[row.doc_id] = pages
    return PageIndex(docs)


# --------------------------------------------------------------------------- loading
@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    where: str  # qid, "line N" or "file"
    message: str


@dataclass
class Report:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    summary: dict[str, dict[str, int]] = field(default_factory=dict)
    n_questions: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, where: str, message: str) -> None:
        self.errors.append(Issue("error", where, message))

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(Issue("warning", where, message))


def load_records(path: Path, report: Report) -> list[tuple[int, dict]]:
    """Parse the JSONL file. Blank lines are skipped; a bad line is reported (without its content)."""
    records: list[tuple[int, dict]] = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                report.error(f"line {number}", f"not valid JSON ({exc.msg} at column {exc.colno})")
                continue
            if not isinstance(obj, dict):
                report.error(f"line {number}", "each line must be a JSON object")
                continue
            records.append((number, obj))
    return records


# --------------------------------------------------------------------------- validation
def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _norm_question(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _enum(report: Report, where: str, rec: dict, name: str, allowed: tuple[str, ...]) -> bool:
    value = rec.get(name)
    if value in allowed:  # unhashable values (lists) fall through to the error below
        return True
    shown = value if isinstance(value, str) and len(value) <= 30 else type(value).__name__
    report.error(where, f"{name}: {shown!r} is not one of {', '.join(allowed)}")
    return False


def _check_gold(report: Report, where: str, rec: dict, index: PageIndex) -> list[tuple[str, int]]:
    """Validate ``gold``; returns the valid (doc_id, pdf_page) pairs for later checks."""
    gold = rec.get("gold")
    pairs: list[tuple[str, int]] = []
    if not isinstance(gold, list):
        report.error(where, "gold: must be a list of {doc_id, pages} objects (empty list if none)")
        return pairs
    seen_docs: set[str] = set()
    for position, entry in enumerate(gold, start=1):
        tag = f"gold[{position}]"
        if not isinstance(entry, dict) or set(entry) != {"doc_id", "pages"}:
            report.error(where, f"{tag}: must be an object with exactly the keys doc_id and pages")
            continue
        doc_id, pages = entry["doc_id"], entry["pages"]
        if not _text(doc_id) or not index.has_doc(doc_id):
            shown = doc_id if isinstance(doc_id, str) and len(doc_id) <= 60 else type(doc_id).__name__
            report.error(where, f"{tag}: doc_id {shown!r} is not in data/manifest.csv")
            continue
        if doc_id in seen_docs:
            report.error(where, f"{tag}: doc_id {doc_id} appears more than once; list its pages together")
            continue
        seen_docs.add(doc_id)
        if not isinstance(pages, list) or not pages:
            report.error(where, f"{tag}: pages must be a non-empty list of PDF page numbers")
            continue
        if any(isinstance(p, bool) or not isinstance(p, int) for p in pages):
            report.error(where, f"{tag}: pages must be integers (PDF page indexes, not printed numbers)")
            continue
        if len(set(pages)) != len(pages):
            report.error(where, f"{tag}: duplicate page numbers for {doc_id}")
        count = index.page_count(doc_id)
        for page in sorted(set(pages)):
            if page < 1 or page > count:
                report.error(where, f"{tag}: page {page} is outside {doc_id} (PDF pages 1-{count})")
            else:
                pairs.append((doc_id, page))
    return pairs


def validate_records(
    records: list[tuple[int, dict]],
    index: PageIndex,
    limits: EvalSetConfig,
    report: Report | None = None,
) -> Report:
    report = report or Report()
    seen_qids: dict[str, int] = {}
    seen_questions: dict[str, str] = {}
    docs_by_qid: dict[str, set[str]] = {}
    counts: dict[str, Counter] = {k: Counter() for k in ("type", "category", "split", "document")}

    for line, rec in records:
        where = rec["qid"] if _text(rec.get("qid")) else f"line {line}"
        unknown = sorted(set(rec) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
        if unknown:
            report.error(where, f"unknown field(s): {', '.join(unknown)}")
        for name in REQUIRED_FIELDS:
            if name not in rec:
                report.error(where, f"missing required field: {name}")

        qid = rec.get("qid")
        if "qid" in rec and not _text(qid):
            report.error(where, "qid: must be a non-empty string")
        elif _text(qid):
            if qid in seen_qids:
                report.error(where, f"qid: duplicate (first used on line {seen_qids[qid]})")
            seen_qids.setdefault(qid, line)

        question = rec.get("question")
        if "question" in rec and not _text(question):
            report.error(where, "question: must be a non-empty string")
        elif _text(question):
            key = _norm_question(question)
            if key in seen_questions:
                report.error(where, f"question: duplicates {seen_questions[key]}")
            else:
                seen_questions[key] = where

        type_ok = "type" in rec and _enum(report, where, rec, "type", TYPES)
        behavior_ok = "expected_behavior" in rec and _enum(report, where, rec, "expected_behavior", BEHAVIORS)
        category_ok = "category" in rec and _enum(report, where, rec, "category", CATEGORIES)
        split_ok = "split" in rec and _enum(report, where, rec, "split", SPLITS)
        if type_ok and behavior_ok and BEHAVIOR_FOR_TYPE[rec["type"]] != rec["expected_behavior"]:
            report.error(
                where,
                f"expected_behavior: {rec['expected_behavior']!r} does not fit type {rec['type']!r} "
                f"(expected {BEHAVIOR_FOR_TYPE[rec['type']]!r})",
            )

        pairs = _check_gold(report, where, rec, index) if "gold" in rec else []
        if type_ok and isinstance(rec.get("gold"), list):
            if rec["type"] == "answerable" and not rec["gold"]:
                report.error(where, "gold: an answerable question needs at least one {doc_id, pages} entry")
            if rec["type"] != "answerable" and rec["gold"]:
                report.error(where, f"gold: must be empty for type {rec['type']!r}")

        if "reference_answer" in rec and not _text(rec["reference_answer"]):
            report.error(where, "reference_answer: must be a non-empty string")
        facts = rec.get("key_facts")
        if "key_facts" in rec:
            if not isinstance(facts, list) or not all(_text(f) for f in facts):
                report.error(where, "key_facts: must be a list of non-empty strings")
            elif not limits.min_key_facts <= len(facts) <= limits.max_key_facts:
                report.error(
                    where,
                    f"key_facts: has {len(facts)} item(s); "
                    f"allowed {limits.min_key_facts}-{limits.max_key_facts}",
                )
        if "notes" in rec and not isinstance(rec["notes"], str):
            report.error(where, "notes: must be a string when present")

        # ---- warnings (never fail validation)
        gold_docs = sorted({d for d, _ in pairs})
        docs_by_qid[where] = set(gold_docs)
        for doc_id, page in pairs:
            info = index.info(doc_id, page)
            if info is not None and info.content_type != "body":
                report.warn(
                    where,
                    f"gold page {doc_id} p{page} is tagged content_type '{info.content_type}' "
                    "(kept out of the index by default)",
                )
        if category_ok and rec["category"] == "multi_doc" and len(gold_docs) < 2:
            report.warn(where, "category multi_doc but gold names fewer than two documents")
        if category_ok and rec["category"] == "multi_page":
            per_doc = Counter(d for d, _ in pairs)
            if not any(n >= 2 for n in per_doc.values()):
                report.warn(where, "category multi_page but no document has two or more gold pages")

        if type_ok:
            counts["type"][rec["type"]] += 1
        if category_ok:
            counts["category"][rec["category"]] += 1
        if split_ok:
            counts["split"][rec["split"]] += 1
        for doc_id in gold_docs:
            counts["document"][doc_id] += 1

    per_doc_questions = Counter(doc for docs in docs_by_qid.values() for doc in docs)
    for doc_id, n in sorted(per_doc_questions.items()):
        if n > limits.max_questions_per_doc:
            report.warn(doc_id, f"{n} questions use this document (limit {limits.max_questions_per_doc})")
    if records:
        for doc_id in sorted(index.docs):
            if doc_id not in per_doc_questions:
                report.warn(doc_id, "no question has this document in its gold")

    report.n_questions = len(records)
    report.summary = {k: dict(sorted(v.items())) for k, v in counts.items()}
    return report


def validate_file(path: Path, cfg: Config, index: PageIndex | None = None) -> Report:
    report = Report()
    if path.suffix != ".jsonl":
        report.error("file", f"{path.name} is not a .jsonl file")
        return report
    if not path.exists():
        report.error("file", f"{path} not found (format example: data/eval/EXAMPLE_FORMAT.md)")
        return report
    records = load_records(path, report)
    return validate_records(records, index or build_page_index(cfg), cfg.evalset, report)


def format_report(report: Report) -> str:
    """Plain-text report. Contains qids, field names, doc_ids and counts, never question text."""
    lines: list[str] = []
    for issue in report.errors:
        lines.append(f"[ERROR]   {issue.where}: {issue.message}")
    for issue in report.warnings:
        lines.append(f"[warning] {issue.where}: {issue.message}")
    if report.n_questions:
        lines.append("")
        lines.append(f"{report.n_questions} question(s)")
        titles = {"type": "by type", "category": "by category", "split": "by split"}
        titles["document"] = "by document"
        for key, title in titles.items():
            parts = ", ".join(f"{name}={n}" for name, n in report.summary.get(key, {}).items()) or "-"
            lines.append(f"  {title}: {parts}")
    lines.append(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- pages helper
def format_pages(doc_id: str, index: PageIndex) -> str:
    """pdf_page / page_label / printed_page_guess / content_type for each page (no page text)."""
    pages = index.docs[doc_id]
    rows = [f"{'pdf_page':>8}  {'page_label':<10}  {'printed_page_guess':<18}  content_type"]
    for number in sorted(pages):
        info = pages[number]
        rows.append(
            f"{number:>8}  {info.page_label or '-':<10}  "
            f"{info.printed_page_guess or '-':<18}  {info.content_type}"
        )
    rows.append(f"{len(pages)} page(s). Gold pages use the pdf_page column.")
    return "\n".join(rows)


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.evalset", description="Eval-set tools.")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="validate data/eval/qa_set.jsonl")
    check.add_argument("--file", help="a .jsonl file to validate instead of paths.eval_set")
    pages = sub.add_parser("pages", help="list pdf_page / page_label / printed_page_guess for a document")
    pages.add_argument("doc_id")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.command == "validate":
        report = validate_file(Path(args.file) if args.file else cfg.paths.eval_set, cfg)
        print(format_report(report))
        return 0 if report.ok else 1

    index = build_page_index(cfg)
    if not index.has_doc(args.doc_id):
        print(f"unknown doc_id {args.doc_id!r}; known: {', '.join(sorted(index.docs))}")
        return 2
    print(format_pages(args.doc_id, index))
    return 0


if __name__ == "__main__":
    sys.exit(main())
