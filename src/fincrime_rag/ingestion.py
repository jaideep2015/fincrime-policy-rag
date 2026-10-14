"""Stage 1: manifest-driven download and verification of the source PDFs.

Two commands (``python -m fincrime_rag.ingestion <command>``):

* ``download``: fetch every ``source=script`` document that is not yet on disk, check it is
  really a PDF, and record its sha256 in the manifest.
* ``verify``: re-check every file in ``data/raw`` against the manifest. For ``source=manual``
  files (sites that block scripted downloads) it also fills in a blank sha256.

TLS certificates are always verified. Downloads are limited to hosts listed in
``config.yaml`` (``ingestion.allowed_download_hosts``), including after redirects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from urllib.parse import urlparse

import requests

from fincrime_rag.config import Config, load_config

PDF_MAGIC = b"%PDF-"
CHUNK = 1 << 16

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}/\d{{1,2}}/\d{{4}}(?:\.V\d+)?|(?:{_MONTHS})\s+\d{{1,2}},\s+\d{{4}}"
    rf"|(?:{_MONTHS})\s+\d{{4}})\b"
)


@dataclass
class ManifestRow:
    doc_id: str
    title: str
    publisher: str
    source: str  # "script" or "manual"
    filename: str
    url: str
    date: str  # publication date
    version_date: str  # version/revision date printed in the PDF itself
    sha256: str
    license_note: str
    note: str = ""  # free-text caveats about this document


MANIFEST_FIELDS = [f.name for f in fields(ManifestRow)]


# --------------------------------------------------------------------------- manifest I/O
def read_manifest(path: Path) -> list[ManifestRow]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = (set(MANIFEST_FIELDS) - {"note"}) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = [ManifestRow(**{k: (r.get(k) or "").strip() for k in MANIFEST_FIELDS}) for r in reader]
    ids = [r.doc_id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest has duplicate doc_id values")
    for r in rows:
        if r.source not in {"script", "manual"}:
            raise ValueError(f"{r.doc_id}: source must be 'script' or 'manual', got {r.source!r}")
    return rows


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- file checks
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def has_pdf_header(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(len(PDF_MAGIC)) == PDF_MAGIC


def suggest_dates(path: Path, max_pages: int = 2) -> list[str]:
    """Date-like strings on the first pages, to help fill `date` / `version_date` by hand."""
    import pymupdf  # imported lazily: only needed for this convenience

    found: list[str] = []
    with pymupdf.open(path) as doc:
        for page in doc.pages(0, min(max_pages, doc.page_count)):
            for m in _DATE_RE.findall(page.get_text()):
                if m not in found:
                    found.append(m)
    return found[:6]


# --------------------------------------------------------------------------- download
class DownloadError(Exception):
    pass


def _host_allowed(url: str, cfg: Config) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in cfg.ingestion.allowed_download_hosts


def download_one(row: ManifestRow, cfg: Config, force: bool = False) -> Path:
    """Download one document to data/raw. Raises DownloadError; never leaves a partial file."""
    dest = cfg.paths.raw_dir / row.filename
    if dest.exists() and not force:
        raise DownloadError("already on disk (use --force to re-download)")
    if not _host_allowed(row.url, cfg):
        raise DownloadError(f"host not in ingestion.allowed_download_hosts: {row.url}")

    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(
            row.url,
            stream=True,
            timeout=cfg.ingestion.timeout_seconds,
            headers={"User-Agent": cfg.ingestion.user_agent},
        ) as resp:
            if resp.status_code != 200:
                raise DownloadError(f"HTTP {resp.status_code}")
            if not _host_allowed(resp.url, cfg):
                raise DownloadError(f"redirected to a disallowed host: {resp.url}")
            expected = resp.headers.get("Content-Length")
            written = 0
            with open(part, "wb") as f:
                for block in resp.iter_content(CHUNK):
                    f.write(block)
                    written += len(block)
        if expected is not None and written != int(expected):
            raise DownloadError(f"size mismatch: got {written} bytes, Content-Length {expected}")
        if written < cfg.ingestion.min_pdf_bytes:
            raise DownloadError(f"too small ({written} bytes) to be the expected PDF")
        if not has_pdf_header(part):
            raise DownloadError("response is not a PDF (missing %PDF- header)")
        os.replace(part, dest)
        return dest
    except requests.RequestException as exc:
        raise DownloadError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        part.unlink(missing_ok=True)


def download_all(cfg: Config, force: bool = False) -> int:
    rows = read_manifest(cfg.paths.manifest)
    failures = 0
    for row in rows:
        if row.source != "script":
            continue
        try:
            dest = download_one(row, cfg, force=force)
        except DownloadError as exc:
            if str(exc).startswith("already on disk"):
                print(f"[skip]  {row.doc_id}: {exc}")
                continue
            failures += 1
            print(f"[FAIL]  {row.doc_id}: {exc}\n        download manually from {row.url}\n"
                  f"        and save as data/raw/{row.filename}")
            continue
        row.sha256 = sha256_file(dest)
        print(f"[ok]    {row.doc_id}: {dest.stat().st_size:,} bytes sha256={row.sha256[:12]}...")
        write_manifest(cfg.paths.manifest, rows)  # persist after each file
    return 1 if failures else 0


# --------------------------------------------------------------------------- verify
def verify_all(cfg: Config) -> int:
    rows = read_manifest(cfg.paths.manifest)
    raw = cfg.paths.raw_dir
    problems = 0
    changed = False
    manual_todo: list[ManifestRow] = []

    for row in rows:
        path = raw / row.filename
        if not path.exists():
            problems += 1
            if row.source == "manual":
                manual_todo.append(row)
            print(f"[MISSING]  {row.doc_id}")
            continue
        size = path.stat().st_size
        if size < cfg.ingestion.min_pdf_bytes or not has_pdf_header(path):
            problems += 1
            print(f"[BAD PDF]  {row.doc_id}: {size:,} bytes or no %PDF- header (saved an HTML error page?)")
            continue
        digest = sha256_file(path)
        if not row.sha256:
            if row.source == "manual":
                row.sha256 = digest
                changed = True
                print(f"[FILLED]   {row.doc_id}: sha256 recorded ({digest[:12]}...)")
            else:
                problems += 1
                print(f"[NO SHA]   {row.doc_id}: script-downloaded file has no recorded sha256")
                continue
        elif row.sha256 != digest:
            problems += 1
            print(f"[MISMATCH] {row.doc_id}: manifest {row.sha256[:12]}... != file {digest[:12]}...")
            continue
        else:
            print(f"[ok]       {row.doc_id}: {size:,} bytes")
        if not row.date or not row.version_date:
            hints = suggest_dates(path)
            if hints:
                missing = [n for n, v in (("date", row.date), ("version_date", row.version_date)) if not v]
                print(f"           blank {'/'.join(missing)}; dates seen on first pages: {hints}")

    if changed:
        write_manifest(cfg.paths.manifest, rows)

    known = {r.filename for r in rows}
    for extra in sorted(p.name for p in raw.glob("*") if p.is_file() and p.name not in known | {".gitkeep"}):
        print(f"[EXTRA]    data/raw/{extra} is not in the manifest")

    if manual_todo:
        print("\nFiles to place manually (browser download, then save under exactly this name):")
        for r in manual_todo:
            print(f"  data/raw/{r.filename}\n      from {r.url}")
    print(f"\n{len(rows)} documents in manifest, {problems} problem(s).")
    return 1 if problems else 0


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fincrime_rag.ingestion", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "verify"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="re-download files already on disk")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "download":
        return download_all(cfg, force=args.force)
    return verify_all(cfg)


if __name__ == "__main__":
    sys.exit(main())
