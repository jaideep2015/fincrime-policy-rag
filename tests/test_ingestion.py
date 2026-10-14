import pytest
import requests
from tests.conftest import make_pdf

from fincrime_rag import ingestion
from fincrime_rag.ingestion import (
    DownloadError,
    ManifestRow,
    download_one,
    has_pdf_header,
    read_manifest,
    sha256_file,
    verify_all,
    write_manifest,
)


def _row(doc_id="d1", source="manual", url="https://www.fincen.gov/x.pdf", sha=""):
    return ManifestRow(doc_id, "Title", "Pub", source, f"{doc_id}.pdf", url, "", "", sha, "note")


def test_manifest_roundtrip_and_validation(cfg):
    rows = [_row("a"), _row("b", source="script")]
    write_manifest(cfg.paths.manifest, rows)
    assert read_manifest(cfg.paths.manifest) == rows

    write_manifest(cfg.paths.manifest, [_row("a"), _row("a")])
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(cfg.paths.manifest)

    write_manifest(cfg.paths.manifest, [_row("a", source="carrier-pigeon")])
    with pytest.raises(ValueError, match="source"):
        read_manifest(cfg.paths.manifest)


def test_repo_manifest_is_valid():
    from tests.conftest import ROOT

    rows = read_manifest(ROOT / "data" / "manifest.csv")
    assert 10 <= len(rows) <= 15
    assert {r.source for r in rows} <= {"script", "manual"}
    assert all(r.url.startswith("https://") and r.filename.endswith(".pdf") for r in rows)


def test_sha_and_pdf_header(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", n_pages=2)
    html = tmp_path / "b.pdf"
    html.write_bytes(b"<!DOCTYPE html><html>blocked</html>")
    assert has_pdf_header(pdf) and not has_pdf_header(html)
    assert len(sha256_file(pdf)) == 64


def test_verify_fills_sha_for_manual_files_and_reports_missing(cfg, capsys):
    make_pdf(cfg.paths.raw_dir / "m1.pdf", n_pages=60)  # large enough to pass min size
    write_manifest(cfg.paths.manifest, [_row("m1"), _row("m2")])
    cfg.ingestion.min_pdf_bytes = 1000

    assert verify_all(cfg) == 1  # m2 is missing
    out = capsys.readouterr().out
    assert "[FILLED]   m1" in out and "[MISSING]  m2" in out and "data/raw/m2.pdf" in out
    rows = {r.doc_id: r for r in read_manifest(cfg.paths.manifest)}
    assert rows["m1"].sha256 == sha256_file(cfg.paths.raw_dir / "m1.pdf")
    assert rows["m2"].sha256 == ""


def test_verify_detects_mismatch_html_and_extra_files(cfg, capsys):
    cfg.ingestion.min_pdf_bytes = 1000
    good = make_pdf(cfg.paths.raw_dir / "g.pdf", n_pages=60)
    (cfg.paths.raw_dir / "h.pdf").write_bytes(b"<!DOCTYPE html>" + b"x" * 5000)
    (cfg.paths.raw_dir / "stray.pdf").write_bytes(b"%PDF-1.4 stray")
    write_manifest(
        cfg.paths.manifest,
        [_row("g", source="script", sha="0" * 64), _row("h", source="script", sha="1" * 64)],
    )
    assert verify_all(cfg) == 1
    out = capsys.readouterr().out
    assert "[MISMATCH] g" in out and "[BAD PDF]  h" in out
    assert "[EXTRA]" in out and "stray.pdf" in out
    assert sha256_file(good) != "0" * 64


def test_verify_script_file_without_sha_is_a_problem(cfg, capsys):
    cfg.ingestion.min_pdf_bytes = 1000
    make_pdf(cfg.paths.raw_dir / "s.pdf", n_pages=60)
    write_manifest(cfg.paths.manifest, [_row("s", source="script")])
    assert verify_all(cfg) == 1
    assert "[NO SHA]" in capsys.readouterr().out


def test_download_rejects_disallowed_host_without_network(cfg):
    with pytest.raises(DownloadError, match="allowed_download_hosts"):
        download_one(_row(url="https://www.fatf-gafi.org/x.pdf"), cfg)
    with pytest.raises(DownloadError, match="allowed_download_hosts"):
        download_one(_row(url="http://www.fincen.gov/x.pdf"), cfg)  # not https


class _FakeResponse:
    def __init__(self, body: bytes, url: str, status: int = 200, length: int | None = None):
        self._body, self.url, self.status_code = body, url, status
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    def iter_content(self, _size):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_get(monkeypatch, response):
    monkeypatch.setattr(ingestion.requests, "get", lambda *a, **k: response)


def test_download_accepts_valid_pdf_and_leaves_no_partial_file(cfg, monkeypatch, tmp_path):
    cfg.ingestion.min_pdf_bytes = 100
    body = make_pdf(tmp_path / "src.pdf", n_pages=30).read_bytes()
    row = _row(source="script")
    _patch_get(monkeypatch, _FakeResponse(body, row.url))
    dest = download_one(row, cfg)
    assert dest.read_bytes() == body and not list(cfg.paths.raw_dir.glob("*.part"))
    with pytest.raises(DownloadError, match="already on disk"):
        download_one(row, cfg)


@pytest.mark.parametrize(
    "response, message",
    [
        (_FakeResponse(b"<html>" + b"x" * 500, "https://www.fincen.gov/x.pdf"), "not a PDF"),
        (_FakeResponse(b"%PDF-xxxxx", "https://www.fincen.gov/x.pdf", length=10**6), "size mismatch"),
        (_FakeResponse(b"%PDF-", "https://www.fincen.gov/x.pdf"), "too small"),
        (_FakeResponse(b"", "https://www.fincen.gov/x.pdf", status=403), "HTTP 403"),
        (_FakeResponse(b"%PDF-" + b"x" * 500, "https://evil.example.com/x.pdf"), "disallowed host"),
    ],
)
def test_download_rejects_bad_responses(cfg, monkeypatch, response, message):
    cfg.ingestion.min_pdf_bytes = 100
    _patch_get(monkeypatch, response)
    with pytest.raises(DownloadError, match=message):
        download_one(_row(source="script"), cfg)
    assert not list(cfg.paths.raw_dir.glob("*"))  # nothing left behind


def test_download_wraps_network_errors(cfg, monkeypatch):
    def boom(*_a, **_k):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(ingestion.requests, "get", boom)
    with pytest.raises(DownloadError, match="ConnectionError"):
        download_one(_row(source="script"), cfg)
