"""Streamlit demo tests: a smoke test with the extractive backend (skipped without an index), plus tests with the
service layer replaced by fakes (no index, model or network needed)."""

from pathlib import Path

import pytest

from fincrime_rag.config import load_config
from fincrime_rag.generation import Answer
from fincrime_rag.service import Result, index_available

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
EXTRACTIVE = "Fast mode: extractive (no language model; much weaker)"

needs_index = pytest.mark.skipif(
    not (ROOT / "config.yaml").exists() or not index_available(load_config(ROOT / "config.yaml")),
    reason="index or refusal calibration not built",
)


@pytest.fixture()
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(ROOT)
    return AppTest.from_file(str(APP), default_timeout=120)


@needs_index
def test_notice_fixed_settings_and_extractive_answer(app):
    at = app.run()
    assert not at.exception
    assert any("Research demo; may be wrong" in w.value for w in at.warning)
    assert at.number_input(key="k_fixed").disabled and at.text_input(key="mode_fixed").disabled
    at.radio(key="backend").set_value(EXTRACTIVE)
    at.text_area(key="question").set_value("What must a bank do when it detects suspicious activity?")
    at.button(key="ask").click().run()
    assert not at.exception
    assert any("Elapsed" in c.value for c in at.caption)
    assert len(at.expander) >= 1  # citations and/or retrieved chunks


def _fake_service(monkeypatch, answer_text: str, chunk_ids: list[str]):
    import fincrime_rag.service as service

    chunks = [
        {"chunk_id": cid, "doc_id": "doc_a", "pdf_pages": [n + 1], "text": f"text of {cid}"}
        for n, cid in enumerate(chunk_ids)
    ]
    answer = Answer("answered", answer_text, backend="extractive", context_chunk_ids=chunk_ids)
    monkeypatch.setattr(service, "index_available", lambda cfg: True)
    monkeypatch.setattr(service, "load_resources", lambda *a, **k: object())
    monkeypatch.setattr(service, "ollama_status", lambda cfg: (False, "test: not reachable"))
    monkeypatch.setattr(service, "ask", lambda res, question, backend: Result(answer, 0.1, chunks, {}))


def test_displayed_chunks_are_the_context_chunks(app, monkeypatch):
    ids = ["doc_a:p0001:c0000", "doc_a:p0002:c0001", "doc_a:p0003:c0002"]
    _fake_service(monkeypatch, "plain answer", ids)
    at = app.run()
    at.radio(key="backend").set_value(EXTRACTIVE)
    at.text_area(key="question").set_value("anything")
    at.button(key="ask").click().run()
    assert not at.exception
    shown = [t.value for t in at.text if ", pages " in t.value and ", chunk " in t.value]
    assert [s.split(", chunk ")[1] for s in shown] == ids  # same chunks, same order


def test_markdown_image_syntax_in_an_answer_is_not_rendered(app, monkeypatch):
    payload = "See ![tracker](http://example.invalid/pixel.png) and [link](http://example.invalid)"
    _fake_service(monkeypatch, payload, ["doc_a:p0001:c0000"])
    at = app.run()
    at.radio(key="backend").set_value(EXTRACTIVE)
    at.text_area(key="question").set_value("anything")
    at.button(key="ask").click().run()
    assert not at.exception
    assert any(t.value == payload for t in at.text)  # shown verbatim as plain text
    assert not any("![" in m.value for m in at.markdown)  # never passed to the markdown renderer
    assert len(at.get("imgs")) == 0


def test_ollama_not_chosen_silently(app, monkeypatch):
    _fake_service(monkeypatch, "x", ["doc_a:p0001:c0000"])
    at = app.run()
    assert not at.exception
    assert (
        at.radio(key="backend").value == EXTRACTIVE
    )  # unreachable Ollama: defaults to the explicit fast mode
    assert any("not reachable" in e.value for e in at.sidebar.error)
