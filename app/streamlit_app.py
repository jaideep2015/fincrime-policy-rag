"""Research demo UI for the fraud / AML policy assistant. Run: streamlit run app/streamlit_app.py

Fixed settings (shown, not editable): hybrid retrieval, k = 8, prompt v2, refusal gate from the calibration file.
No network access except the local Ollama server; questions are not written to disk.
"""

import os

# The embedding model must come from the local cache: no calls to the Hugging Face hub at runtime.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import streamlit as st  # noqa: E402

from fincrime_rag.config import load_config  # noqa: E402
from fincrime_rag.llm.base import LLMUnavailable  # noqa: E402
from fincrime_rag.service import ask, index_available, load_resources, ollama_status  # noqa: E402

NOTICE = "Research demo; may be wrong; check the citations; not legal or compliance advice."
OLLAMA_LABEL = "Ollama (qwen2.5:3b-instruct)"
EXTRACTIVE_LABEL = "Fast mode: extractive (no language model; much weaker)"

st.set_page_config(page_title="Fraud & AML policy assistant (research demo)", layout="wide")
st.title("Fraud & financial crime policy assistant")
st.warning(NOTICE)

cfg = load_config()
if not index_available(cfg):
    st.error(
        "The index is not built. Run `python -m fincrime_rag.chunking`, "
        "`python -m fincrime_rag.vectorstore build` and `python -m fincrime_rag.guardrails calibrate` first."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading the index and embedding model...")
def resources():
    return load_resources()


@st.cache_data(ttl=15, show_spinner=False)
def ollama_reachable():
    return ollama_status(cfg)


reachable, detail = ollama_reachable()
with st.sidebar:
    st.header("Settings")
    options = [OLLAMA_LABEL, EXTRACTIVE_LABEL]
    choice = st.radio("Backend", options, index=0 if reachable else 1, key="backend")
    st.number_input("Context chunks (k)", value=cfg.generation.context_k, disabled=True, key="k_fixed")
    st.text_input("Retrieval mode", value=cfg.retrieval.mode, disabled=True, key="mode_fixed")
    st.caption("k and the retrieval mode are fixed to the evaluated defaults.")
    if reachable:
        st.success(f"Ollama reachable: {detail}")
    else:
        st.error(f"Ollama is not reachable: {detail}. Start Ollama and pull the model, or choose Fast mode.")
    st.caption("Documents added to the corpus later are unevaluated.")

backend = "ollama" if choice == OLLAMA_LABEL else "extractive"
if backend == "extractive":
    st.info(
        "Fast mode returns the best-matching sentences from the retrieved text. It cannot say that the context is "
        "insufficient and its answers are much weaker than the language-model backend."
    )

question = st.text_area("Question", height=100, key="question")
if st.button("Ask", type="primary", key="ask") and question.strip():
    if backend == "ollama" and not reachable:
        st.error(
            "Ollama is not reachable, so no answer was generated. Choose Fast mode explicitly to continue."
        )
        st.stop()
    try:
        with st.spinner("Retrieving and generating..."):
            result = ask(resources(), question.strip(), backend)
    except LLMUnavailable as exc:
        st.error(f"The backend became unavailable: {exc}")
        st.stop()
    answer = result.answer
    st.caption(
        f"Elapsed: {result.elapsed_seconds:.1f} s | backend: {answer.backend or 'none (gate)'} | status: {answer.status}"
    )
    if answer.status == "refused":
        reasons = {
            "low_retrieval_score": "the retrieval similarity was below the refusal threshold (no model call)",
            "model_insufficient": "the model reported that the retrieved context is insufficient",
        }
        st.warning(f"No answer: {reasons.get(answer.refusal_reason, answer.refusal_reason)}.")
    elif answer.status == "rejected_injection":
        st.warning("The reply only repeated an embedded instruction, so it was withheld.")
    else:
        if answer.status == "ungrounded":
            st.warning(
                "This answer has no valid citation to the retrieved documents. Treat it as unsupported."
            )
        st.subheader("Answer")
        st.text(answer.text)  # plain text: the model's reply is never rendered as markdown or HTML
    if answer.valid_citations:
        st.subheader("Citations (PDF page numbers)")
        for cite in answer.valid_citations:
            with st.expander(f"{cite.doc_id}, p. {cite.page}"):
                st.text(result.passages[(cite.doc_id, cite.page)])
    if answer.invalid_citations:
        st.caption(
            f"{len(answer.invalid_citations)} citation(s) were removed because they were not in the retrieved context."
        )
    with st.expander(f"Retrieved chunks ({len(result.chunks)})"):
        for rank, chunk in enumerate(result.chunks, start=1):
            pages = ", ".join(str(p) for p in chunk["pdf_pages"])
            st.text(f"{rank}. {chunk['doc_id']}, pages {pages}, chunk {chunk['chunk_id']}")
            st.text(chunk["text"])
