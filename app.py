"""
Streamlit web UI for the 3GPP RAG Chatbot.

    streamlit run app.py

A thin presentation layer over src/retrieve.py + src/generate.py. The heavy
Retriever (embedding model + BM25 index) is built once and cached across reruns
with @st.cache_resource, so only the first page load pays the load cost.

Every answer is shown with its mode badge (claude / extractive / abstained),
retrieval confidence, and an expandable list of the exact clause excerpts it
was grounded in -- the citations are the point, so they're always one click away.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from generate import active_backend, answer  # noqa: E402
from retrieve import Retriever  # noqa: E402

st.set_page_config(page_title="3GPP RAG Chatbot", page_icon="📡", layout="centered")


@st.cache_resource(show_spinner="Loading retriever (embedding model + BM25 index)...")
def get_retriever() -> Retriever:
    return Retriever()


st.title("📡 3GPP RAG Chatbot")
st.caption(
    "Ask about Release 17 5G Standalone specifications. Every answer is grounded "
    "in retrieved clauses and cited; the bot abstains when the specs don't cover "
    "your question."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Corpus:** 12 curated 3GPP Rel-17 5G core specs (~16.8k clause chunks)\n"
        "- **Retrieval:** hybrid dense (all-MiniLM) + BM25, fused with RRF\n"
        "- **Grounding:** answers only from retrieved clauses, with citations\n"
        "- **Abstains** when retrieval confidence is low"
    )
    st.markdown(f"**Answer backend:** {active_backend()}")
    k = st.slider("Clauses to retrieve", min_value=3, max_value=12, value=6)

retriever = get_retriever()

if "history" not in st.session_state:
    st.session_state.history = []  # list of Answer objects

for past in st.session_state.history:
    with st.chat_message("user"):
        st.write(past.query)
    with st.chat_message("assistant"):
        badge = {"gemini": "🔵 Gemini", "claude": "🟢 Claude", "extractive": "🟡 Extractive", "abstained": "⚪ Abstained"}
        st.caption(f"{badge.get(past.mode, past.mode)}  ·  confidence {past.confidence:.2f}")
        st.markdown(past.text)
        with st.expander(f"Sources ({len(past.hits)} clauses)"):
            for i, hit in enumerate(past.hits, start=1):
                st.markdown(f"**[{i}] {hit.citation}**")
                st.text(hit.text[:1200] + ("..." if len(hit.text) > 1200 else ""))

if query := st.chat_input("Ask about the 5G specs, e.g. 'What is the AMF?'"):
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving clauses and composing a grounded answer..."):
            result = answer(query, retriever, k=k)
        badge = {"gemini": "🔵 Gemini", "claude": "🟢 Claude", "extractive": "🟡 Extractive", "abstained": "⚪ Abstained"}
        st.caption(f"{badge.get(result.mode, result.mode)}  ·  confidence {result.confidence:.2f}")
        st.markdown(result.text)
        with st.expander(f"Sources ({len(result.hits)} clauses)"):
            for i, hit in enumerate(result.hits, start=1):
                st.markdown(f"**[{i}] {hit.citation}**")
                st.text(hit.text[:1200] + ("..." if len(hit.text) > 1200 else ""))
    st.session_state.history.append(result)
