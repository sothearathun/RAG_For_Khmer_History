"""
RAG-Based AI Search System — starter interface.

Run with:
    streamlit run app.py

This gives you a working, end-to-end demo today: document loading, TF-IDF based
retrieval, and an extractive answer — all wired into a real web interface. Build
your final project by upgrading each piece (see the TODOs in rag/embed_store.py
and rag/generate.py) without needing to touch this file's overall structure.
"""

import time

import streamlit as st

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer

DATA_FOLDER = "data"
OVERVIEW_PHRASES = (
    "all khmer history",
    "all cambodian history",
    "history of cambodia",
    "khmer history overview",
    "cambodian history overview",
)
OVERVIEW_TOP_K = 8
COMPREHENSIVE_PHRASES = (
    "all ",
    "important events",
    "major events",
    "key events",
    "timeline",
    "chronology",
    "overview",
    "summarize",
    "summary",
    "tell me about",
    "describe",
    "explain",
    "key developments",
)
COMPREHENSIVE_TOP_K = 8

st.set_page_config(page_title="RAG Search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_store():
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


store, docs, chunks = load_store()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    mode = st.radio("Answer mode", ["extractive", "llm"], index=0,
                     help="Extractive works with no setup. LLM mode needs DEEPSEEK_API_KEY set.")
    st.divider()
    st.caption(f"Indexed **{len(docs)}** documents → **{len(chunks)}** chunks")
    with st.expander("Documents in this index"):
        for d in docs:
            st.write(f"- {d['title']}")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

st.title("🔎 RAG-Based AI Search System - Khmer History")
st.caption("Ask a question about the indexed documents (Khmer History) below. In LLM mode, follow-up "
           "questions can refer back to earlier turns in this conversation.")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        st.caption(
            f"⏱️ retrieval {turn['retrieval_ms']:.0f} ms · "
            f"generation {turn['generation_ms']:.0f} ms · "
            f"total {turn['retrieval_ms'] + turn['generation_ms']:.0f} ms"
        )
        with st.expander("Sources"):
            for chunk, score in turn["retrieved"]:
                st.markdown(f"**{chunk.doc_title}** · similarity {score:.2f}")
                st.write(chunk.text)

query = st.chat_input("Ask a question, e.g. Who led the Khmer Rouge?")

if query and query.strip():
    history_pairs = [(t["query"], t["answer"]) for t in st.session_state.history]

    t0 = time.perf_counter()
    normalized_query = query.lower()
    is_overview = any(phrase in normalized_query for phrase in OVERVIEW_PHRASES)
    is_comprehensive = any(
        phrase in normalized_query for phrase in COMPREHENSIVE_PHRASES
    )
    if is_overview:
        retrieved = store.query_diverse(query, top_k=OVERVIEW_TOP_K)
    elif is_comprehensive:
        retrieved = store.query_mmr(query, top_k=COMPREHENSIVE_TOP_K)
    else:
        retrieved = store.query(query, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    answer = generate_answer(
        query,
        retrieved,
        mode=mode,
        history=history_pairs,
        overview=is_overview,
        comprehensive=is_comprehensive,
    )
    generation_ms = (time.perf_counter() - t1) * 1000

    st.session_state.history.append({
        "query": query,
        "answer": answer,
        "retrieved": retrieved,
        "mode": mode,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
    })
    st.rerun()
