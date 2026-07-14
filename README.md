# RAG-Based AI Search System — Starter Project

This is a runnable starting point for your **final project**. It is intentionally
minimal: every piece works today with zero API keys, and every piece has a clearly
marked upgrade path so you can grow it into your real final submission.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and ask a question
like *"How does content-based filtering rank items?"* against the sample documents.

## What's already working

- **Ingestion** (`rag/ingest.py`) — loads `.txt` files, strips Wikipedia scrape
  artifacts, and splits them into paragraph-aware, section-tagged, overlapping
  chunks (see the module docstring for the chunking rationale).
- **Retrieval** (`rag/embed_store.py`) — embeds chunks with `sentence-transformers`
  (`all-MiniLM-L6-v2`) and ranks them by cosine similarity (dot product over
  L2-normalized vectors), searched in-memory with numpy.
- **Generation** (`rag/generate.py`) — an `extractive` mode that needs no API key,
  plus an `llm` mode stub ready for you to wire up a real model.
- **Interface** (`app.py`) — a Streamlit search UI: query box, answer panel, and an
  expandable, scored list of source chunks. Sidebar controls `top_k` and answer mode.

## Design decisions

- **Vector index: in-memory numpy, not FAISS/Chroma.** The corpus (19 Cambodian
  history articles) produces 1,003 chunks → a `(1003, 384)` float32 matrix
  (~1.5 MB). A full linear-scan query — embed the question + dot product against
  every chunk — measures ~50ms end-to-end, dominated by encoding the query text,
  not the search itself. FAISS/Chroma exist to avoid an O(n) scan at tens of
  thousands+ chunks; at this scale they'd add a dependency and an on-disk index
  with no measurable benefit. `VectorStore.build`/`.query` keep the same interface
  either way, so swapping in FAISS later is a localized change to
  `rag/embed_store.py` only, not a rewrite.

## Project structure

```
final_project_starter/
├── app.py                  # Streamlit interface
├── requirements.txt
├── data/sample_docs/        # replace with your own domain's documents
└── rag/
    ├── ingest.py            # load + chunk documents
    ├── embed_store.py       # vectorize + similarity search
    └── generate.py          # turn retrieved chunks into an answer
```

## Your upgrade path (this is most of the final project)

1. ~~**Swap in your own dataset.**~~ Done — `data/` holds 19 Wikipedia articles on
   Cambodian history (replacing the `data/sample_docs/` samples).
2. ~~**Upgrade retrieval from TF-IDF to real embeddings.**~~ Done — `rag/embed_store.py`
   uses `sentence-transformers` (`all-MiniLM-L6-v2`). The `VectorStore.build` /
   `.query` interface is unchanged, so `app.py` needed no edits.
3. ~~**Move to a real vector database once your corpus is large.**~~ Evaluated and
   skipped — see [Design decisions](#design-decisions): 1,003 chunks searched
   in-memory in ~50ms, well under the scale where FAISS/Chroma would pay off.
4. **Wire up an LLM in `rag/generate.py`'s `llm_answer`** so answers are generated
   and grounded in the retrieved context, with citations back to source documents.
5. **Extend the interface**: file upload for new documents, highlighting matched
   terms, a settings panel for chunk size, response latency display, conversation
   history, etc.
6. **Add an evaluation section**: a small set of test queries with expected sources,
   and a short write-up of what worked, what didn't, and why.

## Why start from this

The retrieval mechanics here (chunk \u2192 vectorize \u2192 cosine similarity \u2192 rank) are
exactly the ones you practiced in the Week 14 content-based filtering lab, just
applied to document chunks instead of movies. Getting this skeleton running today
means every future class session is an upgrade to a working system, not a fresh
start.
