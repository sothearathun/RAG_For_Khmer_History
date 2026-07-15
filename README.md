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
like *"Who led the Khmer Rouge?"* against the indexed Cambodian history articles.

## What's already working

- **Ingestion** (`rag/ingest.py`) — loads `.txt` files, strips Wikipedia scrape
  artifacts, and splits them into paragraph-aware, section-tagged, overlapping
  chunks (see the module docstring for the chunking rationale).
- **Retrieval** (`rag/embed_store.py`) — embeds chunks with `sentence-transformers`
  (`all-MiniLM-L6-v2`) and ranks them by cosine similarity (dot product over
  L2-normalized vectors), searched in-memory with numpy.
- **Generation** (`rag/generate.py`) — an `extractive` mode that needs no API key,
  plus an `llm` mode that calls DeepSeek (`deepseek-v4-flash`) for grounded, cited
  answers, including multi-turn conversation context.
- **Interface** (`app.py`) — a Streamlit chat UI: conversation history rendered as
  chat bubbles, expandable scored source chunks, a retrieval/generation latency
  caption per turn, and a Clear conversation button. Sidebar controls `top_k` and
  answer mode.

## Design decisions

- **Dataset: 21 Wikipedia articles on Cambodian history.** Chosen because it's a
  single coherent domain with a natural chronological structure (Funan → Chenla →
  Khmer Empire → French protectorate → Khmer Rouge era → UNTAC → present) —
  enough real inter-document relationships (cross-references, shared figures
  like Norodom Sihanouk, sequential causality) to make retrieval quality and
  citation correctness meaningfully testable, unlike a bag of unrelated
  articles. The two most recent additions (`Norodom-Sihanouk.txt`,
  `1997-Cambodian-coup-d-etat.txt`) were picked to fill concrete gaps: the
  corpus already cross-referenced the 1997 coup from two other articles without
  containing it, and Sihanouk is the one figure who threads through nearly every
  era but had no dedicated source.
- **Chunking: paragraph-aware with sentence-level fallback, not fixed word-count.**
  `rag/ingest.py` cleans Wikipedia scrape noise (banners, tables, citation
  markers, trailing References/See also sections) before chunking, then merges
  whole paragraphs up to a ~150-word budget, tags each chunk with its section
  header, and carries a 30-word overlap into the next chunk. Paragraphs that
  are much larger than the budget are split by sentence instead of by raw word
  count, so no chunk ever cuts a sentence in half. Full rationale in the
  `rag/ingest.py` module docstring.
- **Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`), not TF-IDF.**
  Semantic vectors instead of bag-of-words — needed because several evaluation
  queries (e.g. "What was the Khmer Empire known for?") share almost no literal
  vocabulary with their answer passages ("hydraulic cities," "Angkor"), which a
  keyword-overlap method like TF-IDF would miss. Free, runs locally, no API key.
  See [EVALUATION.md](EVALUATION.md) for the retrieval-accuracy payoff.
- **Vector index: in-memory numpy, not FAISS/Chroma.** The corpus produces
  ~1,100 chunks → roughly a `(1100, 384)` float32 matrix (~1.7 MB). A full
  linear-scan query — embed the question + dot product against every chunk —
  measures ~50ms end-to-end, dominated by encoding the query text, not the
  search itself. FAISS/Chroma exist to avoid an O(n) scan at tens of
  thousands+ chunks; at this scale they'd add a dependency and an on-disk
  index with no measurable benefit. `VectorStore.build`/`.query` keep the same
  interface either way, so swapping in FAISS later is a localized change to
  `rag/embed_store.py` only, not a rewrite.
- **LLM: DeepSeek (`deepseek-v4-flash`), called via the OpenAI-compatible API.**
  Free-tier friendly for a course project (originally built against Gemini —
  see [EVALUATION.md](EVALUATION.md) for that history and why it changed).
  `rag/generate.py`'s `llm_answer()` sends prior conversation turns as real
  chat history (not just string-concatenated into the prompt), so multi-turn
  follow-ups resolve correctly.
- **Evaluation results: 8/10 fully correct and grounded** on a 10-query test set
  (9 in-domain + 1 adversarial out-of-domain). The one hallucinated citation
  found — an out-of-domain query answered from the model's general knowledge
  but cited to an unrelated source — was root-caused and **fixed** with a
  similarity-score floor (`rag/generate.py`) that refuses before ever calling
  the LLM, verified live afterward. Two corpus/chunking edge cases were also
  found and documented (not fixed — cosmetic, don't affect answer quality).
  Full table, scoring, and analysis in [EVALUATION.md](EVALUATION.md).

## Project structure

```
searchengine_finalproject/
├── app.py                  # Streamlit interface
├── requirements.txt
├── EVALUATION.md            # test queries + retrieval/groundedness results
├── data/                    # 21 Cambodian history .txt sources
└── rag/
    ├── ingest.py            # load + chunk documents
    ├── embed_store.py       # vectorize + similarity search
    └── generate.py          # turn retrieved chunks into an answer
```

## Your upgrade path (this is most of the final project)

1. ~~**Swap in your own dataset.**~~ Done — `data/` holds 21 Wikipedia articles on
   Cambodian history (replacing the `data/sample_docs/` samples).
2. ~~**Upgrade retrieval from TF-IDF to real embeddings.**~~ Done — `rag/embed_store.py`
   uses `sentence-transformers` (`all-MiniLM-L6-v2`). The `VectorStore.build` /
   `.query` interface is unchanged, so `app.py` needed no edits.
3. ~~**Move to a real vector database once your corpus is large.**~~ Evaluated and
   skipped — see [Design decisions](#design-decisions): ~1,100 chunks searched
   in-memory in ~50ms, well under the scale where FAISS/Chroma would pay off.
4. ~~**Wire up an LLM in `rag/generate.py`'s `llm_answer`**~~ Done — DeepSeek
   (`deepseek-v4-flash`) generates grounded, cited answers; see
   [Design decisions](#design-decisions).
5. ~~**Extend the interface**~~ Done — response latency display (retrieval +
   generation time per turn) and conversation history (multi-turn chat, resolves
   follow-up questions using prior turns) — see `app.py`.
6. ~~**Add an evaluation section**~~ Done — see [EVALUATION.md](EVALUATION.md):
   10 test queries with expected sources, retrieval/groundedness scoring, and a
   write-up of what worked, what didn't, and why (including a real hallucinated-citation
   failure found on an out-of-domain adversarial query).

## Why start from this

The retrieval mechanics here (chunk \u2192 vectorize \u2192 cosine similarity \u2192 rank) are
exactly the ones you practiced in the Week 14 content-based filtering lab, just
applied to document chunks instead of movies. Getting this skeleton running today
means every future class session is an upgrade to a working system, not a fresh
start.
