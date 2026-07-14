Final Project: What It Is & Full Implementation Plan
The goal (from the brief + README): build a working RAG system — your own documents → chunked → embedded → retrieved by similarity → answered by an LLM with citations → all through a real UI. The repo you have is a deliberately minimal skeleton that already runs end-to-end using TF-IDF + extractive answers. Your job is to upgrade each stage without breaking the interface between them.

Current state, confirmed by reading the code:

ingest.py — loads .txt only, chunks by fixed word count (80 words, 20 overlap)
embed_store.py — TfidfVectorizer + cosine similarity (not real embeddings)
generate.py — extractive_answer works; llm_answer is a stub with commented-out Anthropic call
app.py — Streamlit UI already wired to all three, with a top_k slider and extractive/llm mode radio
data/sample_docs/ — 4 sample .txt files (need 20+ real docs in your domain)
requirements.txt — core deps installed; optional upgrade deps commented out
Step 0 — Get it running today

pip install -r requirements.txt
streamlit run app.py
Confirm the sample docs return sensible answers in extractive mode before changing anything. This is your safety net / regression baseline.

Step 1 — Pick your domain and swap the dataset
Choose a real domain you have 20+ text-heavy documents for (e.g. course notes, product docs, papers, articles).
Drop .txt files into data/sample_docs/ (or a new data/ folder), replacing the samples.
If your source docs are PDFs/HTML/Markdown, extend load_documents() in ingest.py with loaders (pypdf for PDF, BeautifulSoup for HTML) — normalize everything to plain text before chunking.

Step 2 — Defend your chunking strategy
Current chunker in ingest.py is naive fixed word-count. Decide and justify: sentence-aware, paragraph-aware, or token-aware chunking; pick a chunk size/overlap and be ready to explain why (this is graded — "defensible strategy").
Keep Chunk(chunk_id, doc_title, text) shape, or extend with metadata (source, section, date) if useful for citations.

Step 3 — Real embeddings (replace TF-IDF)
In embed_store.py, swap TfidfVectorizer for sentence-transformers (all-MiniLM-L6-v2 is free/local) or an API embedding model.
Keep the VectorStore.build() / .query() method signatures identical so app.py needs zero changes.

Uncomment sentence-transformers in requirements.txt.

Step 4 — Move to a real vector index (if corpus grows large)
Only needed once chunk count is large (thousands+). Swap the in-memory cosine_similarity call for FAISS or Chroma, same interface.
For a normal course-project corpus size, this step is optional — say so explicitly in your write-up if you skip it.

Step 5 — Wire up a real LLM for generation
In generate.py, llm_answer() already has the prompt constructed and the Anthropic call commented out. Uncomment it, pip install anthropic, set ANTHROPIC_API_KEY.

Make sure the answer cites source titles (the prompt already asks for this) — verify citations actually appear and are correct, not hallucinated.

Step 6 — Extend the interface
Pick at least one meaningful addition beyond what's there (README calls for "at least one setting" — sliders/radio already exist, so add something more):

File upload widget for adding documents at runtime
Highlight matched terms in the source excerpts
Response latency display
Conversation history / multi-turn
Chunk size control exposed as a sidebar setting

Step 7 — Evaluation section
Write 8–10 test queries with the expected source document(s) for each.
Run them through your pipeline, record whether the right source was retrieved and whether the answer was grounded/correct.
Short write-up: what worked, what didn't, why (e.g. TF-IDF vs embeddings tradeoffs, chunking edge cases, retrieval failures).

Step 8 — Polish & submit
Re-run the full flow end-to-end one more time after all swaps to confirm nothing broke.
Make sure README/write-up documents: dataset choice, chunking rationale, embedding choice, LLM choice, and evaluation results.
Suggested order of work matches the steps above — get data in first, then upgrade retrieval, then generation, then polish. Each step is independently testable against the working baseline from Step 0, so you always have something demoable.

Want me to start implementing any of these steps now (e.g. wiring up the Anthropic call in Step 5, or upgrading to sentence-transformers in Step 3)?