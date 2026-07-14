"""
Vector store: turn chunks into vectors and support similarity search over them.

Backend: sentence-transformers (all-MiniLM-L6-v2), a small (~80MB), free, local
embedding model. It runs on CPU with no API key, and captures semantic similarity
(synonyms, paraphrases) that TF-IDF's bag-of-words overlap misses — important here
since queries about this corpus ("who ruled Cambodia in the 1980s") rarely reuse
the exact wording of the source articles.

Embeddings are L2-normalized at encode time, so cosine similarity reduces to a
plain dot product between the query vector and the chunk matrix.

Upgrade path (only needed once the corpus grows past a few thousand chunks):
- Swap the in-memory dot-product search below for FAISS or Chroma.
- Keep the VectorStore interface (`build`, `query`) the same so app.py doesn't change.
"""

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(_MODEL_NAME)
        self.matrix = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunk text and store the resulting (normalized) matrix."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.matrix = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string."""
        if self.matrix is None:
            raise RuntimeError("VectorStore.build() must be called before query().")
        query_vec = self.model.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores = self.matrix @ query_vec
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]
