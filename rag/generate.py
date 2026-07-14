"""
Generation: turn retrieved chunks + a query into a final answer.

Two modes are provided:
- "extractive" (default): no API key needed, works immediately. Just stitches
  together the retrieved chunks so you can verify retrieval quality before wiring
  up an LLM.
- "llm": calls Gemini (Google AI) to write a grounded answer from the
  retrieved context, citing source titles. Requires the `google-genai` package
  and a GEMINI_API_KEY environment variable (loaded from .env via python-dotenv).
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv
from google import genai

from .ingest import Chunk

load_dotenv()

_MODEL = "gemini-2.5-flash"


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "No relevant passages were found for that query."
    lines = [f"Top passages related to: \u201c{query}\u201d\n"]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.doc_title}, score={score:.2f}] {chunk.text}\n")
    return "\n".join(lines)


def llm_answer(
    query: str,
    retrieved: List[Tuple[Chunk, float]],
    history: List[Tuple[str, str]] = None,
) -> str:
    """Ask Gemini to answer the query, grounded only in the retrieved chunks.

    `history` is prior (question, answer) turns from this session, sent as real
    conversation turns so follow-ups ("what about after that?") resolve correctly.
    Retrieval itself is still keyed off the current query text only.
    """
    if not retrieved:
        return "No relevant passages were found for that query."

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            "[LLM mode not configured] Set GEMINI_API_KEY to enable grounded LLM "
            "answers. Falling back to extractive mode:\n\n" + extractive_answer(query, retrieved)
        )

    context = "\n\n".join(f"Source: {c.doc_title}\n{c.text}" for c, _ in retrieved)
    prompt = (
        "Answer the question using ONLY the sources below. Cite the source title(s) "
        f"you used.\n\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    contents = []
    for prior_query, prior_answer in (history or []):
        contents.append({"role": "user", "parts": [{"text": prior_query}]})
        contents.append({"role": "model", "parts": [{"text": prior_answer}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=_MODEL, contents=contents)
    return response.text


def generate_answer(
    query: str,
    retrieved: List[Tuple[Chunk, float]],
    mode: str = "extractive",
    history: List[Tuple[str, str]] = None,
) -> str:
    if mode == "llm":
        return llm_answer(query, retrieved, history=history)
    return extractive_answer(query, retrieved)
