"""
Generation: turn retrieved chunks + a query into a final answer.

Two modes are provided:
- "extractive" (default): no API key needed, works immediately. Just stitches
  together the retrieved chunks so you can verify retrieval quality before wiring
  up an LLM.
- "llm": calls DeepSeek (OpenAI-compatible API) to write a grounded answer from
  the retrieved context, citing source titles. Requires the `openai` package
  and a DEEPSEEK_API_KEY environment variable (loaded from .env via python-dotenv).
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from .ingest import Chunk

load_dotenv()

_MODEL = "deepseek-v4-flash"
_BASE_URL = "https://api.deepseek.com"

# Similarity floor below which retrieved chunks are treated as "no real match"
# rather than passed to the LLM. Chosen from EVALUATION.md: on-topic queries
# scored 0.59-0.76 top-hit similarity; an out-of-domain query ("What is the
# capital of France?") topped out at 0.35 and, without this gate, still got
# answered from the model's own knowledge with a fabricated citation to an
# unrelated source. 0.45 sits in the gap between those two clusters.
_MIN_RELEVANCE_SCORE = 0.45
_NOT_COVERED_MESSAGE = (
    "I don't have information about this in the indexed documents. The "
    "closest match (“{title}”) only scored {score:.2f} similarity, "
    "below the relevance threshold, so I'm not going to guess."
)


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
    """Ask DeepSeek to answer the query, grounded only in the retrieved chunks.

    `history` is prior (question, answer) turns from this session, sent as real
    conversation turns so follow-ups ("what about after that?") resolve correctly.
    Retrieval itself is still keyed off the current query text only.
    """
    if not retrieved:
        return "No relevant passages were found for that query."

    top_chunk, top_score = retrieved[0]
    if top_score < _MIN_RELEVANCE_SCORE:
        return _NOT_COVERED_MESSAGE.format(title=top_chunk.doc_title, score=top_score)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return (
            "[LLM mode not configured] Set DEEPSEEK_API_KEY to enable grounded LLM "
            "answers. Falling back to extractive mode:\n\n" + extractive_answer(query, retrieved)
        )

    context = "\n\n".join(f"Source: {c.doc_title}\n{c.text}" for c, _ in retrieved)
    prompt = (
        "Answer clearly and directly, like a knowledgeable person explaining "
        "something to a colleague — warm but not effusive, no forced enthusiasm "
        "or exclamation points, no flattery or preamble like \"Great question!\". "
        "Get to the point, structure the answer so it's easy to follow, and be "
        "honest about what is and isn't well-supported by the sources rather "
        "than padding with filler.\n\n"
        "Answer the question using ONLY the sources below. Do not use outside "
        "knowledge. If the sources do not actually contain the information needed "
        "to answer, respond with exactly: \"I don't have information about this in "
        "the provided sources.\" Never cite a source unless it genuinely supports "
        "the specific claim you're attaching it to. Cite the source title(s) you "
        f"used.\n\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    messages = []
    for prior_query, prior_answer in (history or []):
        messages.append({"role": "user", "content": prior_query})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": prompt})

    client = OpenAI(api_key=api_key, base_url=_BASE_URL)
    response = client.chat.completions.create(model=_MODEL, messages=messages)
    return response.choices[0].message.content


def generate_answer(
    query: str,
    retrieved: List[Tuple[Chunk, float]],
    mode: str = "extractive",
    history: List[Tuple[str, str]] = None,
) -> str:
    if mode == "llm":
        return llm_answer(query, retrieved, history=history)
    return extractive_answer(query, retrieved)
