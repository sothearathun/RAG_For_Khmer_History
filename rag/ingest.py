"""
Ingestion: load raw documents from disk, clean Wikipedia scrape artifacts,
and split them into overlapping, paragraph-aware chunks.

Chunking strategy (defended in the write-up):
- The corpus is scraped Wikipedia articles: donation banners, infobox tables,
  inline citation markers, and trailing References/External links sections
  are noise that a naive word-count chunker would happily slice into chunks.
  `clean_text()` strips that before anything gets chunked.
- Articles are naturally organised into paragraphs under short section
  headers (e.g. "Etymology", "Decline"). `chunk_document()` merges whole
  paragraphs up to a word budget instead of cutting mid-sentence, tags each
  chunk with its section header, and carries a small word-overlap into the
  next chunk so context isn't lost at a boundary.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str
    section: str = ""


_BANNER_MARKER = "From Wikipedia, the free encyclopedia"
_TRAILING_SECTION_HEADERS = {"References", "External links", "Bibliography", "See also", "Notes"}

_CITATION_LINK_RE = re.compile(r"\[\\\[.*?\\\]\]\([^)]*\)")
_BRACKET_NOTE_RE = re.compile(r"\\?\[[^\[\]]{0,60}\\\]")
_IMAGE_LINE_RE = re.compile(r"^\[\]\(.*\)$")
_INLINE_IMAGE_RE = re.compile(r"\[\]\([^)]*\)")
_TABLE_LINE_RE = re.compile(r"^\|")

# Filenames mangled by the scraper (en-dashes became literal "E2-80-93" byte
# sequences, and a few names were truncated). Small, fixed corpus -> fixed here.
_TITLE_OVERRIDES = {
    "cambodian-e2-80-93vietnamese-war": "Cambodian-Vietnamese War",
    "history-of-cambodia-1993-e2-80-93prese": "History of Cambodia (1993-present)",
    "kingdom-of-cambodia-1953-e2-80-931970": "Kingdom of Cambodia (1953-1970)",
    "united-nations-e2-80-93administered-ca": "United Nations Administered Cambodia",
    "1997-cambodian-coup-d-etat": "1997 Cambodian coup d'état",
}


def clean_text(raw: str) -> str:
    """Strip Wikipedia scrape artifacts before chunking."""
    text = raw
    marker = text.find(_BANNER_MARKER)
    if marker != -1:
        text = text[marker + len(_BANNER_MARKER):]

    lines = text.split("\n")
    cutoff = len(lines)
    for i, line in enumerate(lines):
        if line.strip() in _TRAILING_SECTION_HEADERS:
            cutoff = i
            break
    lines = lines[:cutoff]

    kept = [
        line for line in lines
        if not _TABLE_LINE_RE.match(line.strip()) and not _IMAGE_LINE_RE.match(line.strip())
    ]
    text = "\n".join(kept)

    text = _CITATION_LINK_RE.sub("", text)
    text = _INLINE_IMAGE_RE.sub("", text)
    text = _BRACKET_NOTE_RE.sub("", text)
    text = text.replace("<br>", " ").replace("\\", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(folder: str) -> List[dict]:
    """Load every .txt file in `folder`, clean it, and return {"title", "text"} dicts."""
    docs = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        stem = os.path.splitext(filename)[0]
        if stem.lower() in _TITLE_OVERRIDES:
            title = _TITLE_OVERRIDES[stem.lower()]
        else:
            title = re.sub(r"-([sS])-", r"'\1 ", stem)  # "People-s-Republic" -> "People's Republic"
            title = title.replace("_", " ").replace("-", " ")
            title = re.sub(r"\s+", " ", title).strip().title().replace("'S", "'s")

        docs.append({"title": title, "text": clean_text(raw)})
    return docs


def _is_section_header(paragraph: str) -> bool:
    """Heuristic: Wikipedia section headers are short, standalone lines with no
    terminal sentence punctuation (e.g. "Etymology", "Golden age of Khmer civilisation")."""
    if "\n" in paragraph:
        return False
    words = paragraph.split()
    if not (1 <= len(words) <= 8):
        return False
    return paragraph[-1] not in ".?!\"'”"


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _split_sentences(paragraph: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", paragraph) if s]


def chunk_document(text: str, target_words: int = 150, overlap_words: int = 30) -> List[Tuple[str, str]]:
    """Paragraph-aware chunker: merge whole paragraphs up to ~target_words,
    tagging each chunk with the section header it falls under. Paragraphs much
    larger than the target are split by sentence instead of by raw word count,
    so no chunk ever cuts a sentence in half. Returns (section, chunk_text) pairs."""
    paragraphs = _split_paragraphs(text)

    tagged: List[Tuple[str, str]] = []
    current_section = ""
    for p in paragraphs:
        if _is_section_header(p):
            current_section = p
            continue
        tagged.append((current_section, p))

    units: List[Tuple[str, str]] = []
    for section, paragraph in tagged:
        if len(paragraph.split()) > target_words * 1.5:
            units.extend((section, sentence) for sentence in _split_sentences(paragraph))
        else:
            units.append((section, paragraph))

    chunks: List[Tuple[str, str]] = []
    buf_words: List[str] = []
    buf_section = units[0][0] if units else ""

    def flush():
        if buf_words:
            chunks.append((buf_section, " ".join(buf_words)))

    for section, unit_text in units:
        unit_words = unit_text.split()
        if buf_words and len(buf_words) + len(unit_words) > target_words:
            flush()
            overlap = buf_words[-overlap_words:] if overlap_words else []
            buf_words = overlap + unit_words
            buf_section = section
        else:
            if not buf_words:
                buf_section = section
            buf_words.extend(unit_words)

    flush()
    return chunks


def build_chunk_records(docs: List[dict], target_words: int = 150, overlap_words: int = 30) -> List[Chunk]:
    """Turn loaded documents into a flat list of Chunk records ready for embedding."""
    records = []
    for doc in docs:
        pieces = chunk_document(doc["text"], target_words=target_words, overlap_words=overlap_words)
        for i, (section, piece_text) in enumerate(pieces):
            records.append(Chunk(
                chunk_id=f"{doc['title']}::{i}",
                doc_title=doc["title"],
                text=piece_text,
                section=section,
            ))
    return records
