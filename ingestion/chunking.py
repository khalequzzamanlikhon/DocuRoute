"""
Recursive, overlap-aware chunking. Deliberately simple and dependency-free
(no need for LangChain just for this) so the chunk boundaries are easy to
reason about and debug when retrieval quality looks off.
"""
from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, field

from ingestion.loaders import PageText

# Try to split on paragraph boundaries first, then sentences, then hard cut.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_number: int
    text: str
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size or not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    chunks, current = [], ""
    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_split_text(part, chunk_size, rest))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = overlapped[i - 1][-overlap:]
        overlapped.append(tail + chunks[i])
    return overlapped


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """
    Splits recursively (paragraph -> line -> sentence -> word boundary) and
    then re-stitches a small overlap between adjacent chunks so a fact that
    straddles a chunk boundary is still retrievable from either side.
    """
    all_chunks: list[Chunk] = []
    for page in pages:
        raw_chunks = _split_text(page.text, chunk_size, _SEPARATORS)
        raw_chunks = _add_overlap(raw_chunks, overlap)
        for raw in raw_chunks:
            cleaned = re.sub(r"\s+", " ", raw).strip()
            if not cleaned:
                continue
            all_chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=page.doc_id,
                    page_number=page.page_number,
                    text=cleaned,
                    metadata={"doc_id": page.doc_id, "page_number": page.page_number},
                )
            )
    return all_chunks
