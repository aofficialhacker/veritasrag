"""
chunking.py  -  Part of Phase 2 (Indexing, Section 4.3).
========================================================
Splits SourcePage text into overlapping word-windows. Overlap preserves context
that would otherwise be severed at a chunk boundary. Each Chunk keeps a back-
reference to its document and page so the UI can render a citation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import RETRIEVAL
from .ingestion import SourcePage


@dataclass
class Chunk:
    """A retrievable unit of text with provenance."""

    chunk_id: int
    doc_name: str
    page_number: int
    text: str
    # Filled in later by the vector store / retriever
    metadata: dict = field(default_factory=dict)

    def short_source(self) -> str:
        return f"{self.doc_name} (p.{self.page_number})"


def _split_words(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]

    step = max(1, size - overlap)
    windows: List[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if window:
            windows.append(" ".join(window))
        if start + size >= len(words):
            break
    return windows


def chunk_pages(pages: List[SourcePage]) -> List[Chunk]:
    """Turn a list of pages into a flat list of overlapping chunks."""
    size = RETRIEVAL.chunk_size_words
    overlap = RETRIEVAL.chunk_overlap_words

    chunks: List[Chunk] = []
    cid = 0
    for page in pages:
        for window in _split_words(page.text, size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    text=window,
                )
            )
            cid += 1
    return chunks
