"""
ingestion.py  -  Phase 1 of the RAG pipeline (Section 4.3).
============================================================
Extracts plain text from uploaded documents while preserving page structure so
that citations can later point back to a specific page. Supports PDF, TXT and
Markdown. Each returned unit is a `SourcePage` carrying the document name, page
number and cleaned text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from pypdf import PdfReader


@dataclass
class SourcePage:
    """A single page (or logical section) of a source document."""

    doc_name: str
    page_number: int
    text: str


_WHITESPACE_RE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise whitespace without destroying paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    # de-hyphenate words broken across line ends: "inter-\nnal" -> "internal"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def _read_pdf(path: Path) -> List[SourcePage]:
    reader = PdfReader(str(path))
    pages: List[SourcePage] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append(SourcePage(doc_name=path.name, page_number=i, text=cleaned))
    return pages


def _read_text(path: Path) -> List[SourcePage]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)
    return [SourcePage(doc_name=path.name, page_number=1, text=cleaned)] if cleaned else []


def load_document(path: Union[str, Path]) -> List[SourcePage]:
    """Load one document into a list of SourcePage objects."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return _read_text(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(paths: List[Union[str, Path]]) -> List[SourcePage]:
    """Load several documents, concatenating their pages."""
    pages: List[SourcePage] = []
    for p in paths:
        pages.extend(load_document(p))
    return pages
