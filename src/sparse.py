"""
sparse.py  -  BM25 lexical retrieval (Section 4.4, "Sparse (BM25)").
====================================================================
Bag-of-words retrieval that excels on exact-token / identifier-heavy queries.
Kept as its own component so the retriever can fuse it with dense results
(hybrid retrieval) and so the Retrieval-Comparison tab can show each signal in
isolation.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """A BM25 index over the same chunk list used by the dense store."""

    def __init__(self, chunk_texts: List[str]):
        self.corpus_tokens = [tokenize(t) for t in chunk_texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Return (chunk_index, bm25_score) pairs, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(s)) for idx, s in ranked[:top_k] if s > 0.0]
