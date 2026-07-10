"""
reranker.py  -  Cross-encoder reranking (Section 5.3).
======================================================
A cross-encoder scores the (query, passage) pair jointly, which is more precise
than the bi-encoder used for first-stage retrieval but too slow to run over the
whole corpus. So it is applied only to the fused top-k candidates - the classic
two-stage "retrieve then rerank" pattern the paper recommends.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from .config import RETRIEVAL


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or RETRIEVAL.reranker_model
        self.model = _load_cross_encoder(self.model_name)

    def rerank(
        self, query: str, candidates: List[Tuple[int, str]]
    ) -> List[Tuple[int, float]]:
        """
        candidates: list of (chunk_index, chunk_text).
        Returns (chunk_index, relevance_score) sorted best-first.
        """
        if not candidates:
            return []
        pairs = [[query, text] for _idx, text in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip((idx for idx, _ in candidates), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(int(idx), float(score)) for idx, score in ranked]
