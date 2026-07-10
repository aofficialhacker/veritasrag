"""
retriever.py  -  Phase 3, the heart of the pipeline (Section 4.4).
==================================================================
Builds an index from chunks and exposes four retrieval strategies so that the
Retrieval-Comparison tab can show them side by side, reproducing Table 1 /
Figure 4 of the research paper:

    * sparse   - BM25 only
    * dense    - FAISS HNSW only
    * hybrid   - BM25 + dense fused with Reciprocal Rank Fusion
    * rerank   - hybrid candidates re-scored by a cross-encoder  (the winner)

`retrieve()` is the production path used by the chat: hybrid + rerank.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .chunking import Chunk
from .config import RETRIEVAL
from .embeddings import Embedder
from .fusion import reciprocal_rank_fusion
from .reranker import Reranker
from .sparse import BM25Index
from .vector_store import HNSWVectorStore


@dataclass
class RetrievedPassage:
    """A chunk returned by retrieval, annotated with its score and rank."""

    rank: int
    score: float
    chunk: Chunk

    @property
    def citation(self) -> str:
        return self.chunk.short_source()


@dataclass
class RetrievalResult:
    """Bundles the passages returned by a strategy plus timing telemetry."""

    strategy: str
    passages: List[RetrievedPassage]
    latency_ms: float


class HybridRetriever:
    """Owns the corpus, both indexes and the reranker."""

    def __init__(self, lazy_reranker: bool = True):
        self.chunks: List[Chunk] = []
        self.embedder: Embedder | None = None
        self.vector_store: HNSWVectorStore | None = None
        self.bm25: BM25Index | None = None
        self._reranker: Reranker | None = None
        self._lazy_reranker = lazy_reranker

    # -- build -------------------------------------------------------------
    def build(self, chunks: List[Chunk]) -> Dict[str, float]:
        """Index a list of chunks. Returns simple build telemetry."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        t0 = time.perf_counter()
        self.embedder = Embedder()
        vectors = self.embedder.encode(texts)
        self.vector_store = HNSWVectorStore(dim=self.embedder.dim)
        self.vector_store.add(vectors)
        t_dense = time.perf_counter()

        self.bm25 = BM25Index(texts)
        t_sparse = time.perf_counter()

        if not self._lazy_reranker:
            _ = self.reranker  # force load

        return {
            "num_chunks": len(chunks),
            "embed_seconds": round(t_dense - t0, 2),
            "bm25_seconds": round(t_sparse - t_dense, 2),
        }

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks) and self.vector_store is not None

    # -- individual strategies --------------------------------------------
    def _dense(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        qv = self.embedder.encode_one(query)
        return self.vector_store.search(qv, top_k)

    def _sparse(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        return self.bm25.search(query, top_k)

    def _to_passages(self, ranked: List[Tuple[int, float]]) -> List[RetrievedPassage]:
        return [
            RetrievedPassage(rank=i + 1, score=score, chunk=self.chunks[idx])
            for i, (idx, score) in enumerate(ranked)
        ]

    # -- public: run one named strategy (used by comparison tab) ----------
    def run_strategy(self, strategy: str, query: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or RETRIEVAL.top_k_final
        t0 = time.perf_counter()

        if strategy == "sparse":
            ranked = self._sparse(query, top_k)
        elif strategy == "dense":
            ranked = self._dense(query, top_k)
        elif strategy == "hybrid":
            dense = self._dense(query, RETRIEVAL.top_k_dense)
            sparse = self._sparse(query, RETRIEVAL.top_k_sparse)
            ranked = reciprocal_rank_fusion([dense, sparse])[:top_k]
        elif strategy == "rerank":
            dense = self._dense(query, RETRIEVAL.top_k_dense)
            sparse = self._sparse(query, RETRIEVAL.top_k_sparse)
            fused = reciprocal_rank_fusion([dense, sparse])
            candidate_idxs = [idx for idx, _ in fused[: RETRIEVAL.top_k_dense + RETRIEVAL.top_k_sparse]]
            candidates = [(idx, self.chunks[idx].text) for idx in candidate_idxs]
            ranked = self.reranker.rerank(query, candidates)[:top_k]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            strategy=strategy,
            passages=self._to_passages(ranked),
            latency_ms=round(latency_ms, 1),
        )

    # -- public: production retrieval (hybrid + rerank) -------------------
    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        return self.run_strategy("rerank", query, top_k)
