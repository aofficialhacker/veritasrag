"""
vector_store.py  -  FAISS HNSW index (Section 4.5, Table 2).
============================================================
Wraps a FAISS HNSW graph index. HNSW is chosen as the default because its query
latency stays near-constant as the corpus grows, which is exactly the property
highlighted in the accompanying research paper. Inner-product metric is used on
normalised vectors (== cosine similarity).
"""
from __future__ import annotations

from typing import List, Tuple

import faiss
import numpy as np

from .config import RETRIEVAL


class HNSWVectorStore:
    """A FAISS HNSW index over document-chunk embeddings."""

    def __init__(self, dim: int):
        self.dim = dim
        # HNSW with inner-product metric.
        self.index = faiss.IndexHNSWFlat(dim, RETRIEVAL.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = RETRIEVAL.hnsw_ef_construction
        self.index.hnsw.efSearch = RETRIEVAL.hnsw_ef_search
        self._size = 0

    def add(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        self._size += vectors.shape[0]

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Return (chunk_index, similarity) pairs, best first."""
        if self._size == 0:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._size)
        scores, ids = self.index.search(q, k)
        results: List[Tuple[int, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx != -1:
                results.append((int(idx), float(score)))
        return results

    def __len__(self) -> int:
        return self._size
