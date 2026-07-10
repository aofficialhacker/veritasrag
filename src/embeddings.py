"""
embeddings.py  -  Dense representation (Section 4.4, "Dense (DPR/BGE)").
=======================================================================
Thin wrapper around a sentence-transformers bi-encoder. The model is loaded
once and cached. Vectors are L2-normalised so that inner-product search is
equivalent to cosine similarity.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from .config import RETRIEVAL


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    # Imported lazily so that `import config` stays cheap and the heavy
    # torch import only happens when embeddings are actually needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class Embedder:
    """Encodes text into normalised dense vectors."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or RETRIEVAL.embedding_model
        self.model = _load_model(self.model_name)
        # Method was renamed across sentence-transformers versions; support both.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
