"""
config.py
=========
Central configuration for VeritasRAG. All tunable parameters live here so the
rest of the codebase never hard-codes a magic number. Values are sourced from
the .env file where present, otherwise sensible defaults are used.

The chunking defaults (350 tokens, ~15% overlap) follow the sweet spot reported
in the RAG literature survey that this project accompanies (Section 5.3).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
INDEX_CACHE_DIR = DATA_DIR / "index_cache"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"

# Load environment variables from .env (safe if the file is absent)
load_dotenv(ROOT_DIR / ".env")


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class RetrievalConfig:
    """Parameters that govern chunking, embedding and retrieval."""

    # Chunking (Section 5.3 of the research paper: ~300-400 tokens, 10-20% overlap)
    chunk_size_words: int = 220          # ~ 300 tokens
    chunk_overlap_words: int = 40        # ~ 15% overlap

    # Embedding model (free, local, ~90MB). 384-dim vectors.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Cross-encoder reranker (free, local, ~80MB).
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval breadth
    top_k_dense: int = 10                # candidates from FAISS dense search
    top_k_sparse: int = 10               # candidates from BM25
    rrf_k: int = 60                      # reciprocal-rank-fusion constant
    top_k_final: int = 5                 # passages sent to the generator after rerank

    # FAISS HNSW graph parameters (Section 4.5 of the paper)
    hnsw_m: int = 32                     # neighbours per node
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64


@dataclass
class GeneratorConfig:
    """Which backend produces the final answer."""

    # "gemini" (cloud, default) | "ollama" (local) | "extractive" (no LLM)
    backend: str = field(default_factory=lambda: _get("GENERATOR_BACKEND", "gemini"))

    # Gemini
    gemini_api_key: str = field(default_factory=lambda: _get("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _get("GEMINI_MODEL", "gemini-2.5-flash"))

    # Ollama
    ollama_host: str = field(default_factory=lambda: _get("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _get("OLLAMA_MODEL", "llama3.2:1b"))

    # Decoding — low temperature keeps the model grounded (Section 5.2)
    temperature: float = 0.1
    max_output_tokens: int = 1024


# Singletons imported across the app
RETRIEVAL = RetrievalConfig()
GENERATOR = GeneratorConfig()

# Backends the UI is allowed to offer
AVAILABLE_BACKENDS = ["gemini", "ollama", "extractive"]
