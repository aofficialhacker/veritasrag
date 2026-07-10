"""
fusion.py  -  Reciprocal Rank Fusion (Section 4.4, "Hybrid (RRF)").
===================================================================
Combines two (or more) ranked lists into one without needing their raw scores to
be on the same scale. Each item receives sum(1 / (k + rank)) across the lists it
appears in. This is the fusion method the research paper cites for hybrid search.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .config import RETRIEVAL


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int | None = None,
) -> List[Tuple[int, float]]:
    """Fuse ranked lists of (chunk_index, score) into one (chunk_index, rrf_score)."""
    k = k if k is not None else RETRIEVAL.rrf_k
    fused: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_idx, _score) in enumerate(ranked):
            fused[chunk_idx] = fused.get(chunk_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
