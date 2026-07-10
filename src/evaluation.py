"""
evaluation.py  -  Phase 6, Evaluation (Section 4.3 & 5.7).
==========================================================
Implements Ragas-style metrics without any paid API, using the same local
embedding model as retrieval. Three metrics are reported, mirroring the research
paper's description of Ragas (faithfulness, answer relevance, context precision):

  * context_precision - of the passages we retrieved, how many are actually
    relevant to the question? (signal quality of retrieval)
  * faithfulness      - is the generated answer grounded in the retrieved
    context, or did the model drift? (hallucination check)
  * answer_relevance  - does the answer actually address the question?

Each metric is a cosine-similarity proportion in [0, 1]. They are deterministic,
free and offline, which makes them ideal for a reproducible dashboard.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np

from .embeddings import Embedder
from .retriever import HybridRetriever, RetrievedPassage

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 3]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # vectors are already L2-normalised


@dataclass
class QuestionScore:
    question: str
    context_precision: float
    faithfulness: float
    answer_relevance: float

    def as_dict(self) -> Dict:
        return asdict(self)


class RagEvaluator:
    def __init__(self, embedder: Embedder | None = None, relevance_threshold: float = 0.35):
        self.embedder = embedder or Embedder()
        self.threshold = relevance_threshold

    def context_precision(self, question: str, passages: List[RetrievedPassage]) -> float:
        if not passages:
            return 0.0
        q_vec = self.embedder.encode_one(question)
        ctx_vecs = self.embedder.encode([p.chunk.text for p in passages])
        relevant = [1 for v in ctx_vecs if _cos(q_vec, v) >= self.threshold]
        return len(relevant) / len(passages)

    def faithfulness(self, answer: str, passages: List[RetrievedPassage]) -> float:
        sentences = _split_sentences(answer)
        if not sentences or not passages:
            return 0.0
        ctx_vecs = self.embedder.encode([p.chunk.text for p in passages])
        sent_vecs = self.embedder.encode(sentences)
        supported = 0
        for sv in sent_vecs:
            best = max(_cos(sv, cv) for cv in ctx_vecs)
            if best >= self.threshold:
                supported += 1
        return supported / len(sentences)

    def answer_relevance(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0
        q_vec = self.embedder.encode_one(question)
        a_vec = self.embedder.encode_one(answer)
        # Map cosine [-1,1] to [0,1] for a friendlier dashboard scale.
        return max(0.0, min(1.0, (_cos(q_vec, a_vec) + 1.0) / 2.0))

    def score_one(
        self, question: str, answer: str, passages: List[RetrievedPassage]
    ) -> QuestionScore:
        return QuestionScore(
            question=question,
            context_precision=round(self.context_precision(question, passages), 3),
            faithfulness=round(self.faithfulness(answer, passages), 3),
            answer_relevance=round(self.answer_relevance(question, answer), 3),
        )


def aggregate(scores: List[QuestionScore]) -> Dict[str, float]:
    if not scores:
        return {"context_precision": 0.0, "faithfulness": 0.0, "answer_relevance": 0.0}
    return {
        "context_precision": round(np.mean([s.context_precision for s in scores]), 3),
        "faithfulness": round(np.mean([s.faithfulness for s in scores]), 3),
        "answer_relevance": round(np.mean([s.answer_relevance for s in scores]), 3),
    }
