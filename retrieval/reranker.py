"""
Reranking: re-scores the RRF-fused candidate set with a cross-encoder,
which reads (query, chunk) pairs jointly rather than comparing independently
computed embeddings. This is what actually fixes cases where a chunk is
lexically/semantically similar but not truly relevant to the question.

Also enforces the "refuse if confidence too low" guardrail: callers should
check `passes_confidence_gate()` before handing context to the synthesizer.
"""
from __future__ import annotations
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from config import settings
from retrieval.hybrid_retriever import RetrievedChunk

_model: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(settings.reranker_model)
    return _model


@dataclass
class RankedChunk:
    chunk_id: str
    doc_id: str
    page_number: int
    text: str
    rerank_score: float


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int | None = None) -> list[RankedChunk]:
    if not candidates:
        return []
    top_k = top_k or settings.top_k_final
    model = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)  # raw logits; higher = more relevant

    # Normalize to (0, 1) with sigmoid so downstream thresholds are interpretable.
    import math

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)[:top_k]
    return [
        RankedChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            page_number=c.page_number,
            text=c.text,
            rerank_score=sigmoid(float(score)),
        )
        for c, score in ranked
    ]


def passes_confidence_gate(ranked: list[RankedChunk]) -> bool:
    """Guardrail: if even the top reranked chunk is below threshold, refuse to answer."""
    if not ranked:
        return False
    return ranked[0].rerank_score >= settings.min_rerank_score
