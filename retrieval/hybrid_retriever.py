"""
Hybrid Search: fuses BM25 (sparse/keyword) results with dense vector
results using Reciprocal Rank Fusion (RRF). RRF is used instead of a raw
score-weighted sum because BM25 scores and cosine similarities live on
incomparable scales — RRF sidesteps that by fusing on *rank*, not score.

Supports multi-query retrieval: if multiple query variants are supplied
(from query_rewriter.py), each is run independently through both BM25 and
vector search, and all result lists are fused together in one RRF pass.
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from config import settings
from ingestion.chunking import Chunk
from ingestion.index_builder import get_embedder, get_qdrant_client


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    page_number: int
    text: str
    score: float  # fused RRF score


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    """
    Holds an in-memory BM25 index (rebuilt from chunk metadata stored in
    Qdrant, or loaded from a pickle) alongside a handle to the Qdrant client
    for dense search. Keeping BM25 in-process avoids standing up Elasticsearch
    for a portfolio project while still giving true sparse retrieval.
    """

    def __init__(self, client: QdrantClient | None = None):
        self.client = client or get_qdrant_client()
        self.embedder = get_embedder()
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build_bm25_from_chunks(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def save_bm25(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f)

    def load_bm25(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._bm25 = data["bm25"]

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        # qdrant-client >= 1.7 replaced .search() with .query_points()
        result = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=vec,
            limit=top_k,
        )
        return [(h.payload["chunk_id"], h.score) for h in result.points]

    def _sparse_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_bm25_from_chunks() or load_bm25().")
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunks[i].chunk_id, float(scores[i])) for i in ranked]

    def _chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for c in self._chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def retrieve(
        self,
        queries: list[str],
        top_k_each: int | None = None,
        top_k_fused: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Runs every query variant through both dense and sparse search, then
        fuses ALL resulting ranked lists with RRF: score(d) = sum_over_lists(
        1 / (k + rank_in_that_list) ). Documents appearing near the top of
        multiple lists (agreement across BM25/vector AND across query
        rephrasings) rise to the top — that agreement is the whole point.
        """
        top_k_each = top_k_each or settings.top_k_dense
        top_k_fused = top_k_fused or settings.top_k_fused
        k = settings.rrf_k

        rrf_scores: dict[str, float] = {}
        for q in queries:
            dense_hits = self._dense_search(q, top_k_each)
            sparse_hits = self._sparse_search(q, top_k_each)

            for rank, (chunk_id, _) in enumerate(dense_hits):
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            for rank, (chunk_id, _) in enumerate(sparse_hits):
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_fused]

        results = []
        for chunk_id, score in ranked_ids:
            chunk = self._chunk_by_id(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=score,
                )
            )
        return results
