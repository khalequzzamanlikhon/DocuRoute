"""
FastAPI serving layer. Run with:
    uvicorn api.main:app --reload --port 8000

POST /query  {"question": "..."}
GET  /health
GET  /debug/tables   <- new: shows every table in DuckDB with column types
"""
from __future__ import annotations
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import settings
from ingestion.index_builder import get_qdrant_client, get_table_schemas
from pipeline import answer_question
from retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocuRoute API", version="1.0.0")

_retriever: HybridRetriever | None = None


class QueryRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    marker: str
    kind: str
    doc_id: str | None = None
    page_number: int | None = None
    sql: str | None = None


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    route: str
    route_reasoning: str
    citations: list[CitationOut]
    generated_sql: str | None = None
    sql_error: str | None = None        # ← now always visible so you can debug failures


@app.on_event("startup")
def load_indices():
    global _retriever
    bm25_path = os.path.join(settings.processed_data_dir, "bm25_index.pkl")
    if not os.path.exists(bm25_path):
        logger.warning(
            "No BM25 index at %s — run `python ingest.py` first.", bm25_path
        )
        return
    _retriever = HybridRetriever(client=get_qdrant_client())
    _retriever.load_bm25(bm25_path)
    logger.info("Indices loaded. API ready.")


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": _retriever is not None}


@app.get("/debug/tables")
def debug_tables():
    """
    Shows every table DocuRoute extracted from your PDFs, with column types.
    Use this to verify ingestion worked before asking structured questions.
    """
    return {"tables": get_table_schemas()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if _retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run `python ingest.py` first."
        )
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = answer_question(req.question, _retriever)
    return QueryResponse(
        answer=result.answer,
        refused=result.refused,
        route=result.trace.route,
        route_reasoning=result.trace.route_reasoning,
        citations=[
            CitationOut(
                marker=c.marker,
                kind=c.kind,
                doc_id=c.doc_id,
                page_number=c.page_number,
                sql=c.sql,
            )
            for c in result.citations
        ],
        generated_sql=result.trace.generated_sql,
        sql_error=result.trace.sql_error,   # ← always returned, None if all OK
    )
