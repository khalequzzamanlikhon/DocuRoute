"""
Builds:
  - A Qdrant collection of dense embeddings for text chunks (BAAI/bge-large-en-v1.5,
    open source, runs locally so ingestion never needs an embedding API key).
  - A DuckDB database containing every extracted table, one SQL table per
    ExtractedTable, plus a `_table_registry` metadata table so the
    text-to-SQL agent can discover schemas at query time.

Qdrant runs in embedded/local mode (on-disk, no server) by default, which
is what makes `docker-compose up` optional rather than required for a demo.
"""
from __future__ import annotations
import logging
import re

import duckdb

from config import settings
from ingestion.chunking import Chunk
from ingestion.loaders import ExtractedTable

logger = logging.getLogger(__name__)

_EMBED_DIM = {"BAAI/bge-large-en-v1.5": 1024, "BAAI/bge-base-en-v1.5": 768}


def get_qdrant_client():
    from qdrant_client import QdrantClient

    if settings.qdrant_mode == "server":
        return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return QdrantClient(path=settings.qdrant_storage_path)


def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def build_vector_index(chunks: list[Chunk], client=None) -> None:
    from qdrant_client.http import models as qmodels

    client = client or get_qdrant_client()
    embedder = get_embedder()
    dim = _EMBED_DIM.get(settings.embedding_model, embedder.get_sentence_embedding_dimension())

    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )

    # bge models expect a "passage: " style prefix is NOT required for bge (unlike e5);
    # bge recommends a query-side instruction only, so passages are embedded raw.
    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    points = [
        qmodels.PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "page_number": c.page_number,
                "text": c.text,
            },
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info("Indexed %d chunks into Qdrant collection '%s'", len(points), settings.qdrant_collection)


_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_]")


def _safe_table_name(name: str) -> str:
    return _SAFE_NAME.sub("_", name).lower()


def _coerce_numeric_columns(df):
    """
    Camelot always extracts cell values as strings. Before loading into DuckDB,
    try to convert any column where every non-empty value parses as a number.
    This means Gemini can write AVG(revenue) instead of AVG(TRY_CAST(revenue AS DOUBLE))
    — much more natural SQL that is less likely to fail or need correction.
    """
    import pandas as pd
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only switch the column if at least one value converted and none
        # that were non-empty became NaN (i.e., the column is truly numeric).
        non_empty = df[col].str.strip().astype(bool)
        if non_empty.any() and converted[non_empty].notna().all():
            df[col] = converted
    return df


def build_structured_index(tables: list[ExtractedTable], db_path: str | None = None) -> None:
    db_path = db_path or settings.duckdb_path
    con = duckdb.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS _table_registry (
            table_name VARCHAR PRIMARY KEY,
            doc_id VARCHAR,
            page_number INTEGER,
            columns VARCHAR
        )
        """
    )
    for t in tables:
        safe_name = _safe_table_name(t.table_name)
        df = _coerce_numeric_columns(t.df.copy())
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {safe_name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
        col_types = ", ".join(
            f"{c}({'DOUBLE' if str(df[c].dtype).startswith('float') else 'VARCHAR'})"
            for c in df.columns
        )
        con.execute(
            "INSERT OR REPLACE INTO _table_registry VALUES (?, ?, ?, ?)",
            [safe_name, t.doc_id, t.page_number, col_types],
        )
        logger.info("Loaded table '%s' (%d rows) into DuckDB", safe_name, len(df))
    con.close()


def get_table_schemas(db_path: str | None = None) -> list[dict]:
    """Used by the text-to-SQL agent to know what tables/columns exist."""
    db_path = db_path or settings.duckdb_path
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute("SELECT table_name, doc_id, page_number, columns FROM _table_registry").fetchall()
    except duckdb.CatalogException:
        rows = []
    con.close()
    return [
        {"table_name": r[0], "doc_id": r[1], "page_number": r[2], "columns": r[3]}
        for r in rows
    ]
