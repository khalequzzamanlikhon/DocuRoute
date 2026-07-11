"""
Builds:
  - A Qdrant collection of dense embeddings for text chunks.
  - A DuckDB database containing every extracted table, one SQL table per
    ExtractedTable, plus a `_table_registry` metadata table so the
    text-to-SQL agent can discover schemas at query time.
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


def _clean_column_name(col: str) -> str:
    """
    Camelot extracts multi-line header cells with literal \n characters,
    producing names like 'Weighted-\nAverage\nRemaining Life\n(in years)'.
    Flatten them to a single readable line so Gemini can reference them
    in SQL without quoting newline characters.
    """
    cleaned = re.sub(r"\s+", " ", col.replace("\n", " ")).strip()
    return cleaned or col  # never return empty string


def _clean_value_for_numeric(val: str) -> str:
    """
    Real financial PDFs store numbers like:
      "1,234,567"    → commas as thousand separators
      "$ 42,000"     → dollar signs
      "(123,456)"    → parentheses mean negative in accounting notation
      "—" or "—"    → em-dash means zero/not applicable
    Strip all of this so pd.to_numeric can parse the underlying number.
    """
    v = str(val).strip()
    # em-dash / en-dash / plain dash meaning zero → "0"
    if v in ("—", "–", "-", "—", ""):
        return "0"
    # parentheses mean negative: (1,234) → -1234
    if v.startswith("(") and v.endswith(")"):
        v = "-" + v[1:-1]
    # strip $ , spaces % that surround numbers
    v = re.sub(r"[\$,\s%]", "", v)
    return v


def _coerce_numeric_columns(df):
    """
    Two-pass cleaning for real financial table DataFrames:
      1. Clean column names  — remove embedded newlines from multi-line headers.
      2. Clean cell values   — strip $, commas, handle (negatives), dashes.
      3. Coerce to DOUBLE    — only if every non-empty cell in the column
                               successfully parses as a number after cleaning.
    This means Groq/Gemini can write AVG(revenue) or revenue > 0 directly
    instead of wrapping everything in TRY_CAST(...).
    """
    import pandas as pd

    # Pass 1 — clean column names
    df.columns = [_clean_column_name(c) for c in df.columns]

    # Pass 2 — clean values and coerce
    for col in df.columns:
        cleaned_series = df[col].astype(str).apply(_clean_value_for_numeric)
        converted = pd.to_numeric(cleaned_series, errors="coerce")
        non_empty = df[col].astype(str).str.strip().astype(bool)
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
            doc_id     VARCHAR,
            page_number INTEGER,
            columns    VARCHAR
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
        logger.info("Loaded table '%s' (%d rows, cols: %s)", safe_name, len(df), col_types[:120])
    con.close()


def get_table_schemas(db_path: str | None = None) -> list[dict]:
    """Used by the text-to-SQL agent to know what tables/columns exist."""
    db_path = db_path or settings.duckdb_path
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT table_name, doc_id, page_number, columns FROM _table_registry"
        ).fetchall()
    except duckdb.CatalogException:
        rows = []
    con.close()
    return [
        {"table_name": r[0], "doc_id": r[1], "page_number": r[2], "columns": r[3]}
        for r in rows
    ]
