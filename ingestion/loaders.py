"""
Extracts two parallel streams from the same source PDFs:
  1. Unstructured text, page by page, for the vector/BM25 index.
  2. Structured tables (as pandas DataFrames), for the DuckDB text-to-SQL path.

Why two separate extraction paths from the same file: this is the core
architectural claim of the whole project (mixed data types, not a single
vector index), so it has to actually be true in the ingestion code, not
just the README.
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass

import fitz  # pymupdf
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    doc_id: str
    page_number: int
    text: str


@dataclass
class ExtractedTable:
    doc_id: str
    page_number: int
    table_name: str
    df: pd.DataFrame


def extract_text_pages(pdf_path: str) -> list[PageText]:
    """Extract raw text per page with pymupdf (fast, no external binary deps)."""
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    pages: list[PageText] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(doc_id=doc_id, page_number=i + 1, text=text))
    logger.info("Extracted %d text pages from %s", len(pages), pdf_path)
    return pages


def _repair_merged_header(header_row: list[str]) -> list[str]:
    """
    Camelot's lattice mode occasionally merges an entire header row into a
    single cell (newline-separated) when the header uses a different fill
    color than the data rows, leaving the remaining header cells empty.
    Detect that pattern and redistribute the labels across the empty slots
    so column names come out correct instead of falling back to col_N for
    everything.
    """
    merged_idx = next((i for i, c in enumerate(header_row) if "\n" in c), None)
    if merged_idx is None:
        return header_row
    labels = [p.strip() for p in header_row[merged_idx].split("\n") if p.strip()]
    empty_slots = [i for i, c in enumerate(header_row) if not c.strip()]
    if len(labels) != len(empty_slots) + 1:
        return header_row  # doesn't match the expected pattern; leave as-is
    repaired = list(header_row)
    repaired[merged_idx] = labels[0]
    for slot, label in zip(empty_slots, labels[1:]):
        repaired[slot] = label
    return repaired


def extract_tables(pdf_path: str) -> list[ExtractedTable]:
    """
    Extract tables with camelot. Requires ghostscript installed on the host
    (`apt-get install ghostscript`). Falls back gracefully if camelot/ghostscript
    isn't available so ingestion of text-only corpora still works.
    """
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        import camelot
    except ImportError:
        logger.warning("camelot not installed; skipping table extraction for %s", pdf_path)
        return []

    tables: list[ExtractedTable] = []
    try:
        camelot_tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        if camelot_tables.n == 0:
            camelot_tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
    except Exception as e:  # camelot raises broad exceptions on malformed PDFs
        logger.warning("Table extraction failed for %s: %s", pdf_path, e)
        return []

    for idx, t in enumerate(camelot_tables):
        df = t.df
        header_row = list(df.iloc[0])
        header_row = _repair_merged_header(header_row)
        df.columns = [f"col_{i}" if not c.strip() else c.strip() for i, c in enumerate(header_row)]
        df = df.iloc[1:].reset_index(drop=True)
        table_name = f"{doc_id}_table_{idx}"
        tables.append(
            ExtractedTable(
                doc_id=doc_id,
                page_number=t.page,
                table_name=table_name,
                df=df,
            )
        )
    logger.info("Extracted %d tables from %s", len(tables), pdf_path)
    return tables


def load_corpus(raw_dir: str) -> tuple[list[PageText], list[ExtractedTable]]:
    """Walk a directory of PDFs and extract both streams for every file."""
    all_pages: list[PageText] = []
    all_tables: list[ExtractedTable] = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(raw_dir, fname)
        all_pages.extend(extract_text_pages(path))
        all_tables.extend(extract_tables(path))
    return all_pages, all_tables
