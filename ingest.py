"""
CLI entry point: run `python ingest.py` after dropping PDFs into data/raw/.
Builds the Qdrant vector index, the on-disk BM25 pickle, and the DuckDB
structured tables in one pass.
"""
from __future__ import annotations
import argparse
import logging
import os

from config import settings
from ingestion.chunking import chunk_pages
from ingestion.index_builder import build_structured_index, build_vector_index
from ingestion.loaders import load_corpus
from retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into vector + structured indices.")
    parser.add_argument("--raw-dir", default=settings.raw_data_dir)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    if not os.path.isdir(args.raw_dir) or not os.listdir(args.raw_dir):
        logger.error("No files found in %s. Drop PDFs there first.", args.raw_dir)
        return

    logger.info("Loading corpus from %s ...", args.raw_dir)
    pages, tables = load_corpus(args.raw_dir)
    logger.info("Loaded %d pages, %d tables", len(pages), len(tables))

    chunks = chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    logger.info("Produced %d chunks", len(chunks))

    logger.info("Building vector index ...")
    build_vector_index(chunks)

    logger.info("Building BM25 index ...")
    retriever = HybridRetriever()
    retriever.build_bm25_from_chunks(chunks)
    bm25_path = os.path.join(settings.processed_data_dir, "bm25_index.pkl")
    retriever.save_bm25(bm25_path)
    logger.info("Saved BM25 index to %s", bm25_path)

    if tables:
        logger.info("Building structured (DuckDB) index ...")
        build_structured_index(tables)
    else:
        logger.info("No tables extracted; skipping structured index.")

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
