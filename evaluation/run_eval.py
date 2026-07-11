"""
Runs the golden set through the ACTUAL production pipeline (pipeline.py),
computes Ragas metrics, and appends a row to evaluation/results/history.csv
tagged with a version label you choose (e.g. "baseline_vector_only",
"hybrid_rrf", "hybrid_rrf_rerank"). This history file is the source for the
before/after table in the README.

Usage:
    python evaluation/run_eval.py --version baseline_vector_only
    python evaluation/run_eval.py --version hybrid_rrf_rerank
"""
from __future__ import annotations
import argparse
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from config import settings
from evaluation.metrics import run_ragas_eval
from ingestion.index_builder import get_qdrant_client
from pipeline import answer_question
from retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "results", "history.csv")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


def run(version_label: str) -> pd.DataFrame:
    golden_set = load_golden_set()

    retriever = HybridRetriever(client=get_qdrant_client())
    bm25_path = os.path.join(settings.processed_data_dir, "bm25_index.pkl")
    retriever.load_bm25(bm25_path)

    records = []
    for item in golden_set:
        result = answer_question(item["question"], retriever)
        contexts = [c.text for c in result.trace.reranked_chunks]
        if result.trace.sql_result is not None:
            contexts.append(
                f"SQL: {result.trace.sql_result.sql} -> "
                f"{result.trace.sql_result.columns}: {result.trace.sql_result.rows[:10]}"
            )
        records.append(
            {
                "user_input": item["question"],
                "response": result.answer,
                "retrieved_contexts": contexts or [""],  # ragas requires non-empty list
                "reference": item["ground_truth"],
            }
        )
        logger.info("Answered %s (route=%s, refused=%s)", item["id"], result.trace.route, result.refused)

    scored = run_ragas_eval(records)

    summary = {
        "version": version_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(golden_set),
        "context_precision": scored["context_precision"].mean(),
        "context_recall": scored["context_recall"].mean(),
        "faithfulness": scored["faithfulness"].mean(),
        "answer_relevancy": scored["answer_relevancy"].mean(),
    }

    os.makedirs(RUNS_DIR, exist_ok=True)
    scored.to_csv(os.path.join(RUNS_DIR, f"{version_label}_detailed.csv"), index=False)

    history = pd.DataFrame([summary])
    if os.path.exists(HISTORY_PATH):
        history = pd.concat([pd.read_csv(HISTORY_PATH), history], ignore_index=True)
    history.to_csv(HISTORY_PATH, index=False)

    logger.info("Summary for '%s': %s", version_label, summary)
    return scored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Label for this pipeline configuration, e.g. baseline_vector_only")
    args = parser.parse_args()
    run(args.version)
