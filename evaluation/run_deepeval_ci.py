"""
CI regression gate: runs the golden set through the live pipeline and fails
(non-zero exit code) if faithfulness or answer relevancy drop below the
configured threshold. Wire this into GitHub Actions on every push — that's
what turns "we measured it once" into "we can't silently regress it."

Usage:
    python evaluation/run_deepeval_ci.py
Exit code 0 = pass, 1 = fail (blocks the CI job / PR merge).
"""
from __future__ import annotations
import json
import logging
import os
import sys

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from config import settings
from ingestion.index_builder import get_qdrant_client
from llm_client import complete
from pipeline import answer_question
from retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")


class GeminiJudge(DeepEvalBaseLLM):
    """Lets DeepEval use the same Gemini model as judge instead of defaulting to OpenAI."""

    def load_model(self):
        return None

    def generate(self, prompt: str) -> str:
        return complete(system="You are an expert evaluator.", user=prompt, max_tokens=1024)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return settings.llm_model


def main() -> int:
    with open(GOLDEN_SET_PATH) as f:
        golden_set = json.load(f)

    retriever = HybridRetriever(client=get_qdrant_client())
    bm25_path = os.path.join(settings.processed_data_dir, "bm25_index.pkl")
    retriever.load_bm25(bm25_path)

    judge = GeminiJudge()
    faithfulness_metric = FaithfulnessMetric(threshold=settings.faithfulness_threshold, model=judge)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=judge)

    failures = []
    for item in golden_set:
        result = answer_question(item["question"], retriever)
        contexts = [c.text for c in result.trace.reranked_chunks] or ["(no context retrieved)"]

        test_case = LLMTestCase(
            input=item["question"],
            actual_output=result.answer,
            retrieval_context=contexts,
            expected_output=item["ground_truth"],
        )
        try:
            assert_test(test_case, [faithfulness_metric, relevancy_metric])
            logger.info("PASS %s", item["id"])
        except AssertionError as e:
            logger.error("FAIL %s: %s", item["id"], e)
            failures.append(item["id"])

    if failures:
        logger.error("CI gate FAILED. %d/%d questions failed: %s", len(failures), len(golden_set), failures)
        return 1

    logger.info("CI gate PASSED. All %d questions met thresholds.", len(golden_set))
    return 0


if __name__ == "__main__":
    sys.exit(main())
