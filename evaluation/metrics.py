"""Direct RAG metrics using llm_client.complete() as the judge.

Computes the four standard Ragas metrics (Context Precision, Context Recall,
Faithfulness, Answer Relevancy) without importing the Ragas library at all.
This avoids every langchain / google-genai / pydantic dependency conflict.

Each metric asks the LLM (via the project's own provider chain) to score a
specific aspect on 0-1, then averages across all questions.
"""
from __future__ import annotations

import json
import logging
import re
from statistics import mean
from typing import Any

import pandas as pd

from llm_client import complete

logger = logging.getLogger(__name__)

# ── Scoring prompts ──────────────────────────────────────────────────────────

_CONTEXT_PRECISION_PROMPT = """\
You are evaluating a RAG system's retrieval quality.

Question: {question}
Retrieved context: {context}

On a scale of 0.0 to 1.0, how much of the retrieved context above is actually
relevant to answering the question? Consider each sentence in the context.

Return ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<brief reason>"}}"""

_CONTEXT_RECALL_PROMPT = """\
You are evaluating a RAG system's retrieval completeness.

Question: {question}
Ground-truth answer: {reference}
Retrieved context: {context}

On a scale of 0.0 to 1.0, how much of the information needed to answer the
question (as shown in the ground-truth answer) is covered by the retrieved
context?

Return ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<brief reason>"}}"""

_FAITHFULNESS_PROMPT = """\
You are evaluating whether a generated answer is faithful to the retrieved context.

Question: {question}
Retrieved context: {context}
Generated answer: {response}

On a scale of 0.0 to 1.0, how faithful is the answer — does it ONLY state
things that are directly supported by the retrieved context? Deduct for any
claim in the answer that is not present in or contradicted by the context.

Return ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<brief reason>"}}"""

_ANSWER_RELEVANCY_PROMPT = """\
You are evaluating whether a generated answer actually addresses the question asked.

Question: {question}
Generated answer: {response}

On a scale of 0.0 to 1.0, how relevant is the answer to the question? Does it
directly address what was asked, or does it miss the point?

Return ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<brief reason>"}}"""


# ── Scorers ──────────────────────────────────────────────────────────────────


def _ask_llm(prompt: str, label: str) -> float:
    """Send a prompt to the LLM and extract a 0-1 score from the JSON response."""
    raw = complete(
        system="You are a helpful RAG evaluator. Always respond with valid JSON.",
        user=prompt,
        max_tokens=512,
        temperature=0.0,
    )
    # Extract JSON from the response (handle markdown fences)
    cleaned = raw.strip()
    # Remove markdown code fences and language tags
    cleaned = re.sub(r"^```\w*\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return score
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Bad JSON from LLM for %s: %.100s", label, raw)
        # Fallback: try to find a number in the response
        nums = re.findall(r"(\d+\.?\d*)", raw)
        if nums:
            return max(0.0, min(1.0, float(nums[0])))
        return 0.0


def _score_context_precision(question: str, contexts: list[str]) -> float:
    context = "\n---\n".join(contexts)
    prompt = _CONTEXT_PRECISION_PROMPT.format(question=question, context=context)
    return _ask_llm(prompt, "context_precision")


def _score_context_recall(question: str, reference: str, contexts: list[str]) -> float:
    context = "\n---\n".join(contexts)
    prompt = _CONTEXT_RECALL_PROMPT.format(
        question=question, reference=reference, context=context
    )
    return _ask_llm(prompt, "context_recall")


def _score_faithfulness(question: str, response: str, contexts: list[str]) -> float:
    context = "\n---\n".join(contexts)
    prompt = _FAITHFULNESS_PROMPT.format(
        question=question, context=context, response=response
    )
    return _ask_llm(prompt, "faithfulness")


def _score_answer_relevancy(question: str, response: str) -> float:
    prompt = _ANSWER_RELEVANCY_PROMPT.format(question=question, response=response)
    return _ask_llm(prompt, "answer_relevancy")


# ── Public API ────────────────────────────────────────────────────────────────


def build_dataset(records: list[dict]) -> list[dict]:
    return records


def run_ragas_eval(records: list[dict]) -> pd.DataFrame:
    """Score every record on all 4 metrics and return a DataFrame of per-question scores."""
    rows = []
    for i, rec in enumerate(records):
        logger.info("Scoring %d/%d ...", i + 1, len(records))
        context_precision = _score_context_precision(
            rec["user_input"], rec.get("retrieved_contexts", [""])
        )
        context_recall = _score_context_recall(
            rec["user_input"], rec.get("reference", ""), rec.get("retrieved_contexts", [""])
        )
        faithfulness = _score_faithfulness(
            rec["user_input"], rec.get("response", ""), rec.get("retrieved_contexts", [""])
        )
        answer_relevancy = _score_answer_relevancy(
            rec["user_input"], rec.get("response", "")
        )
        rows.append({
            "user_input": rec["user_input"],
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Averages — context_precision=%.4f, context_recall=%.4f, "
        "faithfulness=%.4f, answer_relevancy=%.4f",
        df["context_precision"].mean(),
        df["context_recall"].mean(),
        df["faithfulness"].mean(),
        df["answer_relevancy"].mean(),
    )
    return df
