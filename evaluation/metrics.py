"""
Wraps Ragas's four core reference metrics:
  - context_precision, context_recall  -> retrieval quality
  - faithfulness, answer_relevancy      -> generation quality

Ragas needs an LLM-as-judge and an embedding model; we point it at the same
Gemini model used elsewhere (via langchain-google-genai) and the same local
BGE embedder used for retrieval (via langchain-huggingface), so eval isn't
silently depending on a different, uncontrolled model.
"""
from __future__ import annotations

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from config import settings


def _judge():
    return LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0, google_api_key=settings.gemini_api_key)
    )


def _embeddings():
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embedding_model))


def build_dataset(records: list[dict]) -> EvaluationDataset:
    """
    Each record must have:
      user_input: str
      response: str
      retrieved_contexts: list[str]
      reference: str   (ground-truth answer from the golden set)
    """
    return EvaluationDataset.from_list(records)


def run_ragas_eval(records: list[dict]) -> pd.DataFrame:
    dataset = build_dataset(records)
    judge = _judge()
    embeddings = _embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(llm=judge),
            ContextRecall(llm=judge),
            Faithfulness(llm=judge),
            AnswerRelevancy(llm=judge, embeddings=embeddings),
        ],
    )
    return result.to_pandas()
