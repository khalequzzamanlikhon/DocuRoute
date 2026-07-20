"""
Central configuration. Every module imports from here instead of reading
os.environ directly, so there is exactly one place that defines defaults.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # --- Primary LLM: Groq (free tier, 14,400 req/day) ---
    # Get free key at https://console.groq.com
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # --- Secondary LLM: OpenRouter (kicks in when Groq rate-limits) ---
    # Get free key at https://openrouter.ai/keys
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # --- Tertiary fallback LLM: Gemini (used when both Groq & OpenRouter fail) ---
    # Get free key at https://aistudio.google.com/apikey
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")  # used by eval wrappers

    # Embeddings / reranker (open-source, run locally — no API key required)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    # Vector store
    qdrant_mode: str = os.getenv("QDRANT_MODE", "local")  # "local" | "server"
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = _int("QDRANT_PORT", 6333)
    qdrant_storage_path: str = os.getenv("QDRANT_STORAGE_PATH", "./data/processed/qdrant_storage")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "documents")

    # Structured store
    duckdb_path: str = os.getenv("DUCKDB_PATH", "./data/processed/tables.duckdb")

    # Paths
    raw_data_dir: str = os.getenv("RAW_DATA_DIR", "./data/raw")
    processed_data_dir: str = os.getenv("PROCESSED_DATA_DIR", "./data/processed")

    # Retrieval tuning
    top_k_dense: int = _int("TOP_K_DENSE", 20)
    top_k_sparse: int = _int("TOP_K_SPARSE", 20)
    top_k_fused: int = _int("TOP_K_FUSED", 10)
    top_k_final: int = _int("TOP_K_FINAL", 5)
    rrf_k: int = _int("RRF_K", 60)  # standard RRF constant

    # Pipeline mode — controls which retrieval stages are active.
    #   "hybrid_rrf_rerank"   = BM25 + vector + RRF + cross-encoder reranker (full)
    #   "hybrid_rrf"          = BM25 + vector + RRF, no reranker
    #   "baseline_vector_only"= dense vector search only, no BM25, no reranker
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid_rrf_rerank")

    # Guardrails
    faithfulness_threshold: float = _float("FAITHFULNESS_THRESHOLD", 0.7)
    min_rerank_score: float = _float("MIN_RERANK_SCORE", 0.15)


settings = Settings()
