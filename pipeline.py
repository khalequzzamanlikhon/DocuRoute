"""
The end-to-end orchestrator: Router -> (Rewriter -> Hybrid Retrieval -> Rerank)
and/or (Text-to-SQL -> Executor) -> Synthesizer.

This is the single function the API and eval harness both call, so the
production path and the evaluated path are guaranteed to be identical.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

from config import settings
from generation.synthesizer import SynthesisResult, synthesize
from ingestion.index_builder import get_table_schemas
from retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from retrieval.query_rewriter import all_query_variants
from retrieval.reranker import RankedChunk, passes_confidence_gate, rerank
from retrieval.router import route_query
from sql_agent.executor import SQLResult, UnsafeSQLError, execute_sql
from sql_agent.text_to_sql import generate_sql

logger = logging.getLogger(__name__)


@dataclass
class PipelineTrace:
    """Everything an evaluator or debugging UI needs, component by component."""
    route: str = ""
    route_reasoning: str = ""
    query_variants: list[str] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    reranked_chunks: list[RankedChunk] = field(default_factory=list)
    generated_sql: str | None = None
    sql_result: SQLResult | None = None
    sql_error: str | None = None          # populated on ANY sql failure
    confidence_gate_passed: bool = True


@dataclass
class PipelineResult:
    answer: str
    citations: list
    refused: bool
    trace: PipelineTrace


def answer_question(question: str, retriever: HybridRetriever) -> PipelineResult:
    trace = PipelineTrace()
    table_schemas = get_table_schemas()

    routing = route_query(question, table_schemas)
    trace.route = routing["route"]
    trace.route_reasoning = routing["reasoning"]

    reranked: list[RankedChunk] = []
    sql_result: SQLResult | None = None

    if routing["route"] in ("unstructured", "both"):
        variants = all_query_variants(question)
        trace.query_variants = variants
        candidates = retriever.retrieve(variants)
        trace.retrieved_chunks = candidates
        if settings.retrieval_mode in ("hybrid_rrf", "baseline_vector_only"):
            # Skip cross-encoder reranker — use top-k from RRF directly.
            reranked = [
                RankedChunk(
                    chunk_id=c.chunk_id, doc_id=c.doc_id,
                    page_number=c.page_number, text=c.text,
                    rerank_score=1.0 - i / max(len(candidates), 1),
                )
                for i, c in enumerate(candidates[:settings.top_k_final])
            ]
        else:
            reranked = rerank(question, candidates)
        trace.reranked_chunks = reranked
        trace.confidence_gate_passed = passes_confidence_gate(reranked)
        if not trace.confidence_gate_passed:
            reranked = []

    if routing["route"] in ("structured", "both"):
        sql_gen = generate_sql(question, table_schemas, routing.get("target_tables"))
        trace.generated_sql = sql_gen.get("sql")
        if trace.generated_sql:
            try:
                sql_result = execute_sql(trace.generated_sql)
                trace.sql_result = sql_result
                # If SQL ran but returned NULL (e.g. all rows filtered out),
                # treat it as an error so the synthesizer gets something useful.
                if sql_result.row_count == 0 or (
                    sql_result.row_count == 1
                    and all(v is None for v in sql_result.rows[0])
                ):
                    trace.sql_error = (
                        f"SQL executed but returned no data. "
                        f"Query: {trace.generated_sql}"
                    )
                    sql_result = None
                    trace.sql_result = None
            except UnsafeSQLError as e:
                # Our own validation rejection (forbidden keywords, multi-statement, etc.)
                trace.sql_error = f"Unsafe SQL rejected: {e}"
                logger.warning("Unsafe SQL rejected: %s", e)
            except Exception as e:
                # DuckDB runtime errors: BinderException, CatalogException,
                # ConversionException, etc. — catch ALL so sql_result stays None
                # instead of propagating up and returning a 500 to the user.
                trace.sql_error = f"SQL execution failed: {type(e).__name__}: {e}"
                logger.error("SQL execution error for query %r: %s", trace.generated_sql, e)

    if trace.sql_result is not None:
        logger.info(
            "SQL result for %r -> columns=%s rows=%s",
            question, trace.sql_result.columns, trace.sql_result.rows,
        )

    synthesis: SynthesisResult = synthesize(
        question, chunks=reranked or None, sql_result=sql_result
    )

    return PipelineResult(
        answer=synthesis.answer,
        citations=synthesis.citations,
        refused=synthesis.refused,
        trace=trace,
    )
