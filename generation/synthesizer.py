"""
Grounded Generation: builds a numbered context block from reranked text
chunks and/or SQL results, forces the LLM to cite [C_n]/[T_n] markers back
to specific chunk IDs or SQL result sets, and refuses when there's nothing
to ground on. This is what makes answers auditable rather than "trust me."
"""
from __future__ import annotations
import os
from dataclasses import dataclass

from llm_client import complete
from retrieval.reranker import RankedChunk
from sql_agent.executor import SQLResult

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "synthesis_prompt.txt")
with open(_PROMPT_PATH) as f:
    SYNTHESIS_SYSTEM_PROMPT = f.read()

REFUSAL_TEXT = (
    "I don't have enough information in the provided sources to answer this confidently."
)


@dataclass
class Citation:
    marker: str          # "C1" or "T1"
    kind: str            # "chunk" | "table"
    chunk_id: str | None = None
    doc_id: str | None = None
    page_number: int | None = None
    sql: str | None = None


@dataclass
class SynthesisResult:
    answer: str
    citations: list[Citation]
    refused: bool


def _build_context_block(
    chunks: list[RankedChunk] | None, sql_result: SQLResult | None
) -> tuple[str, list[Citation]]:
    lines: list[str] = []
    citations: list[Citation] = []

    for i, c in enumerate(chunks or [], start=1):
        marker = f"C{i}"
        lines.append(f"[{marker}] (source: {c.doc_id}, page {c.page_number})\n{c.text}")
        citations.append(
            Citation(marker=marker, kind="chunk", chunk_id=c.chunk_id, doc_id=c.doc_id, page_number=c.page_number)
        )

    if sql_result is not None:
        marker = "T1"
        preview_rows = sql_result.rows[:20]
        table_str = " | ".join(sql_result.columns) + "\n" + "\n".join(
            " | ".join(str(v) for v in row) for row in preview_rows
        )
        lines.append(f"[{marker}] SQL query: {sql_result.sql}\nResult:\n{table_str}")
        citations.append(Citation(marker=marker, kind="table", sql=sql_result.sql))

    return "\n\n".join(lines), citations


def synthesize(
    question: str,
    chunks: list[RankedChunk] | None = None,
    sql_result: SQLResult | None = None,
) -> SynthesisResult:
    if not chunks and not sql_result:
        return SynthesisResult(answer=REFUSAL_TEXT, citations=[], refused=True)

    context_block, citations = _build_context_block(chunks, sql_result)
    user_prompt = f"CONTEXT ITEMS:\n{context_block}\n\nQUESTION: {question}"

    answer = complete(SYNTHESIS_SYSTEM_PROMPT, user_prompt, max_tokens=800)
    refused = answer.strip() == REFUSAL_TEXT

    # Only keep citations that the model actually referenced, so the UI
    # doesn't show sources that weren't used.
    used = [c for c in citations if f"[{c.marker}]" in answer]
    return SynthesisResult(answer=answer, citations=used if not refused else [], refused=refused)
