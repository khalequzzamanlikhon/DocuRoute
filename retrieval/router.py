"""
Classifies each incoming question so it can be routed to the right sub-system:
  - "unstructured": needs semantic retrieval over document text
  - "structured":    needs SQL/pandas execution against extracted tables
  - "both":          needs both, answered by concatenating both contexts

This is the "Agentic RAG" piece — routing decisions are made by the LLM
based on table schemas actually present in the corpus, not by keyword rules.
"""
from __future__ import annotations
from enum import Enum

from llm_client import complete_json

ROUTER_SYSTEM_PROMPT = """You are a query router for a retrieval-augmented \
question answering system. Given a user question and a list of available \
structured tables (with their columns), decide how the question should be answered.

Return JSON with this exact shape:
{
  "route": "unstructured" | "structured" | "both",
  "reasoning": "<one sentence>",
  "target_tables": ["<table_name>", ...]   // empty list if route is "unstructured"
}

Rules:
- "structured": the question asks for a specific number, aggregate, average, \
trend, comparison, or lookup that a table's columns could directly answer.
- "unstructured": the question asks about explanations, risks, opinions, \
narrative content, or anything not reducible to a table lookup.
- "both": the question needs a number AND surrounding narrative context \
(e.g. "why did revenue drop in Q3").
- Only reference tables that actually appear in the provided schema list.
"""


class Route(str, Enum):
    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    BOTH = "both"


def route_query(question: str, table_schemas: list[dict]) -> dict:
    schema_desc = (
        "\n".join(f"- {t['table_name']}: columns = {t['columns']}" for t in table_schemas)
        or "(no structured tables available)"
    )
    user_prompt = f"""Available tables:
{schema_desc}

Question: {question}
"""
    result = complete_json(ROUTER_SYSTEM_PROMPT, user_prompt, max_tokens=600)

    # Defensive normalization — never trust LLM output shape blindly.
    route = result.get("route", "unstructured")
    if route not in {r.value for r in Route}:
        route = Route.UNSTRUCTURED.value
    result["route"] = route
    result.setdefault("target_tables", [])
    result.setdefault("reasoning", "")
    return result
