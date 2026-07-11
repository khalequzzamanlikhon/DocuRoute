"""
Query Transformation: rewrites a vague/conversational query into a form
better suited to retrieval, and optionally generates multiple phrasings
("multi-query retrieval") to widen recall before fusion.
"""
from __future__ import annotations

from llm_client import complete_json

REWRITE_SYSTEM_PROMPT = """You rewrite user questions into effective search \
queries for a hybrid (keyword + semantic) retrieval system over a document \
corpus. Expand abbreviations, resolve vague pronouns using context if given, \
and produce queries that mention concrete entities/keywords likely to appear \
in the source text.

Return JSON with this exact shape:
{
  "primary_query": "<best single rewritten query>",
  "variants": ["<alternate phrasing 1>", "<alternate phrasing 2>"]
}

Produce exactly 2 variants in addition to the primary query. Variants should
emphasize different keywords/angles of the same underlying question, not
just reword it trivially.
"""


def rewrite_query(question: str, conversation_context: str | None = None) -> dict:
    user_prompt = f"Question: {question}"
    if conversation_context:
        user_prompt += f"\n\nRecent conversation context:\n{conversation_context}"

    result = complete_json(REWRITE_SYSTEM_PROMPT, user_prompt, max_tokens=600)
    result.setdefault("primary_query", question)
    result.setdefault("variants", [])
    return result


def all_query_variants(question: str, conversation_context: str | None = None) -> list[str]:
    """Convenience helper returning [primary, *variants] for multi-query retrieval."""
    rewritten = rewrite_query(question, conversation_context)
    return [rewritten["primary_query"], *rewritten["variants"]]
