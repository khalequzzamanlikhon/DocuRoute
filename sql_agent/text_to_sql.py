"""
Text-to-SQL: turns a natural-language question plus the actual DuckDB
schema (with column types) into a single SQL query. The prompt is
grounded in real column names AND types pulled from _table_registry
so Gemini knows which columns are DOUBLE (arithmetic safe) vs VARCHAR.
"""
from __future__ import annotations

from llm_client import complete_json

SQL_SYSTEM_PROMPT = """You are a SQL generation assistant for a DuckDB database. \
Given a question and the exact schema of available tables (with column names AND \
their types), write ONE read-only SELECT query that answers the question.

Return JSON with this exact shape:
{
  "sql": "<the SELECT statement, no trailing semicolon>",
  "explanation": "<one sentence on what the query computes>"
}

Rules:
- Only use tables/columns that appear in the provided schema.
- DOUBLE columns can be used in AVG/SUM/arithmetic directly.
  VARCHAR columns must be wrapped: TRY_CAST(col AS DOUBLE) before any arithmetic.
- Only generate SELECT statements. Never INSERT, UPDATE, DELETE, DROP, ALTER,
  ATTACH, DETACH, COPY, PRAGMA, CALL, EXPORT, IMPORT, or VACUUM.
- If the question cannot be answered with the given schema, return:
  {"sql": null, "explanation": "<why it cannot be answered>"}
- Always quote column names with double quotes if they contain spaces, dollar
  signs, parentheses, or mixed case.
- ALWAYS alias aggregate expressions with a clear, human-readable name using AS,
  e.g. AVG("Revenue ($M)") AS average_quarterly_revenue, not a bare AVG(...).
  This matters because a downstream step reads only the column name and value —
  an unlabeled result like avg("Revenue ($M)") = 48.9 is ambiguous, while
  average_quarterly_revenue = 48.9 is not.
"""


def generate_sql(question: str, table_schemas: list[dict], target_tables: list[str] | None = None) -> dict:
    if target_tables:
        table_schemas = [t for t in table_schemas if t["table_name"] in target_tables] or table_schemas

    schema_desc = "\n".join(
        f"- {t['table_name']}({t['columns']})  -- from doc: {t['doc_id']}, page {t['page_number']}"
        for t in table_schemas
    ) or "(no tables available)"

    user_prompt = f"""Schema:
{schema_desc}

Question: {question}
"""
    result = complete_json(SQL_SYSTEM_PROMPT, user_prompt, max_tokens=800)
    result.setdefault("sql", None)
    result.setdefault("explanation", "")
    return result
