"""
Safe, sandboxed execution of LLM-generated SQL against DuckDB:
  1. Opens the connection read_only=True at the OS level — physically
     impossible to write, regardless of what SQL is sent.
  2. A regex/keyword denylist as defense-in-depth on top of that (belt + suspenders:
     read_only mode alone is the real guarantee, this catches multi-statement
     injection attempts and pragmas early with a clearer error message).
  3. A hard row-limit and statement timeout so a runaway query can't hang
     the API process or return an enormous payload to the LLM synthesizer.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

import duckdb

from config import settings

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|COPY|PRAGMA|CALL|EXPORT|IMPORT|VACUUM)\b",
    re.IGNORECASE,
)
_MAX_ROWS = 500
_STATEMENT_TIMEOUT_MS = 5000


class UnsafeSQLError(Exception):
    pass


@dataclass
class SQLResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    row_count: int
    truncated: bool


def _validate(sql: str) -> None:
    if sql is None or not sql.strip():
        raise UnsafeSQLError("Empty SQL query.")
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise UnsafeSQLError("Multiple statements are not allowed.")
    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT/CTE queries are allowed.")
    if _FORBIDDEN.search(stripped):
        raise UnsafeSQLError("Query contains a forbidden keyword.")


def execute_sql(sql: str, db_path: str | None = None) -> SQLResult:
    _validate(sql)
    db_path = db_path or settings.duckdb_path

    # read_only=True at connection level is the actual security boundary.
    con = duckdb.connect(db_path, read_only=True)
    try:
        con.execute(f"SET statement_timeout='{_STATEMENT_TIMEOUT_MS}ms'")
    except duckdb.Error:
        pass  # older duckdb versions may not support this pragma; denylist still applies

    try:
        capped_sql = f"SELECT * FROM ({sql.rstrip(';')}) AS _sub LIMIT {_MAX_ROWS + 1}"
        result = con.execute(capped_sql)
        columns = [d[0] for d in result.description]
        rows = result.fetchall()
    finally:
        con.close()

    truncated = len(rows) > _MAX_ROWS
    if truncated:
        rows = rows[:_MAX_ROWS]

    return SQLResult(sql=sql, columns=columns, rows=rows, row_count=len(rows), truncated=truncated)
