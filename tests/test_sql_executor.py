import os
import sys

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sql_agent.executor import UnsafeSQLError, execute_sql


@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(db_path)
    df = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4"], "revenue": [100, 120, 90, 150]})
    con.register("tmp_df", df)
    con.execute("CREATE TABLE revenue_table AS SELECT * FROM tmp_df")
    con.close()
    return db_path


def test_valid_select_returns_rows(test_db):
    result = execute_sql("SELECT AVG(revenue) AS avg_rev FROM revenue_table", db_path=test_db)
    assert result.columns == ["avg_rev"]
    assert result.row_count == 1
    assert result.rows[0][0] == pytest.approx(115.0)


def test_rejects_insert(test_db):
    with pytest.raises(UnsafeSQLError):
        execute_sql("INSERT INTO revenue_table VALUES ('Q5', 999)", db_path=test_db)


def test_rejects_drop(test_db):
    with pytest.raises(UnsafeSQLError):
        execute_sql("DROP TABLE revenue_table", db_path=test_db)


def test_rejects_multi_statement(test_db):
    with pytest.raises(UnsafeSQLError):
        execute_sql("SELECT * FROM revenue_table; DROP TABLE revenue_table;", db_path=test_db)


def test_rejects_empty_sql(test_db):
    with pytest.raises(UnsafeSQLError):
        execute_sql("", db_path=test_db)


def test_read_only_connection_physically_blocks_write(test_db):
    """Even if the regex denylist were bypassed, read_only=True should stop writes."""
    with pytest.raises((UnsafeSQLError, duckdb.Error)):
        execute_sql("UPDATE revenue_table SET revenue = 0", db_path=test_db)
