"""SQL-guard tests: one SELECT runs, every write, file or network statement fails.

The interesting cases are the statements duckdb's parser labels SELECT, like
`pragma` and `select * from read_csv_auto(...)`, plus the word-boundary case
where a column named `copies` must not look like a `copy`.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pandas as pd
import pytest

from utils import duck


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    with duck.connect() as connection:
        yield connection


ALLOWED = [
    "select 1 as a",
    "select 1 as copies",  # a column named copies is not a COPY
    "select 1 as loaded",  # a column named loaded is not a LOAD
    "select p.copies from (select 1 as copies) as p",
]

REJECTED = [
    ("insert", "insert into t values (1)"),
    ("update", "update t set a = 1"),
    ("delete", "delete from t"),
    ("drop", "drop table t"),
    ("create", "create table t (a int)"),
    ("attach", "attach 'other.db'"),
    ("detach", "detach other"),
    ("copy", "copy t to 'leak.csv'"),
    ("pragma", "pragma version"),
    ("install", "install httpfs"),
    ("load", "load httpfs"),
    ("call", "call pragma_version()"),
    ("bare read_csv_auto", "read_csv_auto('/etc/passwd')"),
    ("read_csv_auto in a select", "select * from read_csv_auto('/etc/passwd')"),
    ("read_csv", "select * from read_csv('/etc/passwd')"),
    ("read_parquet", "select * from read_parquet('/etc/passwd')"),
    ("parquet_scan", "select * from parquet_scan('/etc/passwd')"),
    ("read_json_auto", "select * from read_json_auto('https://example.com/a.json')"),
    ("read_text", "select * from read_text('/etc/passwd')"),
    ("glob", "select * from glob('/etc/*')"),
    ("sniff_csv", "select * from sniff_csv('/etc/passwd')"),
    ("export", "export database '/tmp/out'"),
    ("import", "import database '/tmp/out.db'"),
    ("two statements", "select 1; select 2"),
    ("empty", ""),
]


@pytest.mark.parametrize(("label", "query"), REJECTED, ids=[case[0] for case in REJECTED])
def test_rejected_statements(label: str, query: str) -> None:
    with pytest.raises(duck.SqlGuardError):
        duck.guard_query(query)


@pytest.mark.parametrize("query", ALLOWED)
def test_allowed_select(query: str) -> None:
    assert duck.guard_query(query) == "SELECT"


def test_connect_disables_and_locks_external_access(conn: duckdb.DuckDBPyConnection) -> None:
    setting = conn.execute("select current_setting('enable_external_access')").fetchone()
    assert setting is not None and setting[0] is False
    with pytest.raises(duckdb.Error):
        conn.execute("set enable_external_access = true")
    with pytest.raises(duckdb.Error):
        conn.execute("set lock_configuration = false")


def test_register_frame_works_with_external_access_disabled(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    duck.register_frame(conn, "t", pd.DataFrame({"a": [1, 2, 3]}))
    outcome = duck.run_select(conn, "select a from t order by a")
    assert outcome.rows == [(1,), (2,), (3,)]
    assert outcome.columns == ["a"]


def test_run_select_caps_rows_and_reports_truncation(conn: duckdb.DuckDBPyConnection) -> None:
    duck.register_frame(conn, "numbers", pd.DataFrame({"n": range(500)}))
    outcome = duck.run_select(conn, "select n from numbers order by n", max_rows=10)
    assert outcome.statement_type == "SELECT"
    assert outcome.truncated is True
    assert len(outcome.rows) == 10
    assert outcome.rows[0] == (0,)
    assert outcome.elapsed_s >= 0


def test_run_select_is_not_truncated_under_the_cap(conn: duckdb.DuckDBPyConnection) -> None:
    duck.register_frame(conn, "numbers", pd.DataFrame({"n": range(500)}))
    outcome = duck.run_select(conn, "select n from numbers where n < 3", max_rows=10)
    assert outcome.truncated is False
    assert outcome.rows == [(0,), (1,), (2,)]


def test_run_select_refuses_a_guarded_statement(conn: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(duck.SqlGuardError):
        duck.run_select(conn, "drop table numbers")


def test_run_select_interrupts_a_slow_query(conn: duckdb.DuckDBPyConnection) -> None:
    interrupted = []

    class RecordingConnection:
        """A connection that remembers whether the timer reached interrupt."""

        def execute(self, query: str, *args: object):
            return conn.execute(query, *args)

        def interrupt(self) -> None:
            interrupted.append(True)
            conn.interrupt()

    with pytest.raises(duckdb.Error):
        duck.run_select(
            RecordingConnection(),
            "select count(*) from range(100000) as a, range(100000) as b",
            timeout_s=0.3,
        )
    assert interrupted == [True]
