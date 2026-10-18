"""A locked-down duckdb connection and a statement guard over it.

duckdb will read a file, open a socket and install an extension if a statement
names one. A model writes the SQL, so the connection and the SQL both have to
refuse that. `connect` opens an in-memory database with external access off,
unsigned extensions off, and the whole configuration locked, so a later
statement cannot turn any of it back on. `guard_query` parses the text with
duckdb's own parser and allows one SELECT; a word-boundary screen over the text
catches the file-and-network functions that parse as a SELECT anyway.

The lock is the point. A regex alone loses to an encoding the author did not
imagine, and a parser alone loses because `read_csv_auto('/etc/passwd')` is a
valid SELECT. Both run, and both fail closed.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import duckdb
import pandas as pd

__all__ = [
    "QueryOutcome",
    "SqlGuardError",
    "connect",
    "guard_query",
    "register_frame",
    "run_select",
]

# The one statement type `run_select` may run. duckdb reports `pragma` as a
# SELECT, so this check alone is not enough and the screen below does real work.
_ALLOWED_STATEMENT = duckdb.StatementType.SELECT

# File, network and extension entry points. Longest branch first so the boundary
# after the longer name still matches. Word boundaries keep a column named
# `copies` out.
_FORBIDDEN = re.compile(
    r"\b("
    r"read_csv_auto|read_json_auto|read_parquet|read_csv|read_json|read_text|"
    r"parquet_scan|sniff_csv|glob|install|load|attach|detach|copy|"
    r"export|import|pragma|set|call"
    r")\b",
    re.IGNORECASE,
)


class SqlGuardError(ValueError):
    """A statement this project refuses to run."""


@dataclass(frozen=True)
class QueryOutcome:
    """The rows a guarded SELECT returned, with the accounting around them."""

    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool
    statement_type: str
    elapsed_s: float


@contextmanager
def connect() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield an in-memory connection with external access off and config locked."""
    conn = duckdb.connect(
        ":memory:",
        config={
            "enable_external_access": False,
            "allow_unsigned_extensions": False,
            "lock_configuration": True,
        },
    )
    try:
        yield conn
    finally:
        conn.close()


def guard_query(query: str) -> str:
    """Return the single allowed statement type, or raise SqlGuardError."""
    try:
        statements = duckdb.extract_statements(query)
    except duckdb.Error as exc:
        raise SqlGuardError(f"query does not parse: {exc}") from exc
    if len(statements) != 1:
        raise SqlGuardError(f"exactly one statement is allowed, found {len(statements)}")
    statement_type = statements[0].type
    if statement_type != _ALLOWED_STATEMENT:
        raise SqlGuardError(f"only SELECT is allowed, not {statement_type.name}")
    match = _FORBIDDEN.search(query)
    if match is not None:
        raise SqlGuardError(f"query references {match.group(1)!r}, which is not allowed")
    return statement_type.name


def register_frame(conn: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame) -> None:
    """Register a pandas frame under a name so a SELECT can read it."""
    conn.register(name, frame)


def run_select(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    *,
    max_rows: int = 200,
    timeout_s: float = 10.0,
) -> QueryOutcome:
    """Run one guarded SELECT, capped at max_rows and interrupted after timeout_s.

    The query is wrapped in an outer `limit max_rows + 1` so the extra row is the
    truncation signal. A timer thread interrupts the connection on the timeout,
    so a slow query fails instead of hanging and the interrupt is always the
    thing that stopped it.
    """
    statement_type = guard_query(query)
    inner = query.strip().rstrip(";").strip()
    wrapped = f"select * from ({inner}) as _guarded limit {int(max_rows) + 1}"

    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.daemon = True
    started = time.perf_counter()
    timer.start()
    try:
        cursor = conn.execute(wrapped)
        columns = [description[0] for description in cursor.description]
        fetched = [tuple(row) for row in cursor.fetchall()]
    finally:
        timer.cancel()
    elapsed_s = time.perf_counter() - started

    return QueryOutcome(
        columns=columns,
        rows=fetched[:max_rows],
        truncated=len(fetched) > max_rows,
        statement_type=statement_type,
        elapsed_s=elapsed_s,
    )
