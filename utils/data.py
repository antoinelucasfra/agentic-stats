"""The dataset: a real split-plot trial, fetched at runtime and cached as parquet.

Nothing is committed. `fetch()` pulls the oats split-plot experiment from
Rdatasets over HTTPS, converts it with `duckdb` straight into parquet, and every
tool reads that parquet. Paths and the source URL live in `utils.config`; two
environment variables move them:

- `AGENTIC_STATS_DATA`  an explicit dataset path (CSV or parquet), no fetching.
- `AGENTIC_STATS_CACHE` the cache directory, the system temp dir by default.

The trial is Yates (1935), as analysed by Pinheiro and Bates (2000): six blocks,
three oat varieties in the whole plots, four nitrogen rates in the subplots, 72
rows. Column names are lowercased, and `yield` becomes `grain`, because patsy
cannot parse `yield` inside a formula: it is a Python keyword.

    python -m utils.data [--out PATH] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from utils.config import CACHE_NAME, SELECT, SOURCE, default_path

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["CACHE_NAME", "SOURCE", "default_path", "ensure", "fetch", "read_parquet"]

# Kept for the transports that read a module-level path.
DATA_PATH = default_path()


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def fetch(dest: Path | None = None, force: bool = False) -> Path:
    """Download the trial, write it as parquet, and return the path.

    The parquet is written to a temporary file and moved into place, so a
    concurrent reader never sees a half-written file. `force=False` skips the
    work when the destination already exists.
    """
    dest = Path(dest) if dest is not None else default_path()
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(dest.name + ".part")
    conn = duckdb.connect()
    try:
        conn.execute(
            f"copy ({SELECT.format(source=_quote(SOURCE))}) to {_quote(str(staging))} "
            "(format parquet)"
        )
        rows = int(
            conn.execute(f"select count(*) from read_parquet({_quote(str(staging))})").fetchone()[0]
        )
    finally:
        conn.close()

    staging.replace(dest)
    _write_meta(dest, rows)
    return dest


def _write_meta(dest: Path, rows: int) -> None:
    """Provenance next to the cache: what was fetched, when, and its digest."""
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    meta = {
        "source": SOURCE,
        "fetched_at": datetime.now(UTC).isoformat(),
        "rows": rows,
        "sha256": digest,
    }
    dest.with_suffix(dest.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def read_parquet(path: Path | str) -> pd.DataFrame:
    """Load a parquet file into pandas through duckdb, so pyarrow is not needed."""
    conn = duckdb.connect()
    try:
        return conn.execute("select * from read_parquet(?)", [str(path)]).df()
    finally:
        conn.close()


def ensure(path: Path | None = None) -> Path:
    """Return a readable dataset path, fetching the cache when it is missing."""
    path = Path(path) if path is not None else default_path()
    if path.exists():
        return path
    return fetch(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache the oats split-plot trial")
    parser.add_argument("--out", type=Path, default=None, help="destination parquet")
    parser.add_argument("--force", action="store_true", help="refetch even if cached")
    args = parser.parse_args()

    written = fetch(args.out, force=args.force or args.out is not None)
    conn = duckdb.connect()
    try:
        rows = conn.execute(
            f"select count(*) from read_parquet({_quote(str(written))})"
        ).fetchone()[0]
        names = [
            row[0]
            for row in conn.execute(
                f"describe select * from read_parquet({_quote(str(written))})"
            ).fetchall()
        ]
    finally:
        conn.close()
    print(f"{written} -> {rows} rows x {len(names)} columns {names}")


if __name__ == "__main__":
    main()
