"""Where a public table can come from: a curated list and the Rdatasets index.

The app never invents a dataset URL. It offers a short curated list of famous
stable files, plus the searchable Rdatasets index of a few thousand R datasets,
and anything else has to be pasted as an explicit URL that still passes the
fetch guard in `utils.fetch`.

The index URL is hardcoded https and goes through the same `assert_public_url`
gate as a pasted URL; redirects are not followed for a fixed host.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import httpx

from utils import config
from utils.fetch import assert_public_url

__all__ = ["INDEX_URL", "CatalogEntry", "curated", "fetch_index", "parse_index", "search"]

INDEX_URL = "https://vincentarelbundock.github.io/Rdatasets/datasets.csv"


@dataclass(frozen=True)
class CatalogEntry:
    """One loadable table: a name, what it holds, and where the CSV lives."""

    name: str
    package: str
    title: str
    csv_url: str
    doc_url: str = ""
    rows: int | None = None


def curated() -> list[CatalogEntry]:
    """Famous stable files, each checked to return 200 when this list was written."""
    return [
        CatalogEntry(
            "penguins",
            "palmerpenguins",
            "Size measurements for three penguin species",
            "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv",
            rows=344,
        ),
        CatalogEntry(
            "gapminder",
            "gapminder",
            "Life expectancy, GDP and population by country and year",
            "https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv",
            rows=1704,
        ),
        CatalogEntry(
            "titanic",
            "seaborn",
            "Titanic passengers: class, fare, survival",
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
            rows=891,
        ),
        CatalogEntry(
            "tips",
            "seaborn",
            "Restaurant tips: bill, party size and day",
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
            rows=244,
        ),
        CatalogEntry(
            "oats",
            "nlme",
            "The split-plot trial this app demos with",
            "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Oats.csv",
            rows=72,
        ),
    ]


def parse_index(text: str) -> list[CatalogEntry]:
    """The Rdatasets index CSV as entries. The CSV column carries the link."""
    entries: list[CatalogEntry] = []
    for row in csv.DictReader(text.splitlines()):
        package = (row.get("Package") or "").strip()
        item = (row.get("Item") or "").strip()
        link = (row.get("CSV") or "").strip()
        if not package or not item or not link.startswith("https://"):
            continue
        try:
            rows = int(row.get("Rows") or 0) or None
        except ValueError:
            rows = None
        entries.append(
            CatalogEntry(
                name=item.lower(),
                package=package,
                title=(row.get("Title") or "").strip(),
                csv_url=link,
                doc_url=(row.get("Doc") or "").strip(),
                rows=rows,
            )
        )
    return entries


def index_path() -> Path:
    """Where the cached index lives, next to the dataset cache."""
    return config.cache_dir() / config.CATALOG_INDEX_NAME


def fetch_index(*, force: bool = False, timeout_s: float = 30.0) -> list[CatalogEntry]:
    """The Rdatasets index, cached on disk and refetched when missing or forced."""
    path = index_path()
    if not path.exists() or force:
        assert_public_url(INDEX_URL)
        response = httpx.get(INDEX_URL, timeout=timeout_s, follow_redirects=False)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(path.suffix + ".part")
        staging.write_text(response.text)
        staging.replace(path)
    return parse_index(path.read_text())


def search(entries: list[CatalogEntry], text: str, *, limit: int = 20) -> list[CatalogEntry]:
    """Case-insensitive substring match over name, package and title.

    Every word has to match somewhere; a word in the name outranks one in the
    title, and names break the remaining ties.
    """
    words = [word for word in text.lower().split() if word]
    if not words:
        return entries[:limit]
    ranked: list[tuple[int, CatalogEntry]] = []
    for entry in entries:
        haystack = f"{entry.name} {entry.package} {entry.title}".lower()
        if all(word in haystack for word in words):
            score = sum(word in entry.name for word in words) + sum(
                word in entry.package for word in words
            )
            ranked.append((score, entry))
    ranked.sort(key=lambda item: (-item[0], item[1].name))
    return [entry for _, entry in ranked[:limit]]
