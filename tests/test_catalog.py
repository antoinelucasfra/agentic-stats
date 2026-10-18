"""The catalog: a curated list and the Rdatasets index, both offline-testable.

The index fetch itself is the only part that touches the network, so it is
tested behind a patched transport. Everything else, parsing, ranking and the
curated list, is pure.
"""

from __future__ import annotations

import httpx
import pytest

from utils import catalog

SAMPLE = """\
"Package","Item","Title","Rows","Cols","CSV","Doc"
"datasets","AirPassengers","Monthly Airline Passenger Numbers 1949-1960",144,2,"https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv","https://vincentarelbundock.github.io/Rdatasets/doc/datasets/AirPassengers.html"
"datasets","BOD","Biochemical Oxygen Demand",6,2,"https://vincentarelbundock.github.io/Rdatasets/csv/datasets/BOD.csv",""
"broken","NoLink","A row without a link",10,2,"",""
"evil","Http","A row with a plain http link",10,2,"http://example.test/x.csv",""
"""


def test_parse_index_keeps_the_good_rows_and_drops_the_bad():
    entries = catalog.parse_index(SAMPLE)

    assert [entry.name for entry in entries] == ["airpassengers", "bod"]
    assert entries[0].package == "datasets"
    assert entries[0].rows == 144
    assert entries[0].csv_url.endswith("AirPassengers.csv")
    assert entries[1].doc_url == ""


def test_search_requires_every_word_and_ranks_names_first():
    entries = catalog.parse_index(SAMPLE)

    assert [entry.name for entry in catalog.search(entries, "airpassengers")] == ["airpassengers"]
    assert [entry.name for entry in catalog.search(entries, "datasets")] == [
        "airpassengers",
        "bod",
    ]
    assert catalog.search(entries, "airpassengers bod") == []
    assert catalog.search(entries, "") == entries


def test_curated_entries_are_stable_public_urls():
    entries = catalog.curated()
    names = [entry.name for entry in entries]

    assert len(names) == len(set(names))
    for entry in entries:
        assert entry.csv_url.startswith("https://"), entry.name
        assert entry.csv_url.rsplit("?", 1)[0].lower().endswith(".csv"), entry.name
        assert entry.title, entry.name


def _response(status: int, text: str) -> httpx.Response:
    """A response `raise_for_status` can read, without a transport."""
    return httpx.Response(status, text=text, request=httpx.Request("GET", catalog.INDEX_URL))


def test_fetch_index_caches_and_reuses(monkeypatch, tmp_path):
    """The transport is patched; the cache behaviour is what is under test."""
    monkeypatch.setenv("AGENTIC_STATS_CACHE", str(tmp_path))
    calls: list[str] = []

    def fake_get(url: str, **kwargs):
        calls.append(url)
        return _response(200, SAMPLE)

    monkeypatch.setattr(httpx, "get", fake_get)

    first = catalog.fetch_index()
    second = catalog.fetch_index()

    assert [entry.name for entry in first] == ["airpassengers", "bod"]
    assert calls == [catalog.INDEX_URL]
    assert second == first
    assert catalog.index_path().exists()


def test_fetch_index_refetches_when_forced(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_STATS_CACHE", str(tmp_path))
    calls: list[str] = []

    def fake_get(url: str, **kwargs):
        calls.append(url)
        return _response(200, SAMPLE)

    monkeypatch.setattr(httpx, "get", fake_get)

    catalog.fetch_index()
    catalog.fetch_index(force=True)

    assert calls == [catalog.INDEX_URL, catalog.INDEX_URL]


def test_fetch_index_goes_through_the_fetch_gate(monkeypatch, tmp_path):
    """The gate resolves the host before any request, and redirects stay off."""
    monkeypatch.setenv("AGENTIC_STATS_CACHE", str(tmp_path))
    monkeypatch.setattr("utils.fetch._resolve", lambda host: ["93.184.216.34"])
    seen: dict = {}

    def fake_get(url: str, **kwargs):
        seen.update(kwargs, url=url)
        return _response(200, SAMPLE)

    monkeypatch.setattr(httpx, "get", fake_get)

    catalog.fetch_index()

    assert seen["url"] == catalog.INDEX_URL
    assert seen["follow_redirects"] is False


def test_fetch_index_refuses_a_non_public_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_STATS_CACHE", str(tmp_path))
    monkeypatch.setattr("utils.fetch._resolve", lambda host: ["127.0.0.1"])

    def fail_get(url: str, **kwargs):
        raise AssertionError("the request left before the gate approved the host")

    monkeypatch.setattr(httpx, "get", fail_get)

    with pytest.raises(Exception, match="non-public"):
        catalog.fetch_index()


def test_fetch_index_reports_a_failed_download(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_STATS_CACHE", str(tmp_path))

    def fake_get(url: str, **kwargs):
        return _response(500, "nope")

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(httpx.HTTPStatusError):
        catalog.fetch_index()
