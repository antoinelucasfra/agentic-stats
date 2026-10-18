"""Fetch-guard tests: the resolver is patched, so nothing here touches the network.

Every case exercises a way in which a URL is hostile, and each has to fail
closed. The happy path is one file under its sha256, and the transport-level
cases run through `httpx.MockTransport` so no socket is ever opened.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from utils import fetch

PUBLIC = "93.184.216.34"


@pytest.fixture(autouse=True)
def no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test reaches the real resolver, so a leak is visible."""

    def guard(host: str) -> list[str]:
        raise AssertionError(f"test resolved {host!r} for real; patch fetch._resolve")

    monkeypatch.setattr(fetch, "_resolve", guard)


def resolve_to(mapping: dict[str, list[str]], default: str = PUBLIC):
    def resolver(host: str) -> list[str]:
        return mapping.get(host, [default])

    return resolver


def serve(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Route `fetch_table` through a MockTransport built on the handler."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        fetch.httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs)
    )


def public_serve(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    monkeypatch.setattr(fetch, "_resolve", resolve_to({}))
    serve(monkeypatch, handler)


def test_plain_http_is_rejected() -> None:
    with pytest.raises(fetch.FetchError):
        fetch.assert_public_url("http://example.com/data.csv")


def test_credentials_in_the_url_are_rejected() -> None:
    with pytest.raises(fetch.FetchError):
        fetch.assert_public_url("https://user:secret@example.com/data.csv")


def test_fragment_is_rejected() -> None:
    with pytest.raises(fetch.FetchError):
        fetch.assert_public_url("https://example.com/data.csv#rows")


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # loopback
        "10.0.0.1",  # RFC1918
        "172.16.4.2",  # RFC1918
        "192.168.10.4",  # RFC1918
        "169.254.169.254",  # link-local, the cloud metadata address
        "100.64.0.7",  # CGNAT
        "0.0.0.0",  # unspecified
        "224.0.0.1",  # multicast
        "::1",  # IPv6 loopback
        "fc00::1",  # IPv6 unique-local
        "fe80::1",  # IPv6 link-local
        "ff02::1",  # IPv6 multicast
    ],
)
def test_non_public_addresses_are_rejected(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    monkeypatch.setattr(fetch, "_resolve", lambda host: [address])
    with pytest.raises(fetch.FetchError):
        fetch.assert_public_url("https://example.com/data.csv")


def test_one_non_public_address_rejects_the_whole_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "_resolve", lambda host: [PUBLIC, "10.0.0.1"])
    with pytest.raises(fetch.FetchError):
        fetch.assert_public_url("https://example.com/data.csv")


def test_a_public_host_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "_resolve", resolve_to({}))
    assert fetch.assert_public_url("https://example.com/data.csv") == (
        "https://example.com/data.csv"
    )


def test_redirect_to_a_private_address_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        fetch, "_resolve", resolve_to({"example.com": [PUBLIC], "internal.example": ["10.0.0.1"]})
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://internal.example/secret.csv"})

    serve(monkeypatch, handler)
    with pytest.raises(fetch.FetchError):
        fetch.fetch_table("https://example.com/data.csv", dest_dir=tmp_path)


def test_redirect_to_a_public_host_is_followed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        fetch, "_resolve", resolve_to({"example.com": [PUBLIC], "cdn.example": [PUBLIC]})
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(301, headers={"location": "https://cdn.example/data.csv"})
        return httpx.Response(200, headers={"content-type": "text/csv"}, content=b"a,b\n1,2\n")

    serve(monkeypatch, handler)
    result = fetch.fetch_table("https://example.com/data.csv", dest_dir=tmp_path)
    assert result.url == "https://cdn.example/data.csv"
    assert result.path.suffix == ".csv"


def test_redirect_loop_is_capped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fetch, "_resolve", resolve_to({}))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.com/again.csv"})

    serve(monkeypatch, handler)
    with pytest.raises(fetch.FetchError):
        fetch.fetch_table("https://example.com/data.csv", dest_dir=tmp_path)


def test_oversized_body_is_aborted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    body = b"a,b\n" * 10_000
    public_serve(
        monkeypatch,
        lambda request: httpx.Response(200, headers={"content-type": "text/csv"}, content=body),
    )
    with pytest.raises(fetch.FetchError):
        fetch.fetch_table("https://example.com/big.csv", dest_dir=tmp_path, max_bytes=1024)
    assert list(tmp_path.iterdir()) == []


def test_unsupported_content_type_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    public_serve(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"}, content=b"<html></html>"
        ),
    )
    with pytest.raises(fetch.FetchError):
        fetch.fetch_table("https://example.com/data.csv", dest_dir=tmp_path)


def test_error_status_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    public_serve(monkeypatch, lambda request: httpx.Response(404))
    with pytest.raises(fetch.FetchError):
        fetch.fetch_table("https://example.com/data.csv", dest_dir=tmp_path)


def test_happy_path_writes_one_file_named_by_sha256(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = b"a,b\n1,2\n3,4\n"
    public_serve(
        monkeypatch,
        lambda request: httpx.Response(200, headers={"content-type": "text/csv"}, content=body),
    )
    result = fetch.fetch_table("https://example.com/data.csv", dest_dir=tmp_path)
    digest = hashlib.sha256(body).hexdigest()
    assert result.sha256 == digest
    assert result.format == "csv"
    assert result.content_type == "text/csv"
    assert result.n_bytes == len(body)
    assert result.url == "https://example.com/data.csv"
    assert result.path == tmp_path / f"{digest}.csv"
    assert result.path.read_bytes() == body
    assert [path.name for path in tmp_path.iterdir()] == [f"{digest}.csv"]
    assert result.fetched_at.endswith("+00:00")


def test_same_bytes_land_on_one_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    body = b"a,b\n1,2\n"
    public_serve(
        monkeypatch,
        lambda request: httpx.Response(200, headers={"content-type": "text/csv"}, content=body),
    )
    first = fetch.fetch_table("https://example.com/data.csv", dest_dir=tmp_path)
    second = fetch.fetch_table("https://example.com/data.csv", dest_dir=tmp_path)
    assert first.path == second.path
    assert [path.name for path in tmp_path.iterdir()] == [first.path.name]


def test_generic_content_type_falls_back_to_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    public_serve(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, content=b"PAR1"
        ),
    )
    result = fetch.fetch_table("https://example.com/data.parquet", dest_dir=tmp_path)
    assert result.format == "parquet"
    assert result.path.suffix == ".parquet"


def test_jsonl_content_type_maps_to_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    public_serve(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-type": "application/x-ndjson"}, content=b'{"a": 1}\n'
        ),
    )
    result = fetch.fetch_table("https://example.com/data.jsonl", dest_dir=tmp_path)
    assert result.format == "json"
    assert result.path.suffix == ".json"
