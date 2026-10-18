"""Fetch-guard tests: the resolver is patched, so nothing here touches the network.

Every case exercises a way in which a URL is hostile, and each has to fail
closed. The happy path is one file under its sha256, and the transport-level
cases run through `httpx.MockTransport` so no socket is ever opened.
"""

from __future__ import annotations

import hashlib
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


def test_a_private_first_resolution_raises_before_any_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(fetch, "_resolve", lambda host: ["10.0.0.1"])

    with pytest.raises(fetch.FetchError):
        fetch.fetch_table("https://example.com/data.csv", dest_dir=tmp_path)


def test_pin_dns_answers_only_with_verified_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A name that re-binds between the check and the connect still resolves to the check."""

    def lying_getaddrinfo(hostname, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))]

    monkeypatch.setattr(socket, "getaddrinfo", lying_getaddrinfo)

    with fetch._pin_dns("example.com", [PUBLIC]):
        infos = socket.getaddrinfo("example.com", 443)
        assert [(info[0], info[4][0], info[4][1]) for info in infos] == [
            (socket.AF_INET, PUBLIC, 443)
        ]
        # a different host keeps the underlying resolver
        assert socket.getaddrinfo("db.internal", 5432)[0][4][0] == "127.0.0.1"

    assert socket.getaddrinfo("example.com", 443)[0][4][0] == "127.0.0.1"


def test_pin_dns_with_no_verified_addresses_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [])

    with fetch._pin_dns("example.com", []):
        assert socket.getaddrinfo("example.com", 443) == []


def test_dns_rebinding_at_connect_time_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The gate sees a public address; the resolver re-binds to loopback for the connect."""
    hits = 0

    class Attacker(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal hits
            hits += 1
            self.send_response(200)
            self.send_header("content-type", "text/csv")
            self.end_headers()
            self.wfile.write(b"a,b\n1,2\n")

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Attacker)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(fetch, "_resolve", lambda host: [PUBLIC])
        underlying = socket.getaddrinfo

        def rebinding_getaddrinfo(hostname, port_, *args, **kwargs):
            if hostname == "attacker.example":
                return [
                    (
                        socket.AF_INET,
                        socket.SOCK_STREAM,
                        socket.IPPROTO_TCP,
                        "",
                        ("127.0.0.1", port_),
                    )
                ]
            return underlying(hostname, port_, *args, **kwargs)

        monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)

        with pytest.raises((fetch.FetchError, httpx.TransportError)):
            fetch.fetch_table("https://attacker.example/data.csv", dest_dir=tmp_path, timeout_s=5.0)
        assert hits == 0, "the connect must use the verified address, not the re-bound one"
    finally:
        server.shutdown()
        server.server_close()
