"""Fetch a public table without letting a URL reach the machine or the network.

A model writes the URL, so the URL is hostile input until proven otherwise. This
module is the single gate that reads one: it refuses anything that is not HTTPS,
expands the hostname and refuses any resolved address that is not globally
routable, re-checks every redirect hop, streams the body through a byte cap, and
caps the file type to CSV, parquet or JSON. Nothing here trusts a name, a
scheme, or a `Content-Length`.

The output is a `FetchedFile`: the bytes on disk under their sha256, plus the
provenance a later analysis needs. `utils.data` follows the same write-then-
rename discipline, so a concurrent reader never sees a half-written file.
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

__all__ = ["FetchError", "FetchedFile", "assert_public_url", "fetch_table"]

Format = Literal["csv", "parquet", "json"]

# Only these content types name a format outright. Anything else informative is
# rejected; the generic ones fall back to the URL path.
_CONTENT_TYPE_FORMAT: dict[str, Format] = {
    "text/csv": "csv",
    "application/csv": "csv",
    "application/parquet": "parquet",
    "application/vnd.apache.parquet": "parquet",
    "application/x-parquet": "parquet",
    "application/json": "json",
    "text/json": "json",
    "application/ndjson": "json",
    "application/x-ndjson": "json",
    "text/x-ndjson": "json",
    "application/jsonlines": "json",
    "application/jsonl": "json",
}

# Content types that say nothing about the format, so the path decides.
_GENERIC_CONTENT_TYPES = frozenset(
    {"", "application/octet-stream", "binary/octet-stream", "text/plain"}
)

_PATH_FORMAT: dict[str, Format] = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
}

_SUFFIX: dict[str, str] = {"csv": ".csv", "parquet": ".parquet", "json": ".json"}

_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 5
_CHUNK_BYTES = 65_536

# Ranges a name must never resolve to. `ipaddress.is_global` already refuses most
# of them, but not multicast or the CGNAT block, so the list is explicit.
_BLOCKED_V4 = (
    ipaddress.ip_network("0.0.0.0/32"),  # unspecified
    ipaddress.ip_network("10.0.0.0/8"),  # RFC1918
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT
    ipaddress.ip_network("127.0.0.0/8"),  # loopback
    ipaddress.ip_network("169.254.0.0/16"),  # link-local
    ipaddress.ip_network("172.16.0.0/12"),  # RFC1918
    ipaddress.ip_network("192.168.0.0/16"),  # RFC1918
    ipaddress.ip_network("224.0.0.0/4"),  # multicast
)
_BLOCKED_V6 = (
    ipaddress.ip_network("::/128"),  # unspecified
    ipaddress.ip_network("::1/128"),  # loopback
    ipaddress.ip_network("fc00::/7"),  # unique-local
    ipaddress.ip_network("fe80::/10"),  # link-local
    ipaddress.ip_network("ff00::/8"),  # multicast
)


class FetchError(ValueError):
    """A URL or response this project refuses to read."""


@dataclass(frozen=True)
class FetchedFile:
    """One downloaded table and where it came from."""

    path: Path
    url: str
    sha256: str
    n_bytes: int
    content_type: str
    fetched_at: str
    format: Format


def _resolve(host: str) -> list[str]:
    """Every address a hostname maps to, in one patchable seam for the tests."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(f"cannot resolve host {host!r}: {exc}") from exc
    return [info[4][0] for info in infos]


def _is_public(address: str) -> bool:
    """True only when the address is globally routable and not in a blocked range."""
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    blocked = _BLOCKED_V6 if ip.version == 6 else _BLOCKED_V4
    return ip.is_global and not any(ip in network for network in blocked)


def _check_public(host: str) -> list[str]:
    """Resolve the host and return the addresses, or raise if any one is not public."""
    addresses = _resolve(host)
    for address in addresses:
        if not _is_public(address):
            raise FetchError(f"{host!r} resolves to the non-public address {address}")
    return addresses


def _approve(url: str) -> tuple[str, list[str]]:
    """assert_public_url plus the verified addresses, so the connect can be pinned to them."""
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise FetchError(f"only https URLs are fetched, not {parsed.scheme or 'no scheme'!r}")
    if parsed.username is not None or parsed.password is not None:
        raise FetchError("credentials are not allowed in a fetched URL")
    if parsed.fragment:
        raise FetchError("a fragment is not allowed in a fetched URL")
    host = parsed.hostname
    if not host:
        raise FetchError(f"no host in URL {url!r}")
    return urlunsplit(parsed), _check_public(host)


def assert_public_url(url: str) -> str:
    """Return the normalized https URL, or raise FetchError."""
    return _approve(url)[0]


@contextmanager
def _pin_dns(host: str, addresses: list[str]):
    """Pin one hostname to exactly the addresses the gate approved, per request.

    httpcore resolves through socket.getaddrinfo, so for the duration of one
    request the resolver is swapped for a table that answers `host` with the
    checked addresses and nothing else. A name that re-binds between the check
    and the connect still reaches only what passed `_is_public`; every returned
    address was approved. Other hosts fall through to the real resolver.
    """
    real = socket.getaddrinfo
    verified = []
    for address in addresses:
        ip = ipaddress.ip_address(address)
        family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        # Flowinfo and scopeid stay zero; the connect port is filled in per call.
        verified.append(
            (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, () if ip.version == 4 else (0, 0))
        )

    def pinned(hostname, port, *args, **kwargs):
        if hostname == host:
            return [
                (family, sock_type, proto, "", (address, port, *extra))
                for (family, sock_type, proto, extra), address in zip(
                    verified, addresses, strict=True
                )
            ]
        return real(hostname, port, *args, **kwargs)

    socket.getaddrinfo = pinned
    try:
        yield
    finally:
        socket.getaddrinfo = real


def _detect_format(media_type: str, url: str) -> Format:
    """The format named by the content type, else by the URL path, else nothing."""
    known = _CONTENT_TYPE_FORMAT.get(media_type)
    if known is not None:
        return known
    if media_type in _GENERIC_CONTENT_TYPES:
        path_format = _PATH_FORMAT.get(PurePosixPath(urlsplit(url).path).suffix.lower())
        if path_format is not None:
            return path_format
    raise FetchError(f"unsupported content type {media_type!r} for {url}")


def _store(
    response: httpx.Response, url: str, dest_dir: Path, fmt: Format, media_type: str, max_bytes: int
) -> FetchedFile:
    """Stream the open response to a .part file, then rename it under its digest."""
    digest = hashlib.sha256()
    n_bytes = 0
    staging: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=dest_dir, suffix=".part", delete=False) as handle:
            staging = Path(handle.name)
            for chunk in response.iter_bytes(_CHUNK_BYTES):
                n_bytes += len(chunk)
                if n_bytes > max_bytes:
                    raise FetchError(f"{url} is larger than the {max_bytes} byte cap")
                digest.update(chunk)
                handle.write(chunk)
        sha256 = digest.hexdigest()
        final = dest_dir / f"{sha256}{_SUFFIX[fmt]}"
        if final.exists():
            staging.unlink()
        else:
            staging.replace(final)
    except BaseException:
        if staging is not None:
            staging.unlink(missing_ok=True)
        raise
    return FetchedFile(
        path=final,
        url=url,
        sha256=sha256,
        n_bytes=n_bytes,
        content_type=media_type,
        fetched_at=datetime.now(UTC).isoformat(),
        format=fmt,
    )


def fetch_table(
    url: str,
    *,
    dest_dir: Path,
    max_bytes: int = 52_428_800,
    timeout_s: float = 30.0,
) -> FetchedFile:
    """Download a public CSV, parquet or JSON file to dest_dir.

    Redirects are followed by hand so every hop is checked by `assert_public_url`
    before it is read. The body is streamed and abandoned the moment it crosses
    `max_bytes`, so a server that lies about its size cannot fill memory.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target, verified = _approve(url)

    with httpx.Client(follow_redirects=False, timeout=timeout_s) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            with (
                _pin_dns(urlsplit(target).hostname, verified),
                client.stream("GET", target) as response,
            ):
                if response.status_code in _REDIRECT_STATUS:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError(f"{target} redirected without a location")
                    target, verified = _approve(urljoin(target, location))
                    continue
                if response.status_code != httpx.codes.OK:
                    raise FetchError(f"{target} returned status {response.status_code}")
                media_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                fmt = _detect_format(media_type, target)
                return _store(response, target, dest_dir, fmt, media_type, max_bytes)
    raise FetchError(f"too many redirects starting at {url}")
