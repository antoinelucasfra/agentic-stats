"""Embedding backends: an OpenAI-compatible endpoint or a local fastembed model.

`EmbedFn` is the whole contract the rest of the package knows: a list of texts
in, a (n, d) float array out. Two backends cover the two deployment stories —
the same provider that answers the chat also embeds, or a small local model
keeps the tool offline. `resolve()` picks one from the environment:

- `AGENTIC_STATS_EMBED`       "local" or "presets", forcing a backend.
- `AGENTIC_STATS_EMBED_BASE_URL`  OpenAI-compatible root, default OpenAI's.
- `AGENTIC_STATS_EMBED_MODEL`     the model name the endpoint expects.
- `AGENTIC_STATS_EMBED_API_KEY`   bearer token; falls back to the chat key.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable

import httpx
import numpy as np

EmbedFn = Callable[[list[str]], "np.ndarray"]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

EXTRA_HINT = (
    "local embeddings need the fastembed package. Install the rag extra "
    "(uv sync --extra rag), or point AGENTIC_STATS_EMBED=presets at an "
    "OpenAI-compatible endpoint instead."
)


def presets_embed(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> EmbedFn:
    """Embed via POST {base_url}/embeddings, the OpenAI-compatible request shape.

    Arguments win over the environment, so tests can pin every part of the call.
    The returned callable carries its model name on `.model` for index metadata.
    """
    base_url = base_url or os.environ.get("AGENTIC_STATS_EMBED_BASE_URL", DEFAULT_BASE_URL)
    model = model or os.environ.get("AGENTIC_STATS_EMBED_MODEL", "")
    if not model:
        raise ValueError(
            "presets embeddings need a model name: set AGENTIC_STATS_EMBED_MODEL "
            "(the endpoint's model id, e.g. text-embedding-3-small)."
        )
    api_key = (
        api_key
        or os.environ.get("AGENTIC_STATS_EMBED_API_KEY")
        or os.environ.get("AGENTIC_STATS_LLM_API_KEY")
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = httpx.Client(base_url=base_url, headers=headers, timeout=60.0)

    def embed(texts: list[str]) -> np.ndarray:
        response = client.post("/embeddings", json={"model": model, "input": texts})
        response.raise_for_status()
        # The API returns items with an explicit index; sort rather than trust order.
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return np.asarray([item["embedding"] for item in data], dtype=np.float32)

    embed.model = f"presets:{model}"  # type: ignore[attr-defined]
    return embed


def local_embed() -> EmbedFn:
    """Embed locally with fastembed's bge-small; requires the `rag` extra."""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # a missing extra, not a bug: say so plainly
        raise RuntimeError(EXTRA_HINT) from exc
    model = TextEmbedding(model_name=LOCAL_MODEL)

    def embed(texts: list[str]) -> np.ndarray:
        return np.asarray(list(model.embed(texts)), dtype=np.float32)

    embed.model = f"local:{LOCAL_MODEL}"  # type: ignore[attr-defined]
    return embed


def resolve() -> EmbedFn:
    """Pick a backend: AGENTIC_STATS_EMBED wins, else presets when a key exists."""
    backend = os.environ.get("AGENTIC_STATS_EMBED", "").strip().lower()
    if backend not in ("", "local", "presets"):
        raise ValueError(f"AGENTIC_STATS_EMBED must be 'local' or 'presets', got '{backend}'.")
    if not backend:
        has_key = bool(
            os.environ.get("AGENTIC_STATS_EMBED_API_KEY")
            or os.environ.get("AGENTIC_STATS_LLM_API_KEY")
        )
        backend = "presets" if has_key else "local"
    # resolve() reads the env on every call, but the built backend (and its
    # httpx.Client) is memoised per configuration: one process, one client.
    return _resolved(
        backend,
        os.environ.get("AGENTIC_STATS_EMBED_BASE_URL", ""),
        os.environ.get("AGENTIC_STATS_EMBED_MODEL", ""),
        os.environ.get("AGENTIC_STATS_EMBED_API_KEY")
        or os.environ.get("AGENTIC_STATS_LLM_API_KEY")
        or "",
    )


@functools.cache
def _resolved(backend: str, base_url: str, model: str, api_key: str) -> EmbedFn:
    """Build (once) the backend for one configuration; see resolve()."""
    if backend == "presets":
        return presets_embed(
            base_url=base_url or None, model=model or None, api_key=api_key or None
        )
    return local_embed()
