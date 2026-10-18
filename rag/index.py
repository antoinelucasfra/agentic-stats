"""The chunk index: build it, save it as parquet, search it by cosine.

The book corpus is cached as one parquet file next to a `.meta.json` sidecar,
the same conventions as `utils.data`: tempdir/agentic_stats/ by default,
`AGENTIC_STATS_CACHE` to move it, and an explicit `AGENTIC_STATS_BOOK_INDEX`
path for tests. A session-local index over pasted documents is the same object
kept in memory and never written to disk.

Vectors are stored and compared L2-normalised, so a dot product is a cosine.
Zero vectors (empty chunks, empty queries) stay zero and score zero instead of
producing NaNs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from rag.chunk import Chunk, chunk_markdown
from rag.embed import EmbedFn
from utils import config

INDEX_NAME = "book_index.parquet"


@dataclass(frozen=True)
class Document:
    """One named text document, already read by the caller."""

    source: str
    text: str


@dataclass
class DocIndex:
    """Chunks and their row-aligned normalised vectors."""

    chunks: list[Chunk]
    vectors: np.ndarray
    model: str
    built: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Hit:
    """One cosine result: the chunk, where it came from, and how close."""

    chunk: str
    source: str
    heading: str
    id: str
    score: float


def build_index(docs: list[Document], embed: EmbedFn) -> DocIndex:
    """Chunk every document, embed the chunks, and normalise the vectors."""
    counted = [(doc, chunk_markdown(doc.text, doc.source)) for doc in docs]
    chunks = [chunk for _, doc_chunks in counted for chunk in doc_chunks]
    vectors = np.zeros((0, 0), dtype=np.float32)
    if chunks:
        vectors = _normalise(embed([chunk.text for chunk in chunks]))
    return DocIndex(
        chunks=chunks,
        vectors=vectors,
        model=str(getattr(embed, "model", "unknown")),
        built=datetime.now(UTC).isoformat(timespec="seconds"),
        sources=[
            {"source": doc.source, "n_chunks": str(len(doc_chunks))} for doc, doc_chunks in counted
        ],
    )


def search(index: DocIndex, query: str, embed: EmbedFn, k: int = 5) -> list[Hit]:
    """Top-k chunks by cosine similarity, best first."""
    if len(index.chunks) == 0:
        return []
    query_vector = _normalise(embed([query]))[0]
    scores = index.vectors @ query_vector
    order = np.argsort(-scores)[: max(k, 0)]
    return [
        Hit(
            chunk=index.chunks[i].text,
            source=index.chunks[i].source,
            heading=index.chunks[i].heading,
            id=index.chunks[i].id,
            score=float(scores[i]),
        )
        for i in order
    ]


def save_index(index: DocIndex, path: Path | str) -> Path:
    """Write the parquet and its .meta.json sidecar, atomically enough for a cache.

    The parquet lands on a .tmp sibling first and `os.replace` moves it into
    place, so two writers can never leave a half-written cache behind. The COPY
    string is built, not bound, so a quote in the path is escaped doubled.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "text": [c.text for c in index.chunks],
            "source": [c.source for c in index.chunks],
            "heading": [c.heading for c in index.chunks],
            "id": [c.id for c in index.chunks],
            "vector": [v.tolist() for v in index.vectors],
        }
    )
    staging = path.with_suffix(".tmp")
    connection = duckdb.connect()
    connection.register("chunks_df", frame)
    target = staging.as_posix().replace("'", "''")
    connection.execute(f"COPY chunks_df TO '{target}' (FORMAT parquet)")
    connection.close()
    os.replace(staging, path)
    meta = {
        "model": index.model,
        "dim": int(index.vectors.shape[1]) if index.vectors.size else 0,
        "n_chunks": len(index.chunks),
        "sources": index.sources,
        "built": index.built,
    }
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    sidecar_staging = sidecar.with_suffix(".tmp")
    sidecar_staging.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(sidecar_staging, sidecar)
    return path


def load_index(path: Path | str) -> DocIndex:
    """Read back what save_index wrote; the sidecar carries the metadata."""
    path = Path(path)
    connection = duckdb.connect()
    frame = connection.execute(
        "select text, source, heading, id, vector from read_parquet(?)", [str(path)]
    ).df()
    connection.close()
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    vectors = np.asarray([np.asarray(v, dtype=np.float32) for v in frame["vector"]])
    if vectors.size == 0:
        vectors = np.zeros((0, int(meta.get("dim", 0))), dtype=np.float32)
    return DocIndex(
        chunks=[
            Chunk(text=t, source=s, heading=h, id=i)
            for t, s, h, i in zip(
                frame["text"], frame["source"], frame["heading"], frame["id"], strict=True
            )
        ],
        vectors=vectors,
        model=str(meta.get("model", "unknown")),
        built=str(meta.get("built", "")),
        sources=list(meta.get("sources", [])),
    )


def corpus_index_path() -> Path:
    """Where the book corpus index lives, overridable for tests."""
    override = os.environ.get("AGENTIC_STATS_BOOK_INDEX")
    return Path(override) if override else Path(config.cache_dir()) / INDEX_NAME


def corpus_docs() -> list[Document]:
    """The vendored book corpus, sorted so the index is reproducible."""
    # Anchor the default to the repo root, not the cwd: the MCP transport
    # spawns the server with cwd=transports/, where a relative path reads empty.
    default = Path(__file__).resolve().parents[1] / "docs" / "corpus"
    corpus = Path(os.environ.get("AGENTIC_STATS_DOCS_DIR") or default)
    return [
        Document(source=page.name, text=page.read_text(encoding="utf-8"))
        for page in sorted(corpus.glob("*.md"))
    ]


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)
