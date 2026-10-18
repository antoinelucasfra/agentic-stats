"""The docs tools: search the book, ask it questions, index your own text.

Non-inferential by design, like `run_sql`: retrieval over documents writes no
ledger stage and requires none, so the workflow tools' `require` calls are the
only gate and these handlers never touch one.

`ask_docs` composes its answer from the retrieved chunks alone. Sentences are
quoted from the top hits and each carries its citation (`file.md#n`); when
retrieval finds nothing above `MIN_RELEVANCE`, the result says so in
`cannot_say` instead of inventing an answer. A session-local index built by
`index_documents` is searched alongside the book corpus, and never cached to
disk.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, Field

from rag import embed as embedding
from rag import index as rag_index
from rag.index import DocIndex, Hit
from stats.errors import ToolError

# Below this cosine the chunk is treated as noise: an answer quoted from it
# would be an answer the docs do not contain.
MIN_RELEVANCE = 0.3
MAX_ANSWER_HITS = 3

_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Session state: the lazily built book index and the index_documents index.
_book: DocIndex | None = None
_user: DocIndex | None = None


def reset_session() -> None:
    """Drop both cached indexes. New session, or new embeddings, in tests."""
    global _book, _user
    _book = None
    _user = None


def _embed():
    """The embedding backend, resolved per call so tests can swap it."""
    return embedding.resolve()


def _book_index() -> DocIndex | None:
    """The corpus index, built once per process and cached as parquet."""
    global _book
    if _book is None:
        path = rag_index.corpus_index_path()
        if path.exists():
            _book = rag_index.load_index(path)
        else:
            docs = rag_index.corpus_docs()
            if not docs:
                return None
            _book = rag_index.build_index(docs, _embed())
            rag_index.save_index(_book, path)
    return _book


def _combined() -> DocIndex | None:
    """Book corpus plus whatever the session indexed, vectors stacked."""
    indexes = [one for one in (_book_index(), _user) if one is not None and one.chunks]
    if not indexes:
        return None
    if len(indexes) == 1:
        return indexes[0]
    return DocIndex(
        chunks=[c for one in indexes for c in one.chunks],
        vectors=_concat([one.vectors for one in indexes]),
        model=";".join(one.model for one in indexes),
        built=max(one.built for one in indexes),
        sources=[s for one in indexes for s in one.sources],
    )


def _concat(vectors: list[np.ndarray]) -> np.ndarray:
    vectors = [v for v in vectors if v.size]
    if len(vectors) == 1:
        return vectors[0]
    dims = {v.shape[1] for v in vectors}
    if len(dims) > 1:
        raise ToolError(
            "Session documents were embedded with a different model than the book "
            "index. Call index_documents again, or reset the session."
        )
    return np.concatenate(vectors, axis=0)


class DocHit(BaseModel):
    chunk_id: str = Field(description="Citable chunk id, as 'file.md#n'")
    source: str
    heading: str
    score: float
    text: str


class SearchDocsResult(BaseModel):
    query: str
    n_hits: int
    hits: list[DocHit]
    citations: list[str] = Field(description="Chunk ids to quote, in score order")


class Citation(BaseModel):
    chunk_id: str
    source: str


class AskDocsResult(BaseModel):
    question: str
    answer: str = Field(description="Sentences quoted from the docs, each with its citation")
    citations: list[Citation] = Field(default_factory=list)
    cannot_say: str = Field(
        default="",
        description="Non-empty when the docs hold nothing relevant; carries the refusal.",
    )


class IndexDocumentsResult(BaseModel):
    names: list[str]
    n_documents: int
    n_chunks: int
    dim: int
    model: str


def search_docs(query: str, k: int = 5) -> SearchDocsResult:
    """Search the book corpus and session documents; returns chunks with citations.

    Cosine top-k over embedded markdown chunks. Use it to find passages when you
    can read them yourself; use ask_docs when you want an answer to quote.
    """
    combined = _require_index()
    hits = rag_index.search(combined, query, _embed(), k=k)
    return SearchDocsResult(
        query=query,
        n_hits=len(hits),
        hits=[
            DocHit(
                chunk_id=h.id,
                source=h.source,
                heading=h.heading,
                score=round(h.score, 4),
                text=h.chunk,
            )
            for h in hits
        ],
        citations=[h.id for h in hits],
    )


def ask_docs(question: str) -> AskDocsResult:
    """Answer a question from the docs, quoting chunks with per-claim citations.

    Every sentence of the answer comes from a retrieved chunk and names its
    source file and chunk id. When nothing relevant is retrieved, `cannot_say`
    explains that and the answer stays empty: the docs are never paraphrased
    into certainty.
    """
    combined = _require_index()
    embed = _embed()
    hits = rag_index.search(combined, question, embed, k=5)
    relevant = [h for h in hits if h.score >= MIN_RELEVANCE]
    if not relevant:
        return AskDocsResult(
            question=question,
            answer="",
            citations=[],
            cannot_say=(
                "The docs contain nothing relevant to this question "
                f"(best similarity {hits[0].score:.2f} below {MIN_RELEVANCE:.2f})."
                if hits
                else "The docs contain nothing relevant to this question."
            ),
        )
    answer_parts: list[str] = []
    citations: list[Citation] = []
    for hit in relevant[:MAX_ANSWER_HITS]:
        sentence = _best_sentence(hit, question)
        if sentence:
            answer_parts.append(f"{sentence} [{hit.id}]")
            citations.append(Citation(chunk_id=hit.id, source=hit.source))
    if not answer_parts:
        first = relevant[0]
        lead = _sentences(first.chunk)[0]
        answer_parts.append(f"{lead} [{first.id}]")
        citations.append(Citation(chunk_id=first.id, source=first.source))
    return AskDocsResult(question=question, answer=" ".join(answer_parts), citations=citations)


def index_documents(documents: list[str], names: list[str] | None = None) -> IndexDocumentsResult:
    """Index pasted text documents for this session; search_docs and ask_docs see them.

    Text only: the caller reads files and passes their contents. The index lives
    in memory for the session and is never written to disk.
    """
    global _user
    if not documents:
        raise ToolError(
            "index_documents needs at least one document: pass its text in `documents`."
        )
    names = names if names is not None else [f"document-{i + 1}.md" for i in range(len(documents))]
    if len(names) != len(documents):
        raise ToolError(
            f"Got {len(documents)} documents and {len(names)} names; pass one name per document."
        )
    docs = [
        rag_index.Document(source=name, text=text)
        for name, text in zip(names, documents, strict=True)
    ]
    _user = rag_index.build_index(docs, _embed())
    return IndexDocumentsResult(
        names=names,
        n_documents=len(docs),
        n_chunks=len(_user.chunks),
        dim=int(_user.vectors.shape[1]) if _user.vectors.size else 0,
        model=_user.model,
    )


def _require_index() -> DocIndex:
    combined = _combined()
    if combined is None:
        raise ToolError(
            "Nothing to search: the book corpus is empty and no session documents "
            "are indexed. Call index_documents(document_text) first."
        )
    return combined


def _sentences(chunk_text: str) -> list[str]:
    body = "\n".join(line for line in chunk_text.splitlines() if not line.lstrip().startswith("#"))
    return [s.strip() for s in _SENTENCE.split(body) if s.strip()]


def _best_sentence(hit: Hit, question: str) -> str:
    """The chunk sentence sharing the most question words, or '' when none do."""
    stop = {
        "the",
        "a",
        "an",
        "of",
        "is",
        "are",
        "was",
        "to",
        "in",
        "on",
        "and",
        "or",
        "what",
        "how",
        "does",
        "do",
    }
    wanted = {w for w in re.findall(r"\w+", question.lower()) if w not in stop}
    best, best_overlap = "", 0
    for sentence in _sentences(hit.chunk):
        overlap = len(wanted & {w for w in re.findall(r"\w+", sentence.lower())})
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    return best
