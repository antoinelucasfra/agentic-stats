"""The rag package: chunking, cosine search and the three docs tools.

Every test runs on a fake bag-of-words `EmbedFn` (dimension 1024, sparse
enough that unrelated words rarely collide) plus a mocked httpx client, so the
suite never touches the network and never needs fastembed.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from rag import chunk, embed
from rag import index as rag_index
from rag import tools as rag_tools
from stats import registry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "rag_docs"

DIM = 1024


def fake_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            out[i, zlib.crc32(word.encode()) % DIM] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


fake_embed.model = "fake:bag-of-words"


@pytest.fixture
def book(tmp_path, monkeypatch):
    """The corpus points at the fixtures, the cache at tmp, the embed at the fake."""
    monkeypatch.setenv("AGENTIC_STATS_DOCS_DIR", str(FIXTURES))
    monkeypatch.setenv("AGENTIC_STATS_BOOK_INDEX", str(tmp_path / "book.parquet"))
    monkeypatch.setattr(rag_tools, "_embed", lambda: fake_embed)
    rag_tools.reset_session()
    yield tmp_path / "book.parquet"
    rag_tools.reset_session()


# --- chunking ---------------------------------------------------------------


def test_chunking_is_deterministic_and_keeps_sections_apart():
    text = FIXTURES.joinpath("oats.md").read_text(encoding="utf-8")
    first = chunk.chunk_markdown(text, "oats.md")
    assert first == chunk.chunk_markdown(text, "oats.md")

    nitrogen = [c for c in first if "Nitrogen" in c.heading]
    blocks = [c for c in first if "Blocks" in c.heading]
    assert nitrogen and blocks
    assert "random effect" not in nitrogen[0].text  # Blocks content stays out
    assert "subplots" in nitrogen[0].text


def test_a_chunk_without_a_heading_gets_the_document_title():
    text = "Bare notes with no heading at all, just a paragraph.\n\nAnd a second one."
    chunks = chunk.chunk_markdown(text, "notes.md")
    assert len(chunks) == 1
    assert chunks[0].text.startswith("# notes")
    assert chunks[0].heading == "notes"


def test_chunks_hit_the_size_target_and_overlap():
    line = "The same moderate sentence about oats and nitrogen and blocks."
    text = "# Trial\n\n" + "\n\n".join(line for _ in range(40))  # ~2.4k chars in paragraphs
    chunks = chunk.chunk_markdown(text, "big.md")
    assert len(chunks) > 1
    for piece in chunks:
        assert len(piece.text) <= chunk.TARGET + len(line) + 120
    # The second chunk opens with text the first one also carries.
    first_body = chunks[0].text.split("\n\n", 1)[1]
    assert first_body[-40:] in chunks[1].text


# --- index and search -------------------------------------------------------


def _marker_docs() -> list[rag_index.Document]:
    markers = ("alpha", "beta", "gamma")
    return [
        rag_index.Document(
            source=f"{word}.md", text=f"# {word}\n\nThe {word} chapter. " + f"{word} " * 30
        )
        for word in markers
    ]


def test_cosine_topk_orders_known_vectors():
    index = rag_index.build_index(_marker_docs(), fake_embed)
    hits = rag_index.search(index, "beta", fake_embed, k=3)
    assert next(h.source for h in hits) == "beta.md"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].id == "beta.md#0"
    assert hits[0].heading == "beta"


def test_search_on_an_empty_index_returns_nothing():
    index = rag_index.build_index([], fake_embed)
    assert rag_index.search(index, "anything", fake_embed, k=5) == []


def test_save_and_load_round_trips_through_parquet_and_meta(tmp_path):
    index = rag_index.build_index(_marker_docs(), fake_embed)
    path = rag_index.save_index(index, tmp_path / "idx.parquet")
    sidecar = tmp_path / "idx.parquet.meta.json"
    assert sidecar.exists()
    assert '"model": "fake:bag-of-words"' in sidecar.read_text(encoding="utf-8")

    loaded = rag_index.load_index(path)
    assert [c.text for c in loaded.chunks] == [c.text for c in index.chunks]
    original = rag_index.search(index, "gamma", fake_embed, k=3)
    reloaded = rag_index.search(loaded, "gamma", fake_embed, k=3)
    assert [h.id for h in original] == [h.id for h in reloaded]


def test_save_index_escapes_a_quote_in_the_path(tmp_path):
    index = rag_index.build_index(_marker_docs(), fake_embed)
    saved = rag_index.save_index(index, tmp_path / "bo'ok.parquet")
    loaded = rag_index.load_index(saved)
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]


# --- embeddings -------------------------------------------------------------


def test_resolve_follows_the_env_switch(monkeypatch):
    for name in ("AGENTIC_STATS_EMBED", "AGENTIC_STATS_EMBED_API_KEY", "AGENTIC_STATS_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    # Stub the backends, so the test exercises only the env-switch logic.
    def stub(model_name):
        def fn(texts):
            raise AssertionError("resolve must not embed")

        fn.model = model_name
        return fn

    monkeypatch.setattr(embed, "local_embed", lambda: stub("local:stubbed"))
    monkeypatch.setattr(embed, "presets_embed", lambda: stub("presets:stubbed"))

    monkeypatch.setenv("AGENTIC_STATS_EMBED", "local")
    assert embed.resolve().model == "local:stubbed"

    monkeypatch.setenv("AGENTIC_STATS_EMBED", "presets")
    assert embed.resolve().model == "presets:stubbed"

    # No override: presets when a key exists, local when it does not.
    monkeypatch.delenv("AGENTIC_STATS_EMBED")
    monkeypatch.setenv("AGENTIC_STATS_LLM_API_KEY", "sk")
    assert embed.resolve().model == "presets:stubbed"

    monkeypatch.delenv("AGENTIC_STATS_LLM_API_KEY")
    assert embed.resolve().model == "local:stubbed"

    monkeypatch.setenv("AGENTIC_STATS_EMBED_API_KEY", "sk")
    assert embed.resolve().model == "presets:stubbed"

    with pytest.raises(ValueError, match="AGENTIC_STATS_EMBED"):
        monkeypatch.setenv("AGENTIC_STATS_EMBED", "banana")
        embed.resolve()


class _FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        # Out of order on purpose: the client must sort by index.
        return {
            "data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
        }


class _FakeClient:
    calls: ClassVar[list[tuple[str, dict, dict]]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def post(self, url, json=None):
        _FakeClient.calls.append((url, json, self.kwargs.get("headers", {})))
        return _FakeResponse()


def test_presets_embed_request_shape(monkeypatch):
    monkeypatch.setattr(embed.httpx, "Client", _FakeClient)
    _FakeClient.calls.clear()

    fn = embed.presets_embed(
        base_url="https://x.example/v1", model="text-embedding-3-small", api_key="sk"
    )
    vectors = fn(["hello", "world"])

    url, payload, headers = _FakeClient.calls[0]
    assert url == "/embeddings"
    assert payload == {"model": "text-embedding-3-small", "input": ["hello", "world"]}
    assert headers == {"Authorization": "Bearer sk"}
    assert vectors.shape == (2, 2)
    assert vectors[0].tolist() == [1.0, 0.0]  # index 0, not response order


def test_presets_embed_needs_a_model():
    with pytest.raises(ValueError, match="AGENTIC_STATS_EMBED_MODEL"):
        embed.presets_embed(model="")


def test_local_embed_names_the_missing_extra():
    with pytest.raises(RuntimeError, match="rag extra"):
        # fastembed is deliberately not installed in the dev environment.
        embed.local_embed()


# --- registry tools ---------------------------------------------------------


def _spec_call(name: str, **arguments: object):
    """Round-trip the arguments through the derived request model, like an agent does."""
    tool = registry.get(name)
    request = tool.input_model.model_validate(arguments)
    return tool.handler(**request.model_dump())


def test_search_docs_round_trip(book, monkeypatch):
    monkeypatch.setattr(rag_tools, "MIN_RELEVANCE", 0.1)
    result = _spec_call("search_docs", query="How are nitrogen rates applied?", k=2)
    assert result.n_hits == 2
    assert result.hits[0].source == "oats.md"
    assert result.citations[0].startswith("oats.md#")
    assert book.exists()  # the lazy build cached the corpus index
    assert "n_chunks" in book.with_name(book.name + ".meta.json").read_text(encoding="utf-8")


def test_ask_docs_answers_with_citations(book, monkeypatch):
    monkeypatch.setattr(rag_tools, "MIN_RELEVANCE", 0.1)
    result = _spec_call("ask_docs", question="How are nitrogen rates applied?")
    assert result.cannot_say == ""
    assert "[oats.md#" in result.answer
    assert all(citation.source == "oats.md" for citation in result.citations)
    assert all(citation.chunk_id in result.answer for citation in result.citations)


def test_ask_docs_refuses_when_retrieval_finds_nothing_relevant(book, monkeypatch):
    monkeypatch.setattr(rag_tools, "MIN_RELEVANCE", 0.99)
    result = _spec_call("ask_docs", question="How are nitrogen rates applied?")
    assert result.answer == ""
    assert result.citations == []
    assert "nothing relevant" in result.cannot_say


def test_index_documents_builds_a_session_local_index(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_STATS_DOCS_DIR", str(tmp_path / "absent"))  # no book corpus
    monkeypatch.setattr(rag_tools, "_embed", lambda: fake_embed)
    rag_tools.reset_session()
    try:
        result = _spec_call(
            "index_documents",
            documents=["Magnesium moves with the transpiration stream.", "Second document text."],
            names=None,
        )
        assert result.n_documents == 2
        assert result.names == ["document-1.md", "document-2.md"]
        assert result.dim == DIM
        assert result.model == "fake:bag-of-words"

        found = _spec_call("search_docs", query="magnesium", k=1)
        assert found.hits[0].source == "document-1.md"

        with pytest.raises(Exception, match="one name per document"):
            _spec_call("index_documents", documents=["a", "b"], names=["only-one.md"])
        with pytest.raises(Exception, match="at least one document"):
            _spec_call("index_documents", documents=[], names=None)
    finally:
        rag_tools.reset_session()


def test_search_docs_names_the_remedy_when_nothing_is_indexed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_STATS_DOCS_DIR", str(tmp_path / "absent"))
    monkeypatch.setattr(rag_tools, "_embed", lambda: fake_embed)
    rag_tools.reset_session()
    try:
        with pytest.raises(rag_tools.ToolError, match="index_documents"):
            _spec_call("search_docs", query="anything")
    finally:
        rag_tools.reset_session()


def test_tool_and_attachment_documents_merge(tmp_path, monkeypatch):
    """One session map: an attachment upload keeps tool-indexed docs and vice versa."""
    monkeypatch.setenv("AGENTIC_STATS_DOCS_DIR", str(tmp_path / "absent"))  # no book corpus
    monkeypatch.setattr(rag_tools, "_embed", lambda: fake_embed)
    rag_tools.reset_session()
    docs = rag_tools.SessionDocs()
    try:
        with rag_tools.use_docs(docs):
            _spec_call(
                "index_documents",
                documents=["Magnesium travels in the xylem stream."],
                names=["tool.md"],
            )
            # The attachment path indexes through the same merge, like _index_attachments.
            rag_tools.index_documents(["Zinc deficiency yellows the leaves."], ["upload.md"])
            found = _spec_call("search_docs", query="magnesium zinc", k=5)
            sources = {h.source for h in found.hits}
            assert {"tool.md", "upload.md"} <= sources
            assert all(h.chunk_id.startswith("user:") for h in found.hits)
            assert "user:upload.md#0" in {h.chunk_id for h in found.hits}

            # A later tool call re-indexes; the attachment document survives.
            _spec_call(
                "index_documents",
                documents=["More magnesium notes on translocation."],
                names=["tool-2.md"],
            )
            found = _spec_call("search_docs", query="zinc leaves", k=5)
            assert "upload.md" in {h.source for h in found.hits}
    finally:
        rag_tools.reset_session()


def test_two_sessions_do_not_share_indexed_documents(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_STATS_DOCS_DIR", str(tmp_path / "absent"))  # no book corpus
    monkeypatch.setattr(rag_tools, "_embed", lambda: fake_embed)
    rag_tools.reset_session()
    first, second = rag_tools.SessionDocs(), rag_tools.SessionDocs()
    try:
        with rag_tools.use_docs(first):
            _spec_call(
                "index_documents",
                documents=["Magnesium travels in the xylem stream."],
                names=["first.md"],
            )
        with rag_tools.use_docs(second):
            _spec_call(
                "index_documents",
                documents=["Zinc deficiency yellows the leaves."],
                names=["second.md"],
            )
            found = _spec_call("search_docs", query="magnesium", k=5)
            assert {h.source for h in found.hits} == {"second.md"}
        with rag_tools.use_docs(first):
            found = _spec_call("search_docs", query="magnesium", k=1)
            assert found.hits[0].source == "first.md"
    finally:
        rag_tools.reset_session()


def test_a_cached_book_from_another_embedder_is_rebuilt(book, monkeypatch):
    _spec_call("search_docs", query="How are nitrogen rates applied?", k=1)  # builds the cache
    meta = book.with_name(book.name + ".meta.json")
    assert '"model": "fake:bag-of-words"' in meta.read_text(encoding="utf-8")

    def other_embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[i, zlib.crc32(word.encode()) % 8] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)

    other_embed.model = "fake:other"  # a different dimension: the cache cannot be reused
    monkeypatch.setattr(rag_tools, "_embed", lambda: other_embed)
    rag_tools.reset_session()
    result = _spec_call("search_docs", query="How are nitrogen rates applied?", k=1)
    assert result.n_hits == 1  # rebuilt instead of a dimension-mismatch matmul
    assert '"model": "fake:other"' in meta.read_text(encoding="utf-8")
