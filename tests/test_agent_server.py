"""The agent panel's guards, verify summary, and tool cards.

A public instance cannot let one conversation run forever or three slow streams
starve it, and every tool result has to carry its display card. All of that is
plain Python around the turn, so it is tested without a model and without a
browser.
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from typing import Any

import pytest
from chatlas import ContentToolResult

from stats import verify
from stats.session import SessionStore
from transports import llm
from web.panels.agent import server as agent_server


def test_slots_refuse_rather_than_queue():
    async def main() -> None:
        await agent_server._SLOTS.acquire()
        await agent_server._SLOTS.acquire()
        await agent_server._SLOTS.acquire()
        assert await agent_server._take_slot() is False
        agent_server._SLOTS.release()
        assert await agent_server._take_slot() is True
        agent_server._SLOTS.release()

    asyncio.run(main())


def test_the_budget_and_the_cap_are_named_constants():
    assert agent_server.TURNS_PER_SESSION == 20
    assert agent_server.MAX_CONCURRENT_TURNS == 3


class _FakeClient:
    """Stands in for the chatlas client: records what shinychat would pass."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def stream_async(self, user_input: str, *contents: Any, **kwargs: Any) -> Any:
        # Same shape as chatlas: a coroutine returning the async iterable.
        self.calls.append(((user_input, *contents), kwargs))

        async def stream() -> Any:
            yield "a"
            yield "b"

        return stream()


def _guarded(inner: _FakeClient, turns: int = 0) -> agent_server.GuardedClient:
    return agent_server.GuardedClient(
        inner,
        {"turns": turns, "key": None, "trace": [], "docs": agent_server.rag.SessionDocs()},
        SessionStore,
    )


@pytest.fixture(autouse=True)
def fresh_slots():
    """The module semaphore binds to one loop; give each test its own."""
    agent_server._SLOTS = asyncio.Semaphore(agent_server.MAX_CONCURRENT_TURNS)
    yield


async def _drain(response: Any) -> list[Any]:
    return [chunk async for chunk in response]


def test_guard_refuses_when_slots_are_busy():
    async def main() -> None:
        inner = _FakeClient()
        guarded = _guarded(inner)
        for _ in range(agent_server.MAX_CONCURRENT_TURNS):
            await agent_server._SLOTS.acquire()
        with pytest.raises(ValueError, match=re.escape(agent_server.SLOT_BUSY)):
            await guarded.stream_async("hello", content="all")
        agent_server._SLOTS.release()

    asyncio.run(main())


def test_guard_refuses_on_budget_overrun_and_frees_the_slot():
    async def main() -> None:
        inner = _FakeClient()
        guarded = _guarded(inner, turns=agent_server.TURNS_PER_SESSION)
        with pytest.raises(ValueError, match=re.escape(agent_server.TURN_BUDGET)):
            await guarded.stream_async("hello", content="all")
        # The refused turn must not hold its slot.
        assert await agent_server._take_slot() is True
        agent_server._SLOTS.release()

    asyncio.run(main())


def test_guard_forwards_the_stream_and_releases_the_slot():
    async def main() -> None:
        inner = _FakeClient()
        state = {"turns": 0, "key": None, "trace": [], "docs": agent_server.rag.SessionDocs()}
        guarded = agent_server.GuardedClient(inner, state, SessionStore)
        response = await guarded.stream_async("hello", content="all")
        assert await _drain(response) == ["a", "b"]
        assert inner.calls == [(("hello",), {"content": "all"})]
        assert state["turns"] == 1
        # The slot is held while the stream is consumed, freed afterwards.
        assert await agent_server._take_slot() is True
        agent_server._SLOTS.release()

    asyncio.run(main())


def test_the_budget_counter_survives_a_client_rebuild():
    async def main() -> None:
        state: dict[str, Any] = {
            "turns": 0,
            "key": None,
            "trace": [],
            "docs": agent_server.rag.SessionDocs(),
        }
        first = agent_server.GuardedClient(_FakeClient(), state, SessionStore)
        second = agent_server.GuardedClient(_FakeClient(), state, SessionStore)
        first._state["turns"] = agent_server.TURNS_PER_SESSION
        with pytest.raises(ValueError):
            await second.stream_async("hello", content="all")

    asyncio.run(main())


def test_verify_message_is_verdict_statement_notes():
    report = verify.VerificationReport(
        verdict="review",
        statement="2 figures traced to stored results, 1 did not.",
        notes=["Ungrounded figures: ['7.2']. Quote a stored value."],
    )
    message = agent_server._verify_message(report)
    assert message.splitlines()[0] == "⚠️ Verify: 2 figures traced to stored results, 1 did not."
    assert message.splitlines()[1] == "- Ungrounded figures: ['7.2']. Quote a stored value."
    clean = agent_server._verify_message(
        verify.VerificationReport(verdict="clean", statement="1 figure traced to stored results.")
    )
    assert clean.startswith("✅ Verify:")


async def _tool_extra(handler: Any, **kwargs: Any) -> ContentToolResult:
    return await llm._as_tool(handler)(**kwargs)


class _Att:
    """The attachment fields shinychat passes to on_user_submit."""

    def __init__(self, name: str, mime: str, text: str) -> None:
        self.name = name
        self.mime = mime
        self.data_url = f"data:{mime};base64," + base64.b64encode(text.encode("utf-8")).decode(
            "ascii"
        )


_INDEXED_NOTE = "Indexed 2 document(s): a.md, b.csv \u2014 ask_docs now reads them."
_FAIL_NOTE = agent_server._FAIL_NOTE


def test_attachments_are_decoded_and_indexed_once(monkeypatch: pytest.MonkeyPatch):
    async def main() -> None:
        calls: list[tuple[list[str], list[str]]] = []

        def fake(documents: list[str], names: list[str] | None = None) -> Any:
            calls.append((list(documents), list(names)))
            docs.docs.update(zip(names, documents, strict=True))  # what the real commit does

        monkeypatch.setattr(agent_server.rag, "index_documents", fake)
        docs = agent_server.rag.SessionDocs()
        notes = await agent_server._index_attachments(
            [_Att("a.md", "text/markdown", "# hi"), _Att("b.csv", "text/csv", "x,y")], docs
        )
        assert calls == [(["# hi", "x,y"], ["a.md", "b.csv"])]
        assert notes == [_INDEXED_NOTE]
        assert docs.docs == {"a.md": "# hi", "b.csv": "x,y"}

    asyncio.run(main())


def test_a_duplicate_attachment_is_not_indexed_again(monkeypatch: pytest.MonkeyPatch):
    async def main() -> None:
        calls: list[Any] = []

        def fake(documents: list[str], names: list[str] | None = None) -> Any:
            calls.append((list(documents), list(names)))
            docs.docs.update(zip(names, documents, strict=True))

        monkeypatch.setattr(agent_server.rag, "index_documents", fake)
        docs = agent_server.rag.SessionDocs(docs={"a.md": "# hi", "b.csv": "x,y"})
        notes = await agent_server._index_attachments([_Att("a.md", "text/markdown", "# hi")], docs)
        assert calls == []
        assert notes == []
        # A new upload indexes only the new names: index_documents merges into
        # the session map, so the earlier documents stay searchable.
        await agent_server._index_attachments([_Att("c.txt", "text/plain", "third")], docs)
        assert calls[-1] == (["third"], ["c.txt"])

    asyncio.run(main())


def test_pdfs_and_images_are_skipped_with_a_note(monkeypatch: pytest.MonkeyPatch):
    async def main() -> None:
        calls: list[Any] = []

        def fake(documents: list[str], names: list[str] | None = None) -> Any:
            calls.append(documents)

        monkeypatch.setattr(agent_server.rag, "index_documents", fake)
        notes = await agent_server._index_attachments(
            [_Att("paper.pdf", "application/pdf", "%PDF-"), _Att("pic.png", "image/png", "x")],
            agent_server.rag.SessionDocs(),
        )
        assert calls == []
        assert notes == ["Skipped paper.pdf, pic.png \u2014 only text and CSV files are indexed."]

    asyncio.run(main())


def test_an_indexing_failure_appends_a_note_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
):
    async def main() -> None:
        def broken(documents: list[str], names: list[str] | None = None) -> Any:
            raise RuntimeError("no embeddings")

        monkeypatch.setattr(agent_server.rag, "index_documents", broken)
        docs = agent_server.rag.SessionDocs()
        notes = await agent_server._index_attachments([_Att("a.md", "text/markdown", "hi")], docs)
        assert notes == [_FAIL_NOTE]
        # The failed upload stays unindexed, so a retry can pick it up.
        assert docs.docs == {}

    asyncio.run(main())


def test_sensitivity_result_carries_a_framed_card():
    def sensitivity(result_id: str) -> Any:
        class _R:
            def model_dump(self) -> dict[str, Any]:
                return {
                    "checks": [{"status": "stable"}, {"status": "flipped"}, {"status": "stable"}],
                    "stable_share": 0.6667,
                }

        return _R()

    extra = asyncio.run(_tool_extra(sensitivity, result_id="r1")).extra
    display = extra["display"]
    assert display.title == "Sensitivity check"
    assert display.value_preview == "2/3 checks stable"
    assert display.open_style == "framed"


def test_plot_result_carries_a_spec_label():
    def plot(kind: str, x: str, y: str) -> Any:
        class _R:
            def model_dump(self) -> dict[str, Any]:
                return {"kind": "box", "x": "variety", "y": "height", "color": None}

        return _R()

    display = asyncio.run(llm._as_tool(plot)(kind="box", x="variety", y="height")).extra["display"]
    assert display.title == "Plot"
    assert display.label == "box, variety, height"
    assert display.open_style == "framed"


def test_fit_model_result_previews_the_first_estimate():
    def fit_model(result_id: str) -> Any:
        class _R:
            def model_dump(self) -> dict[str, Any]:
                return {
                    "coefficients": [
                        {
                            "term": "varietyGoldenRain",
                            "estimate": 1.2,
                            "ci_low": 0.4,
                            "ci_high": 2.0,
                        }
                    ]
                }

        return _R()

    display = asyncio.run(_tool_extra(fit_model, result_id="r1")).extra["display"]
    assert display.title == "Fitted model"
    assert display.value_preview == "varietyGoldenRain: 1.2 [0.4, 2.0]"


def test_unknown_tools_get_a_plain_titled_card():
    def analysis_status() -> Any:
        class _R:
            def model_dump(self) -> dict[str, Any]:
                return {"complete": True}

        return _R()

    display = asyncio.run(_tool_extra(analysis_status)).extra["display"]
    assert display.title == "Analysis status"
    assert display.value_preview is None


def test_a_result_without_the_expected_fields_degrades_quietly():
    def sensitivity(result_id: str) -> Any:
        class _R:
            def model_dump(self) -> dict[str, Any]:
                return {}

        return _R()

    display = asyncio.run(_tool_extra(sensitivity, result_id="r1")).extra["display"]
    assert display.title == "Sensitivity check"
    assert display.value_preview is None


def test_tool_handler_runs_off_the_event_loop():
    """The interleaving proof: while the sync handler blocks its thread, the
    loop still turns, so the marker task completes first. With a blocking
    wrapper the handler would finish before `asyncio.sleep` ever resumed.
    """

    async def main() -> None:
        handler_done = asyncio.Event()

        def slow_tool() -> Any:
            time.sleep(0.3)
            handler_done.set()

            class _R:
                def model_dump(self) -> dict[str, Any]:
                    return {}

            return _R()

        task = asyncio.create_task(llm._as_tool(slow_tool)())
        await asyncio.sleep(0.1)
        assert not handler_done.is_set(), "the handler blocked the event loop"
        result = await task
        assert handler_done.is_set()
        assert isinstance(result, ContentToolResult)

    asyncio.run(main())


def test_attachment_names_are_sanitised_in_notes_and_docs(monkeypatch: pytest.MonkeyPatch):
    async def main() -> None:
        calls: list[list[str]] = []

        def fake(documents: list[str], names: list[str] | None = None) -> Any:
            calls.append(list(names))
            docs.docs.update(zip(names, documents, strict=True))

        monkeypatch.setattr(agent_server.rag, "index_documents", fake)
        docs = agent_server.rag.SessionDocs()
        notes = await agent_server._index_attachments(
            [_Att("<img src=x>.md", "text/markdown", "# hi")], docs
        )
        # No raw brackets reach the chat message...
        assert notes == ["Indexed 1 document(s): img src=x.md \u2014 ask_docs now reads them."]
        # ...and the stored name and citation id carry the sanitised form.
        assert calls == [["img src=x.md"]]
        assert docs.docs == {"img src=x.md": "# hi"}

    asyncio.run(main())
