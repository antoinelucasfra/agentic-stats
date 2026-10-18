"""Agent panel server: shinychat drives the stream; guards wrap the client.

The panel shares the connection's store with the Data and Playground panels, so
a table the reader loaded is the table the agent analyses, and a table the
agent fetched is the table the Playground form offers. The tools read the
ledger, so without the shared store two chats would also share a design stage
and a result id.

With `client=` passed to shinychat's `Chat`, shinychat itself calls
`stream_async` on the client, renders tool calls as activity rows, and wires
the stop button. Our own `on_user_submit` handler runs alongside but cannot
veto that stream, so the guards live in `GuardedClient`: a thin wrapper whose
`stream_async` refuses a turn before the stream starts, and shinychat's
`on_error="auto"` renders the refusal as a chat error.

Three guards keep a public instance alive. A turn budget stops a runaway
conversation: twenty turns, then the reader reloads for a fresh session. A
concurrency cap refuses a turn instead of queueing it, so three slow streams
cannot starve the instance. And after a stream settles, a verify pass checks
the finished text: a figure with no stored result, an effect claim resting on
a descriptive one, or causal language without a causal design lands in the
transcript as a warning, not as an answer.
"""

from __future__ import annotations

import asyncio
from typing import Any

from shiny import Inputs, Outputs, Session, module, reactive, ui
from shinychat import Chat

from stats import verify
from stats.session import use_session
from transports import llm

TURNS_PER_SESSION = 20
MAX_CONCURRENT_TURNS = 3

SLOT_BUSY = "All three agent slots are busy. Wait for a turn to finish and retry."
TURN_BUDGET = (
    f"{TURNS_PER_SESSION} turns is the session budget. Reload the page to start a fresh analysis."
)

GREETING = (
    "Does the oat variety matter, once we account for block variation? Ask me \u2014 "
    "I will frame the question, fit, try to break my own conclusion, and verify "
    "every number."
)

_SLOTS: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_TURNS)


async def _take_slot() -> bool:
    """A concurrency slot without queueing: refuse rather than stack turns."""
    try:
        await asyncio.wait_for(_SLOTS.acquire(), timeout=0.05)
    except TimeoutError:
        return False
    return True


def _usage(client: Any) -> str:
    """Token usage for the turn, or nothing when the provider hides it."""
    try:
        return str(client.token_usage())[:200]
    except Exception:
        return ""


class _RefusedEndpoint:
    """Stands in for a client whose endpoint cannot resolve (no key, no model).

    shinychat calls `stream_async` itself and `Chat(client=None)` would leave
    the chat dead, so an unresolvable endpoint gets a client that raises the
    reason on every turn; `on_error="auto"` renders it in the chat.
    """

    def __init__(self, message: str) -> None:
        self.message = message

    async def stream_async(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError(self.message)


class GuardedClient:
    """Slot and turn guards around the chatlas client shinychat drives.

    The budget counter lives in `state`, not on the client, so a client rebuild
    on endpoint change keeps the count. `stream_async` raises before the stream
    starts when a guard trips; shinychat renders the ValueError in the chat.
    """

    def __init__(self, inner: Any, state: dict[str, Any], store: reactive.Value) -> None:
        self.inner = inner
        self._state = state
        self._store = store

    async def stream_async(self, user_input: str, *contents: Any, **kwargs: Any) -> Any:
        if not await _take_slot():
            raise ValueError(SLOT_BUSY)
        self._state["turns"] += 1
        if self._state["turns"] > TURNS_PER_SESSION:
            _SLOTS.release()
            raise ValueError(TURN_BUDGET)
        inner, store = self.inner, self._store

        async def stream() -> Any:
            # The slot is held until the stream is consumed, not merely
            # started: shinychat iterates the generator after this returns.
            try:
                with use_session(store()):
                    response = await inner.stream_async(user_input, *contents, **kwargs)
                    async for chunk in response:
                        yield chunk
            finally:
                _SLOTS.release()

        return stream()


def _verify_message(report: verify.VerificationReport) -> str:
    """One markdown block: verdict, statement, and the notes that back it."""
    mark = "✅" if report.verdict == "clean" else "⚠️"
    lines = [f"{mark} Verify: {report.statement}"]
    lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines)


@module.server
def agent_server(input: Inputs, output: Outputs, session: Session, store: reactive.Value) -> None:
    state: dict[str, Any] = {"key": None, "client": None, "trace": [], "turns": 0}

    def build(preset_id: str, model: str, base_url: str, api_key: str) -> GuardedClient:
        """A guarded client for the given endpoint, refusing turns if it cannot resolve."""
        try:
            endpoint = llm.resolve(preset_id, model, base_url, api_key)
        except ValueError as exc:
            state["key"] = None
            state["client"] = GuardedClient(_RefusedEndpoint(str(exc)), state, store)
            return state["client"]
        if state["key"] != endpoint:
            state["key"] = endpoint
            state["client"] = GuardedClient(
                llm.build_client(
                    model=endpoint[0],
                    base_url=endpoint[1],
                    api_key=endpoint[2] or None,
                    trace=state["trace"],
                ),
                state,
                store,
            )
        return state["client"]

    def build_current() -> GuardedClient:
        with reactive.isolate():
            return build(input.preset(), input.model(), input.base_url(), input.api_key())

    # shinychat needs a client at construction, but module inputs only exist
    # after the first flush. Start from the default preset's static endpoint;
    # `_swap_client` corrects to the real inputs once they are readable.
    _default = llm.preset_by_id(llm.preset_default())
    chat = Chat(
        id="chat",
        client=build(llm.preset_default(), _default.model, _default.base_url, ""),
        greeting=GREETING,
        history=False,
    )

    @reactive.effect
    @reactive.event(input.preset)
    def _apply_preset() -> None:
        preset = llm.preset_by_id(input.preset())
        ui.update_text("model", value=preset.model)
        ui.update_text("base_url", value=preset.base_url)

    @reactive.effect
    def _swap_client() -> None:
        client = build_current()
        if chat.client is not None:
            # Not sync: a changed endpoint starts a fresh model context.
            chat.client.set(client, sync=False)

    @chat.on_user_submit
    async def _on_submit(user_input: str, attachments: list[Any]) -> None:
        """Placeholder for the rag/ integration: index attachments, if any."""

    @reactive.effect
    async def _verify_turn() -> None:
        status = chat.latest_message_stream.status()
        if status != "success":
            return
        text = chat.latest_message_stream.result()
        if not text:
            return
        report = verify.verify_answer(text, store())
        client = state["client"]
        usage = _usage(client.inner) if client is not None else ""
        await chat.append_message(_verify_message(report))
        if usage:
            await chat.append_message(f"Tokens used this turn: `{usage}`")
