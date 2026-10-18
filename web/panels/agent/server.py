"""Agent panel server: build a chatlas client per endpoint, stream replies.

The panel shares the connection's store with the Data and Playground panels, so
a table the reader loaded is the table the agent analyses, and a table the
agent fetched is the table the Playground form offers. The tools read the
ledger, so without the shared store two chats would also share a design stage
and a result id.

Three guards keep a public instance alive. A turn budget stops a runaway
conversation: twenty turns, then the reader reloads for a fresh session. A
concurrency cap refuses a turn instead of queueing it, so three slow streams
cannot starve the instance. And the finished text goes through the verify pass
before the turn is shown as done: a figure with no stored result, an effect
claim resting on a descriptive one, or causal language without a causal design
lands in the transcript as a warning, not as an answer.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any

from shiny import Inputs, Outputs, Session, module, reactive, ui

from stats import verify
from stats.session import use_session
from transports import llm

TURNS_PER_SESSION = 20
MAX_CONCURRENT_TURNS = 3

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


@module.server
def agent_server(input: Inputs, output: Outputs, session: Session, store: reactive.Value) -> None:
    chat = ui.Chat(id="chat")
    state: dict[str, Any] = {"key": None, "client": None, "trace": [], "turns": 0}

    def client() -> Any:
        with reactive.isolate():
            endpoint = llm.resolve(input.preset(), input.model(), input.base_url(), input.api_key())
        if state["key"] != endpoint:
            state["key"] = endpoint
            state["client"] = llm.build_client(
                model=endpoint[0],
                base_url=endpoint[1],
                api_key=endpoint[2] or None,
                trace=state["trace"],
            )
        return state["client"]

    @reactive.effect
    @reactive.event(input.preset)
    def _apply_preset() -> None:
        preset = llm.preset_by_id(input.preset())
        ui.update_text("model", value=preset.model)
        ui.update_text("base_url", value=preset.base_url)

    @chat.on_user_submit
    async def _(user_input: str) -> None:
        if not await _take_slot():
            await chat.append_message(
                "All three agent slots are busy. Wait for a turn to finish and retry."
            )
            return
        try:
            state["turns"] += 1
            if state["turns"] > TURNS_PER_SESSION:
                await chat.append_message(
                    f"{TURNS_PER_SESSION} turns is the session budget. Reload the page "
                    "to start a fresh analysis."
                )
                return
            state["trace"].clear()
            with use_session(store()):
                response = await client().stream_async(user_input, content="all")
                await chat.append_message_stream(response)
                text, current, usage = response.content, store(), _usage(client())
                report = verify.verify_answer(text, current)
            for event in state["trace"]:
                mark = "❌" if event.is_error else "🔧"
                await chat.append_message(
                    f"{mark} `{event.name}({json.dumps(event.arguments)})` → {event.summary}"
                )
            if usage:
                await chat.append_message(f"Tokens used this turn: `{usage}`")
            mark = "✅" if report.verdict == "clean" else "⚠️"
            await chat.append_message(f"{mark} Verify: {report.statement}")
            for note in report.notes:
                await chat.append_message(f"- {note}")
        except Exception as exc:
            traceback.print_exc()
            ui.notification_show(f"The model endpoint failed: {exc}", type="error")
            await chat.append_message(f"The model endpoint failed: `{exc}`")
            return
        finally:
            _SLOTS.release()
