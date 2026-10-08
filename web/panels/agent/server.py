"""Agent panel server: build a chatlas client per endpoint, stream replies."""

from __future__ import annotations

import json
from typing import Any

from shiny import Inputs, Outputs, Session, module, reactive, ui

from transports import llm


@module.server
def agent_server(input: Inputs, output: Outputs, session: Session) -> None:
    chat = ui.Chat(id="chat")
    state: dict[str, Any] = {"key": None, "client": None, "trace": []}

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
        state["trace"].clear()
        try:
            response = await client().stream_async(user_input, content="all")
            await chat.append_message_stream(response)
        except Exception as exc:
            ui.notification_show(f"The model endpoint failed: {exc}", type="error")
            await chat.append_message(f"The model endpoint failed: `{exc}`")
            return
        for event in state["trace"]:
            mark = "❌" if event.is_error else "🔧"
            await chat.append_message(
                f"{mark} `{event.name}({json.dumps(event.arguments)})` → {event.summary}"
            )
