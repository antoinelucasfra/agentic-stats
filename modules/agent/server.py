"""Agent panel server: build a chatlas client per endpoint, stream replies."""

from __future__ import annotations

from typing import Any

from shiny import Inputs, Outputs, Session, module, reactive, ui

import llm


@module.server
def agent_server(input: Inputs, output: Outputs, session: Session) -> None:
    chat = ui.Chat(id="chat")
    state: dict[str, Any] = {"key": None, "client": None}

    def client() -> Any:
        with reactive.isolate():
            key = (input.model(), input.base_url(), input.api_key())
        if state["key"] != key:
            state["key"] = key
            state["client"] = llm.build_client(
                model=key[0], base_url=key[1], api_key=key[2] or None
            )
        return state["client"]

    @chat.on_user_submit
    async def _(user_input: str) -> None:
        try:
            response = await client().stream_async(user_input, content="all")
            await chat.append_message_stream(response)
        except Exception as exc:
            ui.notification_show(f"The model endpoint failed: {exc}", type="error")
            await chat.append_message(f"The model endpoint failed: `{exc}`")
