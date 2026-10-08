"""Agent panel layout: endpoint settings on the left, the chat on the right."""

from __future__ import annotations

from shiny import module, ui

import llm

DEFAULT_MODEL, DEFAULT_BASE_URL, DEFAULT_API_KEY = llm.settings()


@module.ui
def agent_ui() -> ui.NavPanel:
    return ui.nav_panel(
        "Agent",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_text("model", "Model", value=DEFAULT_MODEL),
                ui.input_text("base_url", "Endpoint base URL", value=DEFAULT_BASE_URL),
                ui.input_password(
                    "api_key",
                    "API key",
                    value=DEFAULT_API_KEY,
                    placeholder="optional, kept in this session only",
                ),
                ui.tags.p(
                    "Any OpenAI-compatible endpoint: Ollama, vLLM, LM Studio, Groq, OpenAI. "
                    "Changing these starts a fresh model context.",
                    class_="text-muted small",
                ),
                width=380,
                open="always",
            ),
            ui.card(ui.chat_ui("chat", height="70vh")),
            fillable=True,
        ),
        value="agent",
    )
