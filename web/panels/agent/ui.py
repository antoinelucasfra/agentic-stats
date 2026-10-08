"""Agent panel layout: endpoint settings on the left, the chat on the right."""

from __future__ import annotations

from shiny import module, ui

from transports import llm

PRESET = llm.preset_default()
_DEFAULTS = llm.preset_by_id(PRESET)


@module.ui
def agent_ui() -> ui.NavPanel:
    return ui.nav_panel(
        "Agent",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "preset",
                    "Preset",
                    choices={preset.id: preset.label for preset in llm.PRESETS},
                    selected=PRESET,
                ),
                ui.input_text("model", "Model", value=_DEFAULTS.model),
                ui.input_text("base_url", "Endpoint base URL", value=_DEFAULTS.base_url),
                ui.input_password(
                    "api_key",
                    "API key",
                    value="",
                    placeholder="blank for the Demo and Ollama presets",
                ),
                ui.tags.p(
                    "Demo runs on the server's key: nothing to enter. Ollama runs "
                    "locally with no key. The other presets use the key you type, "
                    "kept in this session only. Changing preset or endpoint starts "
                    "a fresh model context.",
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
