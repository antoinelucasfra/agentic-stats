"""Shiny for Python app over the same registry the MCP server and CLI use.

    uv run shiny run --host 127.0.0.1 --port 8766 app.py

Three panels. Overview paints the six answers the README quotes, computed once
at import. Playground calls any registered tool with a form derived from its
schema, against the shipped dataset or an uploaded CSV. Agent hands the same
tools to a chat model.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from pydantic import ValidationError
from shiny import App, Inputs, Outputs, Session, reactive, render, run_app, ui

from agentic_stats import forms, llm, registry, stats_tools, views
from agentic_stats.stats_tools import DatasetDescription, ToolError

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

RESPONSE = "assay_signal"

# A question, the tool that answers it, and the arguments it is called with.
# The Overview renders these as cards and hands the same arguments to the
# Playground when a reader clicks through.
CARDS: list[dict[str, Any]] = [
    {
        "id": "formulation",
        "question": "Does the formulation change the assay signal?",
        "tool": "anova_effect",
        "args": {"response": RESPONSE, "factor": "formulation"},
    },
    {
        "id": "contrasts",
        "question": "How far apart are the formulations, and how sure are we?",
        "tool": "marginal_means",
        "args": {"response": RESPONSE, "factors": ["formulation"], "covariates": ["dose"]},
    },
    {
        "id": "batch",
        "question": "Is batch-to-batch variation real, or noise?",
        "tool": "fit_mixed_model",
        "args": {"response": RESPONSE, "fixed_effects": ["formulation", "dose"], "group": "batch"},
    },
    {
        "id": "assumptions",
        "question": "Can we trust the model behind those p-values?",
        "tool": "check_assumptions",
        "args": {"response": RESPONSE, "fixed_effects": ["formulation", "dose"]},
    },
    {
        "id": "power",
        "question": "Should we run more pilot batches?",
        "tool": "power_analysis",
        "args": {"response": RESPONSE, "factor": "formulation"},
    },
    {
        "id": "dose",
        "question": "Where does the dose response peak?",
        "tool": "dose_response",
        "args": {"response": RESPONSE, "dose": "dose", "model": "quadratic"},
    },
]

TOOL_CHOICES = {spec.name: spec.name for spec in registry.TOOLS}

# Computed once at import: all seven tools together cost well under a second.
DATASET: DatasetDescription = registry.get("describe_dataset").handler()
OVERVIEW: dict[str, Any] = {
    card["id"]: registry.get(card["tool"]).handler(**card["args"]) for card in CARDS
}

DEFAULT_MODEL, DEFAULT_BASE_URL, DEFAULT_API_KEY = llm.settings()


# --- panels -----------------------------------------------------------------


def _overview_panel() -> ui.NavPanel:
    cards = [
        ui.card(
            ui.card_header(card["question"]),
            ui.tags.p(ui.tags.code(card["tool"]), class_="mb-2"),
            ui.output_ui(f"ov_{card['id']}"),
            ui.input_action_button(
                f"open_{card['id']}",
                "Open in playground",
                class_="btn-sm btn-outline-secondary",
            ),
            height="100%",
        )
        for card in CARDS
    ]
    return ui.nav_panel(
        "Overview",
        ui.output_ui("ov_dataset"),
        ui.layout_column_wrap(*cards, width=1 / 2),
        value="overview",
    )


def _playground_panel() -> ui.NavPanel:
    return ui.nav_panel(
        "Playground",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("tool", "Tool", choices=TOOL_CHOICES),
                ui.output_ui("tool_form"),
                ui.input_action_button("run", "Run", class_="btn-primary"),
                ui.hr(),
                ui.input_file("csv", "Analyse your own CSV", accept=[".csv"]),
                ui.tags.p(
                    "Needs the same columns as the shipped dataset. Read on the server "
                    "and never written to disk.",
                    class_="text-muted small",
                ),
                width=380,
                open="always",
            ),
            ui.output_ui("result"),
            fillable=True,
        ),
        value="playground",
    )


def _agent_panel() -> ui.NavPanel:
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


app_ui = ui.page_navbar(
    _overview_panel(),
    _playground_panel(),
    _agent_panel(),
    title="agentic-stats",
    id="nav",
    fillable=True,
)


# --- form fields ------------------------------------------------------------


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _field_input(field: forms.Field, args: dict[str, Any], dataset: DatasetDescription) -> Any:
    label = f"{field.name} *" if field.required else field.name
    value = args.get(field.name)
    if value is None:
        value = (
            field.default if field.default is not None else forms.suggestion(field.name, dataset)
        )

    if field.kind == "enum":
        control = ui.input_select(
            field.name, label, choices=field.choices, selected=str(value) if value else None
        )
    elif field.kind == "boolean":
        control = ui.input_checkbox(field.name, label, value=bool(value))
    elif field.kind in ("number", "integer"):
        control = ui.input_numeric(field.name, label, value=_number(value))
    elif field.kind == "array":
        joined = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
        control = ui.input_text(field.name, label, value=joined)
    else:
        control = ui.input_text(field.name, label, value=str(value or ""))

    children: list[Any] = [control]
    if field.description:
        children.append(ui.tags.small(field.description, class_="text-muted d-block"))
    return ui.tags.div(*children, class_="mb-3")


def _upload_problem(info: dict[str, Any]) -> str | None:
    """Reject an upload before it is read. The browser name is never a path."""
    if not str(info["name"]).lower().endswith(".csv"):
        return "Only .csv files are accepted."
    if info["size"] > MAX_UPLOAD_BYTES:
        return f"Keep the file under {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
    return None


# --- server -----------------------------------------------------------------


def server(input: Inputs, output: Outputs, session: Session) -> None:
    # The arguments a card handed to the Playground, until the tool changes.
    pending = reactive.value(None)
    uploaded = reactive.value(None)

    @reactive.calc
    def frame() -> pd.DataFrame:
        override = uploaded()
        return override if override is not None else stats_tools.load_dataset()

    @reactive.calc
    def description() -> DatasetDescription:
        with stats_tools.use_frame(frame()):
            return stats_tools.describe_dataset()

    @reactive.effect
    @reactive.event(input.csv)
    def _load_upload() -> None:
        files = input.csv()
        if not files:
            uploaded.set(None)
            return
        info = files[0]
        problem = _upload_problem(info)
        if problem:
            ui.notification_show(problem, type="error")
            uploaded.set(None)
            return
        try:
            loaded = pd.read_csv(info["datapath"])
        except Exception as exc:
            ui.notification_show(f"Could not read that CSV: {exc}", type="error")
            uploaded.set(None)
            return
        if loaded.empty:
            ui.notification_show("That CSV has no rows.", type="error")
            uploaded.set(None)
            return
        uploaded.set(loaded)
        ui.notification_show(f"Loaded {info['name']}. Run a tool to analyse it.", type="message")

    # --- overview --------------------------------------------------------
    @render.ui
    def ov_dataset() -> Any:
        return ui.layout_column_wrap(
            ui.value_box("Rows", f"{DATASET.n_rows:,}"),
            ui.value_box("Columns", str(DATASET.n_columns)),
            ui.value_box("Factors", ", ".join(DATASET.factors)),
            ui.value_box("Numeric", ", ".join(DATASET.numeric)),
            width=1 / 4,
        )

    for card in CARDS:
        output(id=f"ov_{card['id']}")(_overview_renderer(card))
        _register_open_handler(card, input, pending)

    # --- playground ------------------------------------------------------
    @reactive.effect
    @reactive.event(input.tool)
    def _clear_pending() -> None:
        entry = pending()
        if entry is not None and entry["tool"] != input.tool():
            pending.set(None)

    @render.ui
    def tool_form() -> Any:
        spec = registry.get(input.tool())
        entry = pending()
        args = entry["args"] if entry and entry["tool"] == spec.name else {}
        dataset = description()
        fields = forms.fields_for(spec.input_schema(), spec.input_model.model_fields)
        return [
            ui.tags.p(spec.description, class_="text-muted small"),
            *[_field_input(field, args, dataset) for field in fields],
        ]

    @reactive.calc
    @reactive.event(input.run)
    def outcome() -> tuple[str, Any]:
        spec = registry.get(input.tool())
        fields = forms.fields_for(spec.input_schema(), spec.input_model.model_fields)
        try:
            raw = {field.name: input[field.name]() for field in fields}
            arguments = forms.arguments(fields, raw)
            with stats_tools.use_frame(frame()):
                return spec.name, spec.handler(**arguments)
        except KeyError:
            return spec.name, ToolError("The form is still loading. Try again.")
        except (ToolError, ValidationError) as exc:
            return spec.name, exc

    @render.ui
    def result() -> Any:
        name, payload = outcome()
        if isinstance(payload, Exception):
            return ui.div(str(payload), class_="alert alert-danger")
        return [
            *views.full_view(name, payload),
            ui.tags.details(
                ui.tags.summary("Raw JSON"),
                ui.tags.pre(json.dumps(payload.model_dump(), indent=2)),
            ),
        ]

    # --- agent -----------------------------------------------------------
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


def _overview_renderer(card: dict[str, Any]) -> Any:
    @render.ui
    def card_output() -> list[Any]:
        return views.card_view(card["tool"], OVERVIEW[card["id"]])

    return card_output


def _register_open_handler(card: dict[str, Any], input: Inputs, pending: Any) -> None:
    """Hand a card's tool and arguments to the Playground, then jump to it."""

    @reactive.effect
    @reactive.event(input[f"open_{card['id']}"])
    def _open() -> None:
        ui.update_navset("nav", selected="playground")
        ui.update_select("tool", selected=card["tool"])
        pending.set({"tool": card["tool"], "args": card["args"]})


app = App(app_ui, server)


def main() -> None:
    """Serve the app without the Shiny CLI, for a wheel install."""
    run_app(
        app,
        host=os.environ.get("AGENTIC_STATS_HOST", "127.0.0.1"),
        port=int(os.environ.get("AGENTIC_STATS_PORT", "8000")),
    )
