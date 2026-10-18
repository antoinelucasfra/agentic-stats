"""Playground panel server: the form, the upload, and running a tool by hand.

The panel holds one session store for the life of the connection, so the ledger
the tools read is the ledger the reader sees. An upload replaces the store,
because new data invalidates every stage of the analysis that came before it.

The ledger strip above the form is the point of the panel now. A tool that
refuses out of order is a dead end unless the reader can see which call unblocks
it, so `analysis_status` is rendered next to the picker rather than hidden in a
result.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from stats import forms, registry, stats_tools, views, workflow
from stats.report import html_report
from stats.session import use_session
from stats.stats_tools import DatasetDescription, ToolError
from web.panels.playground.helpers import field_input, read_uploaded, upload_problem


@module.server
def playground_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    pending: reactive.Value,
    store: reactive.Value,
) -> None:
    @reactive.effect
    @reactive.event(input.csv)
    def _load_upload() -> None:
        files = input.csv()
        if not files:
            store.set(stats_tools.env_store())
            return
        info = files[0]
        problem = upload_problem(info)
        if problem:
            ui.notification_show(problem, type="error")
            return
        try:
            loaded = read_uploaded(info["datapath"])
        except Exception as exc:
            ui.notification_show(f"Could not read that file: {exc}", type="error")
            return
        if loaded.empty:
            ui.notification_show("That file has no rows.", type="error")
            return
        current = store()
        current.add_table(loaded, name="upload", source=f"upload:{info['name']}", primary=True)
        current.reset_analysis()
        store.set(current.refreshed())
        ui.notification_show(
            f"Loaded {info['name']}. Frame the question, then profile the table.",
            type="message",
        )

    @reactive.calc
    def description() -> DatasetDescription:
        with use_session(store()):
            return stats_tools.describe_dataset()

    @reactive.effect
    @reactive.event(pending)
    def _apply_pending() -> None:
        """A card click arrives as a value; selecting the tool is this module's job."""
        entry = pending()
        if entry is not None:
            ui.update_select("tool", selected=entry["tool"])

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
            *[field_input(field, args, dataset) for field in fields],
        ]

    @reactive.calc
    @reactive.event(input.run)
    def outcome() -> tuple[str, Any]:
        spec = registry.get(input.tool())
        fields = forms.fields_for(spec.input_schema(), spec.input_model.model_fields)
        try:
            raw = {field.name: input[field.name]() for field in fields}
            arguments = forms.arguments(fields, raw)
            with use_session(store()):
                name, payload = spec.name, spec.handler(**arguments)
            if spec.name == "load_table":
                # The tool mutated the shared store in place, and a reactive
                # only notices a new object. Without the replacement the form
                # would keep offering the old table's columns.
                store.set(store().refreshed())
            return name, payload
        except KeyError:
            return spec.name, ToolError("The form is still loading. Try again.")
        except (ToolError, ValidationError) as exc:
            return spec.name, exc

    def _ledger_strip() -> Any:
        """The ledger as a banner, shown above every result.

        It lives inside the result output rather than in its own, because a
        second output registered ahead of `result` starves it: Shiny's flush
        renders `ledger_view` and leaves `result` silent for that cycle.
        """
        status = workflow.analysis_status()
        if status.complete:
            return ui.div(
                ui.tags.strong("Analysis ledger: every stage recorded."),
                ui.tags.p(status.recommended_next),
                class_="alert alert-success",
            )
        return ui.div(
            ui.tags.strong("Analysis ledger: what is still missing"),
            ui.tags.p(status.recommended_next),
            ui.tags.ul(*[ui.tags.li(item) for item in status.missing]),
            class_="alert alert-warning",
        )

    @render.ui
    def result() -> Any:
        name, payload = outcome()
        with use_session(store()):
            strip = _ledger_strip()
        if isinstance(payload, Exception):
            return [strip, ui.div(str(payload), class_="alert alert-danger")]
        return [
            strip,
            *views.full_view(name, payload),
            ui.tags.details(
                ui.tags.summary("Raw JSON"),
                ui.tags.pre(json.dumps(payload.model_dump(), indent=2)),
            ),
        ]

    @render.download_button(filename="agentic-stats-report.html")
    async def download_report():
        """The session as one self-contained file: tables, ledger, findings."""
        with use_session(store()):
            yield html_report(store())
