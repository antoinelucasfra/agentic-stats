"""Playground panel server: the form, the upload, and running a tool by hand."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import ValidationError
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from stats import forms, registry, stats_tools, views
from stats.stats_tools import DatasetDescription, ToolError
from web.panels.playground.helpers import field_input, read_uploaded, upload_problem


@module.server
def playground_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    pending: reactive.Value,
) -> None:
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
        problem = upload_problem(info)
        if problem:
            ui.notification_show(problem, type="error")
            uploaded.set(None)
            return
        try:
            loaded = read_uploaded(info["datapath"])
        except Exception as exc:
            ui.notification_show(f"Could not read that file: {exc}", type="error")
            uploaded.set(None)
            return
        if loaded.empty:
            ui.notification_show("That file has no rows.", type="error")
            uploaded.set(None)
            return
        uploaded.set(loaded)
        ui.notification_show(f"Loaded {info['name']}. Run a tool to analyse it.", type="message")

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
