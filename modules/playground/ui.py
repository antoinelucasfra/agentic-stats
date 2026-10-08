"""Playground panel layout: the tool picker, its form, and the upload box."""

from __future__ import annotations

from shiny import module, ui

import registry
from utils.config import UPLOAD_SUFFIXES

TOOL_CHOICES = {spec.name: spec.name for spec in registry.TOOLS}


@module.ui
def playground_ui() -> ui.NavPanel:
    return ui.nav_panel(
        "Playground",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("tool", "Tool", choices=TOOL_CHOICES),
                ui.output_ui("tool_form"),
                ui.input_action_button("run", "Run", class_="btn-primary"),
                ui.hr(),
                ui.input_file(
                    "csv",
                    "Analyse your own CSV or parquet",
                    accept=list(UPLOAD_SUFFIXES),
                ),
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
