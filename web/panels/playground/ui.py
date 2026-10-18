"""Playground panel layout: the tool picker, its form, and the upload box."""

from __future__ import annotations

from shiny import module, ui

from stats import registry
from utils.config import UPLOAD_SUFFIXES

TOOL_CHOICES = registry.grouped()


@module.ui
def playground_ui() -> ui.NavPanel:
    return ui.nav_panel(
        "Playground",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("tool", "Tool", choices=TOOL_CHOICES),
                ui.output_ui("tool_form"),
                ui.input_action_button("run", "Run", class_="btn-primary"),
                ui.download_button("download_report", "Download report"),
                ui.hr(),
                ui.input_file(
                    "csv",
                    "Analyse your own CSV or parquet",
                    accept=list(UPLOAD_SUFFIXES),
                ),
                ui.tags.p(
                    "The workflow tools run first: frame the question, then profile the "
                    "table. Everything downstream refuses to run before those two. "
                    "Uploads are read on the server and never written to disk.",
                    class_="text-muted small",
                ),
                ui.hr(),
                ui.accordion(
                    ui.accordion_panel(
                        "Notes to table",
                        ui.input_text_area(
                            "notes",
                            "Trial notes",
                            value="",
                            height="150px",
                            placeholder=(
                                "Paste messy notes here: one trial or plot per row, any layout."
                            ),
                        ),
                        ui.input_text(
                            "notes_columns",
                            "Columns (comma-separated)",
                            value="block, variety, nitrogen, grain",
                        ),
                        ui.input_action_button("extract", "Extract table", class_="btn-primary"),
                        ui.output_ui("notes_result"),
                        ui.download_button("download_notes_csv", "Download CSV"),
                    ),
                    id="notes_accordion",
                    open=None,
                ),
                width=380,
                open="always",
            ),
            ui.output_ui("result"),
            fillable=True,
        ),
        value="playground",
    )
