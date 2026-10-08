"""Playground panel layout: the tool picker, its form, and the upload box."""

from __future__ import annotations

from shiny import module, ui

from utils.config import UPLOAD_SUFFIXES

TOOL_CHOICES = {
    "Profile the data": {
        "describe_dataset": "describe_dataset",
        "correlate": "correlate",
    },
    "Compare levels": {
        "anova_effect": "anova_effect",
        "marginal_means": "marginal_means",
    },
    "Model with grouping": {
        "fit_mixed_model": "fit_mixed_model",
        "check_assumptions": "check_assumptions",
    },
    "Design size": {"power_analysis": "power_analysis"},
    "Dose response": {
        "dose_response": "dose_response",
        "fit_curve_family": "fit_curve_family",
    },
}


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
