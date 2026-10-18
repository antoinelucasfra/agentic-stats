"""Data panel layout: import by URL or catalog, tables, and roles."""

from __future__ import annotations

from shiny import module, ui


@module.ui
def data_ui() -> ui.NavPanel:
    return ui.nav_panel(
        "Data",
        ui.layout_sidebar(
            ui.sidebar(
                ui.tags.h5("Load a table"),
                ui.input_text(
                    "url",
                    "Public CSV, parquet or JSON URL",
                    placeholder="https://…",
                ),
                ui.input_action_button("fetch", "Fetch URL", class_="btn-primary"),
                ui.tags.p(
                    "HTTPS only. Private addresses, oversized files and unknown "
                    "formats are refused before anything is read.",
                    class_="text-muted small",
                ),
                ui.hr(),
                ui.tags.h5("Catalog"),
                ui.input_text("catalog_q", "Search Rdatasets and the curated list"),
                ui.input_action_button("catalog_search", "Search"),
                ui.input_select(
                    "catalog_pick", "Pick a table", choices={"": "Search the catalog first"}
                ),
                ui.input_action_button("catalog_load", "Load"),
                ui.tags.p(
                    "The index is cached on disk. Offline, the curated list still works.",
                    class_="text-muted small",
                ),
                ui.hr(),
                ui.tags.h5("Loaded tables"),
                ui.input_select("table_pick", "Primary table", choices={"primary": "primary"}),
                ui.input_action_button("make_primary", "Make primary"),
                width=380,
                open="always",
            ),
            ui.output_ui("tables_view"),
            ui.output_ui("roles_view"),
            fillable=True,
        ),
        value="data",
    )
