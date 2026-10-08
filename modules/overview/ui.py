"""Overview panel layout. One card per question in `utils.config.CARDS`."""

from __future__ import annotations

from shiny import module, ui

from utils.config import CARDS


@module.ui
def overview_ui() -> ui.NavPanel:
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
