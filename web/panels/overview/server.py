"""Overview panel server: compute the six answers once, then render them.

The tools are cheap (all seven cost well under a second), so the results are
computed at import and the renderers only format them. Clicking a card hands its
tool and arguments to the Playground through the shared `pending` value.
"""

from __future__ import annotations

from typing import Any

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from stats import registry, views
from stats.stats_tools import DatasetDescription
from utils.config import CARDS

# Computed once at import: all seven tools together cost well under a second.
DATASET: DatasetDescription = registry.get("describe_dataset").handler()
OVERVIEW: dict[str, Any] = {
    card["id"]: registry.get(card["tool"]).handler(**card["args"]) for card in CARDS
}


@module.server
def overview_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    pending: reactive.Value,
) -> None:
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


def _overview_renderer(card: dict[str, Any]) -> Any:
    @render.ui
    def card_output() -> list[Any]:
        return views.card_view(card["tool"], OVERVIEW[card["id"]])

    return card_output


def _register_open_handler(card: dict[str, Any], input: Inputs, pending: reactive.Value) -> None:
    """Hand a card's tool and arguments to the Playground.

    The panel jump and the tool select are applied where those ids live: the
    navbar in `server.py`, the picker in the Playground module.
    """

    @reactive.effect
    @reactive.event(input[f"open_{card['id']}"])
    def _open() -> None:
        pending.set({"tool": card["tool"], "args": card["args"]})
