"""Overview panel server: compute the nine answers once, then render them.

The results are computed lazily per process, on the first session that renders
a card, and cached in a module-level dict: a fresh server with an empty cache
and no network must still boot, and a failed compute renders an error card
naming the fetch command instead of crashing the app. Clicking a card hands
its tool and arguments to the Playground through the shared `pending` value.
"""

from __future__ import annotations

import threading
from typing import Any

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from stats import registry, views
from stats.stats_tools import DatasetDescription
from utils.config import CARDS

_FETCH_COMMAND = "python -m utils.data"

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _build() -> dict[str, Any]:
    """The cached dataset description and card results, computing them once.

    A `threading.Lock` because two sessions can reach the empty cache on the
    same loop tick; the compute itself is sync and finishes in under a second.
    Only successes are cached: a failed compute is retried on the next render
    instead of being frozen for the process lifetime.
    """
    with _LOCK:
        if not _CACHE:
            try:
                dataset = registry.get("describe_dataset").handler()
                cards = {
                    card["id"]: registry.get(card["tool"]).handler(**card["args"]) for card in CARDS
                }
                _CACHE.update({"dataset": dataset, "cards": cards, "error": None})
            except Exception as exc:
                # Not cached: the next render (or reload) retries the compute.
                return {"dataset": None, "cards": {}, "error": str(exc)}
        return _CACHE


def _error_card(error: str) -> list[Any]:
    return [
        ui.div(
            ui.tags.strong("The overview data is not loaded yet."),
            ui.tags.p(
                f"Fetch it with {_FETCH_COMMAND} on the server; the retry runs when "
                "the panel re-renders, so a reload also works."
            ),
            ui.tags.p(error, class_="text-muted small"),
            class_="alert alert-warning",
        )
    ]


@module.server
def overview_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    pending: reactive.Value,
) -> None:
    overview = _build()

    @render.ui
    def ov_dataset() -> Any:
        if overview["error"] is not None:
            return _error_card(overview["error"])
        dataset: DatasetDescription = overview["dataset"]
        return ui.layout_column_wrap(
            ui.value_box("Rows", f"{dataset.n_rows:,}"),
            ui.value_box("Columns", str(dataset.n_columns)),
            ui.value_box("Factors", ", ".join(dataset.factors)),
            ui.value_box("Numeric", ", ".join(dataset.numeric)),
            width=1 / 4,
        )

    for card in CARDS:
        output(id=f"ov_{card['id']}")(_overview_renderer(card, overview))
        _register_open_handler(card, input, pending)


def _overview_renderer(card: dict[str, Any], overview: dict[str, Any]) -> Any:
    @render.ui
    def card_output() -> list[Any]:
        if overview["error"] is not None:
            return _error_card(overview["error"])
        return views.card_view(card["tool"], overview["cards"][card["id"]])

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
