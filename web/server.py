"""Top-level server: the shared state the panel modules talk through."""

from __future__ import annotations

from shiny import Inputs, Outputs, Session, reactive, ui

from stats import stats_tools
from utils.config import AGENT_ID, DATA_ID, NAV_ID, OVERVIEW_ID, PLAYGROUND_ID
from web.panels.agent import agent_server
from web.panels.data import data_server
from web.panels.overview import overview_server
from web.panels.playground import playground_server


def server(input: Inputs, output: Outputs, session: Session) -> None:
    # A card in the Overview hands its tool and arguments to the Playground.
    pending = reactive.value(None)
    # One store for the connection: the Data panel loads into it, the Playground
    # and the Agent read from it, and no reader's tables, ids or ledger leak
    # into another's. The panels mutate it in place and replace the object when
    # the UI has to notice, because a reactive only fires on a new object.
    store = reactive.value(stats_tools.env_store())

    overview_server(OVERVIEW_ID, pending=pending)
    data_server(DATA_ID, store=store)
    playground_server(PLAYGROUND_ID, pending=pending, store=store)
    agent_server(AGENT_ID, store=store)

    @reactive.effect
    @reactive.event(pending)
    def _jump_to_playground() -> None:
        """The navbar is this module's, so the panel jump happens here."""
        if pending() is not None:
            ui.update_navset(NAV_ID, selected=PLAYGROUND_ID)
