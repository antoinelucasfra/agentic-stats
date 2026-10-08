"""Top-level server: the shared state the panel modules talk through."""

from __future__ import annotations

from shiny import Inputs, Outputs, Session, reactive, ui

from utils.config import AGENT_ID, NAV_ID, OVERVIEW_ID, PLAYGROUND_ID
from web.panels.agent import agent_server
from web.panels.overview import overview_server
from web.panels.playground import playground_server


def server(input: Inputs, output: Outputs, session: Session) -> None:
    # A card in the Overview hands its tool and arguments to the Playground.
    pending = reactive.value(None)

    overview_server(OVERVIEW_ID, pending=pending)
    playground_server(PLAYGROUND_ID, pending=pending)
    agent_server(AGENT_ID)

    @reactive.effect
    @reactive.event(pending)
    def _jump_to_playground() -> None:
        """The navbar is this module's, so the panel jump happens here."""
        if pending() is not None:
            ui.update_navset(NAV_ID, selected=PLAYGROUND_ID)
