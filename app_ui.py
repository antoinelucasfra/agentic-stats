"""Top-level UI: the navbar that assembles the panel modules."""

from __future__ import annotations

from shiny import ui

from modules.agent import agent_ui
from modules.overview import overview_ui
from modules.playground import playground_ui
from utils.config import AGENT_ID, NAV_ID, OVERVIEW_ID, PLAYGROUND_ID

app_ui = ui.page_navbar(
    overview_ui(OVERVIEW_ID),
    playground_ui(PLAYGROUND_ID),
    agent_ui(AGENT_ID),
    title="agentic-stats",
    id=NAV_ID,
    fillable=True,
)
