"""Entry point: the Shiny app, assembled from `app_ui.py` and `server.py`.

    uv run shiny run --host 127.0.0.1 --port 8766 app.py

Three panels. Overview paints the six answers the README quotes, computed once
at import. Playground calls any registered tool with a form derived from its
schema, against the fetched dataset or an uploaded CSV. Agent hands the same
tools to a chat model.
"""

from __future__ import annotations

import os

from shiny import App, run_app

from web.app_ui import app_ui
from web.server import server

app = App(app_ui, server)


def main() -> None:
    """Serve the app without the Shiny CLI."""
    run_app(
        app,
        host=os.environ.get("AGENTIC_STATS_HOST", "127.0.0.1"),
        port=int(os.environ.get("AGENTIC_STATS_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
