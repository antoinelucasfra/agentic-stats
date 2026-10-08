"""Entry point for `uv run shiny run app.py`.

The app lives inside the package so tests can import it; Shiny's CLI needs a
path, so this shim exists and nothing else.
"""

from agentic_stats.app import app

__all__ = ["app"]
