"""Point the suite at the committed fixture, so no test touches the network.

The app fetches its dataset at runtime and caches it; `AGENTIC_STATS_DATA` makes
every tool read the 72-row fixture instead. Set at import time, because
`app.py` computes its Overview cards while it is being imported.
"""

from __future__ import annotations

import os
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "oats.csv"

os.environ.setdefault("AGENTIC_STATS_DATA", str(FIXTURE))
os.environ.setdefault("AGENTIC_STATS_CACHE", str(FIXTURE.parent / "_cache"))
