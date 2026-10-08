"""Constants, paths and configuration shared by the app, the tools and the tests.

Two kinds of thing live here: what the app knows about the dataset (where it
comes from, where the cache goes, which column answers the questions) and the
limits the UI enforces before it reads anything.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# --- dataset ----------------------------------------------------------------

SOURCE = "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Oats.csv"
CACHE_NAME = "oats.parquet"

# `lower(...)` normalises the R column names. `yield` is the grain yield, and
# renaming it to `grain` keeps the formula parser happy: it is a Python keyword.
SELECT = (
    "select lower(Block) as block, lower(Variety) as variety, "
    "yield as grain, nitro as nitrogen from read_csv_auto({source})"
)

# The response every card and every form falls back to.
RESPONSE = "grain"

# --- uploads ----------------------------------------------------------------

UPLOAD_SUFFIXES = (".csv", ".parquet")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# --- panel and input ids ------------------------------------------------------

# The navbar and its panel ids, in one place. A module cannot address another
# module's inputs by id, so the cross-panel handoff goes through a shared value:
# the sender writes it, the receiving module updates its own inputs.
NAV_ID = "nav"
OVERVIEW_ID = "overview"
PLAYGROUND_ID = "playground"
AGENT_ID = "agent"

# --- the six Overview cards --------------------------------------------------

# A question, the tool that answers it, and the arguments it is called with.
# The Overview renders these as cards and hands the same arguments to the
# Playground when a reader clicks through.
CARDS: list[dict[str, Any]] = [
    {
        "id": "nitrogen",
        "question": "Does the nitrogen rate raise the grain yield?",
        "tool": "anova_effect",
        "args": {"response": RESPONSE, "factor": "nitrogen"},
    },
    {
        "id": "variety",
        "question": "How far apart are the varieties, once nitrogen is held level?",
        "tool": "marginal_means",
        "args": {"response": RESPONSE, "factors": ["variety"], "covariates": ["nitrogen"]},
    },
    {
        "id": "block",
        "question": "Is block-to-block variation real, or noise?",
        "tool": "fit_mixed_model",
        "args": {"response": RESPONSE, "fixed_effects": ["variety", "nitrogen"], "group": "block"},
    },
    {
        "id": "assumptions",
        "question": "Can we trust the model behind those p-values?",
        "tool": "check_assumptions",
        "args": {"response": RESPONSE, "fixed_effects": ["variety", "nitrogen"]},
    },
    {
        "id": "power",
        "question": "Should we run more blocks?",
        "tool": "power_analysis",
        "args": {"response": RESPONSE, "factor": "variety"},
    },
    {
        "id": "dose",
        "question": "How much nitrogen does each variety need to reach half its maximum?",
        "tool": "dose_response",
        "args": {"response": RESPONSE, "dose": "nitrogen", "model": "4pl", "by": "variety"},
    },
]


# --- paths ------------------------------------------------------------------


def cache_dir() -> Path:
    """Where the parquet lives, overridable with AGENTIC_STATS_CACHE."""
    override = os.environ.get("AGENTIC_STATS_CACHE")
    return Path(override) if override else Path(tempfile.gettempdir()) / "agentic_stats"


def default_path() -> Path:
    """The dataset path: AGENTIC_STATS_DATA if set, else the parquet cache."""
    override = os.environ.get("AGENTIC_STATS_DATA")
    return Path(override) if override else cache_dir() / CACHE_NAME


FETCH_COMMAND = "python -m utils.data"
