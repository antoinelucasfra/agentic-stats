"""Server wiring: overview cards, the playground round trip, file uploads.

Module ids are namespaced, so the Playground's `tool` input is
`playground-tool` and the Overview's first card output is `overview-ov_nitrogen`.
"""

from __future__ import annotations

import pandas as pd
import pytest
from shiny.testserver import test_server

from app import app
from stats import stats_tools
from utils.config import CARDS

TOOL = "playground-tool"
RUN = "playground-run"
FORM = "playground-tool_form"
RESULT = "playground-result"
UPLOAD = "playground-csv"

# The marker each card has to show, from its own renderer.
CARD_MARKERS = {
    "nitrogen": "Omega squared",
    "variety": "Marginal means with 95% confidence intervals",
    "block": "ICC",
    "assumptions": "Normal QQ plot",
    "power": "Power achieved",
    "dose": "EC50",
}


@pytest.fixture
def ts():
    # Rendering the six Overview cards draws several plotnine figures; 30s leaves
    # room for a loaded machine where the 5s default times out.
    with test_server(app, timeout_secs=30) as server:
        yield server


def rendered(value) -> str:
    """The HTML a UI output would send to the browser."""
    return value.value["html"] if isinstance(value.value, dict) else str(value.value)


def test_overview_renders_every_card(ts):
    ts.set_inputs(**{TOOL: "describe_dataset"})  # flush: outputs render in batches

    for card in CARDS:
        output = ts.get_output(f"overview-ov_{card['id']}")

        assert output.status == "ok", f"{card['id']} rendered {output.status}: {output.error}"
        assert CARD_MARKERS[card["id"]] in rendered(output)


def test_overview_dataset_value_boxes(ts):
    html = rendered(ts.get_output("overview-ov_dataset"))

    assert "Rows" in html
    assert f"{stats_tools.describe_dataset().n_rows:,}" in html


def test_tool_form_follows_the_picker(ts):
    ts.set_inputs(**{TOOL: "anova_effect"})
    html = rendered(ts.get_output(FORM))
    assert 'id="playground-response"' in html
    assert 'id="playground-factor"' in html

    ts.set_inputs(**{TOOL: "power_analysis"})
    html = rendered(ts.get_output(FORM))
    assert 'id="playground-target_power"' in html
    assert 'id="playground-factor"' in html


def test_run_renders_a_result(ts):
    ts.set_inputs(
        **{
            TOOL: "anova_effect",
            "playground-response": "grain",
            "playground-factor": "nitrogen",
            RUN: 1,
        }
    )

    assert "Omega squared" in rendered(ts.get_output(RESULT))


def test_run_reports_a_tool_error(ts):
    ts.set_inputs(
        **{
            TOOL: "anova_effect",
            "playground-response": "grain",
            "playground-factor": "nitrogen_id",
            RUN: 1,
        }
    )
    html = rendered(ts.get_output(RESULT))

    assert "alert-danger" in html
    assert "Available columns" in html


def test_run_reports_missing_required_fields(ts):
    ts.set_inputs(
        **{TOOL: "anova_effect", "playground-response": "", "playground-factor": "", RUN: 1}
    )

    assert "Fill in: response, factor" in rendered(ts.get_output(RESULT))


def test_open_in_playground_prefills_the_form(ts):
    ts.set_inputs(**{TOOL: "describe_dataset"})
    ts.set_inputs(**{"overview-open_nitrogen": 1})
    # The browser applies the card's update_select; test_server does not, so the
    # test sends the tool change the client would send.
    ts.set_inputs(**{TOOL: "anova_effect"})

    html = rendered(ts.get_output(FORM))
    assert '<option value="grain" selected' in html
    assert '<option value="nitrogen" selected' in html


def test_switching_tools_clears_the_prefilled_arguments(ts):
    ts.set_inputs(**{TOOL: "describe_dataset"})
    ts.set_inputs(**{"overview-open_nitrogen": 1})
    ts.set_inputs(**{TOOL: "anova_effect"})
    assert '<option value="nitrogen" selected' in rendered(ts.get_output(FORM))

    # Away and back: the card's arguments are gone, the suggestion is back.
    ts.set_inputs(**{TOOL: "describe_dataset"})
    ts.set_inputs(**{TOOL: "anova_effect"})
    html = rendered(ts.get_output(FORM))
    assert '<option value="nitrogen" selected' not in html
    assert '<option value="variety" selected' in html


def test_uploaded_csv_replaces_the_dataset(ts, tmp_path):
    subset = pd.read_csv(stats_tools.dataset_path()).head(60)
    path = tmp_path / "subset.csv"
    subset.to_csv(path, index=False)

    ts.set_inputs(
        **{
            UPLOAD: [
                {
                    "name": "subset.csv",
                    "datapath": str(path),
                    "size": path.stat().st_size,
                    "type": "text/csv",
                }
            ]
        }
    )
    ts.set_inputs(**{TOOL: "describe_dataset", RUN: 1})

    assert "60" in rendered(ts.get_output(RESULT))


def test_upload_rejects_anything_but_csv_or_parquet(ts, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a csv")

    ts.set_inputs(
        **{
            UPLOAD: [
                {
                    "name": "notes.txt",
                    "datapath": str(path),
                    "size": path.stat().st_size,
                    "type": "text/plain",
                }
            ]
        }
    )
    ts.set_inputs(**{TOOL: "describe_dataset", RUN: 1})

    html = rendered(ts.get_output(RESULT))
    assert str(stats_tools.describe_dataset().n_rows) in html
