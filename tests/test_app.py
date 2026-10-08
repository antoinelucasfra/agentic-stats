"""Server wiring: overview cards, the playground round trip, CSV uploads."""

from __future__ import annotations

import pandas as pd
import pytest
from shiny.testserver import test_server

from agentic_stats import stats_tools
from agentic_stats.app import CARDS, app

# The marker each card has to show, from its own renderer.
CARD_MARKERS = {
    "formulation": "Omega squared",
    "contrasts": "Marginal means with 95% confidence intervals",
    "batch": "ICC",
    "assumptions": "Normal QQ plot",
    "power": "Power achieved",
    "dose": "Peak dose",
}


@pytest.fixture
def ts():
    with test_server(app) as server:
        yield server


def rendered(value) -> str:
    """The HTML a UI output would send to the browser."""
    return value.value["html"] if isinstance(value.value, dict) else str(value.value)


def test_overview_renders_every_card(ts):
    ts.set_inputs(tool="describe_dataset")  # flush: outputs render in batches

    for card in CARDS:
        output = ts.get_output(f"ov_{card['id']}")

        assert output.status == "ok", f"{card['id']} rendered {output.status}: {output.error}"
        assert CARD_MARKERS[card["id"]] in rendered(output)


def test_overview_dataset_value_boxes(ts):
    html = rendered(ts.get_output("ov_dataset"))

    assert "Rows" in html
    assert f"{stats_tools.describe_dataset().n_rows:,}" in html


def test_tool_form_follows_the_picker(ts):
    ts.set_inputs(tool="anova_effect")
    html = rendered(ts.get_output("tool_form"))
    assert 'id="response"' in html
    assert 'id="factor"' in html

    ts.set_inputs(tool="power_analysis")
    html = rendered(ts.get_output("tool_form"))
    assert 'id="target_power"' in html
    assert 'id="factor"' in html


def test_run_renders_a_result(ts):
    ts.set_inputs(tool="anova_effect", response="assay_signal", factor="formulation", run=1)

    assert "Omega squared" in rendered(ts.get_output("result"))


def test_run_reports_a_tool_error(ts):
    ts.set_inputs(tool="anova_effect", response="assay_signal", factor="formulation_id", run=1)
    html = rendered(ts.get_output("result"))

    assert "alert-danger" in html
    assert "Available columns" in html


def test_run_reports_missing_required_fields(ts):
    ts.set_inputs(tool="anova_effect", response="", factor="", run=1)

    assert "Fill in: response, factor" in rendered(ts.get_output("result"))


def test_open_in_playground_prefills_the_form(ts):
    ts.set_inputs(tool="describe_dataset")
    ts.set_inputs(open_formulation=1)
    # The browser applies the card's update_select; test_server does not, so the
    # test sends the tool change the client would send.
    ts.set_inputs(tool="anova_effect")

    html = rendered(ts.get_output("tool_form"))
    assert 'value="assay_signal"' in html
    assert 'value="formulation"' in html


def test_switching_tools_clears_the_prefilled_arguments(ts):
    ts.set_inputs(tool="describe_dataset")
    ts.set_inputs(open_formulation=1)
    ts.set_inputs(tool="anova_effect")
    assert 'value="assay_signal"' in rendered(ts.get_output("tool_form"))

    # Away and back: the card's arguments are gone, the suggestion is back.
    ts.set_inputs(tool="describe_dataset")
    ts.set_inputs(tool="anova_effect")
    html = rendered(ts.get_output("tool_form"))
    assert 'value="assay_signal"' not in html
    assert 'value="dose"' in html


def test_uploaded_csv_replaces_the_dataset(ts, tmp_path):
    subset = pd.read_csv(stats_tools.dataset_path()).head(300)
    path = tmp_path / "subset.csv"
    subset.to_csv(path, index=False)

    ts.set_inputs(
        csv=[
            {
                "name": "subset.csv",
                "datapath": str(path),
                "size": path.stat().st_size,
                "type": "text/csv",
            }
        ]
    )
    ts.set_inputs(tool="describe_dataset", run=1)

    assert "300" in rendered(ts.get_output("result"))


def test_upload_rejects_anything_but_csv(ts, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a csv")

    ts.set_inputs(
        csv=[
            {
                "name": "notes.txt",
                "datapath": str(path),
                "size": path.stat().st_size,
                "type": "text/plain",
            }
        ]
    )
    ts.set_inputs(tool="describe_dataset", run=1)

    html = rendered(ts.get_output("result"))
    assert str(stats_tools.describe_dataset().n_rows) in html
