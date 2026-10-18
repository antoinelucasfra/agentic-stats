"""The reading tools: the chart's features, and a reading that keeps its figures.

A chart description is the easiest place for a model to invent a trend, so the
features are computed and the test checks them against the data. The reading
tests use synthetic stored payloads, so they do not depend on the estimation
module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from stats import reading, session
from stats.errors import ToolError


class QueryPayload(BaseModel):
    """The shape a query result has, as far as the reading is concerned."""

    columns: list[str]
    rows: list[list[object]]
    row_count: int = 0
    truncated: bool = False
    statement_type: str = "SELECT"
    elapsed_s: float = 0.0
    note: str = "Descriptive only."
    evidence_class: str = "descriptive"


class ProfilePayload(BaseModel):
    """A result the reading tool has nothing to say about."""

    table: str = "primary"


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = np.arange(30, dtype=float)
    return pd.DataFrame(
        {
            "x": x,
            "y": 2.5 * x + rng.normal(0, 3, size=30),
            "group": ["a", "b", "c"] * 10,
            "label": [f"row-{index:02d}" for index in range(30)],
        }
    )


@pytest.fixture
def store(frame: pd.DataFrame) -> session.SessionStore:
    store = session.SessionStore.from_frame(frame, source="synthetic")
    store.state.record("design", result_id=None, note="synthetic")
    store.state.record("quality", result_id=None, note="synthetic")
    return store


def test_a_scatter_reports_the_trend_it_actually_has(store):
    with session.use_session(store):
        result = reading.plot("scatter", x="x", y="y")

    slope = float(np.polyfit(store.frame()["x"], store.frame()["y"], 1)[0])
    assert result.features.trend_slope == pytest.approx(slope)
    assert "upward" in result.features.trend_direction
    assert result.image.startswith("data:image/png;base64,")
    assert result.alt_text
    assert result.result_id.startswith("p_")


def test_a_chart_is_stored_as_descriptive(store):
    with session.use_session(store):
        result = reading.plot("hist", x="y")

    assert store.get(result.result_id).evidence_class == "descriptive"


def test_a_line_chart_reports_its_largest_gap(store):
    sparse = store.frame().drop(index=[10, 11, 12])
    store.tables["primary"] = session.TableRef(name="primary", frame=sparse, source="synthetic")

    with session.use_session(store):
        result = reading.plot("line", x="x", y="y")

    assert result.features.largest_gap == 4.0


def test_an_unknown_column_is_refused_with_the_available_ones(store):
    with session.use_session(store), pytest.raises(ToolError, match="Available columns"):
        reading.plot("hist", x="nope")


def test_a_non_numeric_y_is_refused_for_a_scatter(store):
    with session.use_session(store), pytest.raises(ToolError, match="numeric y"):
        reading.plot("scatter", x="x", y="label")


def test_a_categorical_axis_above_the_limit_is_refused(store):
    with session.use_session(store), pytest.raises(ToolError, match="above the"):
        reading.plot("bar", x="label", y="y", max_levels=5)


def test_a_heatmap_needs_a_fill_column(store):
    with session.use_session(store), pytest.raises(ToolError, match="color column"):
        reading.plot("heatmap", x="x", y="group")


def test_a_plot_without_a_quality_stage_is_refused(frame):
    bare = session.SessionStore.from_frame(frame)
    with session.use_session(bare), pytest.raises(ToolError, match="frame_question"):
        reading.plot("hist", x="y")


def test_the_reading_of_a_plot_repeats_the_features(store):
    with session.use_session(store):
        chart = reading.plot("scatter", x="x", y="y")
        explanation = reading.explain_result(chart.result_id)

    assert explanation.kind == "plot"
    assert explanation.reading
    assert explanation.effects == []
    assert explanation.cannot_say
    assert explanation.causal_claims_permitted is False
    assert store.state.has("interpretation")


def test_a_query_result_is_read_as_ranked_contributions(store):
    payload = QueryPayload(
        columns=["region", "revenue"],
        rows=[["north", 60.0], ["south", 30.0], ["east", 10.0]],
        row_count=3,
    )
    stored = store.put(payload, kind="query", evidence_class="descriptive")

    with session.use_session(store):
        explanation = reading.explain_result(stored.result_id)

    assert any("north" in line for line in explanation.reading)
    assert any("60.0% of the total" in line for line in explanation.reading)
    assert any("single grouping" in line for line in explanation.reading)


def test_a_query_whose_values_are_not_parts_of_a_whole_says_so(store):
    payload = QueryPayload(
        columns=["region", "change"],
        rows=[["north", -5.0], ["south", 3.0]],
        row_count=2,
    )
    stored = store.put(payload, kind="query", evidence_class="descriptive")

    with session.use_session(store):
        explanation = reading.explain_result(stored.result_id)

    assert any("not a set of parts of a whole" in line for line in explanation.reading)


def test_a_result_with_nothing_to_read_is_refused(store):
    stored = store.put(ProfilePayload(), kind="profile", evidence_class="descriptive")

    with session.use_session(store), pytest.raises(ToolError, match="Nothing to interpret"):
        reading.explain_result(stored.result_id)


def test_an_inferential_reading_needs_the_whole_chain(store):
    """Sensitivity gates interpretation, so a fit cannot be read before it runs."""
    stored = store.put(QueryPayload(columns=["a", "b"], rows=[[1, 2]]), kind="fit")
    for stage in ("assumptions", "estimate"):
        store.state.record(stage, result_id=stored.result_id)

    with session.use_session(store), pytest.raises(ToolError, match="sensitivity"):
        reading.explain_result(stored.result_id)
