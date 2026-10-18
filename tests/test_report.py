"""The report: a session's tables, ledger and findings, escaped.

The export is the artifact a reader hands to someone else, so the tests hold two
properties: everything user-controlled is escaped, and every stored figure can
be traced back. A column named `<script>` must read as text, never run.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import BaseModel

from stats import report, session


class _Effect(BaseModel):
    statement: str = ""


class _Fit(BaseModel):
    effects: list[_Effect] = []
    caveats: list[str] = []
    cannot_say: list[str] = []


@pytest.fixture
def store() -> session.SessionStore:
    frame = pd.DataFrame({"region": ["north", "south"], "revenue": [60.0, 30.0]})
    store = session.SessionStore.from_frame(
        frame, name="sales", source="https://example.test/sales.csv", sha256="abc"
    )
    stored = store.put(
        _Fit(
            effects=[_Effect(statement="north vs south: 30.00 (10.00 to 50.00)")],
            caveats=["Two rows is not an analysis."],
            cannot_say=["Direction of the effect."],
        ),
        kind="fit",
        summary="Regional comparison",
    )
    store.state.record("design", result_id=stored.result_id, note="compare regions")
    store.state.record("quality", result_id=stored.result_id)
    return store


def test_markdown_lists_tables_ledger_and_findings(store):
    text = report.markdown_report(store)

    assert "sales" in text
    assert "https://example.test/sales.csv" in text
    assert "design: ok" in text
    assert "Regional comparison" in text
    assert "north vs south: 30.00 (10.00 to 50.00)" in text
    assert "Caveat: Two rows is not an analysis." in text
    assert "Cannot say: Direction of the effect." in text
    assert "30.00" in text


def test_html_escapes_user_controlled_content():
    frame = pd.DataFrame({"<script>alert(1)</script>": [1.0]})
    store = session.SessionStore.from_frame(frame, source='"><img src=x onerror=y>')

    text = report.html_report(store)

    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text
    assert '"><img' not in text


def test_html_embeds_a_plot_image():
    from stats.reading import PlotFeatures, PlotResult

    store = session.SessionStore.from_frame(pd.DataFrame({"x": [1.0, 2.0]}))
    store.put(
        PlotResult(
            kind="hist",
            table="primary",
            x="x",
            image="data:image/png;base64,AAAA",
            alt_text="histogram",
            features=PlotFeatures(n_rows=2, series=1),
        ),
        kind="plot",
        evidence_class="descriptive",
    )

    text = report.html_report(store)

    assert 'src="data:image/png;base64,AAAA"' in text
    assert 'alt="histogram"' in text


def test_an_empty_session_still_reports():
    store = session.SessionStore()

    assert "No tables are loaded." in report.markdown_report(store)
    assert "No results are stored." in report.markdown_report(store)
