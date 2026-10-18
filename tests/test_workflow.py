"""The workflow tools: the estimand, the quality screen, the ledger and SQL.

These tests hold the tools to the contract in docs/analysis-tools.md: the design
stage is written before anything else, a causal target on a cross-section is
refused rather than softened, the leakage check blocks instead of warning, and a
descriptive SQL result never carries an effect. The oats fixture is the success
path; the failure cases get synthetic frames with one known defect each.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from stats import session, stats_tools, workflow
from stats.errors import ToolError


@pytest.fixture
def oats() -> pd.DataFrame:
    return stats_tools.load_dataset()


def test_frame_question_reads_the_oats_design_from_the_block(oats):
    store = session.SessionStore.from_frame(oats)

    with session.use_session(store):
        plan = workflow.frame_question("Does grain differ between varieties?")

    assert plan.outcome == "grain"
    assert plan.target == "comparison"
    assert plan.design == "clustered"
    assert plan.unit_of_analysis == "block"
    assert plan.estimand == "the difference in mean grain between levels of variety"
    assert plan.predictors == ["variety", "nitrogen"]
    assert store.state.has("design")
    assert store.get(plan.result_id).kind == "plan"
    assert "profile_table" in plan.next


def test_frame_question_refuses_causal_language_without_structure():
    frame = pd.DataFrame(
        {
            "reading": [float(i % 7) for i in range(40)],
            "score": [float(i) for i in range(40)],
        }
    )
    store = session.SessionStore.from_frame(frame)

    with session.use_session(store), pytest.raises(ToolError, match="causal"):
        workflow.frame_question("Does the reading cause score to rise?", target="causal")


def test_a_grouping_column_does_not_license_a_causal_claim():
    """Clustered data with no assigned exposure and no time is observational."""
    frame = pd.DataFrame(
        {
            "region": ["north", "south"] * 20,
            "sales": [float(i) for i in range(40)],
        }
    )
    store = session.SessionStore.from_frame(frame)

    with session.use_session(store), pytest.raises(ToolError, match="assigned exposure"):
        workflow.frame_question("Do regions cause sales to rise?", target="causal")


def test_a_named_treatment_column_licenses_a_causal_claim():
    frame = pd.DataFrame(
        {
            "treatment": ["drug", "placebo"] * 20,
            "score": [float(i) for i in range(40)],
        }
    )
    store = session.SessionStore.from_frame(frame)

    with session.use_session(store):
        plan = workflow.frame_question("Does the drug raise score?", target="causal")

    assert plan.target == "causal"
    assert plan.design in ("clustered", "independent")


def test_frame_question_refuses_an_unknown_outcome_and_lists_the_columns(oats):
    store = session.SessionStore.from_frame(oats)

    with session.use_session(store), pytest.raises(ToolError) as excinfo:
        workflow.frame_question("Compare yield between varieties", outcome="yield")

    assert "Available columns" in str(excinfo.value)
    assert "grain" in str(excinfo.value)


def test_profile_table_blocks_a_column_that_copies_the_outcome():
    frame = pd.DataFrame({"y": [float(i) for i in range(40)]})
    frame["y_copy"] = frame["y"]
    store = session.SessionStore.from_frame(frame)

    with session.use_session(store):
        workflow.frame_question("What is the distribution of y?", outcome="y")
        profile = workflow.profile_table(outcome="y")
        blocked = [v for v in profile.verdicts if v.status == "blocked"]

    assert any("y_copy" in v.remedy for v in blocked)
    assert any("copy of itself" in v.statement for v in blocked)

    # The ledger refuses the next stage and carries the remedy into the error.
    store.state.record("assumptions")
    with pytest.raises(ToolError, match="y_copy"):
        store.state.require("estimate")


def test_profile_table_blocks_a_column_that_is_mostly_missing():
    frame = pd.DataFrame(
        {
            "y": [float(i) for i in range(50)],
            "mostly_missing": [float(i) if i < 20 else None for i in range(50)],
        }
    )
    store = session.SessionStore.from_frame(frame)

    with session.use_session(store):
        store.state.record("design")
        profile = workflow.profile_table(outcome="y")

    blocked = [v for v in profile.verdicts if v.status == "blocked"]
    assert any(v.check == "missingness" and "mostly_missing" in v.statement for v in blocked)
    assert "mostly_missing" in profile.next


def test_profile_table_blocks_duplicate_rows():
    base = pd.DataFrame({"g": ["a", "b"] * 10, "y": [float(i) for i in range(20)]})
    frame = pd.concat([base, base.head(3)], ignore_index=True)
    store = session.SessionStore.from_frame(frame)

    with session.use_session(store):
        store.state.record("design")
        profile = workflow.profile_table(outcome="y")

    assert any(v.status == "blocked" and "duplicate" in v.check for v in profile.verdicts)


def test_profile_table_leaves_a_clean_table_unblocked(oats):
    store = session.SessionStore.from_frame(oats)

    with session.use_session(store):
        workflow.frame_question("Does grain differ between varieties?", outcome="grain")
        profile = workflow.profile_table(outcome="grain")

    assert profile.verdicts
    assert not [v for v in profile.verdicts if v.status == "blocked"]
    assert "fit_model" in profile.next
    assert store.state.has("quality")
    assert store.get(profile.result_id).rows_total == 72


def test_run_sql_returns_rows_and_tags_them_descriptive(oats):
    store = session.SessionStore.from_frame(oats)

    with session.use_session(store):
        result = workflow.run_sql(
            'select variety, count(*) as n from "primary" group by variety order by variety'
        )

    assert result.columns == ["variety", "n"]
    assert result.row_count == 3
    assert result.truncated is False
    assert result.statement_type == "SELECT"
    assert result.evidence_class == "descriptive"
    assert result.note == (
        "Descriptive only. Cite compare_groups, fit_model or sensitivity "
        "for any claim about an effect."
    )
    # The payload has to survive json, because the store holds JSON.
    json.dumps(result.model_dump())
    stored = store.get(result.result_id)
    assert stored.kind == "query"
    assert stored.evidence_class == "descriptive"


def test_run_sql_caps_rows_and_reports_truncation(oats):
    store = session.SessionStore.from_frame(oats)

    with session.use_session(store):
        result = workflow.run_sql('select * from "primary"', max_rows=10)

    assert result.row_count == 10
    assert result.truncated is True


def test_run_sql_turns_a_guard_violation_into_a_tool_error(oats):
    store = session.SessionStore.from_frame(oats)

    with session.use_session(store), pytest.raises(ToolError, match="SELECT"):
        workflow.run_sql('drop table "primary"')


def test_analysis_status_lists_the_missing_stages_in_order(oats):
    store = session.SessionStore.from_frame(oats)

    with session.use_session(store):
        status = workflow.analysis_status()

    assert status.complete is False
    assert status.missing[:6] == [f"{stage}: not started" for stage in session.STAGE_ORDER]
    assert "frame_question" in status.recommended_next
    assert store.state.records == {}


def _fake_fetch(path, url="https://example.test/titanic.csv", format="csv"):
    """A fetch result pointing at a local file, so no test touches the network."""
    from utils.fetch import FetchedFile

    return FetchedFile(
        path=path,
        url=url,
        sha256="abc",
        n_bytes=10,
        content_type="text/csv",
        fetched_at="2026-01-01T00:00:00+00:00",
        format=format,
    )


def test_load_table_makes_the_fetched_file_primary_and_clears_the_ledger(monkeypatch, tmp_path):
    csv = tmp_path / "titanic.csv"
    pd.DataFrame({"cls": ["a", "b"] * 10, "fare": [float(i) for i in range(20)]}).to_csv(
        csv, index=False
    )
    store = session.SessionStore.from_frame(pd.DataFrame({"x": [1.0]}))
    store.state.record("design")
    monkeypatch.setattr(workflow, "fetch_table", lambda url, **kwargs: _fake_fetch(csv, url))

    with session.use_session(store):
        summary = workflow.load_table("https://example.test/titanic.csv")

    assert summary.name == "titanic"
    assert summary.n_rows == 20
    assert summary.result_id.startswith("t_")
    assert store.primary == "titanic"
    assert store.table().rows == 20
    assert store.get(summary.result_id).evidence_class == "descriptive"
    assert store.state.missing(), "a new table invalidates the old analysis"


def test_load_table_rejects_a_plain_http_url():
    store = session.SessionStore.from_frame(pd.DataFrame({"x": [1.0]}))

    with session.use_session(store), pytest.raises(ToolError, match="https"):
        workflow.load_table("http://example.test/x.csv")


def test_load_table_refuses_an_oversized_table(monkeypatch, tmp_path):
    from utils import config

    big = tmp_path / "big.csv"
    pd.DataFrame({"x": [1.0] * (config.MAX_IMPORT_ROWS + 1)}).to_csv(big, index=False)
    store = session.SessionStore.from_frame(pd.DataFrame({"x": [1.0]}))
    monkeypatch.setattr(workflow, "fetch_table", lambda url, **kwargs: _fake_fetch(big, url))

    with session.use_session(store), pytest.raises(ToolError, match="Aggregate it"):
        workflow.load_table("https://example.test/big.csv")


def test_load_table_reports_an_unreadable_file(monkeypatch, tmp_path):
    bad = tmp_path / "x.parquet"
    bad.write_text("this is not parquet")
    store = session.SessionStore.from_frame(pd.DataFrame({"x": [1.0]}))
    monkeypatch.setattr(
        workflow, "fetch_table", lambda url, **kwargs: _fake_fetch(bad, url, "parquet")
    )

    with session.use_session(store), pytest.raises(ToolError, match="could not read"):
        workflow.load_table("https://example.test/x.parquet")


def test_table_names_come_from_the_url_and_stay_unique():
    store = session.SessionStore.from_frame(pd.DataFrame({"x": [1.0]}), source="first")
    store.add_table(pd.DataFrame({"y": [1.0]}), name="titanic")

    assert workflow._name_from_url("https://example.test/a/Titanic-Data.csv?x=1") == (
        "titanic_data"
    )
    assert workflow._name_from_url("https://example.test/") == "example"
    assert workflow._unique_name(store, "titanic") == "titanic_2"
    assert workflow._unique_name(store, "fresh") == "fresh"


def test_read_imported_reads_csv_parquet_and_json(tmp_path):
    import duckdb

    frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    csv = tmp_path / "t.csv"
    parquet = tmp_path / "t.parquet"
    frame.to_csv(csv, index=False)
    duckdb.connect().execute(
        f"copy (select * from read_csv_auto('{csv.as_posix()}')) "
        f"to '{parquet.as_posix()}' (format parquet)"
    )
    as_json = tmp_path / "t.json"
    as_json.write_text('[{"a": 1.0, "b": "x"}, {"a": 2.0, "b": "y"}]')

    assert list(workflow.read_imported(csv, "csv").columns) == ["a", "b"]
    assert len(workflow.read_imported(parquet, "parquet")) == 2
    assert len(workflow.read_imported(as_json, "json")) == 2
