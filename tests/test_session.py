"""The session store and the ledger: isolation, ordering, and result ids.

The contract these tests hold to is in docs/analysis-tools.md. Two readers must
never share a frame, a result id or a stage; a stage cannot run before the ones
it depends on; and a figure a narrative quotes has to trace back to a stored
result.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import BaseModel

from stats import roles, session, stats_tools
from stats.errors import ToolError
from tests.conftest import FIXTURE


class Payload(BaseModel):
    estimate: float
    ci_low: float
    ci_high: float


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})


def test_two_sessions_do_not_share_a_frame(frame):
    first = session.SessionStore.from_frame(frame, source="first")
    second = session.SessionStore.from_frame(frame.head(2), source="second")

    with session.use_session(first):
        assert session.current_session().table().source == "first"
        assert stats_tools.describe_dataset().n_rows == 4
    with session.use_session(second):
        assert session.current_session().table().source == "second"
        assert stats_tools.describe_dataset().n_rows == 2
    assert session.current_session() is not second


def test_a_tool_reads_the_scoped_session_and_not_the_environment(frame):
    store = session.SessionStore.from_frame(frame)

    with session.use_session(store):
        assert stats_tools.describe_dataset().n_rows == 4
    assert stats_tools.describe_dataset().n_rows == 72


def test_an_empty_table_is_refused():
    with pytest.raises(ToolError, match="no rows"):
        session.SessionStore.from_frame(pd.DataFrame({"x": []}))


def test_an_unknown_table_lists_the_loaded_ones(frame):
    store = session.SessionStore.from_frame(frame)
    with pytest.raises(ToolError, match="primary"):
        store.table("secondary")


def test_results_are_stored_behind_an_id_and_keep_their_figures():
    store = session.SessionStore()
    stored = store.put(
        Payload(estimate=104.5034, ci_low=97.63, ci_high=111.38),
        kind="fit",
        summary="variety means",
    )

    assert stored.result_id.startswith("f_")
    assert store.get(stored.result_id) is stored
    assert store.ids() == [stored.result_id]
    assert "104.5034" in stored.quotable
    assert "104.50" in stored.quotable
    assert stored.as_payload()["payload"]["estimate"] == 104.5034


def test_a_figure_nobody_computed_is_not_quotable():
    store = session.SessionStore()
    stored = store.put(Payload(estimate=104.5034, ci_low=97.63, ci_high=111.38), kind="fit")

    assert stored.quotable, "the quotable set must not be empty"
    assert "104.5034" in stored.quotable
    assert "0.47" not in stored.quotable


def test_a_descriptive_kind_is_tagged_descriptive_without_being_told():
    """The class follows from the kind, so a query cannot be stored as evidence."""
    store = session.SessionStore()

    assert store.put(Payload(estimate=1.0, ci_low=0.0, ci_high=2.0), kind="query").evidence_class
    assert store.put(
        Payload(estimate=1.0, ci_low=0.0, ci_high=2.0), kind="plot"
    ).evidence_class == ("descriptive")
    assert store.put(Payload(estimate=1.0, ci_low=0.0, ci_high=2.0), kind="fit").evidence_class == (
        "inferential"
    )


def test_a_remedy_can_clear_a_block():
    """Verdicts belong to their stage, so re-running a tool replaces them."""
    state = session.AnalysisState()
    blocked = session.Verdict(
        check="leakage",
        status="blocked",
        statement="y_copy copies the outcome.",
        remedy="Drop y_copy and call profile_table again.",
        severity="blocking",
    )
    state.record("design")
    state.record("quality", status="blocked", verdicts=[blocked])
    assert state.blocking() == [blocked]

    state.record("quality", verdicts=[])

    assert state.blocking() == []
    assert state.status().complete is False  # later stages are still missing
    assert not any("leakage" in item for item in state.missing())


def test_an_unknown_result_id_lists_the_stored_ones():
    store = session.SessionStore()
    store.put(Payload(estimate=1.0, ci_low=0.0, ci_high=2.0), kind="fit")

    with pytest.raises(ToolError, match="Stored ids"):
        store.get("f_nope")


def test_a_query_result_carries_its_evidence_class():
    store = session.SessionStore()
    stored = store.put(
        Payload(estimate=0.61, ci_low=0.61, ci_high=0.61),
        kind="query",
        evidence_class="descriptive",
    )

    assert stored.evidence_class == "descriptive"


def test_a_table_can_be_tagged_with_its_source():
    store = session.SessionStore()
    ref = store.add_table(
        pd.DataFrame({"x": [1.0, 2.0]}),
        name="penguins",
        source="https://example.org/penguins.csv",
        sha256="abc",
        license_note="CC0",
    )

    assert ref.rows == 2
    assert store.primary == "penguins"
    assert store.table("penguins").license_note == "CC0"


def test_the_ledger_refuses_to_run_out_of_order():
    store = session.SessionStore()

    with pytest.raises(ToolError, match="frame_question"):
        store.state.require("quality")

    store.state.record("design")
    with pytest.raises(ToolError, match="profile_table"):
        store.state.require("assumptions")


def test_the_ledger_names_the_next_call_until_nothing_is_left():
    state = session.AnalysisState()

    assert "frame_question" in state.recommended_next()
    state.record("design", result_id="a_1")
    assert "profile_table" in state.recommended_next()
    for stage in ("quality", "assumptions", "estimate", "sensitivity", "interpretation"):
        state.record(stage)

    status = state.status()
    assert status.complete
    assert status.missing == []
    assert "Nothing blocking" in state.recommended_next()


def test_a_blocking_verdict_stops_the_next_stage():
    state = session.AnalysisState()
    state.record("design")
    state.record("quality")
    state.record(
        "assumptions",
        status="blocked",
        verdicts=[
            session.Verdict(
                check="leakage",
                status="blocked",
                statement="post_amount is recorded after the outcome.",
                remedy="Drop post_amount from the terms and refit.",
                severity="blocking",
            )
        ],
    )

    with pytest.raises(ToolError, match="Drop post_amount"):
        state.require("estimate")
    assert not state.status().complete


def test_open_items_are_reported_stage_by_stage():
    state = session.AnalysisState()
    state.record("design")
    state.record("quality", missing=["missingness in sex"])

    status = state.status()

    assert "quality: missingness in sex" in status.missing
    assert "assumptions: not started" in status.missing
    assert status.stages[0].status == "ok"
    assert status.stages[2].note == "not started"


def test_the_default_session_follows_the_environment_dataset(monkeypatch, tmp_path, frame):
    small = tmp_path / "small.csv"
    frame.to_csv(small, index=False)

    monkeypatch.setenv("AGENTIC_STATS_DATA", str(small))
    assert stats_tools.describe_dataset().n_rows == 4
    assert stats_tools.describe_dataset().path.endswith("small.csv")

    monkeypatch.setenv("AGENTIC_STATS_DATA", str(FIXTURE))
    assert stats_tools.describe_dataset().n_rows == 72


def test_an_edited_file_reloads(tmp_path, monkeypatch):
    """The env key carries mtime and size, so editing the file is picked up."""
    path = tmp_path / "oats.csv"
    path.write_text(FIXTURE.read_text())
    monkeypatch.setenv("AGENTIC_STATS_DATA", str(path))
    assert stats_tools.describe_dataset().n_rows == 72

    trimmed = pd.read_csv(FIXTURE).head(60)
    trimmed.to_csv(path, index=False)
    assert stats_tools.describe_dataset().n_rows == 60


def test_roles_read_the_active_session():
    wide = pd.DataFrame({"x": [float(i) for i in range(12)], "g": ["a", "b"] * 6})
    store = session.SessionStore.from_frame(wide)
    with session.use_session(store):
        profile = roles.infer_profile(session.current_session().frame())

    assert profile.best("response") == "x"
    assert profile.best("factor") == "g"


async def _read_after_await() -> session.SessionStore:
    """Read the session the way a chat tool does: after an await."""
    import asyncio

    await asyncio.sleep(0)
    return session.current_session()


def test_the_session_survives_an_await():
    """chatlas runs its tool callbacks inside the streaming task.

    The agent panel scopes its store around the whole turn, so the contextvar has
    to survive an await for the tools to see that store rather than the
    process-wide one.
    """
    import asyncio

    store = session.SessionStore.from_frame(pd.DataFrame({"x": [1.0, 2.0]}), source="awaited")
    with session.use_session(store):
        seen = asyncio.run(_read_after_await())

    assert seen is store
    assert seen.table().source == "awaited"


def test_a_table_can_carry_confirmed_roles(frame):
    store = session.SessionStore.from_frame(frame)
    ref = store.table().with_roles({"response": "x"})

    assert ref.roles == {"response": "x"}
    assert ref.name == "primary"
    assert store.table().roles == {}


def test_set_primary_switches_the_table(frame):
    store = session.SessionStore.from_frame(frame)
    store.add_table(frame.head(2), name="small")

    assert store.set_primary("small").rows == 2
    assert len(store.frame()) == 2

    with pytest.raises(ToolError, match="small"):
        store.set_primary("nope")


def test_reset_analysis_clears_results_and_the_ledger(frame):
    store = session.SessionStore.from_frame(frame)
    stored = store.put(Payload(estimate=1.0, ci_low=0.0, ci_high=2.0), kind="fit")
    store.state.record("design", result_id=stored.result_id)
    assert store.ids()

    store.reset_analysis()

    assert store.ids() == []
    assert store.state.missing()
    assert store.table().rows == 4


def test_refreshed_returns_a_new_object_with_the_same_content(frame):
    store = session.SessionStore.from_frame(frame)
    fresh = store.refreshed()

    assert fresh is not store
    assert set(fresh.tables) == set(store.tables)
    assert fresh.primary == store.primary


def test_a_figure_written_into_a_sentence_is_quotable():
    """A p-value that lives only in a notes string is still the tool's number."""

    class Noted(BaseModel):
        notes: list[str]

    store = session.SessionStore()
    stored = store.put(Noted(notes=["Omnibus p 0.2993 across the three levels."]), kind="fit")

    assert "0.2993" in stored.quotable


def test_a_result_id_is_not_a_figure():
    """Digits glued to letters must not enter the quotable set."""

    class Noted(BaseModel):
        next: str = ""

    store = session.SessionStore()
    stored = store.put(Noted(next="Call profile_table() then see f_3a91."), kind="fit")

    assert not any("91" in item for item in stored.quotable)
