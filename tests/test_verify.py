"""The verify pass: a figure in the prose has to exist in a stored result.

This is the check that makes "no invented numbers" mechanical. It is
deterministic, so it needs no model and no network, and the allowance list is
tested so its size stays a decision rather than an accident.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import BaseModel

from stats import session, verify


class Plan(BaseModel):
    """The part of an analysis plan the verify pass reads."""

    target: str
    design: str = "independent"


class Numbers(BaseModel):
    estimate: float
    ci_low: float
    ci_high: float


@pytest.fixture
def store() -> session.SessionStore:
    store = session.SessionStore.from_frame(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    store.put(Numbers(estimate=104.5034, ci_low=97.63, ci_high=111.38), kind="fit")
    return store


def test_a_stored_figure_is_grounded(store):
    grounded, ungrounded, _excused = verify.ungrounded_numbers("The mean was 104.50 grams.", store)

    assert "104.50" in grounded
    assert ungrounded == []


def test_an_invented_figure_is_reported(store):
    _grounded, ungrounded, _excused = verify.ungrounded_numbers("The mean was 118.42 grams.", store)

    assert ungrounded == ["118.42"]


def test_a_structural_figure_is_excused_rather_than_grounded(store):
    grounded, ungrounded, excused = verify.ungrounded_numbers(
        "The 95 percent interval at alpha 0.05 excludes zero.", store
    )

    assert "95" in excused
    assert "0.05" in excused
    assert ungrounded == []
    assert grounded == []


def test_a_year_is_excused(store):
    _grounded, ungrounded, excused = verify.ungrounded_numbers("Collected in 1935.", store)

    assert ungrounded == []
    assert "1935" in excused


def test_a_thousands_separator_still_matches(store):
    store.put(Numbers(estimate=104503.0, ci_low=1.0, ci_high=2.0), kind="fit")

    _grounded, ungrounded, _excused = verify.ungrounded_numbers("Total 104,503 units.", store)

    assert ungrounded == []


def test_causal_language_is_flagged_when_the_design_never_permitted_it(store):
    report = verify.verify_answer("Nitrogen causes higher yield.", store)

    assert report.verdict == "review"
    assert "causes" in report.causal_terms
    assert not report.causal_claims_permitted


def test_causal_language_passes_once_the_design_stage_permits_it(store):
    plan = store.put(Plan(target="causal", design="clustered"), kind="plan")
    store.state.record("design", result_id=plan.result_id)

    report = verify.verify_answer("Nitrogen causes higher yield.", store)

    assert report.causal_claims_permitted
    assert report.causal_terms
    assert report.verdict == "clean"


def test_attributional_phrasing_is_reported_separately(store):
    report = verify.verify_answer("The effect of variety on yield is small.", store)

    assert report.soft_terms
    assert report.verdict == "clean"


def test_a_clean_answer_reports_clean(store):
    report = verify.verify_answer("The mean was 104.50 with interval 97.63 to 111.38.", store)

    assert report.verdict == "clean"
    assert report.ungrounded == []
    assert len(report.grounded) == 3


def test_a_figure_from_a_descriptive_result_is_not_enough_for_an_effect_claim(store):
    """The contract's headline rule: a SQL difference is descriptive, not evidence."""
    query = store.put(Numbers(estimate=3.21, ci_low=3.21, ci_high=3.21), kind="query")
    assert store.get(query.result_id).evidence_class == "descriptive"

    report = verify.verify_answer("Nitrogen raises yield by 3.21 units.", store)

    assert report.verdict == "review"
    assert report.misclassified == ["3.21"]
    assert any("descriptive result" in note for note in report.notes)


def test_the_same_figure_is_fine_in_a_descriptive_sentence(store):
    store.put(Numbers(estimate=3.21, ci_low=3.21, ci_high=3.21), kind="query")

    report = verify.verify_answer("The total for the north region was 3.21.", store)

    assert report.verdict == "clean"
    assert report.misclassified == []


def test_an_effect_claim_with_nothing_inferential_is_reviewed(store):
    empty = session.SessionStore.from_frame(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))

    report = verify.verify_answer("Variety raises yield.", empty)

    assert report.verdict == "review"
    assert not report.inferential_available
    assert any("no inferential result" in note for note in report.notes)


def test_a_figure_stored_by_an_inferential_result_passes_an_effect_claim(store):
    report = verify.verify_answer("Nitrogen raises yield by 104.50 units.", store)

    assert report.inferential_available
    assert report.misclassified == []
    assert report.verdict == "clean"
