"""Role inference: what a table can answer, and the reasons for each guess.

The point of these tests is the defaults, because a wrong default is what makes
a generic tool useless on a new dataset. The oats trial has a nuisance column
that is also the best factor candidate, and a business table has a binary column
that looks like a factor but is the outcome.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stats import roles, stats_tools


@pytest.fixture
def oats() -> pd.DataFrame:
    return stats_tools.load_dataset()


@pytest.fixture
def orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(40)],
            "created_at": pd.date_range("2024-01-01", periods=40, freq="D"),
            "revenue": [float(i) * 3 + 10 for i in range(40)],
            "country": ["FR", "DE", "ES", "IT"] * 10,
            "converted": [0, 1] * 20,
        }
    )


def test_oats_columns_get_the_roles_the_tools_need(oats):
    profile = roles.infer_profile(oats)

    assert profile.best("response") == "grain"
    assert profile.best("factor") == "variety"
    assert profile.best("group") == "block"
    assert profile.best("dose") == "nitrogen"
    assert profile.best("time") is None


def test_a_nuisance_column_does_not_also_win_the_factor_role(oats):
    """Block is the grouping column; variety is the factor a question is about."""
    profile = roles.infer_profile(oats)

    assert "block" in profile.columns("factor")
    assert profile.best("factor") != profile.best("group")


def test_a_business_table_gets_time_identifier_and_leakage_roles(orders):
    profile = roles.infer_profile(orders)

    assert profile.best("time") == "created_at"
    assert profile.best("response") == "revenue"
    assert profile.best("factor") == "country"
    assert "order_id" in profile.columns("identifier")
    assert "converted" in profile.columns("leakage")
    assert any("leakage" in note for note in profile.notes)


def test_a_binary_column_is_offered_as_an_outcome(orders):
    profile = roles.infer_profile(orders)
    converted = [g for g in profile.guesses if g.column == "converted"]

    assert any(g.role == "response" and g.confidence == 0.6 for g in converted)


def test_every_guess_carries_its_reason(oats):
    profile = roles.infer_profile(oats)

    assert profile.guesses
    assert all(guess.reason for guess in profile.guesses)
    assert all(0 < guess.confidence <= 1 for guess in profile.guesses)


def test_a_table_with_no_grouping_column_says_so():
    solo = pd.DataFrame({"value": [float(i) for i in range(30)]})

    assert roles.infer_profile(solo).best("group") is None
    assert any("grouping" in note for note in roles.infer_profile(solo).notes)


def test_a_repeated_categorical_column_is_offered_as_a_group():
    frame = pd.DataFrame(
        {"value": [float(i) for i in range(30)], "treatment": ["a", "b", "c"] * 10}
    )
    profile = roles.infer_profile(frame)

    assert profile.best("group") == "treatment"
    assert not any("grouping" in note for note in profile.notes)


def test_a_single_categorical_column_can_fill_more_than_one_role():
    """Sharing is allowed when no confident alternative exists."""
    frame = pd.DataFrame({"value": [float(i) for i in range(30)], "cohort": ["a", "b"] * 15})
    profile = roles.infer_profile(frame)

    assert profile.best("factor") == "cohort"
    assert profile.best("group") == "cohort"


def test_missing_values_and_constants_are_reported():
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, None, 4.0],
            "flat": ["x", "x", "x", "x"],
        }
    )
    profile = roles.infer_profile(frame)

    assert any("missing values" in note for note in profile.notes)
    assert any("Constant columns" in note for note in profile.notes)


def test_a_unique_integer_column_is_a_measurement_not_an_identifier():
    """Order counts and prices in cents are all distinct; they are not ids."""
    frame = pd.DataFrame({"order_count": [1, 2, 3, 4, 5, 6, 7, 8], "region": ["a", "b"] * 4})
    profile = roles.infer_profile(frame)

    assert "order_count" not in profile.columns("identifier")
    assert profile.best("response") == "order_count"


def test_a_unique_text_column_is_still_an_identifier():
    frame = pd.DataFrame({"code": [f"c{index}" for index in range(12)]})
    profile = roles.infer_profile(frame)

    assert "code" in profile.columns("identifier")


def test_a_column_whose_values_parse_as_dates_is_time():
    frame = pd.DataFrame(
        {
            "when": [f"2024-01-{day:02d}" for day in range(1, 21)],
            "value": [float(day) for day in range(1, 21)],
        }
    )
    profile = roles.infer_profile(frame)

    assert profile.best("time") == "when"


def test_a_confirmed_role_wins_over_the_guess(oats):
    profile = roles.infer_profile(oats, overrides={"response": "nitrogen"})

    assert profile.best("response") == "nitrogen"
    assert any("confirmed as nitrogen" in note for note in profile.notes)


def test_an_override_for_a_missing_column_is_reported_not_applied(oats):
    profile = roles.infer_profile(oats, overrides={"response": "yield", "group": "block"})

    assert profile.best("response") == "grain"
    assert profile.best("group") == "block"
    assert any("no such column" in note for note in profile.notes)


def test_an_override_for_a_non_role_is_reported(oats):
    profile = roles.infer_profile(oats, overrides={"outcome": "grain"})

    assert any("not a role" in note for note in profile.notes)
