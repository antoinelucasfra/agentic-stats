"""Tests for the statistical tools against the oats split-plot trial.

The fixture is the real experiment: six blocks, three varieties, four nitrogen
rates, 72 rows. Reference values below come from that data, so a change in the
tools that shifts the numbers fails here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stats import stats_tools


@pytest.fixture(scope="module")
def description():
    return stats_tools.describe_dataset()


def test_dataset_shape_and_columns(description):
    assert description.n_rows == 72
    assert set(description.factors) == {"block", "variety"}
    assert set(description.numeric) == {"grain", "nitrogen"}
    assert description.columns[0].missing == 0
    assert "oats" in description.path


def test_mixed_model_recovers_block_variance_and_the_nitrogen_effect():
    result = stats_tools.fit_mixed_model("grain", ["variety", "nitrogen"], group="block")

    assert result.converged
    assert result.n_obs == 72
    assert result.group_variance == pytest.approx(245.05, rel=0.01)
    assert result.residual_variance == pytest.approx(234.73, rel=0.01)
    assert result.icc == pytest.approx(0.5108, abs=0.005)

    effects = {effect.term: effect for effect in result.fixed_effects}
    assert effects["nitrogen"].estimate == pytest.approx(73.67, abs=0.5)
    assert effects["nitrogen"].p_value < 1e-10
    assert effects["nitrogen"].ci_low < effects["nitrogen"].ci_high
    # Variety moves the mean, but not by enough for this many rows to settle.
    assert effects["variety[T.marvellous]"].p_value > 0.05
    assert effects["variety[T.victory]"].p_value > 0.05


def test_mixed_model_does_not_collapse_without_the_covariate():
    """lbfgs returned llf=inf and a zero variance for this exact model.

    The blocks differ by roughly 220 units of variance, so a reported zero would
    be a silent wrong answer rather than a boundary estimate.
    """
    result = stats_tools.fit_mixed_model("grain", ["variety"], group="block")

    assert result.converged
    assert result.group_variance > 100
    assert result.icc > 0.2
    assert not result.notes


def test_mixed_model_reports_icc_and_group_count():
    result = stats_tools.fit_mixed_model("grain", ["variety", "nitrogen"], group="block")

    assert result.n_groups == 6
    assert 0 < result.icc < 1
    assert result.icc == pytest.approx(
        result.group_variance / (result.group_variance + result.residual_variance), abs=1e-3
    )


def test_mixed_model_keeps_fixed_effects_in_the_order_asked_for():
    result = stats_tools.fit_mixed_model("grain", ["nitrogen", "variety"], group="block")
    assert [effect.term for effect in result.fixed_effects] == [
        "Intercept",
        "nitrogen",
        "variety[T.marvellous]",
        "variety[T.victory]",
    ]


def test_a_keyword_column_name_is_quoted_instead_of_crashing():
    """`yield` is a Python keyword, so patsy needs it quoted. Real data has it."""
    frame = pd.DataFrame(
        {
            "yield": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "lot": ["a", "a", "a", "b", "b", "b"],
            "day": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    with stats_tools.use_frame(frame):
        result = stats_tools.fit_mixed_model("yield", ["day"], group="lot")

    assert result.converged
    assert result.formula == 'Q("yield") ~ day'
    assert "day" in [effect.term for effect in result.fixed_effects]


def test_anova_effect_reports_significant_nitrogen_effect():
    result = stats_tools.anova_effect("grain", "nitrogen")

    assert result.p_value < 1e-5
    assert result.omega_squared == pytest.approx(0.3548, abs=0.01)
    assert result.level_means["0.6"] > result.level_means["0.0"]


def test_unknown_column_error_lists_available_columns():
    with pytest.raises(stats_tools.ToolError, match="Available columns"):
        stats_tools.anova_effect("grain", "block_id")


def test_non_numeric_response_is_rejected():
    with pytest.raises(stats_tools.ToolError, match="not numeric"):
        stats_tools.anova_effect("variety", "nitrogen")


def test_use_frame_overrides_the_dataset_for_the_block_only():
    subset = pd.read_csv(stats_tools.dataset_path()).head(60)

    assert stats_tools.describe_dataset().n_rows == 72
    with stats_tools.use_frame(subset):
        assert stats_tools.describe_dataset().n_rows == 60
    assert stats_tools.describe_dataset().n_rows == 72


def test_a_parquet_dataset_is_read_through_duckdb(monkeypatch, tmp_path):
    """Production reads the parquet cache, so the parquet branch needs a test."""
    import duckdb

    parquet = tmp_path / "oats.parquet"
    duckdb.connect().execute(
        f"copy (select * from read_csv_auto('{stats_tools.dataset_path().as_posix()}')) "
        f"to '{parquet.as_posix()}' (format parquet)"
    )
    monkeypatch.setenv("AGENTIC_STATS_DATA", str(parquet))

    description = stats_tools.describe_dataset()
    assert description.n_rows == 72
    assert set(description.factors) == {"block", "variety"}
    assert description.path.endswith("oats.parquet")
    assert stats_tools.anova_effect("grain", "nitrogen").p_value < 1e-5
