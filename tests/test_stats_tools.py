"""Tests for the statistical tools against the synthetic DOE dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from agentic_stats import stats_tools


@pytest.fixture(scope="module")
def description():
    return stats_tools.describe_dataset()


def test_dataset_shape_and_columns(description):
    assert description.n_rows == 12 * 3 * 4 * 3 * 4
    assert set(description.factors) >= {"batch", "formulation", "operator"}
    assert "assay_signal" in description.numeric
    assert description.columns[0].missing == 0


def test_mixed_model_recovers_batch_variance_and_formulation_effect():
    result = stats_tools.fit_mixed_model("assay_signal", ["formulation", "dose"], group="batch")

    assert result.converged
    assert result.n_obs == 12 * 3 * 4 * 3 * 4
    assert result.group_variance > 0.5  # batch-to-batch variation is real in the generator
    assert result.residual_variance > 0.5

    effects = {effect.term: effect for effect in result.fixed_effects}
    assert effects["formulation[T.B]"].estimate > 0  # B is the strongest formulation
    assert effects["formulation[T.B]"].p_value < 0.01
    assert effects["dose"].estimate > 0
    assert effects["dose"].ci_low < effects["dose"].ci_high


def test_mixed_model_does_not_collapse_without_the_dose_term():
    """lbfgs returned llf=inf and a zero variance for this exact model.

    The generator gives batches an SD of 2.0, so a reported variance of 0 is a
    silent wrong answer rather than a boundary estimate.
    """
    result = stats_tools.fit_mixed_model("assay_signal", ["formulation"], group="batch")

    assert result.converged
    assert result.group_variance > 2.0
    assert result.icc > 0.3
    assert not result.notes


def test_mixed_model_reports_icc_and_group_count():
    result = stats_tools.fit_mixed_model("assay_signal", ["formulation", "dose"], group="batch")

    assert result.n_groups == 12
    assert 0 < result.icc < 1
    assert result.icc == pytest.approx(
        result.group_variance / (result.group_variance + result.residual_variance), abs=1e-3
    )


def test_mixed_model_keeps_fixed_effects_in_the_order_asked_for():
    result = stats_tools.fit_mixed_model("assay_signal", ["dose", "formulation"], group="batch")
    assert [effect.term for effect in result.fixed_effects] == [
        "Intercept",
        "dose",
        "formulation[T.B]",
        "formulation[T.C]",
    ]


def test_anova_effect_reports_significant_formulation_effect():
    result = stats_tools.anova_effect("assay_signal", "formulation")

    assert result.p_value < 0.01
    assert result.omega_squared > 0.01
    assert result.level_means["B"] > result.level_means["A"]


def test_unknown_column_error_lists_available_columns():
    with pytest.raises(stats_tools.ToolError, match="Available columns"):
        stats_tools.anova_effect("assay_signal", "batch_id")


def test_non_numeric_response_is_rejected():
    with pytest.raises(stats_tools.ToolError, match="not numeric"):
        stats_tools.anova_effect("formulation", "dose")


def test_use_frame_overrides_the_dataset_for_the_block_only():
    subset = pd.read_csv(stats_tools.dataset_path()).head(60)

    assert stats_tools.describe_dataset().n_rows == 12 * 3 * 4 * 3 * 4
    with stats_tools.use_frame(subset):
        assert stats_tools.describe_dataset().n_rows == 60
    assert stats_tools.describe_dataset().n_rows == 12 * 3 * 4 * 3 * 4
