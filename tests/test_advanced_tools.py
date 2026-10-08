"""Tests for marginal means, assumption checks, power and dose-response.

The dose-response tests write a synthetic dataset with a known EC50 or peak and
point AGENTIC_STATS_DATA at it, so the assertions are against a ground truth
rather than against whatever the fitted model happens to return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_stats import stats_tools

REAL_LEVEL_MEANS = {"A": 21.4384, "B": 24.7755, "C": 19.711}


def use_csv(monkeypatch, tmp_path, frame: pd.DataFrame) -> None:
    """Point the tools at a temporary dataset for one test."""
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setenv("AGENTIC_STATS_DATA", str(path))


@pytest.fixture
def sigmoid(monkeypatch, tmp_path):
    """A 4PL with a known bottom, top, EC50 and Hill slope, plus batch noise."""
    rng = np.random.default_rng(7)
    bottom, top, ec50, hill = 5.0, 100.0, 8.0, 1.5
    rows = []
    for batch in ["B01", "B02", "B03"]:
        batch_effect = rng.normal(0, 0.4)
        for dose in [0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0]:
            for _ in range(4):
                signal = (
                    bottom
                    + (top - bottom) * dose**hill / (ec50**hill + dose**hill)
                    + batch_effect
                    + rng.normal(0, 1.0)
                )
                rows.append({"batch": batch, "dose": dose, "signal": signal})
    use_csv(monkeypatch, tmp_path, pd.DataFrame(rows))
    return {"bottom": bottom, "top": top, "ec50": ec50, "hill": hill}


@pytest.fixture
def peaked(monkeypatch, tmp_path):
    """A parabola with its maximum at dose 6."""
    rng = np.random.default_rng(11)
    rows = [
        {"dose": float(dose), "signal": 10 + 4 * dose - (4 / 12) * dose**2 + rng.normal(0, 0.3)}
        for dose in np.linspace(0, 12, 13)
        for _ in range(5)
    ]
    use_csv(monkeypatch, tmp_path, pd.DataFrame(rows))
    return 6.0


def test_marginal_means_match_the_raw_level_means_on_a_balanced_design():
    result = stats_tools.marginal_means("assay_signal", ["formulation"], covariates=["dose"])

    assert result.model == "ols"
    by_level = {mean.level: mean.estimate for mean in result.marginal_means}
    assert by_level == pytest.approx(REAL_LEVEL_MEANS, abs=0.01)
    assert by_level["B"] > by_level["A"] > by_level["C"]


def test_marginal_means_confidence_intervals_bracket_the_estimate():
    result = stats_tools.marginal_means("assay_signal", ["formulation"])
    for mean in result.marginal_means:
        assert mean.ci_low < mean.estimate < mean.ci_high
        assert mean.std_error > 0
        assert mean.n_obs == 576


def test_marginal_means_pairwise_uses_tukey_and_finds_every_pair_different():
    result = stats_tools.marginal_means("assay_signal", ["formulation"])

    assert result.adjustment == "tukey"
    assert len(result.pairwise) == 3
    assert all(contrast.significant for contrast in result.pairwise)
    assert any("Tukey" in note for note in result.notes)

    b_vs_a = next(
        contrast
        for contrast in result.pairwise
        if {contrast.level_a, contrast.level_b} == {"A", "B"}
    )
    assert abs(b_vs_a.difference) == pytest.approx(3.3371, abs=0.01)


def test_marginal_means_can_use_a_mixed_model():
    result = stats_tools.marginal_means("assay_signal", ["formulation"], group="batch")

    assert result.model == "mixedlm"
    # A random intercept widens the interval: it carries between-batch variance.
    ols = stats_tools.marginal_means("assay_signal", ["formulation"])
    mixed_se = {m.level: m.std_error for m in result.marginal_means}
    ols_se = {m.level: m.std_error for m in ols.marginal_means}
    assert mixed_se["B"] > ols_se["B"]


def test_marginal_means_accepts_a_numeric_factor_at_its_observed_levels():
    result = stats_tools.marginal_means("assay_signal", ["dose"])
    assert [mean.level for mean in result.marginal_means] == ["0.0", "5.0", "10.0", "15.0"]


def test_marginal_means_rejects_an_unknown_adjustment():
    with pytest.raises(stats_tools.ToolError, match="Unknown adjustment"):
        stats_tools.marginal_means("assay_signal", ["formulation"], adjust="wat")


def test_marginal_means_rejects_an_unknown_column():
    with pytest.raises(stats_tools.ToolError, match="Available columns"):
        stats_tools.marginal_means("assay_signal", ["formulation_id"])


def test_check_assumptions_reports_the_expected_diagnostics():
    result = stats_tools.check_assumptions("assay_signal", ["formulation", "dose"])

    assert result.model == "ols"
    assert result.normality.test in {"shapiro_wilk", "dagostino_pearson"}
    assert len(result.qq) == 300
    assert len(result.residuals) == 300
    assert len(result.outliers) == 5
    names = {test.name for test in result.tests}
    assert {"breusch_pagan", "levene:formulation"} <= names
    assert result.vif["dose"] == pytest.approx(1.0, abs=0.01)
    assert result.max_vif == pytest.approx(1.3333, abs=0.01)


def test_unmodelled_batch_variation_is_what_breaks_normality():
    """The mixed model fixes the normality the OLS residual test rejects."""
    ols = stats_tools.check_assumptions("assay_signal", ["formulation", "dose"])
    mixed = stats_tools.check_assumptions("assay_signal", ["formulation", "dose"], group="batch")

    assert ols.normality.p_value < 0.05
    assert mixed.normality.p_value > 0.05
    assert mixed.model == "mixedlm"
    assert any("random intercept" in note for note in mixed.notes)


def test_check_assumptions_output_is_json_safe():
    """A nan or inf would make the payload invalid JSON for an MCP client."""
    payload = stats_tools.check_assumptions(
        "assay_signal", ["formulation", "dose"]
    ).model_dump_json()
    assert "NaN" not in payload
    assert "Infinity" not in payload


def test_power_analysis_reports_a_well_powered_design():
    result = stats_tools.power_analysis("assay_signal", "formulation")

    assert result.cohens_f > 0.5
    assert result.partial_eta_squared == pytest.approx(0.389, abs=0.01)
    assert result.achieved_power > 0.99
    assert result.required_n_per_group < result.n_per_group["A"]
    assert result.minimum_detectable_f < result.cohens_f


def test_power_analysis_sample_size_is_self_consistent():
    result = stats_tools.power_analysis("assay_signal", "formulation")

    # The n it recommends must actually reach the target power.
    from statsmodels.stats.power import FTestAnovaPower

    achieved = FTestAnovaPower().power(
        effect_size=result.cohens_f, nobs=result.required_n_total, alpha=0.05, k_groups=3
    )
    assert achieved >= result.target_power


def test_power_analysis_needs_more_rows_for_a_smaller_target_effect():
    observed = stats_tools.power_analysis("assay_signal", "formulation")
    smaller = stats_tools.power_analysis("assay_signal", "formulation", target_effect=0.5)
    assert smaller.required_n_total > observed.required_n_total
    assert any("response units" in note for note in smaller.notes)


def test_power_analysis_rejects_a_single_level_factor(monkeypatch, tmp_path):
    use_csv(
        monkeypatch,
        tmp_path,
        pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0], "only": ["x"] * 4}),
    )
    with pytest.raises(stats_tools.ToolError, match="fewer than 2"):
        stats_tools.power_analysis("signal", "only")


def test_power_analysis_accepts_a_multi_level_factor():
    assert stats_tools.power_analysis("assay_signal", "replicate").n_per_group


def test_four_pl_recovers_a_known_ec50(sigmoid):
    result = stats_tools.dose_response("signal", "dose", model="4pl")
    fit = result.fits[0]

    assert fit.converged
    assert fit.ec50 == pytest.approx(sigmoid["ec50"], rel=0.05)
    assert fit.ec50_ci_low < sigmoid["ec50"] < fit.ec50_ci_high
    assert fit.parameters[3].name == "hill"
    assert fit.parameters[3].estimate == pytest.approx(sigmoid["hill"], abs=0.3)
    assert fit.r_squared > 0.98
    assert len(fit.curve) == 50
    assert fit.aic is not None


def test_four_pl_runs_one_fit_per_group(sigmoid):
    result = stats_tools.dose_response("signal", "dose", model="4pl", by="batch")
    assert [fit.group for fit in result.fits] == ["B01", "B02", "B03"]
    assert all(fit.converged for fit in result.fits)


def test_four_pl_handles_a_zero_dose(sigmoid):
    """The safe algebraic form is what lets dose 0 into the fit at all."""
    fit = stats_tools.dose_response("signal", "dose", model="4pl").fits[0]
    zero = next(point for point in fit.curve if point.dose == 0.0)
    assert zero.predicted == pytest.approx(fit.parameters[0].estimate, abs=0.5)


def test_quadratic_finds_a_known_peak(peaked):
    fit = stats_tools.dose_response("signal", "dose", model="quadratic").fits[0]

    assert fit.converged
    assert fit.optimum_dose == pytest.approx(peaked, abs=0.2)
    assert fit.optimum_ci_low < peaked < fit.optimum_ci_high
    assert fit.ec50 is None


def test_quadratic_warns_when_the_peak_is_outside_the_tested_range():
    fit = stats_tools.dose_response("assay_signal", "dose", model="quadratic").fits[0]
    assert fit.converged
    assert fit.optimum_dose is not None
    assert fit.optimum_dose > 15
    assert any("extrapolation" in note for note in fit.notes)


def test_dose_response_rejects_an_unknown_model():
    with pytest.raises(stats_tools.ToolError, match="Unknown model"):
        stats_tools.dose_response("assay_signal", "dose", model="emax")


def test_dose_response_rejects_a_negative_dose(monkeypatch, tmp_path):
    use_csv(
        monkeypatch, tmp_path, pd.DataFrame({"dose": [-1.0, 0.0, 1.0], "signal": [1.0, 2.0, 3.0]})
    )
    with pytest.raises(stats_tools.ToolError, match="negative"):
        stats_tools.dose_response("signal", "dose")


def test_dose_response_reports_a_failed_fit_instead_of_raising(monkeypatch, tmp_path):
    """A flat response cannot define a curve; that is a result, not an exception."""
    use_csv(
        monkeypatch, tmp_path, pd.DataFrame({"dose": [0.0, 1.0, 2.0] * 4, "signal": [5.0] * 12})
    )
    fit = stats_tools.dose_response("signal", "dose", model="4pl").fits[0]
    assert not fit.converged
    assert fit.parameters == []
    assert fit.notes


def test_anova_effect_reports_effect_size_and_variance_check():
    result = stats_tools.anova_effect("assay_signal", "formulation")
    assert result.partial_eta_squared == pytest.approx(0.389, abs=0.01)
    assert 0 <= result.levene_p_value <= 1
