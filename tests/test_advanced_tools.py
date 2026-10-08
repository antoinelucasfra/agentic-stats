"""Tests for marginal means, assumption checks, power and dose-response.

The dose-response tests write a synthetic dataset with a known EC50 or peak and
point AGENTIC_STATS_DATA at it, so the assertions are against a ground truth
rather than against whatever the fitted model happens to return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stats import stats_tools

REAL_LEVEL_MEANS = {"golden rain": 104.5, "marvellous": 109.7917, "victory": 97.625}


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
    result = stats_tools.marginal_means("grain", ["variety"], covariates=["nitrogen"])

    assert result.model == "ols"
    by_level = {mean.level: mean.estimate for mean in result.marginal_means}
    assert by_level == pytest.approx(REAL_LEVEL_MEANS, abs=0.01)
    assert by_level["marvellous"] > by_level["golden rain"] > by_level["victory"]


def test_marginal_means_confidence_intervals_bracket_the_estimate():
    result = stats_tools.marginal_means("grain", ["variety"])
    for mean in result.marginal_means:
        assert mean.ci_low < mean.estimate < mean.ci_high
        assert mean.std_error > 0
        assert mean.n_obs == 24


def test_marginal_means_pairwise_uses_tukey_and_finds_no_winner():
    """Yates' oats: variety really is not distinguishable, nitrogen is."""
    result = stats_tools.marginal_means("grain", ["variety"])

    assert result.adjustment == "tukey"
    assert len(result.pairwise) == 3
    assert not any(contrast.significant for contrast in result.pairwise)
    assert any("Tukey" in note for note in result.notes)

    rain_vs_marvellous = next(
        contrast
        for contrast in result.pairwise
        if {contrast.level_a, contrast.level_b} == {"golden rain", "marvellous"}
    )
    assert abs(rain_vs_marvellous.difference) == pytest.approx(5.2917, abs=0.01)


def test_marginal_means_can_use_a_mixed_model():
    """Variety is a whole-plot factor, so block variance widens its interval."""
    result = stats_tools.marginal_means("grain", ["variety"], group="block")

    assert result.model == "mixedlm"
    ols = stats_tools.marginal_means("grain", ["variety"])
    mixed_se = {m.level: m.std_error for m in result.marginal_means}
    ols_se = {m.level: m.std_error for m in ols.marginal_means}
    assert mixed_se["marvellous"] > ols_se["marvellous"]


def test_marginal_means_accepts_a_numeric_factor_at_its_observed_levels():
    result = stats_tools.marginal_means("grain", ["nitrogen"])
    assert [mean.level for mean in result.marginal_means] == ["0.0", "0.2", "0.4", "0.6"]


def test_marginal_means_rejects_an_unknown_adjustment():
    with pytest.raises(stats_tools.ToolError, match="Unknown adjustment"):
        stats_tools.marginal_means("grain", ["variety"], adjust="wat")


def test_marginal_means_rejects_an_unknown_column():
    with pytest.raises(stats_tools.ToolError, match="Available columns"):
        stats_tools.marginal_means("grain", ["variety_id"])


def test_check_assumptions_reports_the_expected_diagnostics():
    result = stats_tools.check_assumptions("grain", ["variety", "nitrogen"])

    assert result.model == "ols"
    assert result.normality.test in {"shapiro_wilk", "dagostino_pearson"}
    assert len(result.qq) == 72
    assert len(result.residuals) == 72
    assert len(result.outliers) == 5
    names = {test.name for test in result.tests}
    assert {"breusch_pagan", "levene:variety"} <= names
    assert result.vif["nitrogen"] == pytest.approx(1.0, abs=0.01)
    assert result.max_vif == pytest.approx(1.3333, abs=0.01)


def test_unmodelled_block_variation_is_what_breaks_normality():
    """The mixed model fixes the normality the OLS residual test rejects."""
    ols = stats_tools.check_assumptions("grain", ["variety", "nitrogen"])
    mixed = stats_tools.check_assumptions("grain", ["variety", "nitrogen"], group="block")

    assert ols.normality.p_value < 0.05
    assert mixed.normality.p_value > 0.05
    assert mixed.model == "mixedlm"
    assert any("random intercept" in note for note in mixed.notes)


def test_check_assumptions_output_is_json_safe():
    """A nan or inf would make the payload invalid JSON for an MCP client."""
    payload = stats_tools.check_assumptions("grain", ["variety", "nitrogen"]).model_dump_json()
    assert "NaN" not in payload
    assert "Infinity" not in payload


def test_power_analysis_reports_a_well_powered_design():
    result = stats_tools.power_analysis("grain", "nitrogen")

    assert result.cohens_f > 0.5
    assert result.partial_eta_squared == pytest.approx(0.3851, abs=0.01)
    assert result.achieved_power > 0.99
    assert result.required_n_per_group < result.n_per_group["0.0"]
    assert result.minimum_detectable_f < result.cohens_f


def test_power_analysis_sample_size_is_self_consistent():
    result = stats_tools.power_analysis("grain", "nitrogen")

    # The n it recommends must actually reach the target power.
    from statsmodels.stats.power import FTestAnovaPower

    achieved = FTestAnovaPower().power(
        effect_size=result.cohens_f, nobs=result.required_n_total, alpha=0.05, k_groups=4
    )
    assert achieved >= result.target_power


def test_power_analysis_needs_more_rows_for_a_smaller_target_effect():
    observed = stats_tools.power_analysis("grain", "nitrogen")
    smaller = stats_tools.power_analysis("grain", "nitrogen", target_effect=0.5)
    assert smaller.required_n_total > observed.required_n_total
    assert any("response units" in note for note in smaller.notes)


def test_power_analysis_rejects_a_single_level_factor(monkeypatch, tmp_path):
    use_csv(
        monkeypatch,
        tmp_path,
        pd.DataFrame({"grain": [1.0, 2.0, 3.0, 4.0], "only": ["x"] * 4}),
    )
    with pytest.raises(stats_tools.ToolError, match="fewer than 2"):
        stats_tools.power_analysis("grain", "only")


def test_power_analysis_accepts_a_multi_level_factor():
    assert stats_tools.power_analysis("grain", "block").n_per_group


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
    fit = stats_tools.dose_response("grain", "nitrogen", model="quadratic").fits[0]
    assert fit.converged
    assert fit.optimum_dose is not None
    assert fit.optimum_dose > 0.6
    assert any("extrapolation" in note for note in fit.notes)


def test_dose_response_rejects_an_unknown_model():
    with pytest.raises(stats_tools.ToolError, match="Unknown model"):
        stats_tools.dose_response("grain", "nitrogen", model="emax")


def test_dose_response_rejects_a_negative_dose(monkeypatch, tmp_path):
    use_csv(
        monkeypatch,
        tmp_path,
        pd.DataFrame({"nitrogen": [-1.0, 0.0, 1.0], "grain": [1.0, 2.0, 3.0]}),
    )
    with pytest.raises(stats_tools.ToolError, match="negative"):
        stats_tools.dose_response("grain", "nitrogen")


def test_dose_response_reports_a_failed_fit_instead_of_raising(monkeypatch, tmp_path):
    """A flat response cannot define a curve; that is a result, not an exception."""
    use_csv(
        monkeypatch,
        tmp_path,
        pd.DataFrame({"nitrogen": [0.0, 1.0, 2.0] * 4, "grain": [5.0] * 12}),
    )
    fit = stats_tools.dose_response("grain", "nitrogen", model="4pl").fits[0]
    assert not fit.converged
    assert fit.parameters == []
    assert fit.notes


def test_anova_effect_reports_effect_size_and_variance_check():
    result = stats_tools.anova_effect("grain", "nitrogen")
    assert result.partial_eta_squared == pytest.approx(0.3851, abs=0.01)
    assert 0 <= result.levene_p_value <= 1


def test_curve_family_picks_the_supported_shape_on_oats():
    result = stats_tools.fit_curve_family("grain", "nitrogen")

    assert len(result.comparisons) == 1
    comparison = result.comparisons[0]
    assert {candidate.model for candidate in comparison.candidates} == {
        "linear",
        "quadratic",
        "4pl",
    }
    assert all(candidate.converged for candidate in comparison.candidates)
    assert comparison.best in {"linear", "quadratic", "4pl"}
    assert set(comparison.delta_aic) == {candidate.model for candidate in comparison.candidates} - {
        comparison.best
    }
    assert all(gap >= 0 for gap in comparison.delta_aic.values())


def test_curve_family_repeats_the_comparison_per_group():
    result = stats_tools.fit_curve_family("grain", "nitrogen", by="variety")

    assert [comparison.group for comparison in result.comparisons] == [
        "golden rain",
        "marvellous",
        "victory",
    ]
    assert all(comparison.best for comparison in result.comparisons)


def test_curve_family_names_the_saturating_shape(sigmoid):
    comparison = stats_tools.fit_curve_family("signal", "dose").comparisons[0]

    assert comparison.best == "4pl"
    assert comparison.delta_aic["linear"] > 10


def test_curve_family_rejects_a_negative_dose(monkeypatch, tmp_path):
    use_csv(
        monkeypatch,
        tmp_path,
        pd.DataFrame({"nitrogen": [-1.0, 0.0, 1.0], "grain": [1.0, 2.0, 3.0]}),
    )
    with pytest.raises(stats_tools.ToolError, match="negative"):
        stats_tools.fit_curve_family("grain", "nitrogen")


def test_correlate_reports_pearson_and_spearman():
    result = stats_tools.correlate()

    assert set(result.columns) == {"grain", "nitrogen"}
    assert result.n_obs == 72
    assert result.pearson["grain"]["nitrogen"] == pytest.approx(0.613, abs=0.01)
    assert result.spearman["grain"]["grain"] == 1.0
    assert any("nitrogen-grain" in note for note in result.notes)


def test_correlate_rejects_non_numeric_and_lone_columns():
    with pytest.raises(stats_tools.ToolError, match="not numeric"):
        stats_tools.correlate(["grain", "variety"])
    with pytest.raises(stats_tools.ToolError, match="at least two"):
        stats_tools.correlate(["grain"])
