"""Tests for the estimation tools: fit, compare, margins, sensitivity.

The oats fixture carries the success paths, because its numbers are known. The
failure paths (separation, unequal variance, non-normal small cells, an effect
driven by two rows) need frames built for them, so those come from a fixed numpy
seed.

Every tool reads the ledger before it computes, so each test walks the stages in
order through `gated()` rather than calling a tool on a bare session.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from stats import session, stats_tools
from stats.errors import ToolError
from stats.estimates import compare_groups, fit_model, marginal_effects, sensitivity

SEED = 20260101


@contextmanager
def gated(frame: pd.DataFrame) -> Iterator[session.SessionStore]:
    """A session whose ledger holds design and quality, which the tools require."""
    store = session.SessionStore.from_frame(frame, source="test")
    store.state.record("design", note="test fixture")
    store.state.record("quality", note="test fixture")
    with session.use_session(store) as active:
        yield active


def oats() -> pd.DataFrame:
    """The fixture, copied: `load_dataset` hands back the session's own frame."""
    return stats_tools.load_dataset().copy()


def unequal_variance() -> pd.DataFrame:
    """Two groups, same mean, very different spreads: Levene must decide the route."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(
        {
            "y": np.concatenate([rng.normal(0.0, 1.0, 30), rng.normal(0.4, 8.0, 30)]),
            "g": ["a"] * 30 + ["b"] * 30,
        }
    )


def small_skewed_cells() -> pd.DataFrame:
    """Three levels of eight rows each, drawn from an exponential: means mislead."""
    rng = np.random.default_rng(SEED)
    shift = {"a": 0.0, "b": 1.0, "c": 8.0}
    rows = [
        {"y": shift[level] + value, "f": level}
        for level in ("a", "b", "c")
        for value in rng.exponential(1.0, 8)
    ]
    return pd.DataFrame(rows)


def two_point_driven_effect() -> pd.DataFrame:
    """A downward slope that two high-leverage rows turn upward."""
    rng = np.random.default_rng(SEED)
    x = np.linspace(0.0, 3.0, 30)
    frame = pd.DataFrame({"x": x, "y": -0.5 * x + rng.normal(0.0, 0.2, 30)})
    frame.loc[30] = [3.0, 200.0]
    frame.loc[31] = [3.0, 210.0]
    return frame


def separated() -> pd.DataFrame:
    """A threshold on x predicts y perfectly: complete separation."""
    frame = pd.DataFrame({"x": np.arange(20, dtype=float)})
    frame["y"] = (frame.x > 10).astype(int)
    return frame


# --------------------------------------------------------------------------- #
# fit_model
# --------------------------------------------------------------------------- #


def test_fit_model_records_both_stages_and_returns_intervals():
    with gated(oats()) as store:
        result = fit_model("grain rises with nitrogen", "grain", ["variety", "nitrogen"])

        assert result.n_obs == 72
        assert result.formula == "grain ~ variety + nitrogen"
        assert result.reference_levels == {"variety": "golden rain"}
        coefficients = {item.term: item for item in result.coefficients}
        assert coefficients["nitrogen"].estimate == pytest.approx(73.67, abs=0.1)
        assert coefficients["nitrogen"].ci_low < coefficients["nitrogen"].estimate
        assert coefficients["nitrogen"].ci_high > coefficients["nitrogen"].estimate
        assert coefficients["nitrogen"].scale == "response"
        assert (
            coefficients["variety[T.marvellous]"].ci_low
            < coefficients["variety[T.marvellous]"].ci_high
        )

        # Both stages carry this result's id, and the store holds it as a fit.
        assert store.state.records["assumptions"].result_id == result.result_id
        assert store.state.records["estimate"].result_id == result.result_id
        stored = store.get(result.result_id)
        assert stored.kind == "fit"
        assert stored.rows_total == 72
        assert stored.rows_used == 72
        assert stored.payload.result_id == result.result_id

        assert result.next.startswith("Call marginal_effects(")
        assert result.rows_total == 72
        assert result.rows_used == 72
        assert result.cannot_say


def test_fit_model_reports_every_named_diagnostic_with_a_remedy():
    with gated(oats()) as store:
        result = fit_model("grain rises with nitrogen", "grain", ["variety", "nitrogen"])
        checks = {verdict.check for verdict in result.diagnostics}
        assert {
            "convergence",
            "heteroscedasticity",
            "residual_normality",
            "influence",
            "collinearity",
            "cell_size",
        } <= checks
        failed = [v for v in result.diagnostics if v.status != "ok"]
        assert failed
        for verdict in failed:
            assert verdict.remedy, verdict.check
            assert "(" in verdict.remedy and ")" in verdict.remedy
        assert any("sensitivity(" in verdict.remedy for verdict in failed)
        assert store.state.records["assumptions"].status == "caution"


def test_fit_model_counts_rows_it_dropped():
    frame = oats()
    frame.loc[frame.index[:4], "nitrogen"] = np.nan
    with gated(frame) as store:
        result = fit_model("grain rises with nitrogen", "grain", ["variety", "nitrogen"])
        assert result.rows_total == 72
        assert result.rows_used == 68
        assert result.exclusions and result.exclusions[0]["n"] == 4
        assert store.get(result.result_id).exclusions[0]["n"] == 4


def test_fit_model_refuses_an_unknown_column():
    with gated(oats()):
        with pytest.raises(ToolError) as caught:
            fit_model("x", "grain", ["variety", "potassium"])
        message = str(caught.value)
        assert "Unknown column" in message
        assert "potassium" in message
        assert "Available columns" in message


def test_fit_model_refuses_a_constant_response():
    with gated(oats().assign(grain=5.0)):
        with pytest.raises(ToolError) as caught:
            fit_model("x", "grain", ["variety"])
        message = str(caught.value)
        assert "constant" in message
        assert "describe_dataset" in message


def test_fit_model_refuses_a_term_with_a_single_level():
    with gated(oats().assign(constant=1.0)):
        with pytest.raises(ToolError) as caught:
            fit_model("x", "grain", ["variety", "constant"])
        message = str(caught.value)
        assert "'constant' has a single value" in message
        assert "Drop it from terms" in message


def test_fit_model_refuses_a_term_that_is_an_exact_linear_combination():
    frame = oats().assign(nitrogen_again=oats().nitrogen * 2)
    with gated(frame):
        with pytest.raises(ToolError) as caught:
            fit_model("x", "grain", ["nitrogen", "nitrogen_again"])
        message = str(caught.value)
        assert "exact linear combination" in message
        assert "nitrogen_again" in message


def test_fit_model_refuses_a_binomial_response_that_is_not_two_valued():
    frame = oats()
    frame["three_way"] = np.where(frame.grain > 110, 2, np.where(frame.grain > 90, 1, 0))
    with gated(frame):
        with pytest.raises(ToolError) as caught:
            fit_model("x", "three_way", ["nitrogen"], family="binomial")
        assert "two-valued" in str(caught.value)
    with gated(oats().assign(coded=np.where(oats().grain > 100, 2, 1))):
        with pytest.raises(ToolError) as caught:
            fit_model("x", "coded", ["nitrogen"], family="binomial")
        assert "two-valued" in str(caught.value)


def test_fit_model_flags_complete_separation_as_blocking():
    with gated(separated()) as store:
        result = fit_model("y rises with x", "y", ["x"], family="binomial")
        separation = [v for v in result.diagnostics if v.check == "separation"]
        assert separation
        assert separation[0].status == "blocked"
        assert separation[0].severity == "blocking"
        assert "complete separation" in separation[0].statement
        assert "fit_model" in separation[0].remedy
        assert store.state.blocking()
        # A blocked fit stops the tools that come after it.
        with pytest.raises(ToolError) as caught:
            sensitivity(result.result_id)
        assert "separation" in str(caught.value)


def test_fit_model_transforms_effects_for_a_logit_fit():
    frame = oats().assign(high=(oats().grain > oats().grain.median()).astype(int))
    with gated(frame):
        result = fit_model("probability of a high yield", "high", ["nitrogen"], family="binomial")
        coefficient = result.coefficients[1]
        effect = result.effects[0]
        assert coefficient.scale == "link"
        assert effect.scale == "response"
        assert effect.estimate == pytest.approx(float(np.exp(coefficient.estimate)), rel=1e-3)
        assert "odds ratio" in effect.contrast
        assert not any(v.status == "blocked" for v in result.diagnostics)


# --------------------------------------------------------------------------- #
# compare_groups
# --------------------------------------------------------------------------- #


def test_compare_groups_routes_to_welch_and_says_why():
    with gated(unequal_variance()):
        result = compare_groups("y", "g")
        assert result.test == "welch_t"
        assert "Levene" in result.routing_reason
        assert "Welch" in result.routing_reason
        assert result.effect_size.name == "cohens_d"
        assert result.effect_size.ci_low < result.effect_size.estimate
        assert result.effect_size.ci_high > result.effect_size.estimate
        variance = {v.check: v for v in result.diagnostics}
        assert variance["equal_variance"].status == "caution"
        assert "welch_t" in variance["equal_variance"].statement


def test_compare_groups_routes_to_kruskal_on_non_normal_small_cells():
    with gated(small_skewed_cells()):
        result = compare_groups("y", "f")
        assert result.test == "kruskal_wallis"
        assert "Shapiro" in result.routing_reason
        assert "Kruskal-Wallis" in result.routing_reason
        assert result.effect_size.name == "epsilon_squared"
        assert result.effect_size.ci_low < result.effect_size.estimate
        assert result.effect_size.ci_high > result.effect_size.estimate
        assert not any(v.status == "blocked" for v in result.diagnostics)


def test_compare_groups_returns_an_effect_size_with_an_interval_on_the_oats_factor():
    with gated(oats()):
        result = compare_groups("grain", "variety")
        assert result.test in {"student_t", "one_way_anova", "welch_anova"}
        assert result.effect_size.name in {"cohens_d", "eta_squared"}
        assert result.effect_size.ci_low < result.effect_size.estimate
        assert result.effect_size.ci_high > result.effect_size.estimate
        assert result.effect_size.interpretation
        assert result.contrasts
        assert all(contrast.method for contrast in result.contrasts)
        payload = result.effect_size
        assert payload.ci_high - payload.ci_low > 0


def test_compare_groups_speaks_equivalence_when_the_interval_excludes_the_threshold():
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame(
        {
            "y": np.concatenate([rng.normal(0.0, 0.2, 20), rng.normal(4.0, 0.2, 20)]),
            "g": ["a"] * 20 + ["b"] * 20,
        }
    )
    with gated(frame):
        told = compare_groups("y", "g", smallest_effect=1.0)
        assert any("smallest effect of interest" in note for note in told.notes)
        assert any("include" not in note for note in told.notes)
    with gated(frame):
        silent = compare_groups("y", "g")
        assert silent.smallest_effect is None
        assert any("does not exclude" in note for note in silent.notes)
        assert not any("smallest effect of interest" in note for note in silent.notes)
        assert "not evidence of no difference" in " ".join(silent.notes)


def test_compare_groups_uses_the_paired_route_when_a_pairing_column_is_given():
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame(
        {"y": rng.normal(0.0, 1.0, 40), "f": ["a"] * 20 + ["b"] * 20, "id": list(range(20)) * 2}
    )
    frame.loc[20:, "y"] += 1.4
    with gated(frame):
        result = compare_groups("y", "f", paired="id")
        assert result.test == "paired_t"
        assert "Paired rows" in result.routing_reason
        assert len(result.contrasts) == 1
        assert "paired" in result.contrasts[0].method
        assert result.contrasts[0].difference == pytest.approx(-1.4, abs=0.15)


def test_compare_groups_refuses_a_factor_with_one_level():
    with gated(oats().assign(only="same")):
        with pytest.raises(ToolError) as caught:
            compare_groups("grain", "only")
        assert "single level" in str(caught.value)


# --------------------------------------------------------------------------- #
# marginal_effects
# --------------------------------------------------------------------------- #


def test_marginal_effects_matches_the_tested_marginal_means_machinery():
    with gated(oats()) as store:
        result = fit_model("grain by variety", "grain", ["variety"], group="block")
        stages = set(store.state.records)
        margins = marginal_effects(result.result_id)
        # It refines an existing estimate, so the ledger keeps the stages it had.
        assert set(store.state.records) == stages
        assert margins.result_id != result.result_id
        expected = stats_tools.marginal_means("grain", ["variety"], group="block")
        assert [cell.level for cell in margins.cells] == [m.level for m in expected.marginal_means]
        for cell, reference in zip(margins.cells, expected.marginal_means, strict=True):
            assert cell.estimate == pytest.approx(reference.estimate, abs=1e-3)
            assert cell.ci_low < cell.estimate < cell.ci_high
        assert store.get(margins.result_id).kind == "marginal"


def test_marginal_effects_returns_an_interaction_contrast_as_a_difference_of_differences():
    with gated(oats()):
        result = fit_model(
            "grain by variety and nitrogen with their interaction",
            "grain",
            ["variety", "nitrogen", "variety:nitrogen"],
        )
        margins = marginal_effects(result.result_id)
        interactions = [c for c in margins.contrasts if c.factor == "variety:nitrogen"]
        assert len(interactions) == 1
        interaction = interactions[0]
        assert "difference at nitrogen=0" in interaction.level_a
        assert "difference at nitrogen=0.6" in interaction.level_b
        assert interaction.method == "interaction (difference of differences)"
        assert interaction.ci_low < interaction.difference < interaction.ci_high
        coefficient = {item.term: item.estimate for item in result.coefficients}[
            "variety[T.marvellous]:nitrogen"
        ]
        # The difference of differences of a linear fit is the interaction slope times the
        # distance it was evaluated over.
        assert interaction.difference == pytest.approx(0.6 * coefficient, abs=1e-3)
        assert any("difference of differences" in note for note in margins.notes)


def test_marginal_effects_pins_a_covariate_and_reports_where_the_average_was_taken():
    with gated(oats()):
        result = fit_model("grain rises with nitrogen", "grain", ["variety", "nitrogen"])
        margins = marginal_effects(result.result_id, at={"nitrogen": [0.2]}, contrasts="none")
        assert margins.contrasts == []
        assert margins.reference_levels["nitrogen"] == "held at 0.2"
        assert margins.reference_levels["variety"] == "golden rain"
        pinned = {cell.level: cell.estimate for cell in margins.cells}
        plain = marginal_effects(result.result_id)
        unpinned = {cell.level: cell.estimate for cell in plain.cells}
        assert pinned["victory"] < unpinned["victory"]
        assert unpinned["victory"] == pytest.approx(97.625, abs=0.01)


def test_marginal_effects_refuses_a_result_that_is_not_a_fit():
    with gated(oats()) as store:
        comparison = compare_groups("grain", "variety")
        assert comparison.result_id != ""
        with pytest.raises(ToolError) as caught:
            marginal_effects(comparison.result_id)
        assert "not a fit" in str(caught.value)
        assert store.get(comparison.result_id).kind == "comparison"


def test_marginal_effects_refuses_an_unknown_term_in_at():
    with gated(oats()):
        result = fit_model("grain rises with nitrogen", "grain", ["variety", "nitrogen"])
        with pytest.raises(ToolError) as caught:
            marginal_effects(result.result_id, at={"block": ["i"]})
        assert "not a term" in str(caught.value)


# --------------------------------------------------------------------------- #
# sensitivity
# --------------------------------------------------------------------------- #


def test_sensitivity_flips_when_two_rows_drive_the_effect():
    with gated(two_point_driven_effect()):
        result = fit_model("y falls with x", "y", ["x"])
        slope = result.coefficients[1]
        assert slope.estimate > 0
        assert slope.p_value < 0.05
        report = sensitivity(result.result_id, checks=["influence"])
        assert report.checks[0].check == "influence"
        assert report.checks[0].status == "flipped"
        assert report.checks[0].estimate < 0
        assert "Cook's distance" in report.checks[0].statement
        assert report.stable_share == 0.0
        assert "fragile" in report.verdict


def test_sensitivity_reports_a_stable_share_and_records_the_stage():
    with gated(oats()) as store:
        result = fit_model("grain rises with nitrogen", "grain", ["variety", "nitrogen"])
        report = sensitivity(result.result_id)
        assert [check.check for check in report.checks] == [
            "influence",
            "robust_se",
            "alternative_family",
            "bootstrap_ci",
            "adjustment_set",
        ]
        assert 0.0 < report.stable_share <= 1.0
        assert report.stable_share == pytest.approx(
            sum(1 for check in report.checks if check.status == "stable") / len(report.checks)
        )
        assert all(check.statement for check in report.checks)
        assert "nitrogen" in report.verdict
        assert any("seed 20260101" in check.statement for check in report.checks)
        assert store.state.records["sensitivity"].result_id == report.result_id
        assert store.state.records["sensitivity"].status == "ok"
        assert report.source_result_id == result.result_id


def test_sensitivity_reports_the_cluster_interval_for_a_grouped_fit():
    with gated(oats()):
        result = fit_model("grain by variety", "grain", ["variety"], group="block")
        report = sensitivity(result.result_id, checks=["robust_se"])
        check = report.checks[0]
        assert check.status == "stable"
        assert "Cluster-robust" in check.statement
        assert "block" in check.statement


def test_sensitivity_refuses_an_unknown_check():
    with gated(oats()):
        result = fit_model("grain rises with nitrogen", "grain", ["variety", "nitrogen"])
        with pytest.raises(ToolError) as caught:
            sensitivity(result.result_id, checks=["vibes"])
        assert "Unknown check" in str(caught.value)


# --------------------------------------------------------------------------- #
# the ledger gates every call
# --------------------------------------------------------------------------- #


def test_calls_out_of_order_name_the_missing_tool():
    bare = session.SessionStore.from_frame(oats(), source="test")
    with session.use_session(bare):
        for call in (
            lambda: fit_model("x", "grain", ["variety"]),
            lambda: compare_groups("grain", "variety"),
        ):
            with pytest.raises(ToolError) as caught:
                call()
            assert "design" in str(caught.value)
            assert "frame_question" in str(caught.value)

    design_only = session.SessionStore.from_frame(oats(), source="test")
    design_only.state.record("design")
    with session.use_session(design_only):
        with pytest.raises(ToolError) as caught:
            fit_model("x", "grain", ["variety"])
        assert "quality" in str(caught.value)
        assert "profile_table" in str(caught.value)

    with gated(oats()):
        with pytest.raises(ToolError) as caught:
            marginal_effects("f_none")
        assert "assumptions" in str(caught.value)
        assert "fit_model" in str(caught.value)
        with pytest.raises(ToolError) as caught:
            sensitivity("f_none")
        assert "assumptions" in str(caught.value)
        assert "fit_model" in str(caught.value)


def test_the_contrasts_say_they_do_not_inherit_a_rank_route():
    """A mean-based post-hoc beside a Kruskal-Wallis test has to be labelled.

    The contract promised rank-biserial intervals and a rank-based alternative for
    the k>2 route. The implementation reports epsilon-squared and mean-based
    contrasts instead, so the result has to say so rather than let a reader assume
    the post-hoc inherits the rank test.
    """
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame(
        {
            "arm": ["a", "b", "c"] * 8,
            "value": np.concatenate(
                [rng.exponential(1.0, 8), rng.exponential(9.0, 8), rng.exponential(4.0, 8)]
            ),
        }
    )
    store = session.SessionStore.from_frame(frame)
    store.state.record("design")
    store.state.record("quality")

    with session.use_session(store):
        result = compare_groups("value", "arm")

    assert result.test == "kruskal_wallis", f"expected the rank route, got {result.test}"
    assert any("do not inherit" in note for note in result.notes)
