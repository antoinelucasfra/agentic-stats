"""Estimation tools: fit, compare, margins, sensitivity, each writing the ledger.

`stats_tools` answers one question per call and forgets it. These four tools are
the analysis itself: each reads the stage it needs from `stats.session`, refuses
to run out of order, computes, and writes the stage it establishes. A fit cannot
be interpreted before its assumptions were checked, and a conclusion cannot be
published before someone tried to break it.

Nothing here re-estimates what `stats_tools` already estimates. Fits go through
`stats_tools._fit` (OLS and random-intercept mixed models) or `smf.glm` (the
binary and count families). Margins use `_design`, `_predict_vector`, `_estimate`
and `_t_critical`; contrasts use `_pairwise` and `_adjust`; diagnostics use
`_normality` and `_vif`. What this module adds is the ordering, the row
accounting, and the `next` sentence that keeps a small model on the path.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pydantic import BaseModel, Field
from scipy import stats
from scipy.optimize import brentq
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.outliers_influence import OLSInfluence

from stats import session, stats_tools
from stats.errors import ToolError
from stats.session import Verdict

__all__ = [
    "CheckResult",
    "Coefficient",
    "ComparisonResult",
    "Contrast",
    "Effect",
    "EffectSize",
    "FitResult",
    "GroupSummary",
    "MarginCell",
    "MarginalEffectsResult",
    "SensitivityReport",
    "compare_groups",
    "fit_model",
    "marginal_effects",
    "sensitivity",
]

ALPHA = 0.05
LEVENE_ALPHA = 0.05
SMALL_CELL = 20
VIF_LIMIT = 10
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260101

Family = Literal["gaussian", "binomial", "poisson", "negbin"]
Adjust = Literal["tukey", "holm", "bonferroni", "fdr_bh"]
Check = Literal["influence", "robust_se", "alternative_family", "bootstrap_ci", "adjustment_set"]
CheckStatus = Literal["stable", "attenuated", "flipped", "inconclusive"]
Scale = Literal["link", "response"]

# Built on demand: NegativeBinomial() warns about its default alpha at import time.
_FAMILIES: dict[str, Callable[[], Any]] = {
    "gaussian": sm.families.Gaussian,
    "binomial": sm.families.Binomial,
    "poisson": sm.families.Poisson,
    "negbin": sm.families.NegativeBinomial,
}

_CHECKS: tuple[str, ...] = (
    "influence",
    "robust_se",
    "alternative_family",
    "bootstrap_ci",
    "adjustment_set",
)

# "variety[T.victory]:nitrogen" reads as term `variety:nitrogen`, level `victory`.
_COLUMN = re.compile(r"^(?P<base>[^\[]+)\[T\.(?P<level>.+?)\](?P<rest>.*)$")


class Coefficient(BaseModel):
    """One fitted term, on the link scale unless the link is the identity."""

    term: str
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    scale: Scale


class Effect(BaseModel):
    """One term read on the response scale, with its contrast spelled out."""

    term: str
    contrast: str
    estimate: float
    ci_low: float
    ci_high: float
    scale: Scale


class FitResult(BaseModel):
    estimand: str
    formula: str
    family: Family
    response: str
    terms: list[str]
    group: str | None = None
    n_obs: int
    converged: bool
    coefficients: list[Coefficient] = Field(default_factory=list)
    effects: list[Effect] = Field(default_factory=list)
    reference_levels: dict[str, str] = Field(default_factory=dict)
    diagnostics: list[Verdict] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    cannot_say: list[str] = Field(default_factory=list)
    rows_total: int
    rows_used: int
    exclusions: list[dict] = Field(default_factory=list)
    result_id: str
    next: str
    notes: list[str] = Field(default_factory=list)


class GroupSummary(BaseModel):
    level: str
    n: int
    mean: float
    sd: float
    median: float


class EffectSize(BaseModel):
    name: str
    estimate: float
    ci_low: float
    ci_high: float
    interpretation: str


class Contrast(BaseModel):
    factor: str
    level_a: str
    level_b: str
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    adjusted_p_value: float
    method: str
    n_obs: int


class ComparisonResult(BaseModel):
    response: str
    factor: str
    test: str
    routing_reason: str
    n_obs: int
    groups: list[GroupSummary] = Field(default_factory=list)
    effect_size: EffectSize
    contrasts: list[Contrast] = Field(default_factory=list)
    diagnostics: list[Verdict] = Field(default_factory=list)
    smallest_effect: float | None = None
    result_id: str
    next: str
    notes: list[str] = Field(default_factory=list)


class MarginCell(BaseModel):
    factor: str
    level: str
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    n_obs: int


class MarginalEffectsResult(BaseModel):
    source_result_id: str
    cells: list[MarginCell] = Field(default_factory=list)
    contrasts: list[Contrast] = Field(default_factory=list)
    adjust: str
    reference_levels: dict[str, str] = Field(default_factory=dict)
    result_id: str
    next: str
    notes: list[str] = Field(default_factory=list)


class CheckResult(BaseModel):
    check: str
    estimate: float
    ci_low: float
    ci_high: float
    status: CheckStatus
    statement: str


class SensitivityReport(BaseModel):
    source_result_id: str
    checks: list[CheckResult] = Field(default_factory=list)
    stable_share: float
    verdict: str
    result_id: str
    next: str
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# formula and column plumbing
# --------------------------------------------------------------------------- #


def _formula_term(term: str) -> str:
    """One term as a formula fragment, quoting each part of an interaction.

    `stats_tools._term` quotes a whole term, so `variety:nitrogen` would become
    `Q("variety:nitrogen")`, and patsy would look for a column with that name.
    """
    chunks = []
    for chunk in term.split("*"):
        chunks.append(":".join(stats_tools._term(part.strip()) for part in chunk.split(":")))
    return "*".join(chunks)


def _family(name: str) -> Any:
    """The statsmodels family for one name."""
    return _FAMILIES[name]()


_LINKS: dict[str, str] = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "negbin": "log",
}


def _link_name(name: str) -> str:
    """The link as a word, since the family object prints as a memory address."""
    return _LINKS[name]


def _plain_term(term: str) -> bool:
    """Whether a term is a single column name, the only shape `_fit` can quote."""
    return stats_tools._term(term) == term


def _rhs(terms: list[str]) -> str:
    return " + ".join(_formula_term(term) for term in terms)


def _design_terms(terms: list[str]) -> list[str]:
    """The terms patsy builds, with `a*b` expanded into a, b and a:b."""
    expanded: list[str] = []
    for term in terms:
        chunks = [part.strip() for part in term.split("*")]
        for size in range(1, len(chunks) + 1):
            for combo in combinations(chunks, size):
                name = ":".join(combo)
                if name not in expanded:
                    expanded.append(name)
    return expanded


def _base_columns(terms: list[str]) -> list[str]:
    """The data columns a term list touches, an interaction split into its parts."""
    columns: list[str] = []
    for term in _design_terms(terms):
        for part in term.split(":"):
            if part not in columns:
                columns.append(part)
    return columns


def _split_column(column: str) -> tuple[str, str | None]:
    """Split a fitted column name into its term and the level it contrasts."""
    bases: list[str] = []
    levels: list[str] = []
    for part in column.split(":"):
        match = _COLUMN.match(part)
        if match is None:
            bases.append(part)
            continue
        bases.append(match.group("base"))
        levels.append(match.group("level") + match.group("rest"))
    return ":".join(bases), " @ ".join(level for level in levels if level) or None


def _levels(series: pd.Series) -> list[str]:
    """A column's levels, in the order patsy encodes them."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(value) for value in series.cat.categories]
    return [str(value) for value in sorted(series.dropna().unique())]


def _numeric_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]


def _columns_of(
    response: str, terms: list[str], group: str | None = None, weights: str | None = None
) -> list[str]:
    columns = [response, *_base_columns(terms)]
    columns += [name for name in (group, weights) if name]
    return columns


def _complete_rows(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Rows the model can use, with what was dropped and why."""
    missing = frame[columns].isna().any(axis=1)
    if not missing.any():
        return frame, []
    reason = f"missing values in {', '.join(columns)}"
    return frame.loc[~missing], [{"reason": reason, "n": int(missing.sum())}]


def _pin_value(frame: pd.DataFrame, column: str, value: Any) -> None:
    """Set one column to a fixed value, keeping every level in the encoding.

    `stats_tools._pin` keeps the original categories on the column, which is what
    lets patsy rebuild the same dummy columns for a pinned factor; assigning a
    single string would drop them and the design would lose its terms.
    """
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series):
        try:
            frame[column] = float(value)
        except (TypeError, ValueError):
            raise ToolError(
                f"'{column}' is numeric, so `at` needs a number for it, not {value!r}."
            ) from None
        return
    names = _levels(series)
    if str(value) not in names:
        raise ToolError(f"Unknown level {value!r} for '{column}'. Observed levels: {names}.")
    frame[column] = stats_tools._pin(series, str(value))


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def _wrap(result: Any, *, formula: str, rhs: str, params: pd.Series, kind: str) -> stats_tools._Fit:
    """Adapt a statsmodels result to `_Fit`, the shape the shared machinery reads."""
    cov = result.cov_params()
    return stats_tools._Fit(
        formula=formula,
        rhs=rhs,
        result=result,
        params=params,
        cov=cov.loc[params.index, params.index].astype(float),
        df_resid=float(result.df_resid),
        kind=kind,
    )


def _model(
    frame: pd.DataFrame,
    response: str,
    terms: list[str],
    group: str | None = None,
    family: str = "gaussian",
    weights: str | None = None,
) -> stats_tools._Fit:
    """Fit the requested model.

    The ordinary path is `stats_tools._fit`, which the suite already covers. It
    is bypassed only where that function cannot express the request: interaction
    terms, whose quoting turns `a:b` into a column lookup, and row weights. The
    fallback calls the same statsmodels entry points with the same arguments.
    """
    if family == "gaussian" and weights is None and all(_plain_term(t) for t in terms):
        return stats_tools._fit(frame, response, terms, group)

    rhs = _rhs(terms)
    formula = f"{stats_tools._term(response)} ~ {rhs}"
    try:
        if family != "gaussian":
            extra = {"var_weights": frame[weights]} if weights else {}
            result = smf.glm(formula, frame, family=_family(family), **extra).fit()
            params, kind = result.params, family
        elif weights:
            result = smf.wls(formula, frame, weights=frame[weights]).fit()
            params, kind = result.params, "wls"
        elif group:
            result = smf.mixedlm(formula, frame, groups=frame[group]).fit(
                reml=True, method=stats_tools.MIXED_METHOD
            )
            params, kind = result.fe_params, "mixedlm"
        else:
            result = smf.ols(formula, frame).fit()
            params, kind = result.params, "ols"
    except Exception as exc:  # singular design, non-convergence, bad encoding
        raise ToolError(
            f"Model '{formula}' failed to fit ({type(exc).__name__}: {exc}). Check that every "
            "group holds enough observations and that the terms are not collinear."
        ) from exc
    return _wrap(result, formula=formula, rhs=rhs, params=params, kind=kind)


def _converged(fit: stats_tools._Fit) -> bool:
    if fit.kind == "mixedlm":
        return bool(fit.result.converged) and fit.healthy
    return bool(getattr(fit.result, "converged", True))


def _interval(fit: stats_tools._Fit, column: str, confidence_level: float) -> tuple[float, float]:
    """The fitted model's own interval for one column, on the scale it was fitted."""
    conf = fit.result.conf_int(alpha=1 - confidence_level)
    if isinstance(conf, pd.DataFrame):
        conf = conf.loc[fit.params.index]
        return float(conf.loc[column, 0]), float(conf.loc[column, 1])
    names = list(getattr(fit.result.model, "exog_names", ()))
    row = np.asarray(conf)[names.index(column)]
    return float(row[0]), float(row[1])


def _statistics(fit: stats_tools._Fit, column: str, confidence_level: float) -> dict[str, float]:
    """Estimate, interval and p-value for one fitted column."""
    low, high = _interval(fit, column, confidence_level)
    return {
        "estimate": float(fit.params[column]),
        "ci_low": low,
        "ci_high": high,
        "p_value": float(fit.result.pvalues[column]),
    }


def _refuse_unfittable(frame: pd.DataFrame, response: str, terms: list[str], family: str) -> None:
    """The refusals the contract names, each with the call that fixes it."""
    values = frame[response].dropna()
    if values.nunique(dropna=True) < 2:
        raise ToolError(
            f"Response '{response}' is constant ({values.iloc[0]!r} in every row), so there is "
            "nothing to model. Model a different column, or describe this one with "
            "describe_dataset()."
        )
    for column in _base_columns(terms):
        if frame[column].nunique(dropna=True) < 2:
            value = frame[column].dropna().iloc[0]
            raise ToolError(
                f"Term '{column}' has a single value ({value!r}), so it cannot shift the response. "
                "Drop it from terms and retry fit_model()."
            )
    if family == "binomial":
        unique = set(values.unique().tolist())
        if len(unique) != 2 or not unique <= {0, 1}:
            raise ToolError(
                f"Binomial response '{response}' takes {len(unique)} distinct value(s) "
                f"({sorted(unique)[:5]}); it must be two-valued and coded 0/1. Recode it, or fit "
                "family='gaussian' on a proportion."
            )


def _refuse_collinear(frame: pd.DataFrame, terms: list[str]) -> None:
    """An exact linear combination leaves every coefficient unidentified."""
    try:
        matrix = patsy.dmatrix(_rhs(terms), frame, return_type="dataframe")
    except Exception as exc:
        raise ToolError(
            f"Could not build the design matrix for terms {terms} ({type(exc).__name__}: {exc}). "
            "Check the fit_model() terms for typos."
        ) from exc
    values = matrix.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(values))
    if rank >= values.shape[1]:
        return
    redundant = []
    for term in _design_terms(terms):
        owned = [name for name in matrix.columns if _split_column(name)[0] == term]
        rest = [name for name in matrix.columns if name not in owned]
        if not rest:
            continue
        if int(np.linalg.matrix_rank(matrix[rest].to_numpy(dtype=float))) == rank:
            redundant.append(term)
    culprits = redundant or _design_terms(terms)
    raise ToolError(
        f"Term(s) {culprits} are an exact linear combination of the other terms, so none of the "
        "coefficients is identified. Drop one of them and retry fit_model()."
    )


def _reference_levels(frame: pd.DataFrame, terms: list[str], columns: pd.Index) -> dict[str, str]:
    """The level each categorical term is contrasted against."""
    references: dict[str, str] = {}
    for term in _design_terms(terms):
        if ":" in term or term not in frame.columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[term]):
            continue
        owned = [name for name in columns if _split_column(name)[0] == term]
        present = {_split_column(name)[1] for name in owned}
        missing = [level for level in _levels(frame[term]) if level not in present]
        if missing:
            references[term] = missing[0]
    return references


def _exponentiate(value: float) -> float:
    """exp without the overflow warning: a separated fit honestly has an infinite ratio."""
    with np.errstate(over="ignore"):
        return float(np.exp(value))


def _coefficient_list(
    fit: stats_tools._Fit,
    frame: pd.DataFrame,
    terms: list[str],
    family: str,
    confidence_level: float,
) -> tuple[list[Coefficient], list[Effect], dict[str, str]]:
    """Coefficients on the fitted scale, plus the same contrasts on the response scale."""
    references = _reference_levels(frame, terms, fit.params.index)
    scale: Scale = "response" if family == "gaussian" else "link"
    coefficients: list[Coefficient] = []
    effects: list[Effect] = []
    for column in fit.params.index:
        low, high = _interval(fit, column, confidence_level)
        estimate = float(fit.params[column])
        coefficients.append(
            Coefficient(
                term=str(column),
                estimate=round(estimate, 4),
                std_error=round(float(np.sqrt(fit.cov.loc[column, column])), 4),
                ci_low=round(low, 4),
                ci_high=round(high, 4),
                p_value=float(fit.result.pvalues[column]),
                scale=scale,
            )
        )
        if column == "Intercept":
            continue
        term, level = _split_column(str(column))
        ratio = "response scale"
        if scale == "link":
            estimate, low, high = (_exponentiate(v) for v in (estimate, low, high))
            ratio = "odds ratio" if family == "binomial" else "rate ratio"
        effects.append(
            Effect(
                term=term,
                contrast=_contrast_label(term, level, references, ratio),
                estimate=round(estimate, 4),
                ci_low=round(low, 4),
                ci_high=round(high, 4),
                scale="response",
            )
        )
    return coefficients, effects, references


def _contrast_label(term: str, level: str | None, references: dict[str, str], ratio: str) -> str:
    """One coefficient as a phrase a reader can quote."""
    if level is None:
        return f"per 1-unit increase in {term} ({ratio})"
    parts = term.split(":")
    reference = " and ".join(references.get(part, "the reference level") for part in parts)
    if len(parts) > 1:
        return f"{level} vs {reference}, per 1-unit increase in {parts[1]} ({ratio})"
    return f"{level} vs {reference} ({ratio})"


# --------------------------------------------------------------------------- #
# fit diagnostics
# --------------------------------------------------------------------------- #


def _verdict_status(verdicts: list[Verdict]) -> session.Severity:
    if any(v.status == "blocked" or v.severity == "blocking" for v in verdicts):
        return "blocked"
    if any(v.status == "caution" for v in verdicts):
        return "caution"
    return "ok"


def _separation_verdict(fit: stats_tools._Fit, frame: pd.DataFrame) -> Verdict:
    """Complete separation: the likelihood has no finite maximum, so no interval is real."""
    predicted = np.asarray(fit.result.predict(frame), dtype=float)
    at_edge = int(((predicted < 1e-8) | (predicted > 1 - 1e-8)).sum())
    largest = float(np.abs(fit.beta).max())
    remedy = (
        "Refit with fit_model(...) without the term that separates the classes, or pool the levels "
        "that create the empty cell."
    )
    if at_edge == len(frame):
        return Verdict(
            check="separation",
            status="blocked",
            statement=(
                f"All {at_edge} fitted probabilities sit at 0 or 1 (largest |coefficient| "
                f"{largest:.3g}): complete separation, so no coefficient here is identified and no "
                "interval is real."
            ),
            remedy=remedy,
            severity="blocking",
        )
    if at_edge or largest > 10:
        return Verdict(
            check="separation",
            status="caution",
            statement=(
                f"{at_edge} fitted probability(ies) sit at 0 or 1 and the largest |coefficient| is "
                f"{largest:.3g}: the classes are close to separable, so those coefficients are "
                "inflated."
            ),
            remedy=remedy,
            severity="warning",
        )
    return Verdict(
        check="separation",
        status="ok",
        statement=(
            f"No fitted probability reaches 0 or 1 (range {predicted.min():.3g} to "
            f"{predicted.max():.3g}); the classes overlap."
        ),
    )


def _fit_diagnostics(
    *,
    fit: stats_tools._Fit,
    frame: pd.DataFrame,
    response: str,
    terms: list[str],
    group: str | None,
    family: str,
    weights: str | None,
    result_id: str,
    notes: list[str],
) -> list[Verdict]:
    """Named checks with named remedies, in place of a p-value dump."""
    verdicts: list[Verdict] = []
    n_rows = len(frame)

    if _converged(fit):
        verdicts.append(
            Verdict(
                check="convergence",
                status="ok",
                statement=f"The {fit.kind} fit converged on {n_rows} rows.",
            )
        )
    else:
        verdicts.append(
            Verdict(
                check="convergence",
                status="caution",
                statement=(
                    "The optimizer did not converge, so these coefficients may sit at the wrong "
                    "optimum."
                ),
                remedy=(
                    f"Refit with fit_model(estimand=..., response='{response}', terms={terms}, "
                    "group=None) to drop the random intercept, or remove the weakest term."
                ),
                severity="warning",
            )
        )

    exog = np.asarray(fit.result.model.exog, dtype=float)
    resid = np.asarray(
        fit.result.resid if family == "gaussian" else fit.result.resid_pearson, dtype=float
    )
    residual_kind = "residual" if family == "gaussian" else "Pearson residual"
    try:
        bp_p = float(het_breuschpagan(resid, exog)[1])
    except (ValueError, IndexError) as exc:
        bp_p = float("nan")
        notes.append(f"Breusch-Pagan skipped: {exc}")
    if np.isnan(bp_p):
        verdicts.append(
            Verdict(
                check="heteroscedasticity",
                status="caution",
                statement=f"Breusch-Pagan could not run on the {residual_kind} variance.",
                remedy=(
                    "Refit with fewer terms via fit_model(...) so the test has the rows it needs, "
                    "then read the interval as approximate."
                ),
                severity="warning",
            )
        )
    elif bp_p < ALPHA:
        verdicts.append(
            Verdict(
                check="heteroscedasticity",
                status="caution",
                statement=(
                    f"The {residual_kind} variance changes with the design (Breusch-Pagan "
                    f"p = {bp_p:.3g}), so the usual standard errors are optimistic."
                ),
                remedy=(
                    f"Call sensitivity('{result_id}', checks=['robust_se']) for a "
                    "heteroscedasticity-robust or cluster-robust interval."
                ),
                severity="warning",
            )
        )
    else:
        verdicts.append(
            Verdict(
                check="heteroscedasticity",
                status="ok",
                statement=(
                    f"The {residual_kind} variance looks constant across the design "
                    f"(Breusch-Pagan p = {bp_p:.3g})."
                ),
            )
        )

    if family == "gaussian":
        normality = stats_tools._normality(np.asarray(fit.result.resid, dtype=float))
        if normality.p_value < ALPHA:
            verdicts.append(
                Verdict(
                    check="residual_normality",
                    status="caution",
                    statement=(
                        f"Residuals depart from normality ({normality.test} "
                        f"p = {normality.p_value:.3g}), so the p-values are approximate."
                    ),
                    remedy=(
                        f"Call sensitivity('{result_id}', checks=['bootstrap_ci']) for an interval "
                        "that does not assume normal residuals, or compare_groups() for a "
                        "rank-based test of the same factor."
                    ),
                    severity="warning",
                )
            )
        else:
            verdicts.append(
                Verdict(
                    check="residual_normality",
                    status="ok",
                    statement=(
                        f"Residuals are consistent with normality ({normality.test} "
                        f"p = {normality.p_value:.3g})."
                    ),
                )
            )
    else:
        verdicts.append(
            Verdict(
                check="residual_normality",
                status="ok",
                statement=(
                    f"Normal residuals are not assumed for a {family} fit; the deviance and the "
                    "dispersion carry that role."
                ),
            )
        )

    design = fit
    if family != "gaussian" or group or weights:
        design = _model(frame, response, terms, None, "gaussian")
        notes.append(
            "Influence and collinearity describe the design, so they are computed on an OLS fit of "
            "the same terms"
            + (f"; the random intercept for {group} absorbs them." if group else ".")
        )
    influence = OLSInfluence(design.result)
    cook = np.asarray(influence.cooks_distance[0], dtype=float)
    cutoff = 4 / n_rows
    influential = int((cook > cutoff).sum())
    if influential:
        verdicts.append(
            Verdict(
                check="influence",
                status="caution",
                statement=(
                    f"{influential} of {n_rows} rows have a Cook's distance above 4/n = "
                    f"{cutoff:.4g} (largest {float(cook.max()):.3g}), so a few rows move the fit."
                ),
                remedy=(
                    f"Call sensitivity('{result_id}', checks=['influence']) to see the estimate "
                    "without those rows."
                ),
                severity="warning",
            )
        )
    else:
        verdicts.append(
            Verdict(
                check="influence",
                status="ok",
                statement=(
                    f"No row has a Cook's distance above 4/n = {cutoff:.4g} (largest "
                    f"{float(cook.max()):.3g})."
                ),
            )
        )

    vif = stats_tools._vif(exog, list(design.params.index), notes)
    worst = max(vif.items(), key=lambda item: item[1], default=None)
    if worst is not None and worst[1] > VIF_LIMIT:
        verdicts.append(
            Verdict(
                check="collinearity",
                status="caution",
                statement=(
                    f"Variance inflation factor for '{worst[0]}' is {worst[1]:.1f}, above the "
                    f"{VIF_LIMIT} limit, so that coefficient is unstable."
                ),
                remedy=(
                    "Drop one of the correlated terms and refit with fit_model(), or keep only the "
                    "combination the question needs."
                ),
                severity="warning",
            )
        )
    elif vif:
        verdicts.append(
            Verdict(
                check="collinearity",
                status="ok",
                statement=(
                    f"Every variance inflation factor is below {VIF_LIMIT} "
                    f"(largest {worst[1]:.1f})."
                ),
            )
        )

    if family == "binomial":
        verdicts.append(_separation_verdict(fit, frame))

    thin: list[str] = []
    for column in _base_columns(terms):
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        counts = frame[column].astype(str).value_counts()
        thin += [f"{column}={level} ({int(count)})" for level, count in counts.items() if count < 2]
    if group:
        counts = frame[group].astype(str).value_counts()
        thin += [f"{group}={level} ({int(count)})" for level, count in counts.items() if count < 2]
    if thin:
        verdicts.append(
            Verdict(
                check="cell_size",
                status="caution",
                statement=(
                    f"Cell(s) {thin} hold fewer than two rows, so the terms they carry rest on a "
                    "single observation."
                ),
                remedy=(
                    "Collect more rows, or pool that level into another one and refit with "
                    "fit_model()."
                ),
                severity="warning",
            )
        )
    else:
        verdicts.append(
            Verdict(
                check="cell_size",
                status="ok",
                statement="Every level of every term holds at least two rows.",
            )
        )
    return verdicts


def _fit_caveats(
    *,
    frame: pd.DataFrame,
    response: str,
    terms: list[str],
    group: str | None,
    family: str,
    weights: str | None,
    coefficients: list[Coefficient],
) -> list[str]:
    caveats: list[str] = []
    base = _base_columns(terms)
    if group and family == "gaussian":
        caveats.append(
            f"Rows repeat within {group}, and the random intercept handles that dependence; the "
            "terms stay associations, not interventions."
        )
    elif group:
        caveats.append(
            f"Rows repeat within {group} and this {family} fit ignores that: every coefficient is "
            "an association computed as if the rows were independent."
        )
    else:
        caveats.append(
            "The fit is cross-sectional: it quantifies an association between the terms and the "
            "response, not the effect of changing any of them."
        )
    for column in _numeric_columns(frame, base):
        caveats.append(
            f"'{column}' effects are per 1-unit change and hold only over the observed range "
            f"({float(frame[column].min()):.4g} to {float(frame[column].max()):.4g})."
        )
    wide = [
        coefficient.term
        for coefficient in coefficients
        if coefficient.term != "Intercept"
        and coefficient.ci_low <= 0 <= coefficient.ci_high
        and coefficient.ci_high - coefficient.ci_low > 2 * abs(coefficient.estimate)
    ]
    if wide:
        caveats.append(
            f"The interval for {wide} spans zero and is wider than twice its estimate, so this fit "
            "cannot resolve those effects."
        )
    counts = frame[base[0]].astype(str).value_counts() if base else pd.Series(dtype=int)
    if not counts.empty and int(counts.max()) > 2 * max(int(counts.min()), 1):
        caveats.append(
            "Cells are unbalanced, so each level mean is weighted by how many rows it holds."
        )
    if family != "gaussian":
        caveats.append(
            f"Coefficients are on the {_link_name(family)} scale; the effects list holds the "
            "same contrasts back-transformed to the response scale."
        )
    if weights:
        caveats.append(
            f"Rows are weighted by '{weights}'; refitting from the stored formula does not reapply "
            "those weights, so pass the column again."
        )
    if group and family != "gaussian":
        caveats.append(
            f"group='{group}' is recorded but unused: a {family} GLM here is a fixed-effects model "
            "that treats the rows as independent."
        )
    return caveats


def _fit_cannot_say(
    *, frame: pd.DataFrame, response: str, terms: list[str], group: str | None, family: str
) -> list[str]:
    base = _base_columns(terms)
    structure = (
        f"Rows repeat within {group}, which makes this a clustered fit, not an experiment: the "
        "design still has no treatment assignment and no time order"
        if group
        else "The design has no treatment assignment and no time order"
    )
    cannot_say = [f"'{response}' is not shown to be caused by {terms}: {structure}."]
    if group:
        cannot_say.append(
            f"Between-{group} differences are absorbed by the random intercept, so this fit "
            f"estimates no {group}-level effect."
        )
    for column in _numeric_columns(frame, base):
        cannot_say.append(
            f"Nothing outside the observed range of '{column}' "
            f"({float(frame[column].min()):.4g} to {float(frame[column].max()):.4g}) is "
            "covered, so a prediction there is extrapolation."
        )
    others = [column for column in frame.columns if column not in [*base, response, group]]
    if others:
        cannot_say.append(
            f"Terms not in this fit are not estimated: {others[:6]}"
            + (" and more." if len(others) > 6 else ".")
        )
    if family == "binomial":
        cannot_say.append(
            "A log-odds fit does not give the risk difference of a 1-unit change, which depends on "
            "where the change starts."
        )
    return cannot_say


def _reseal(stored: session.StoredResult) -> None:
    """Recompute the quotable figures once the payload carries its id and remedies."""
    stored.quotable = session.quotable_numbers(stored.payload)


def fit_model(
    estimand: str,
    response: str,
    terms: list[str],
    group: str | None = None,
    family: Family = "gaussian",
    weights: str | None = None,
    confidence_level: float = 0.95,
) -> FitResult:
    """Estimate `response` from `terms`, with the assumption verdicts attached.

    One entry point for OLS, GLM and random-intercept mixed models: `family` and
    `group` choose the engine, never the caller. A term may name an interaction
    ('variety:nitrogen' or 'variety*nitrogen'); each categorical term is
    contrasted against the reference level named in `reference_levels`. Pass
    `group` when rows repeat inside a block or a subject. `weights` are row
    weights: `smf.wls` for a gaussian fit, `var_weights` for a GLM.

    Both are analytic (precision) weights, so a row with weight 2 counts twice as
    much without inflating the residual degrees of freedom.

    Call this after `frame_question` and `profile_table`, since it refuses out of
    order. It writes both the assumptions and the estimate stage. Read `caveats`
    and `cannot_say` before writing a sentence, and never quote a coefficient
    without its interval.
    """
    store = session.current_session()
    store.state.require("assumptions")

    if family not in _FAMILIES:
        raise ToolError(f"Unknown family '{family}'. Use one of {sorted(_FAMILIES)}.")
    if not terms:
        raise ToolError("Provide at least one term, e.g. ['variety', 'nitrogen'].")
    if not 0 < confidence_level < 1:
        raise ToolError(f"confidence_level must be between 0 and 1, got {confidence_level}.")

    frame = stats_tools._frame()
    columns = _columns_of(response, terms, group, weights)
    stats_tools._require(frame, columns)
    stats_tools._require_numeric(frame, response, "Response")

    data, exclusions = _complete_rows(frame, columns)
    if len(data) < 3:
        raise ToolError(
            f"Only {len(data)} complete row(s) remain for {response} and {terms}: too few to fit. "
            "Repair the missing values first, or drop the offending column."
        )
    _refuse_unfittable(data, response, terms, family)
    _refuse_collinear(data, terms)

    fit = _model(data, response, terms, group, family, weights)
    coefficients, effects, references = _coefficient_list(
        fit, data, terms, family, confidence_level
    )

    notes: list[str] = []
    if weights:
        notes.append(
            f"Row weights come from '{weights}': a gaussian fit uses smf.wls, a GLM uses "
            "var_weights, so they are analytic (precision) weights. Pass the column again to "
            "reproduce the fit from the stored formula."
        )
    if group and family != "gaussian":
        notes.append(
            f"group='{group}' is recorded but unused: a {family} GLM here is a "
            "fixed-effects model. Call sensitivity(result_id, checks=['robust_se']) if the "
            "clusters must widen the interval."
        )
    if family != "gaussian":
        notes.append(
            f"Coefficients are on the {_link_name(family)} scale; 'effects' are the same "
            "contrasts back-transformed, and marginal_effects() reports the response scale."
        )
    if family == "negbin":
        notes.append(
            "NegativeBinomial keeps statsmodels' default dispersion alpha = 1.0: the dispersion "
            "was not estimated, so read the interval as narrower than it should be."
        )

    payload = FitResult(
        estimand=estimand,
        formula=fit.formula,
        family=family,
        response=response,
        terms=list(terms),
        group=group,
        n_obs=len(data),
        converged=_converged(fit),
        coefficients=coefficients,
        effects=effects,
        reference_levels=references,
        rows_total=len(frame),
        rows_used=len(data),
        exclusions=exclusions,
        result_id="",
        next="",
        notes=notes,
    )
    stored = store.put(
        payload,
        kind="fit",
        summary=f"{family} fit of {response} on {', '.join(terms)} ({len(data)} rows)",
        rows_total=len(frame),
        rows_used=len(data),
        exclusions=exclusions,
    )
    payload.diagnostics = _fit_diagnostics(
        fit=fit,
        frame=data,
        response=response,
        terms=terms,
        group=group,
        family=family,
        weights=weights,
        result_id=stored.result_id,
        notes=notes,
    )
    payload.caveats = _fit_caveats(
        frame=data,
        response=response,
        terms=terms,
        group=group,
        family=family,
        weights=weights,
        coefficients=coefficients,
    )
    payload.cannot_say = _fit_cannot_say(
        frame=data, response=response, terms=terms, group=group, family=family
    )
    payload.next = (
        f"Call marginal_effects('{stored.result_id}') to read these coefficients as adjusted "
        f"means, then sensitivity('{stored.result_id}') to try to break the answer."
    )
    status = _verdict_status(payload.diagnostics)
    blocked = [verdict.check for verdict in payload.diagnostics if verdict.status == "blocked"]
    store.state.record(
        "assumptions",
        status=status,
        result_id=stored.result_id,
        note=f"{family} fit of {response}; blocking checks: {blocked or 'none'}",
        verdicts=payload.diagnostics,
    )
    store.state.record(
        "estimate",
        status=status,
        result_id=stored.result_id,
        note=f"estimand: {estimand}",
        verdicts=payload.diagnostics,
    )
    _reseal(stored)
    return payload


# --------------------------------------------------------------------------- #
# compare_groups
# --------------------------------------------------------------------------- #


def _group_summary(frame: pd.DataFrame, response: str, factor: str, level: str) -> GroupSummary:
    values = frame.loc[frame[factor].astype(str) == level, response].to_numpy(dtype=float)
    return GroupSummary(
        level=level,
        n=int(values.size),
        mean=round(float(values.mean()), 4),
        sd=round(float(values.std(ddof=1)), 4),
        median=round(float(np.median(values)), 4),
    )


def _cohens_d_interval(
    first: np.ndarray, second: np.ndarray, confidence_level: float
) -> tuple[float, float, float]:
    """Cohen's d with the large-sample interval (Hedges & Olkin 1985, eq. 5.18)."""
    n1, n2 = first.size, second.size
    pooled = float(
        np.sqrt(((n1 - 1) * first.var(ddof=1) + (n2 - 1) * second.var(ddof=1)) / (n1 + n2 - 2))
    )
    difference = float(first.mean() - second.mean())
    if pooled == 0:
        edge = 0.0 if difference == 0 else float(np.sign(difference) * np.inf)
        return 0.0, -edge, edge
    d = difference / pooled
    se = float(np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2))))
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    return d, d - z * se, d + z * se


def _paired_d_interval(
    differences: np.ndarray, confidence_level: float
) -> tuple[float, float, float]:
    """Cohen's d_z for matched pairs, on the same large-sample approximation."""
    n = differences.size
    sd = float(differences.std(ddof=1))
    mean = float(differences.mean())
    if sd == 0:
        edge = 0.0 if mean == 0 else float(np.sign(mean) * np.inf)
        return 0.0, -edge, edge
    d = mean / sd
    se = float(np.sqrt(1 / n + d**2 / (2 * n)))
    z = float(stats.norm.ppf(0.5 + confidence_level / 2))
    return d, d - z * se, d + z * se


def _rank_biserial(first: np.ndarray, second: np.ndarray) -> float:
    """Rank-biserial correlation: positive means the first sample runs higher."""
    u = float(stats.mannwhitneyu(first, second, alternative="two-sided").statistic)
    return 2 * u / (first.size * second.size) - 1


def _matched_rank_biserial(differences: np.ndarray) -> float:
    """Rank-biserial for pairs: (W+ - W-) / (W+ + W-), zero differences dropped."""
    differences = differences[differences != 0]
    if differences.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    return (positive - negative) / (positive + negative)


def _epsilon_squared(statistic: float, k: int, n: int) -> float:
    """Kruskal-Wallis effect size: (H - k + 1) / (n - k)."""
    if n <= k:
        return float("nan")
    return (statistic - k + 1) / (n - k)


def _lambda_bound(f: float, df1: float, df2: float, target: float) -> float:
    """Noncentrality whose noncentral F puts the observed F at `target`."""
    if float(stats.ncf.cdf(f, df1, df2, 0.0)) <= target:
        return 0.0
    upper = 1.0
    while float(stats.ncf.cdf(f, df1, df2, upper)) > target:
        upper *= 2.0
    return float(brentq(lambda lam: float(stats.ncf.cdf(f, df1, df2, lam)) - target, 0.0, upper))


def _eta_squared_interval(
    f: float, df1: float, df2: float, n_total: int, confidence_level: float
) -> tuple[float, float, float]:
    """Eta-squared with the noncentral-F interval (Steiger 2004).

    The bounds come from inverting the noncentral F at the observed statistic,
    then reading each noncentrality as eta-squared = lambda / (lambda + N).
    """

    def eta(lam: float) -> float:
        return lam / (lam + n_total)

    alpha = 1 - confidence_level
    observed = f * df1 / (f * df1 + df2)
    low = eta(_lambda_bound(f, df1, df2, 1 - alpha / 2))
    high = eta(_lambda_bound(f, df1, df2, alpha / 2))
    return observed, low, high


def _bootstrap_interval(
    samples: list[np.ndarray],
    statistic: Callable[[list[np.ndarray]], float],
    confidence_level: float,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a rank effect size, at the recorded seed."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draw = [rng.choice(sample, size=sample.size, replace=True) for sample in samples]
        try:
            values.append(float(statistic(draw)))
        except ValueError:  # a resample with no variation left
            continue
    if len(values) < BOOTSTRAP_RESAMPLES // 2:
        return float("nan"), float("nan")
    percentiles = [100 * (1 - confidence_level) / 2, 100 * (1 + confidence_level) / 2]
    low, high = np.percentile(values, percentiles)
    return float(low), float(high)


def _equivalence_notes(
    name: str,
    ci_low: float,
    ci_high: float,
    smallest_effect: float | None,
    confidence_level: float,
) -> list[str]:
    """Equivalence language, or the plain statement of what the interval leaves open."""
    percent = f"{confidence_level:.0%}"
    low, high = min(ci_low, ci_high), max(ci_low, ci_high)
    if smallest_effect is None:
        reach = max(abs(low), abs(high))
        return [
            f"The {percent} interval for {name} is [{low:.4g}, {high:.4g}], which includes zero: "
            "these data are consistent with no difference, and the interval does not exclude an "
            f"effect as large as {reach:.4g} in either direction. A wide interval is not evidence "
            "of no difference, only of a design that cannot resolve this one."
        ]
    threshold = abs(float(smallest_effect))
    if low > threshold or high < -threshold:
        return [
            f"The {percent} interval for {name}, [{low:.4g}, {high:.4g}], excludes the smallest "
            f"effect of interest in magnitude ({threshold:.4g}): every value the data still leave "
            "open is further from zero than that threshold, so the effect is at least that large."
        ]
    if -threshold < low and high < threshold:
        return [
            f"The {percent} interval for {name}, [{low:.4g}, {high:.4g}], lies inside "
            f"±{threshold:.4g}, so it excludes an effect as large as the smallest effect of "
            "interest."
        ]
    return [
        f"The {percent} interval for {name}, [{low:.4g}, {high:.4g}], contains the smallest effect "
        f"of interest ({threshold:.4g}), so the data cannot separate an effect that size from zero "
        "or from a larger one."
    ]


def _omnibus(
    *,
    test: str,
    data: pd.DataFrame,
    response: str,
    factor: str,
    samples: list[np.ndarray],
    pairs: dict[str, Any] | None,
) -> tuple[float, float, str]:
    """The headline test: statistic, p-value and the degrees of freedom it used."""
    first, second = samples[0], samples[1]
    if test == "student_t":
        result = stats.ttest_ind(first, second, equal_var=True)
        return float(result.statistic), float(result.pvalue), f"df = {first.size + second.size - 2}"
    if test == "welch_t":
        result = stats.ttest_ind(first, second, equal_var=False)
        return float(result.statistic), float(result.pvalue), "Welch df"
    if test == "paired_t" and pairs is not None:
        result = stats.ttest_1samp(pairs["differences"], 0.0)
        return (
            float(result.statistic),
            float(result.pvalue),
            f"df = {pairs['differences'].size - 1}",
        )
    if test == "wilcoxon" and pairs is not None:
        result = stats.wilcoxon(pairs["differences"])
        return (
            float(result.statistic),
            float(result.pvalue),
            f"n = {pairs['differences'].size} pairs",
        )
    if test == "mann_whitney":
        result = stats.mannwhitneyu(first, second, alternative="two-sided")
        return float(result.statistic), float(result.pvalue), "U"
    if test == "kruskal_wallis":
        result = stats.kruskal(*samples)
        return float(result.statistic), float(result.pvalue), f"df = {len(samples) - 1}"
    one_way = stats_tools._one_way(data, response, factor)
    if test == "one_way_anova":
        return one_way.f, one_way.p, f"df = {one_way.df_between}, {one_way.df_within}"
    result = anova_oneway(samples, use_var="unequal")
    return (
        float(result.statistic),
        float(result.pvalue),
        f"df = {result.df_num:.4g}, {float(result.df_denom):.4g}",
    )


def _effect_size(
    *,
    test: str,
    data: pd.DataFrame,
    response: str,
    factor: str,
    levels: list[str],
    samples: list[np.ndarray],
    pairs: dict[str, Any] | None,
    notes: list[str],
    confidence_level: float,
) -> EffectSize:
    """The effect size the route calls for, always with an interval."""
    if test in {"student_t", "welch_t"}:
        d, low, high = _cohens_d_interval(samples[0], samples[1], confidence_level)
        notes.append(
            "Cohen's d interval: large-sample normal approximation (Hedges & Olkin 1985), not a "
            "noncentral-t interval."
        )
        return EffectSize(
            name="cohens_d",
            estimate=round(d, 4),
            ci_low=round(low, 4),
            ci_high=round(high, 4),
            interpretation=(
                f"Cohen's d = {d:.3g} [{low:.3g}, {high:.3g}]: the two levels differ by about "
                f"{abs(d):.3g} pooled standard deviations."
            ),
        )
    if test == "paired_t" and pairs is not None:
        d, low, high = _paired_d_interval(pairs["differences"], confidence_level)
        notes.append(
            "Cohen's d_z interval: large-sample normal approximation (Hedges & Olkin 1985)."
        )
        return EffectSize(
            name="cohens_dz",
            estimate=round(d, 4),
            ci_low=round(low, 4),
            ci_high=round(high, 4),
            interpretation=(
                f"Cohen's d_z = {d:.3g} [{low:.3g}, {high:.3g}] over {pairs['n']} pairs: the "
                "within-pair change is that many standard deviations of the differences."
            ),
        )
    if test in {"mann_whitney", "wilcoxon"}:
        if pairs is not None:
            observations = [pairs["differences"]]
            estimate = _matched_rank_biserial(pairs["differences"])

            def statistic(draw: list[np.ndarray]) -> float:
                return _matched_rank_biserial(draw[0])
        else:
            observations = [samples[0], samples[1]]
            estimate = _rank_biserial(samples[0], samples[1])

            def statistic(draw: list[np.ndarray]) -> float:
                return _rank_biserial(draw[0], draw[1])

        low, high = _bootstrap_interval(observations, statistic, confidence_level)
        label = f"{levels[0]} vs {levels[1]}" if len(levels) == 2 else "the two samples"
        notes.append(
            f"Rank-biserial interval: percentile bootstrap, {BOOTSTRAP_RESAMPLES} resamples, seed "
            f"{BOOTSTRAP_SEED}; no closed form for it exists in scipy."
        )
        return EffectSize(
            name="rank_biserial",
            estimate=round(estimate, 4),
            ci_low=round(low, 4),
            ci_high=round(high, 4),
            interpretation=(
                f"Rank-biserial = {estimate:.3g} [{low:.3g}, {high:.3g}]: positive means {label} "
                "tends to come out higher, with 1 the largest separation the data allow."
            ),
        )
    if test == "kruskal_wallis":
        n = sum(sample.size for sample in samples)
        estimate = _epsilon_squared(float(stats.kruskal(*samples).statistic), len(levels), n)

        def statistic(draw: list[np.ndarray]) -> float:
            total = sum(sample.size for sample in draw)
            return _epsilon_squared(float(stats.kruskal(*draw).statistic), len(levels), total)

        low, high = _bootstrap_interval(samples, statistic, confidence_level)
        notes.append(
            "Kruskal-Wallis has no rank-biserial analogue beyond two groups, so the k-level rank "
            f"route reports epsilon-squared, bootstrapped at {BOOTSTRAP_RESAMPLES} resamples, seed "
            f"{BOOTSTRAP_SEED}."
        )
        return EffectSize(
            name="epsilon_squared",
            estimate=round(estimate, 4),
            ci_low=round(low, 4),
            ci_high=round(high, 4),
            interpretation=(
                f"Epsilon-squared = {estimate:.3g} [{low:.3g}, {high:.3g}]: the factor "
                "accounts for that share of the rank variance in the response."
            ),
        )
    one_way = stats_tools._one_way(data, response, factor)
    if test == "welch_anova":
        result = anova_oneway(samples, use_var="unequal")
        f, df1, df2 = float(result.statistic), float(result.df_num), float(result.df_denom)
    else:
        f, df1, df2 = one_way.f, float(one_way.df_between), float(one_way.df_within)
    estimate, low, high = _eta_squared_interval(
        f, df1, df2, sum(sample.size for sample in samples), confidence_level
    )
    notes.append("Eta-squared interval: noncentral-F inversion (Steiger 2004).")
    return EffectSize(
        name="eta_squared",
        estimate=round(estimate, 4),
        ci_low=round(low, 4),
        ci_high=round(high, 4),
        interpretation=(
            f"Eta-squared = {estimate:.3g} [{low:.3g}, {high:.3g}]: the factor accounts for about "
            "that share of the variance in the response."
        ),
    )


def _route(
    *,
    levels: list[str],
    pairs: dict[str, Any] | None,
    levene_p: float,
    shapiro_p: float,
    smallest_cell: int,
) -> tuple[str, str]:
    """Choose the test on the observed violations, and say which number decided it."""
    k = len(levels)
    if pairs is not None:
        normality = stats_tools._normality(pairs["differences"])
        if normality.p_value < ALPHA:
            return (
                "wilcoxon",
                f"Paired rows (paired column given): the within-pair differences depart from "
                f"normality ({normality.test} p = {normality.p_value:.3g}), so Wilcoxon "
                f"signed-rank on {pairs['n']} complete pairs, which needs no "
                "distributional assumption.",
            )
        return (
            "paired_t",
            f"Paired rows (paired column given): the within-pair differences are consistent with "
            f"normality ({normality.test} p = {normality.p_value:.3g}) across "
            f"{pairs['n']} complete pairs, so the paired t-test on the differences.",
        )
    if levene_p < LEVENE_ALPHA and k == 2:
        return (
            "welch_t",
            f"Levene p = {levene_p:.3g} (< {LEVENE_ALPHA}) with {k} levels: the spreads differ, so "
            "Welch's t-test, which does not assume equal variances.",
        )
    if levene_p < LEVENE_ALPHA and k > 2:
        return (
            "welch_anova",
            f"Levene p = {levene_p:.3g} (< {LEVENE_ALPHA}) with {k} levels: unequal spreads, so "
            "Welch's ANOVA (statsmodels.stats.oneway.anova_oneway, use_var='unequal').",
        )
    if shapiro_p < ALPHA and smallest_cell < SMALL_CELL:
        if k == 2:
            return (
                "mann_whitney",
                f"Shapiro p = {shapiro_p:.3g} (< {ALPHA}) and the smallest cell holds "
                f"{smallest_cell} rows (< {SMALL_CELL}), so the mean is not the summary to trust: "
                "Mann-Whitney.",
            )
        return (
            "kruskal_wallis",
            f"Shapiro p = {shapiro_p:.3g} (< {ALPHA}) and the smallest cell holds {smallest_cell} "
            f"rows (< {SMALL_CELL}), so the mean is not the summary to trust: Kruskal-Wallis.",
        )
    if k == 2:
        return (
            "student_t",
            f"Levene p = {levene_p:.3g} and Shapiro p = {shapiro_p:.3g}: equal spreads, "
            "cells of at least 20 rows shaped like a normal sample, so the standard "
            "two-sample t-test.",
        )
    return (
        "one_way_anova",
        f"Levene p = {levene_p:.3g} and Shapiro p = {shapiro_p:.3g}: equal spreads and cells of at "
        f"least 20 rows across {k} levels, so the one-way ANOVA.",
    )


def _contrasts_from(
    pairwise: list[Any], counts: pd.Series, factor: str, method: str, n_obs: int | None = None
) -> list[Contrast]:
    """Re-shape `stats_tools._pairwise` output into the contrast model this module returns."""
    return [
        Contrast(
            factor=factor,
            level_a=item.level_a,
            level_b=item.level_b,
            difference=item.difference,
            ci_low=item.ci_low,
            ci_high=item.ci_high,
            p_value=item.p_value,
            adjusted_p_value=item.p_adjusted,
            method=method,
            n_obs=n_obs
            if n_obs is not None
            else int(counts[item.level_a]) + int(counts[item.level_b]),
        )
        for item in pairwise
    ]


def _fell_back(notes: list[str]) -> bool:
    """Whether the Tukey adjustment declined and Holm took over."""
    return any(note.startswith("Tukey") and "Holm" in note for note in notes)


def _group_contrasts(
    *,
    data: pd.DataFrame,
    response: str,
    factor: str,
    levels: list[str],
    group: str | None,
    confidence_level: float,
    notes: list[str],
) -> tuple[list[Contrast], str]:
    """Differences of level means from a model of the same factor, adjusted for multiplicity."""
    fit = _model(data, response, [factor], group, "gaussian")
    bases: dict[tuple[str, str], np.ndarray] = {}
    for level in levels:
        frame = data.copy()
        _pin_value(frame, factor, level)
        bases[(factor, level)] = stats_tools._predict_vector(fit, frame)

    counts = data[factor].astype(str).value_counts()
    balanced = len(set(int(counts[level]) for level in counts.index)) == 1
    adjust = "tukey" if group is None and balanced else "holm"
    before = len(notes)
    pairwise = stats_tools._pairwise(
        fit,
        response,
        [factor],
        {factor: levels},
        bases,
        confidence_level,
        adjust,
        notes,
        data,
    )
    method = "holm" if _fell_back(notes[before:]) else adjust
    return _contrasts_from(pairwise, counts, factor, method), method


def _pairs(levels: list[str]) -> list[tuple[str, str]]:
    return [(levels[i], levels[j]) for i in range(len(levels)) for j in range(i + 1, len(levels))]


def _paired_contrast(
    *,
    levels: list[str],
    differences: np.ndarray,
    response: str,
    factor: str,
    confidence_level: float,
    n_obs: int,
) -> list[Contrast]:
    """The single within-pair difference, with the paired interval."""
    n = differences.size
    mean = float(differences.mean())
    std_error = float(differences.std(ddof=1) / np.sqrt(n))
    t_crit = stats_tools._t_critical(confidence_level, n - 1)
    statistic = mean / std_error if std_error > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(statistic), n - 1))
    return [
        Contrast(
            factor=factor,
            level_a=levels[0],
            level_b=levels[1],
            difference=round(mean, 4),
            ci_low=round(mean - t_crit * std_error, 4),
            ci_high=round(mean + t_crit * std_error, 4),
            p_value=p_value,
            adjusted_p_value=p_value,
            method="paired difference (single contrast, no multiplicity)",
            n_obs=n_obs,
        )
    ]


def _comparison_diagnostics(
    *,
    levels: list[str],
    summaries: list[GroupSummary],
    levene_p: float,
    shapiro: dict[str, float],
    shapiro_p: float,
    min_cell: int,
    group: str | None,
    data: pd.DataFrame,
    response: str,
    factor: str,
    test: str,
) -> list[Verdict]:
    verdicts: list[Verdict] = []
    if levene_p < LEVENE_ALPHA:
        verdicts.append(
            Verdict(
                check="equal_variance",
                status="caution",
                statement=(
                    f"Levene p = {levene_p:.3g}: the spread differs across levels, which is "
                    f"why the route is {test}."
                ),
                remedy=(
                    f"compare_groups(response='{response}', factor='{factor}') already drops the "
                    "equal-variance assumption; inspect the wide level with run_sql() to see "
                    "whether a few rows cause it."
                ),
                severity="warning",
            )
        )
    else:
        verdicts.append(
            Verdict(
                check="equal_variance",
                status="ok",
                statement=f"Levene p = {levene_p:.3g}: the spread is similar across levels.",
            )
        )
    if not shapiro:
        verdicts.append(
            Verdict(
                check="normality",
                status="caution",
                statement="No level holds 3 rows, so normality could not be checked.",
                remedy=(
                    f"Collect more rows, or fit_model(response='{response}', "
                    f"terms=['{factor}']) and read its residual diagnostics."
                ),
                severity="warning",
            )
        )
    elif shapiro_p < ALPHA:
        thin = sorted(level for level, p in shapiro.items() if p < ALPHA)
        rank_route = test in {"mann_whitney", "kruskal_wallis", "wilcoxon"}
        if rank_route:
            status, severity, statement = (
                "caution",
                "warning",
                f"Shapiro p = {shapiro_p:.3g}: level(s) {thin} depart from normality, which is "
                f"why the route is {test}; that test assumes no distribution.",
            )
            remedy = (
                f"compare_groups(response='{response}', factor='{factor}') already routes around "
                "it; report the medians from the same result."
            )
        elif min_cell >= SMALL_CELL:
            status, severity, statement = (
                "caution",
                "warning",
                f"Shapiro p = {shapiro_p:.3g}: level(s) {thin} depart from normality, but the "
                f"smallest cell holds {min_cell} rows, so the mean is still usable.",
            )
            remedy = (
                f"Call compare_groups(response='{response}', factor='{factor}') on a transformed "
                "response if the skew is structural, or read the medians from the same result."
            )
        else:
            status, severity, statement = (
                "blocked",
                "blocking",
                f"Shapiro p = {shapiro_p:.3g}: level(s) {thin} depart from normality and the "
                f"smallest cell holds only {min_cell} rows, so the mean is not the summary to "
                "report.",
            )
            remedy = (
                f"Call compare_groups(response='{response}', factor='{factor}', paired=...) only "
                "if the rows are matched; otherwise collect more rows or use a rank route."
            )
        verdicts.append(
            Verdict(
                check="normality",
                status=status,
                statement=statement,
                remedy=remedy,
                severity=severity,
            )
        )
    else:
        verdicts.append(
            Verdict(
                check="normality",
                status="ok",
                statement=f"Shapiro p = {shapiro_p:.3g}: each level is consistent with normality.",
            )
        )
    if min_cell < 2:
        verdicts.append(
            Verdict(
                check="cell_size",
                status="blocked",
                statement=(
                    f"The smallest level holds {min_cell} row(s), so nothing can be estimated."
                ),
                remedy=(
                    "Pool that level and retry "
                    f"compare_groups(response='{response}', factor='{factor}')."
                ),
                severity="blocking",
            )
        )
    elif min_cell < SMALL_CELL:
        verdicts.append(
            Verdict(
                check="cell_size",
                status="caution",
                statement=(
                    f"The smallest level holds {min_cell} rows (< {SMALL_CELL}), so the interval "
                    "covers a wide range of effects."
                ),
                remedy=(
                    f"Collect more rows, or call power_analysis(response='{response}', "
                    f"factor='{factor}') to see the effect this design can resolve."
                ),
                severity="warning",
            )
        )
    else:
        verdicts.append(
            Verdict(
                check="cell_size",
                status="ok",
                statement=f"Every level holds at least {min_cell} rows.",
            )
        )
    if group:
        verdicts.append(
            Verdict(
                check="clustering",
                status="caution",
                statement=(
                    f"Rows repeat within '{group}' ({data[group].nunique()} groups); the "
                    f"{test} omnibus test treats them as independent."
                ),
                remedy=(
                    f"Call fit_model(response='{response}', terms=['{factor}'], "
                    f"group='{group}') for a random intercept, which the pairwise "
                    "intervals here do not include."
                ),
                severity="warning",
            )
        )
    return verdicts


def compare_groups(
    response: str,
    factor: str,
    group: str | None = None,
    paired: str | None = None,
    smallest_effect: float | None = None,
    confidence_level: float = 0.95,
) -> ComparisonResult:
    """Compare the levels of `factor` on `response`, with an effect size and intervals.

    The route follows the data, not a preference: unequal spreads give Welch, a
    violated distributional assumption with small cells gives a rank test, and
    `paired` gives the test on within-pair differences. `routing_reason` names
    the number that decided it, so a reader can disagree. `group` names a column
    the rows repeat inside (a block, a site): it is reported and it shapes the
    contrasts, while the omnibus test treats rows as independent.

    Pass `smallest_effect` when the user named one; the result then says whether
    the interval excludes it. Without it the result says what the interval does
    not exclude, and never reads a wide interval as evidence of no difference.
    """
    store = session.current_session()
    store.state.require("assumptions")
    if not 0 < confidence_level < 1:
        raise ToolError(f"confidence_level must be between 0 and 1, got {confidence_level}.")
    if factor == response:
        raise ToolError(f"Factor and response are both '{response}'; pass two different columns.")

    frame = stats_tools._frame()
    extra = [name for name in (group, paired) if name]
    stats_tools._require(frame, [response, factor, *extra])
    stats_tools._require_numeric(frame, response, "Response")

    data, exclusions = _complete_rows(frame, [response, factor, *extra])
    notes: list[str] = []
    if len(data) < 4:
        raise ToolError(
            f"Only {len(data)} complete row(s) remain for {response} and {factor}. Load more rows, "
            "or repair the missing values first."
        )
    if exclusions:
        exclusions[0]["reason"] = f"missing values in {response}, {factor}" + (
            f", {', '.join(extra)}" if extra else ""
        )

    levels = _levels(data[factor])
    if len(levels) < 2:
        raise ToolError(f"Factor '{factor}' has a single level, so there is nothing to compare.")
    if len(levels) > stats_tools.MAX_LEVELS:
        raise ToolError(
            f"Factor '{factor}' has {len(levels)} levels (limit {stats_tools.MAX_LEVELS}). "
            "Pool them into groups first, or pass it to fit_model as a covariate."
        )
    n_obs = len(data)
    summaries = [_group_summary(data, response, factor, level) for level in levels]
    thin = [summary.level for summary in summaries if summary.n < 2]
    if thin:
        raise ToolError(
            f"Level(s) {thin} of '{factor}' hold fewer than two rows, so no spread can be "
            "estimated. Pool them and retry compare_groups()."
        )

    samples = [
        data.loc[data[factor].astype(str) == level, response].to_numpy(dtype=float)
        for level in levels
    ]
    levene_p = float(stats.levene(*samples).pvalue)
    shapiro = {
        level: stats_tools._normality(sample).p_value
        for level, sample in zip(levels, samples, strict=True)
        if sample.size >= 3
    }
    shapiro_p = min(shapiro.values()) if shapiro else 1.0
    min_cell = min(summary.n for summary in summaries)

    pairs: dict[str, Any] | None = None
    if paired:
        if len(levels) != 2:
            raise ToolError(
                f"A paired comparison needs exactly 2 levels of '{factor}', found {len(levels)}. "
                "Select the two levels with run_sql first, or drop the paired argument."
            )
        counts = data.groupby([paired, factor], observed=True)[response].size()
        if int(counts.max()) > 1:
            worst = counts.idxmax()
            raise ToolError(
                f"'{paired}' does not identify one row per level: {worst} carries "
                f"{int(counts.max())} rows. Aggregate first with run_sql, or pick another column."
            )
        wide = data.pivot(index=paired, columns=factor, values=response).dropna()
        wide.columns = [str(column) for column in wide.columns]
        if wide.shape[0] < 3:
            raise ToolError(
                f"Only {wide.shape[0]} value(s) of '{paired}' carry both levels of '{factor}'; a "
                "paired test needs at least 3 complete pairs. Load more rows, or use the unpaired "
                "route."
            )
        dropped = n_obs - 2 * len(wide)
        if dropped:
            exclusions.append({"reason": f"incomplete pairs in {paired}", "n": int(dropped)})
        pairs = {
            "n": int(wide.shape[0]),
            "differences": wide[levels[0]].to_numpy(dtype=float)
            - wide[levels[1]].to_numpy(dtype=float),
        }

    test, reason = _route(
        levels=levels,
        pairs=pairs,
        levene_p=levene_p,
        shapiro_p=shapiro_p,
        smallest_cell=min_cell,
    )
    notes.append(reason)
    statistic, p_value, df_label = _omnibus(
        test=test, data=data, response=response, factor=factor, samples=samples, pairs=pairs
    )
    notes.append(f"Omnibus test: {test} = {statistic:.4g} on {df_label}, p = {p_value:.4g}.")

    effect = _effect_size(
        test=test,
        data=data,
        response=response,
        factor=factor,
        levels=levels,
        samples=samples,
        pairs=pairs,
        notes=notes,
        confidence_level=confidence_level,
    )
    notes += _equivalence_notes(
        effect.name, effect.ci_low, effect.ci_high, smallest_effect, confidence_level
    )

    if pairs is not None:
        contrasts = _paired_contrast(
            levels=levels,
            differences=pairs["differences"],
            response=response,
            factor=factor,
            confidence_level=confidence_level,
            n_obs=n_obs,
        )
        notes.append(
            "The contrast is the mean within-pair difference; a paired design leaves no "
            "multiplicity to adjust."
        )
    else:
        contrasts, method = _group_contrasts(
            data=data,
            response=response,
            factor=factor,
            levels=levels,
            group=group,
            confidence_level=confidence_level,
            notes=notes,
        )
        notes.append(
            f"Pairwise contrasts are differences of level means from an OLS fit of '{factor}', "
            f"adjusted with {method}."
        )
        if test in {"mann_whitney", "kruskal_wallis", "wilcoxon"}:
            notes.append(
                f"The contrasts above are mean-based and do not inherit the {test} route. Read "
                "them as descriptive differences; the inference is the rank test. A rank-based "
                "post-hoc for more than two groups is not implemented here."
            )

    diagnostics = _comparison_diagnostics(
        levels=levels,
        summaries=summaries,
        levene_p=levene_p,
        shapiro=shapiro,
        shapiro_p=shapiro_p,
        min_cell=min_cell,
        group=group,
        data=data,
        response=response,
        factor=factor,
        test=test,
    )

    payload = ComparisonResult(
        response=response,
        factor=factor,
        test=test,
        routing_reason=reason,
        n_obs=n_obs,
        groups=summaries,
        effect_size=effect,
        contrasts=contrasts,
        diagnostics=diagnostics,
        smallest_effect=smallest_effect,
        result_id="",
        next="",
        notes=notes,
    )
    stored = store.put(
        payload,
        kind="comparison",
        summary=f"{test} on {response} by {factor} ({n_obs} rows)",
        rows_total=len(frame),
        rows_used=n_obs,
        exclusions=exclusions,
    )
    payload.next = (
        f"Call sensitivity('{stored.result_id}') to check that this difference survives the "
        "influential rows, or fit_model() followed by marginal_effects() to adjust for other "
        "columns."
    )
    status = _verdict_status(diagnostics)
    store.state.record(
        "assumptions",
        status=status,
        result_id=stored.result_id,
        note=f"{test}: {reason}",
        verdicts=diagnostics,
    )
    store.state.record(
        "estimate",
        status=status,
        result_id=stored.result_id,
        note=f"{effect.name} = {effect.estimate:.4g} [{effect.ci_low:.4g}, {effect.ci_high:.4g}]",
        verdicts=diagnostics,
    )
    _reseal(stored)
    return payload


# --------------------------------------------------------------------------- #
# marginal_effects
# --------------------------------------------------------------------------- #


def _fit_payload(store: session.SessionStore, result_id: str) -> FitResult:
    """The stored fit, refusing anything that is not one."""
    stored = store.get(result_id)
    if stored.kind != "fit" or not isinstance(stored.payload, FitResult):
        raise ToolError(
            f"Result {result_id!r} is a {stored.kind}, not a fit. Pass an id returned by "
            f"fit_model(). Stored fits: "
            f"{[key for key, item in store.results.items() if item.kind == 'fit'] or 'none'}."
        )
    return stored.payload


def _margin_frame(
    data: pd.DataFrame,
    *,
    pins: dict[str, list[Any]],
    covariates: list[str] | None,
    numeric: list[str],
) -> pd.DataFrame:
    """Observed rows with the pinned columns set, and the rest left to vary."""
    frame = data.copy()
    for column, values in pins.items():
        _pin_value(frame, column, values[0])
    if covariates is not None:
        for column in numeric:
            if column not in pins and column not in covariates:
                frame[column] = float(data[column].mean())
    return frame


def _margin_delta(
    fit: stats_tools._Fit,
    components: list[tuple[pd.DataFrame, float]],
    link: Callable[[np.ndarray], np.ndarray] | None,
    confidence_level: float,
) -> tuple[float, float, float, float]:
    """Estimate and delta-method interval for a weighted sum of average predictions.

    With the identity link this is `_estimate` on the mean design row, exactly
    what `stats_tools.marginal_means` reports. Under a link it is the average of
    the predicted responses over the observed rows, with the gradient of that
    average taken numerically.
    """
    matrices = [
        (stats_tools._design(fit.rhs, frame, fit.params.index), weight)
        for frame, weight in components
    ]
    basis = sum(weight * matrix.mean(axis=0) for matrix, weight in matrices)
    if link is None:
        estimate, std_error = stats_tools._estimate(fit, basis)
    else:

        def average(beta: np.ndarray) -> float:
            return float(sum(weight * np.mean(link(matrix @ beta)) for matrix, weight in matrices))

        estimate = average(fit.beta)
        gradient = _numeric_gradient(average, fit.beta)
        variance = float(gradient @ np.asarray(fit.cov, dtype=float) @ gradient)
        std_error = float(np.sqrt(max(variance, 0.0)))
    half_width = stats_tools._t_critical(confidence_level, fit.df_resid) * std_error
    return estimate, std_error, estimate - half_width, estimate + half_width


def _numeric_gradient(function: Callable[[np.ndarray], float], beta: np.ndarray) -> np.ndarray:
    """Central-difference gradient, for the delta method on a nonlinear average."""
    gradient = np.zeros_like(beta, dtype=float)
    for index in range(beta.size):
        shift = np.zeros_like(beta)
        step = 1e-6 * max(abs(float(beta[index])), 1.0)
        shift[index] = step
        gradient[index] = (function(beta + shift) - function(beta - shift)) / (2 * step)
    return gradient


def marginal_effects(
    result_id: str,
    at: dict[str, list[Any]] | None = None,
    covariates: list[str] | None = None,
    contrasts: Literal["pairwise", "vs_reference", "none"] = "pairwise",
    adjust: Adjust = "tukey",
) -> MarginalEffectsResult:
    """Average predicted margins from a fitted model, at each level of each factor.

    This is the tool that stops a coefficient being read as an effect: the
    estimate for a level is the average predicted response over the observed
    rows with that factor set to the level, so covariates are averaged over the
    distribution they really have. Numeric terms vary by default; name
    `covariates` to average over only those and hold the rest at their mean, and
    pass `at` to pin a factor's levels or a covariate's value.

    Contrasts come with intervals and adjusted p-values under `adjust`. Where a
    term is an interaction the contrast is a difference of differences: the
    difference between two levels of one factor at two levels of the other.

    Call this after `fit_model` with the id it returned. It refines that estimate
    and records no new stage.
    """
    store = session.current_session()
    store.state.require("estimate")
    source = _fit_payload(store, result_id)

    frame = stats_tools._frame()
    columns = _columns_of(source.response, source.terms, source.group)
    stats_tools._require(frame, columns)
    data, _ = _complete_rows(frame, columns)

    base = _base_columns(source.terms)
    numeric = _numeric_columns(data, base)
    factors = [column for column in base if column not in numeric]
    pins: dict[str, list[Any]] = {}
    for column, value in (at or {}).items():
        if column not in base:
            raise ToolError(f"'{column}' is not a term of {result_id}. Its terms are {base}.")
        pins[column] = list(value) if isinstance(value, list | tuple) else [value]
        if not pins[column]:
            raise ToolError(f"`at` gives no value for '{column}'; pass at least one.")
    if covariates:
        unknown = [column for column in covariates if column not in numeric]
        if unknown:
            raise ToolError(
                f"covariates must be numeric terms of the fit; {unknown} are not. Numeric terms: "
                f"{numeric or 'none'}."
            )
    if contrasts not in {"pairwise", "vs_reference", "none"}:
        raise ToolError(
            f"Unknown contrasts '{contrasts}'. Use 'pairwise', 'vs_reference' or 'none'."
        )
    if adjust not in {"tukey", "holm", "bonferroni", "fdr_bh"}:
        raise ToolError(
            f"Unknown adjustment '{adjust}'. Use 'tukey', 'holm', 'bonferroni' or 'fdr_bh'."
        )

    fit = _model(data, source.response, source.terms, source.group, source.family)
    link = None if source.family == "gaussian" else fit.result.model.family.link.inverse

    notes = [
        f"Refit from stored result {result_id} on the current table ({fit.formula}, {len(data)} "
        "rows): each cell is the average predicted response over the observed rows, so it is a "
        "margin and not a coefficient."
    ]
    if len(data) != source.rows_used:
        notes.append(
            f"The current table gives {len(data)} complete rows where the stored fit used "
            f"{source.rows_used}."
        )
    averaged = covariates if covariates is not None else numeric
    notes.append(
        "Averaged over the observed distribution of "
        f"{averaged or 'no numeric term'}"
        + (
            f", with {[c for c in numeric if c not in averaged]} held at their mean."
            if averaged != numeric and numeric
            else "."
        )
    )
    if source.group:
        notes.append(
            f"Predictions use the fixed effects only: the random intercept for '{source.group}' is "
            "averaged out, which is what an adjusted mean over the population implies."
        )
    if source.family != "gaussian":
        notes.append(
            "Cells and contrasts stay on the response scale: a contrast is the difference between "
            "two average predictions, and an interaction contrast is the difference of two such "
            "differences."
        )

    cells: list[MarginCell] = []
    order: dict[str, list[str]] = {}
    cell_counts: dict[str, pd.Series] = {}
    for factor in factors:
        order[factor] = [str(value) for value in pins.get(factor, _levels(data[factor]))]
        cell_counts[factor] = data[factor].astype(str).value_counts()
        for level in order[factor]:
            pinned = _margin_frame(
                data, pins={**pins, factor: [level]}, covariates=covariates, numeric=numeric
            )
            estimate, std_error, low, high = _margin_delta(fit, [(pinned, 1.0)], link, 0.95)
            cells.append(
                MarginCell(
                    factor=factor,
                    level=level,
                    estimate=round(estimate, 4),
                    std_error=round(std_error, 4),
                    ci_low=round(low, 4),
                    ci_high=round(high, 4),
                    n_obs=int((data[factor].astype(str) == level).sum()),
                )
            )

    references = dict(source.reference_levels)
    for column in numeric:
        if column in pins:
            references[column] = f"held at {pins[column][0]}"
        elif covariates is not None and column not in covariates:
            references[column] = f"held at its mean {float(data[column].mean()):.4g}"
        else:
            references[column] = "averaged over its observed values"

    contrast_objects: list[Contrast] = []
    method = adjust
    if contrasts != "none":
        request = adjust
        if request == "tukey" and (source.group or source.family != "gaussian"):
            request = "holm"
            notes.append(
                "Tukey's studentized range is defined for a one-way Gaussian comparison, so the "
                "adjustment reported is Holm."
            )
        for factor in factors:
            levels = order[factor]
            if contrasts == "pairwise":
                wanted = _pairs(levels)
            else:
                reference = source.reference_levels.get(factor, _levels(data[factor])[0])
                wanted = [(level, reference) for level in levels if level != reference]
            if not wanted:
                continue
            components = [
                (
                    _margin_frame(
                        data,
                        pins={**pins, factor: [level_a]},
                        covariates=covariates,
                        numeric=numeric,
                    ),
                    _margin_frame(
                        data,
                        pins={**pins, factor: [level_b]},
                        covariates=covariates,
                        numeric=numeric,
                    ),
                )
                for level_a, level_b in wanted
            ]
            raw = [
                _margin_contrast(fit, [(first, 1.0), (second, -1.0)], link, 0.95)
                for first, second in components
            ]
            p_values = np.array([item["p_value"] for item in raw], dtype=float)
            reported, adjusted = stats_tools._adjust(
                p_values,
                request,
                data,
                factor,
                _levels(data[factor]),
                source.response,
                0.95,
                notes,
            )
            method = reported
            for (level_a, level_b), item, adjusted_p in zip(wanted, raw, adjusted, strict=True):
                contrast_objects.append(
                    Contrast(
                        factor=factor,
                        level_a=level_a,
                        level_b=level_b,
                        difference=item["difference"],
                        ci_low=item["ci_low"],
                        ci_high=item["ci_high"],
                        p_value=item["p_value"],
                        adjusted_p_value=float(adjusted_p),
                        method=reported,
                        n_obs=int(cell_counts[factor][level_a]) + int(cell_counts[factor][level_b]),
                    )
                )
        contrast_objects += _interaction_contrasts(
            fit=fit,
            data=data,
            source=source,
            base=base,
            numeric=numeric,
            pins=pins,
            covariates=covariates,
            notes=notes,
        )

    payload = MarginalEffectsResult(
        source_result_id=result_id,
        cells=cells,
        contrasts=contrast_objects,
        adjust=method,
        reference_levels=references,
        result_id="",
        next="",
        notes=notes,
    )
    stored = store.put(
        payload,
        kind="marginal",
        summary=f"margins from {result_id}: {len(cells)} cell(s)",
        rows_total=len(frame),
        rows_used=len(data),
    )
    payload.next = (
        f"Call sensitivity('{result_id}') to test whether these margins survive influential rows "
        f"and a different family, then explain_result('{result_id}')."
    )
    _reseal(stored)
    return payload


def _margin_contrast(
    fit: stats_tools._Fit,
    components: list[tuple[pd.DataFrame, float]],
    link: Callable[[np.ndarray], np.ndarray] | None,
    confidence_level: float,
) -> dict[str, float]:
    """A weighted sum of average predictions minus another, with interval and p-value."""
    difference, std_error, low, high = _margin_delta(fit, components, link, confidence_level)
    statistic = difference / std_error if std_error > 0 else 0.0
    return {
        "difference": round(difference, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "p_value": float(2 * stats.t.sf(abs(statistic), fit.df_resid)),
    }


def _setting_levels(
    name: str, pins: dict[str, list[Any]], data: pd.DataFrame, numeric: list[str]
) -> list[str]:
    """The two settings an interaction contrast varies: pinned if given, else the range."""
    pinned = pins.get(name)
    if pinned is not None and len(pinned) >= 2:
        return [str(pinned[0]), str(pinned[1])]
    if name in numeric:
        values = data[name].to_numpy(dtype=float)
        return [f"{values.min():.4g}", f"{values.max():.4g}"]
    return _levels(data[name])[:2]


def _interaction_contrasts(
    *,
    fit: stats_tools._Fit,
    data: pd.DataFrame,
    source: FitResult,
    base: list[str],
    numeric: list[str],
    pins: dict[str, list[Any]],
    covariates: list[str] | None,
    notes: list[str],
) -> list[Contrast]:
    """Every interaction as a difference of differences, with a delta-method interval."""
    contrasts: list[Contrast] = []
    for term in _design_terms(source.terms):
        parts = term.split(":")
        if len(parts) != 2:
            continue
        first, second = parts
        if first not in base or second not in base:
            continue
        settings_a = _setting_levels(first, pins, data, numeric)
        settings_b = _setting_levels(second, pins, data, numeric)
        if len(settings_a) < 2 or len(settings_b) < 2:
            continue
        frames = {
            (a, b): _margin_frame(
                data,
                pins={**pins, first: [a], second: [b]},
                covariates=covariates,
                numeric=numeric,
            )
            for a in settings_a
            for b in settings_b
        }
        components = [
            (frames[(settings_a[0], settings_b[0])], 1.0),
            (frames[(settings_a[1], settings_b[0])], -1.0),
            (frames[(settings_a[0], settings_b[1])], -1.0),
            (frames[(settings_a[1], settings_b[1])], 1.0),
        ]
        link = None if source.family == "gaussian" else fit.result.model.family.link.inverse
        item = _margin_contrast(fit, components, link, 0.95)
        contrasts.append(
            Contrast(
                factor=term,
                level_a=f"{first} difference at {second}={settings_b[0]}",
                level_b=f"{first} difference at {second}={settings_b[1]}",
                difference=item["difference"],
                ci_low=item["ci_low"],
                ci_high=item["ci_high"],
                p_value=item["p_value"],
                adjusted_p_value=item["p_value"],
                method="interaction (difference of differences)",
                n_obs=len(data),
            )
        )
        notes.append(
            f"'{term}' is an interaction: its contrast is a difference of differences, the "
            f"'{first}' difference at {second}={settings_b[0]} minus the same difference at "
            f"{second}={settings_b[1]}, in response units."
        )
    return contrasts


# --------------------------------------------------------------------------- #
# sensitivity
# --------------------------------------------------------------------------- #


def _focus_term(fit: stats_tools._Fit) -> str:
    """The term the report re-estimates: the smallest p-value among the coefficients."""
    terms = [column for column in fit.params.index if column != "Intercept"]
    if not terms:
        raise ToolError(
            "The fit has only an intercept, so there is no term to re-estimate. Refit with "
            "fit_model() and at least one term."
        )
    return min(terms, key=lambda column: float(fit.result.pvalues[column]))


def _check_status(original: dict[str, float], new: dict[str, float]) -> str:
    """stable, attenuated, flipped or inconclusive, on direction and significance."""
    if original["estimate"] * new["estimate"] < 0:
        return "flipped"
    if original["p_value"] < ALPHA <= new["p_value"]:
        return "attenuated"
    return "stable"


def _check_result(
    check: str,
    new: dict[str, float],
    original: dict[str, float],
    statement: str,
    status: CheckStatus | None = None,
) -> CheckResult:
    return CheckResult(
        check=check,
        estimate=round(new["estimate"], 4),
        ci_low=round(new["ci_low"], 4),
        ci_high=round(new["ci_high"], 4),
        status=status or _check_status(original, new),
        statement=statement,
    )


def _inconclusive(check: str, original: dict[str, float], statement: str) -> CheckResult:
    """A check that could not run: the original estimate stands, marked as such."""
    return _check_result(check, original, original, statement, status="inconclusive")


def _check_influence(
    *,
    source: FitResult,
    fit: stats_tools._Fit,
    data: pd.DataFrame,
    focus: str,
    original: dict[str, float],
    notes: list[str],
) -> CheckResult:
    design = _model(data, source.response, source.terms, None, "gaussian")
    cook = np.asarray(OLSInfluence(design.result).cooks_distance[0], dtype=float)
    cutoff = 4 / len(data)
    keep = cook <= cutoff
    if bool(keep.all()):
        return CheckResult(
            check="influence",
            estimate=round(original["estimate"], 4),
            ci_low=round(original["ci_low"], 4),
            ci_high=round(original["ci_high"], 4),
            status="stable",
            statement=(
                f"No row reaches the Cook's distance cut-off 4/n = {cutoff:.4g} (largest "
                f"{float(cook.max()):.3g}), so there is nothing to drop and the estimate stands at "
                f"{original['estimate']:.4g}."
            ),
        )
    refit = _model(data.loc[keep], source.response, source.terms, source.group, source.family)
    new = _statistics(refit, focus, 0.95)
    dropped = int((~keep).sum())
    notes.append(f"Left out {dropped} row(s) with Cook's distance above {cutoff:.4g}.")
    return _check_result(
        "influence",
        new,
        original,
        f"Dropping the {dropped} row(s) with Cook's distance above 4/n = {cutoff:.4g} moves "
        f"'{focus}' from {original['estimate']:.4g} to {new['estimate']:.4g} "
        f"[{new['ci_low']:.4g}, {new['ci_high']:.4g}] (p = {new['p_value']:.3g}).",
    )


def _check_robust_se(
    *,
    source: FitResult,
    fit: stats_tools._Fit,
    data: pd.DataFrame,
    focus: str,
    original: dict[str, float],
) -> CheckResult:
    if source.family != "gaussian":
        return _inconclusive(
            "robust_se",
            original,
            f"HC3 is a Gaussian sandwich estimator and this stack exposes no robust covariance for "
            f"a {source.family} fit, so the analytic interval stands.",
        )
    if source.group:
        design = _model(data, source.response, source.terms, None, "gaussian")
        result = design.result.get_robustcov_results(
            cov_type="cluster", groups=data[source.group].to_numpy(), use_t=True, df_correction=True
        )
        clusters = int(data[source.group].nunique())
        names = list(design.result.model.exog_names)
        index = names.index(focus)
        new = {
            "estimate": float(np.asarray(result.params)[index]),
            "ci_low": float(np.asarray(result.conf_int())[index][0]),
            "ci_high": float(np.asarray(result.conf_int())[index][1]),
            "p_value": float(np.asarray(result.pvalues)[index]),
        }
        return _check_result(
            "robust_se",
            new,
            original,
            f"Cluster-robust standard errors by '{source.group}' ({clusters} clusters, t on "
            f"{clusters - 1} df) leave '{focus}' at {new['estimate']:.4g} "
            f"[{new['ci_low']:.4g}, {new['ci_high']:.4g}] (p = {new['p_value']:.3g}).",
        )
    result = fit.result.get_robustcov_results(cov_type="HC3")
    names = list(fit.result.model.exog_names)
    index = names.index(focus)
    new = {
        "estimate": float(np.asarray(result.params)[index]),
        "ci_low": float(np.asarray(result.conf_int())[index][0]),
        "ci_high": float(np.asarray(result.conf_int())[index][1]),
        "p_value": float(np.asarray(result.pvalues)[index]),
    }
    return _check_result(
        "robust_se",
        new,
        original,
        f"HC3 standard errors leave '{focus}' at {new['estimate']:.4g} "
        f"[{new['ci_low']:.4g}, {new['ci_high']:.4g}] (p = {new['p_value']:.3g}), so "
        + (
            "heteroscedasticity is not what the interval rests on."
            if new["p_value"] < ALPHA
            else "the analytic interval was carrying the result."
        ),
    )


def _check_alternative_family(
    *, source: FitResult, data: pd.DataFrame, focus: str, original: dict[str, float]
) -> CheckResult:
    frame = data.copy()
    if source.family == "gaussian":
        frame[source.response] = stats.rankdata(data[source.response].to_numpy(dtype=float))
        description = (
            "Rank-based equivalent: the response replaced by its ranks. The estimate is on "
            "the rank scale, so read its sign and its significance, not its size."
        )
    elif source.family == "binomial":
        description = (
            "Different family: a linear probability model (gaussian fit of the same 0/1 response), "
            "which cannot predict outside [0, 1] but does not depend on the logit link."
        )
    else:
        frame[source.response] = np.log1p(data[source.response].to_numpy(dtype=float))
        description = "Different family: gaussian on log1p(response), the usual robust stand-in."
    refit = _model(frame, source.response, source.terms, source.group, "gaussian")
    new = _statistics(refit, focus, 0.95)
    return _check_result(
        "alternative_family",
        new,
        original,
        f"{description} '{focus}' comes out at {new['estimate']:.4g} "
        f"[{new['ci_low']:.4g}, {new['ci_high']:.4g}] (p = {new['p_value']:.3g}).",
    )


def _bootstrap_ci(
    *,
    source: FitResult,
    data: pd.DataFrame,
    focus: str,
    original: dict[str, float],
    notes: list[str],
) -> CheckResult:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_rows = len(data)
    values: list[float] = []
    failures = 0
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = data.iloc[rng.integers(0, n_rows, n_rows)]
        try:
            refit = _model(sample, source.response, source.terms, None, source.family)
            values.append(float(refit.params[focus]))
        except ToolError:
            failures += 1
    if len(values) < BOOTSTRAP_RESAMPLES // 2:
        return _inconclusive(
            "bootstrap_ci",
            original,
            f"Only {len(values)} of {BOOTSTRAP_RESAMPLES} resamples refitted, too few for an "
            "interval; the analytic one stands.",
        )
    low, high = np.percentile(values, [2.5, 97.5])
    if source.group:
        notes.append(
            "Bootstrap resamples are refitted without the random intercept: a mixed fit per "
            "resample costs too much, so this interval understates the cluster uncertainty."
        )
    return _check_result(
        "bootstrap_ci",
        {
            "estimate": original["estimate"],
            "ci_low": float(low),
            "ci_high": float(high),
            "p_value": 0.0 if float(low) * float(high) > 0 else 1.0,
        },
        original,
        f"Percentile bootstrap, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED} "
        f"({failures} refit failures skipped): '{focus}' covers "
        f"[{float(low):.4g}, {float(high):.4g}].",
    )


def _check_adjustment_set(
    *,
    source: FitResult,
    data: pd.DataFrame,
    fit: stats_tools._Fit,
    focus: str,
    original: dict[str, float],
    notes: list[str],
) -> CheckResult:
    others = [column for column in fit.params.index if column not in {"Intercept", focus}]
    if not others:
        return _inconclusive(
            "adjustment_set",
            original,
            "The fit has no other term to drop, so the adjustment set cannot be perturbed.",
        )
    worst = max(others, key=lambda column: float(fit.result.pvalues[column]))
    target = _split_column(worst)[0]
    kept = [term for term in source.terms if target not in _design_terms([term])]
    if not kept:
        return _inconclusive(
            "adjustment_set",
            original,
            f"Dropping '{target}' would leave no term in the model, so nothing to refit.",
        )
    refit = _model(data, source.response, kept, source.group, source.family)
    new = _statistics(refit, focus, 0.95)
    notes.append(f"Dropped '{target}' (p = {float(fit.result.pvalues[worst]):.3g}) and refit.")
    return _check_result(
        "adjustment_set",
        new,
        original,
        f"Dropping the least significant term '{target}' (p = "
        f"{float(fit.result.pvalues[worst]):.3g}) leaves '{focus}' at {new['estimate']:.4g} "
        f"[{new['ci_low']:.4g}, {new['ci_high']:.4g}] (p = {new['p_value']:.3g}).",
    )


def sensitivity(
    result_id: str,
    checks: list[Check] | None = None,
) -> SensitivityReport:
    """Re-run a stored fit under conditions that should not change the answer.

    Five checks, all of them by default: influential rows dropped, robust or
    cluster-robust standard errors, a different family or a rank-based
    equivalent, a bootstrap interval instead of the analytic one, and the least
    significant term dropped from the adjustment set.

    Each check reports the recomputed estimate with its interval and a status of
    stable, attenuated, flipped or inconclusive, and `stable_share` is the share
    that kept both the direction and the significance of the term under scrutiny.
    This is the stage a model never volunteers, so the report reads on its own:
    the verdict says whether the conclusion is durable.

    Call it after `fit_model` or `compare_groups`, with the id one of them
    returned.
    """
    store = session.current_session()
    store.state.require("estimate")
    source = _fit_payload(store, result_id)

    requested = [str(check) for check in (checks or list(_CHECKS))]
    unknown = [check for check in requested if check not in _CHECKS]
    if unknown:
        raise ToolError(f"Unknown check(s) {unknown}. Use a subset of {list(_CHECKS)}.")

    frame = stats_tools._frame()
    columns = _columns_of(source.response, source.terms, source.group)
    stats_tools._require(frame, columns)
    data, _ = _complete_rows(frame, columns)
    if len(data) < 4:
        raise ToolError(
            f"Only {len(data)} complete row(s) remain for {source.response}; nothing can be "
            "re-estimated."
        )

    fit = _model(data, source.response, source.terms, source.group, source.family)
    focus = _focus_term(fit)
    original = _statistics(fit, focus, 0.95)

    notes = [
        f"Every check re-estimates '{focus}', the smallest p-value term of {result_id} "
        f"(p = {original['p_value']:.3g} in the stored fit).",
        "The point estimate is the stored one unless a check reports its own; intervals are "
        "at 95%.",
    ]
    if len(data) != source.rows_used:
        notes.append(
            f"The current table gives {len(data)} complete rows where the stored fit used "
            f"{source.rows_used}."
        )

    results: list[CheckResult] = []
    for check in requested:
        try:
            if check == "influence":
                result = _check_influence(
                    source=source, fit=fit, data=data, focus=focus, original=original, notes=notes
                )
            elif check == "robust_se":
                result = _check_robust_se(
                    source=source, fit=fit, data=data, focus=focus, original=original
                )
            elif check == "alternative_family":
                result = _check_alternative_family(
                    source=source, data=data, focus=focus, original=original
                )
            elif check == "bootstrap_ci":
                result = _bootstrap_ci(
                    source=source, data=data, focus=focus, original=original, notes=notes
                )
            else:
                result = _check_adjustment_set(
                    source=source, fit=fit, data=data, focus=focus, original=original, notes=notes
                )
        except ToolError as exc:
            result = _inconclusive(check, original, f"The check could not run: {exc}")
        results.append(result)

    stable = [result for result in results if result.status == "stable"]
    share = len(stable) / len(results) if results else 0.0
    durability = (
        "the conclusion survived every recomputation"
        if len(stable) == len(results)
        else "treat the conclusion as fragile until the checks that failed are resolved"
    )
    verdict = (
        f"{len(stable)} of {len(results)} checks preserved both the direction and the significance "
        f"of '{focus}' (stored p = {original['p_value']:.3g}), so {durability}."
    )

    payload = SensitivityReport(
        source_result_id=result_id,
        checks=results,
        stable_share=round(share, 4),
        verdict=verdict,
        result_id="",
        next="",
        notes=notes,
    )
    stored = store.put(
        payload,
        kind="sensitivity",
        summary=f"sensitivity of {result_id}: {len(stable)}/{len(results)} checks stable",
        rows_total=len(frame),
        rows_used=len(data),
    )
    payload.next = (
        f"Call explain_result('{result_id}') to write the conclusion from the stored result"
        + (
            ", naming the checks that failed."
            if len(stable) != len(results)
            else "; every check held, so the caveats are the design's, not the data's."
        )
    )
    store.state.record(
        "sensitivity",
        status="ok" if share == 1.0 else "caution",
        result_id=stored.result_id,
        note=(
            f"{len(stable)}/{len(results)} checks stable on '{focus}': "
            f"{[result.status for result in results]}"
        ),
    )
    _reseal(stored)
    return payload
