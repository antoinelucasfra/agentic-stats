"""Statistical tools over a real agronomy split-plot trial.

Pure functions: no MCP, no HTTP. The MCP server, the CLI agent and the Shiny
app all call them, so what the tests cover is what ships.

Errors are raised as `ToolError`, whose message is written for an LLM to act on
(available columns, expected ranges) rather than a bare KeyError.
"""

from __future__ import annotations

import keyword
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from pydantic import BaseModel, Field
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.power import FTestAnovaPower

from utils import config
from utils import data as dataset

# Guards against a tool call that would return more than a model can read.
MAX_LEVELS = 50
MAX_POINTS = 300
MAX_BY_LEVELS = 12
MAX_VIF_TERMS = 20


class ToolError(ValueError):
    """A user- or agent-facing error with an actionable message."""


class ColumnSummary(BaseModel):
    name: str
    dtype: str
    missing: int
    unique: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None


class DatasetDescription(BaseModel):
    path: str
    n_rows: int
    n_columns: int
    columns: list[ColumnSummary]
    factors: list[str] = Field(description="Categorical columns available as fixed effects")
    numeric: list[str] = Field(description="Continuous columns available as responses")


class FixedEffect(BaseModel):
    term: str
    estimate: float
    std_error: float
    p_value: float
    ci_low: float
    ci_high: float


class MixedModelResult(BaseModel):
    formula: str
    response: str
    group: str = Field(description="Column defining the random-effect groups")
    n_obs: int
    n_groups: int
    converged: bool
    group_variance: float = Field(description="Between-group variance component")
    residual_variance: float
    icc: float = Field(description="Share of variance between groups, as a proportion")
    fixed_effects: list[FixedEffect]
    notes: list[str] = Field(default_factory=list)


class AnovaResult(BaseModel):
    formula: str
    factor: str
    n_obs: int
    df_between: int
    f_statistic: float
    p_value: float
    omega_squared: float = Field(description="Bias-adjusted effect size")
    partial_eta_squared: float = Field(description="Variance share of this factor alone")
    levene_p_value: float = Field(description="Equal-variance check across levels")
    level_means: dict[str, float]


class MarginalMean(BaseModel):
    factor: str
    level: str
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    n_obs: int


class PairwiseContrast(BaseModel):
    factor: str
    level_a: str
    level_b: str
    difference: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    significant: bool


class MarginalMeansResult(BaseModel):
    formula: str
    model: str = Field(description="'ols' or 'mixedlm', whichever was fitted")
    confidence_level: float
    adjustment: str
    marginal_means: list[MarginalMean]
    pairwise: list[PairwiseContrast]
    notes: list[str] = Field(default_factory=list)


class NormalityTest(BaseModel):
    test: str
    statistic: float
    p_value: float
    verdict: str


class DiagnosticTest(BaseModel):
    name: str
    statistic: float
    p_value: float
    verdict: str


class OutlierRow(BaseModel):
    row: int
    cook_d: float
    leverage: float
    standardized_residual: float
    detail: dict[str, str]


class QQPoint(BaseModel):
    theoretical: float
    standardized: float


class ResidualPoint(BaseModel):
    fitted: float
    residual: float


class AssumptionChecks(BaseModel):
    formula: str
    model: str
    n_obs: int
    normality: NormalityTest
    skew: float
    excess_kurtosis: float
    tests: list[DiagnosticTest]
    vif: dict[str, float]
    max_vif: float | None = None
    outliers: list[OutlierRow]
    qq: list[QQPoint]
    residuals: list[ResidualPoint]
    notes: list[str] = Field(default_factory=list)


class PowerResult(BaseModel):
    response: str
    factor: str
    alpha: float
    target_power: float
    n_obs: int
    n_per_group: dict[str, int]
    level_means: dict[str, float]
    pooled_sd: float
    cohens_f: float = Field(description="Cohen's f for the observed means")
    partial_eta_squared: float
    achieved_power: float = Field(description="Power at the current sample size")
    minimum_detectable_f: float = Field(
        description="Smallest f this design detects at target power"
    )
    required_n_total: int
    required_n_per_group: int
    notes: list[str] = Field(default_factory=list)


class FitParameter(BaseModel):
    name: str
    estimate: float
    std_error: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class CurvePoint(BaseModel):
    dose: float
    predicted: float


class DoseResponseFit(BaseModel):
    group: str | None = None
    n_obs: int
    model: str
    converged: bool
    parameters: list[FitParameter]
    ec50: float | None = None
    ec50_ci_low: float | None = None
    ec50_ci_high: float | None = None
    optimum_dose: float | None = None
    optimum_ci_low: float | None = None
    optimum_ci_high: float | None = None
    r_squared: float | None = None
    aic: float | None = None
    curve: list[CurvePoint]
    notes: list[str] = Field(default_factory=list)


class DoseResponseResult(BaseModel):
    response: str
    dose: str
    fits: list[DoseResponseFit]


@lru_cache(maxsize=8)
def _read(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    # `mtime_ns` and `size` exist to key the cache: editing the file reloads it.
    frame = dataset.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)
    if frame.empty:
        raise ToolError(f"Dataset at {path} is empty.")
    return frame


def _load(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise ToolError(
            f"Dataset not found at {file}. Fetch it with `{config.FETCH_COMMAND}`, "
            "or point AGENTIC_STATS_DATA at a CSV or parquet with the same columns."
        )
    stat = file.stat()
    return _read(str(file.resolve()), stat.st_mtime_ns, stat.st_size)


def dataset_path() -> Path:
    """Resolve the dataset path, overridable with AGENTIC_STATS_DATA.

    With no override this fetches and caches the parquet on first use, so a
    fresh process has a dataset without a build step.
    """
    override = os.environ.get("AGENTIC_STATS_DATA")
    return Path(override) if override else dataset.ensure()


def load_dataset() -> pd.DataFrame:
    """The active dataset: an uploaded frame in this context, else the CSV."""
    return _frame()


def _frame() -> pd.DataFrame:
    override = _ACTIVE.get()
    return override if override is not None else _load(str(dataset_path()))


# A per-call override, so the Shiny app can analyse an uploaded CSV without
# touching the process-wide AGENTIC_STATS_DATA.
_ACTIVE: ContextVar[pd.DataFrame | None] = ContextVar("agentic_stats_frame", default=None)


@contextmanager
def use_frame(frame: pd.DataFrame) -> Iterator[None]:
    """Analyse `frame` for the duration of the block, whatever the env says."""
    token = _ACTIVE.set(frame)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    unknown = [c for c in columns if c not in frame.columns]
    if unknown:
        raise ToolError(f"Unknown column(s): {unknown}. Available columns: {list(frame.columns)}")


def _require_numeric(frame: pd.DataFrame, column: str, role: str) -> None:
    if not pd.api.types.is_numeric_dtype(frame[column]):
        numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        raise ToolError(f"{role} '{column}' is not numeric. Use one of {numeric}.")


def _checked(response: str, *columns: str) -> pd.DataFrame:
    """Load the dataset, rejecting unknown columns and a non-numeric response."""
    frame = _frame()
    _require(frame, [response, *columns])
    _require_numeric(frame, response, "Response")
    return frame


def _summary(frame: pd.DataFrame, column: str) -> ColumnSummary:
    series = frame[column]
    summary = ColumnSummary(
        name=column,
        dtype=str(series.dtype),
        missing=int(series.isna().sum()),
        unique=int(series.nunique()),
    )
    if pd.api.types.is_numeric_dtype(series):
        summary.minimum = float(series.min())
        summary.maximum = float(series.max())
        summary.mean = float(series.mean())
    return summary


@dataclass
class _Fit:
    """A fitted model plus the pieces every downstream calculation needs."""

    formula: str
    rhs: str
    result: Any
    params: pd.Series
    cov: pd.DataFrame
    df_resid: float
    kind: str

    @property
    def beta(self) -> np.ndarray:
        return self.params.to_numpy(dtype=float)

    @property
    def healthy(self) -> bool:
        """A finite likelihood. lbfgs can report converged=True with llf=inf."""
        return bool(np.isfinite(float(self.result.llf)))


# `method="lbfgs"`, statsmodels' default for MixedLM, silently collapses the
# random-effect variance to zero on some designs (group_var 0.0, llf inf, while
# still reporting converged=True). Powell finds the correct optimum and costs
# about 0.1s on this dataset.
MIXED_METHOD = "powell"

_KEYWORDS = frozenset(keyword.kwlist)


def _term(name: str) -> str:
    """A formula-safe column reference.

    patsy parses the formula as Python, so a column called `yield` raises a
    SyntaxError. Quoting it with patsy's Q() keeps real datasets usable; the
    quoted name shows up verbatim in the fitted coefficient names.
    """
    if name.isidentifier() and name not in _KEYWORDS:
        return name
    return 'Q("{}")'.format(name.replace('"', '\\"'))


def _fit(
    frame: pd.DataFrame,
    response: str,
    fixed_effects: list[str],
    group: str | None = None,
) -> _Fit:
    """Fit OLS, or a random-intercept mixed model when `group` is given."""
    formula = f"{_term(response)} ~ " + " + ".join(_term(effect) for effect in fixed_effects)
    rhs = " + ".join(_term(effect) for effect in fixed_effects)
    try:
        if group:
            result = smf.mixedlm(formula, frame, groups=frame[group]).fit(
                reml=True, method=MIXED_METHOD
            )
            params = result.fe_params
        else:
            result = smf.ols(formula, frame).fit()
            params = result.params
    except Exception as exc:  # convergence / singular design / rank deficiency
        raise ToolError(
            f"Model '{formula}' failed to fit ({type(exc).__name__}: {exc}). "
            "Check that groups have enough observations and the terms are not collinear."
        ) from exc

    cov = result.cov_params()
    return _Fit(
        formula=formula,
        rhs=rhs,
        result=result,
        params=params,
        cov=cov.loc[params.index, params.index].astype(float),
        df_resid=float(result.df_resid),
        kind="mixedlm" if group else "ols",
    )


def _design(rhs: str, data: pd.DataFrame, columns: pd.Index) -> np.ndarray:
    """Rebuild `rhs`'s design matrix for `data`, aligned to the fitted columns."""
    try:
        matrix = patsy.dmatrix(rhs, data, return_type="dataframe")
    except Exception as exc:
        raise ToolError(
            f"Could not rebuild the design matrix for prediction ({type(exc).__name__}: {exc})."
        ) from exc
    missing = [name for name in columns if name not in matrix.columns]
    if missing:
        raise ToolError(
            f"Prediction design is missing term(s) {missing}; a factor level may have "
            "disappeared from the data."
        )
    return matrix[list(columns)].to_numpy(dtype=float)


def _predict_vector(fit: _Fit, data: pd.DataFrame) -> np.ndarray:
    """Mean design row for `data`: the averaged-prediction basis."""
    return _design(fit.rhs, data, fit.params.index).mean(axis=0)


def _estimate(fit: _Fit, basis: np.ndarray) -> tuple[float, float]:
    """Point estimate and standard error of `basis @ beta`."""
    estimate = float(basis @ fit.beta)
    variance = float(basis @ fit.cov.to_numpy(dtype=float) @ basis)
    return estimate, float(np.sqrt(max(variance, 0.0)))


def _t_critical(confidence_level: float, df_resid: float) -> float:
    return float(stats.t.ppf(0.5 + confidence_level / 2, df_resid))


@dataclass
class _OneWay:
    """A one-factor decomposition, computed directly so the pieces stay legible."""

    factor: str
    n: int
    k: int
    means: pd.Series
    counts: pd.Series
    ss_between: float
    ss_within: float
    df_between: int
    df_within: int
    f: float
    p: float
    ms_within: float


def _one_way(frame: pd.DataFrame, response: str, factor: str) -> _OneWay:
    grouped = frame.groupby(factor, observed=True)[response]
    means = grouped.mean()
    counts = grouped.size()
    grand = float(frame[response].mean())
    residuals = frame[response] - grouped.transform("mean")
    ss_between = float((counts * (means - grand) ** 2).sum())
    ss_within = float((residuals**2).sum())
    n, k = len(frame), int(means.size)
    if k < 2:
        raise ToolError(f"Factor '{factor}' has fewer than 2 distinct levels: nothing to compare.")
    df_between, df_within = k - 1, n - k
    if df_within <= 0:
        raise ToolError(f"Factor '{factor}' has {k} levels for {n} rows: no residual variance.")
    f_stat = (ss_between / df_between) / (ss_within / df_within)
    return _OneWay(
        factor=factor,
        n=n,
        k=k,
        means=means,
        counts=counts,
        ss_between=ss_between,
        ss_within=ss_within,
        df_between=df_between,
        df_within=df_within,
        f=float(f_stat),
        p=float(stats.f.sf(f_stat, df_between, df_within)),
        ms_within=ss_within / df_within,
    )


def _order_terms(params: pd.Index, fixed_effects: list[str]) -> list[str]:
    """Present coefficients in the order the caller asked for, not the design's.

    statsmodels orders design columns itself, which puts categorical dummy terms
    ahead of numeric ones. An agent reading the result should see `nitrogen` where it
    asked for `nitrogen`. Anything unmatched is appended so nothing is dropped.
    """
    ordered = [name for name in params if name == "Intercept"]
    for effect in fixed_effects:
        # `_term` is the identity for ordinary names; dedupe so no term lands twice.
        for base in dict.fromkeys((effect, _term(effect))):
            prefix = f"{base}[T."
            ordered += [name for name in params if name == base or name.startswith(prefix)]
    unique = list(dict.fromkeys(ordered))
    return unique + [name for name in params if name not in unique]


def describe_dataset() -> DatasetDescription:
    """Profile the experiment dataset: shape, columns, missing values, ranges.

    Call this first when a question is vague: it tells you which columns exist,
    which are categorical factors, and which continuous columns can be used as
    a model response.
    """
    frame = _frame()
    return DatasetDescription(
        path=str(dataset_path()),
        n_rows=len(frame),
        n_columns=len(frame.columns),
        columns=[_summary(frame, c) for c in frame.columns],
        factors=[c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])],
        numeric=[c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])],
    )


def fit_mixed_model(
    response: str,
    fixed_effects: list[str],
    group: str = "block",
) -> MixedModelResult:
    """Fit a linear mixed model with a random intercept for `group`.

    Use this when repeated measurements are nested in a higher-level unit, e.g.
    subplots grouped within the same field block. `response` must be numeric,
    `fixed_effects` may mix categorical factors and numeric covariates, and
    `group` is the column defining the random-effect groups (a random intercept
    per level). Returns fixed-effect estimates with 95% confidence intervals,
    the between-group and residual variances, and the ICC.

    Prefer this over plain OLS whenever observations repeat within `group`: it
    separates block-to-block variability from the effect you actually care about.
    """
    frame = _checked(response, *fixed_effects, group)
    if not fixed_effects:
        raise ToolError("Provide at least one fixed effect, e.g. ['variety', 'nitrogen'].")

    fit = _fit(frame, response, fixed_effects, group)
    result = fit.result
    notes: list[str] = []
    if not fit.healthy:
        notes.append(
            "The optimizer returned a non-finite likelihood, so this fit is unreliable; "
            "try fewer fixed effects or a different grouping column."
        )
    if not result.converged:
        notes.append("Model did not converge: treat inference as indicative only.")

    group_variance = float(result.cov_re.iloc[0, 0])
    residual_variance = float(result.scale)
    total = group_variance + residual_variance
    if total > 0 and group_variance / total < 1e-6:
        notes.append(
            f"Between-{group} variance is at its lower bound, so a random intercept "
            "adds nothing here; plain OLS would give the same fixed effects."
        )

    conf = result.conf_int()
    effects = [
        FixedEffect(
            term=term,
            estimate=float(fit.params[term]),
            std_error=float(np.sqrt(fit.cov.loc[term, term])),
            p_value=float(result.pvalues[term]),
            ci_low=float(conf.loc[term, 0]),
            ci_high=float(conf.loc[term, 1]),
        )
        for term in _order_terms(fit.params.index, fixed_effects)
    ]

    return MixedModelResult(
        formula=fit.formula,
        response=response,
        group=group,
        n_obs=int(result.nobs),
        n_groups=int(frame[group].nunique()),
        converged=bool(result.converged) and fit.healthy,
        group_variance=group_variance,
        residual_variance=residual_variance,
        icc=round(group_variance / total, 4) if total > 0 else 0.0,
        fixed_effects=effects,
        notes=notes,
    )


def anova_effect(response: str, factor: str) -> AnovaResult:
    """Test whether `factor` shifts the mean of `response`, with effect size.

    Use for a one-factor comparison across levels (variety, nitrogen rate,
    block). Reports the F-test p-value, omega-squared (the share of variance
    attributable to `factor`, corrected for sample size), partial eta-squared,
    and Levene's test for equal variances across levels. Pair it with
    marginal_means for the per-level estimates, or fit_mixed_model when
    replicates are nested within blocks.
    """
    frame = _checked(response, factor)

    one_way = _one_way(frame, response, factor)
    ss_total = one_way.ss_between + one_way.ss_within
    omega_squared = (one_way.ss_between - one_way.df_between * one_way.ms_within) / (
        ss_total + one_way.ms_within
    )

    groups = [
        frame.loc[frame[factor] == level, response].to_numpy(dtype=float)
        for level in one_way.means.index
    ]
    levene = stats.levene(*groups)

    return AnovaResult(
        formula=f"{response} ~ C({factor})",
        factor=factor,
        n_obs=one_way.n,
        df_between=one_way.df_between,
        f_statistic=one_way.f,
        p_value=one_way.p,
        omega_squared=round(omega_squared, 4),
        partial_eta_squared=round(one_way.ss_between / (one_way.ss_between + one_way.ss_within), 4),
        levene_p_value=round(float(levene.pvalue), 4),
        level_means={str(level): round(float(value), 4) for level, value in one_way.means.items()},
    )


def marginal_means(
    response: str,
    factors: list[str],
    covariates: list[str] | None = None,
    group: str | None = None,
    confidence_level: float = 0.95,
    pairwise: bool = True,
    adjust: str = "tukey",
) -> MarginalMeansResult:
    """Average predicted `response` at each level of each factor, with contrasts.

    The estimate for a level is the mean predicted value over the observed rows
    with that factor set to the level, so covariates are averaged over their
    real distribution instead of held at an arbitrary reference value. Use this
    after anova_effect, which tells you a factor matters but not by how much.

    `group` fits a random intercept (as in fit_mixed_model) when rows repeat
    within a unit such as a block. `pairwise=True` adds level-vs-level
    differences with a multiplicity adjustment; `adjust` accepts 'tukey' (a true
    Tukey correction for a one-factor design) or any method from
    statsmodels.stats.multitest, e.g. 'holm', 'bonferroni', 'fdr_bh'.
    """
    covariates = covariates or []
    fixed_effects = [*factors, *covariates]
    if not factors:
        raise ToolError("Provide at least one factor, e.g. ['variety'].")
    frame = _checked(response, *fixed_effects, *([group] if group else []))
    for factor in factors:
        if not pd.api.types.is_numeric_dtype(frame[factor]):
            levels = frame[factor].nunique()
            if levels > MAX_LEVELS:
                raise ToolError(
                    f"Factor '{factor}' has {levels} levels (limit {MAX_LEVELS}). "
                    "Pass it as a covariate instead, or group the levels first."
                )

    fit = _fit(frame, response, fixed_effects, group)
    notes: list[str] = []
    if fit.kind == "mixedlm" and not fit.result.converged:
        notes.append("The mixed model did not converge: treat these estimates as indicative.")
    if not fit.healthy:
        notes.append(
            "The optimizer returned a non-finite likelihood; these estimates are unreliable."
        )
    if adjust not in _ADJUSTMENTS:
        raise ToolError(f"Unknown adjustment '{adjust}'. Use one of {sorted(_ADJUSTMENTS)}.")

    estimates: list[MarginalMean] = []
    bases: dict[tuple[str, str], np.ndarray] = {}
    order: dict[str, list[str]] = {}
    for factor in factors:
        levels = [str(level) for level in sorted(frame[factor].dropna().unique())]
        order[factor] = levels
        for level in levels:
            data = frame.copy()
            data[factor] = _pin(frame[factor], level)
            basis = _predict_vector(fit, data)
            estimate, std_error = _estimate(fit, basis)
            half_width = _t_critical(confidence_level, fit.df_resid) * std_error
            bases[(factor, level)] = basis
            estimates.append(
                MarginalMean(
                    factor=factor,
                    level=level,
                    estimate=round(estimate, 4),
                    std_error=round(std_error, 4),
                    ci_low=round(estimate - half_width, 4),
                    ci_high=round(estimate + half_width, 4),
                    n_obs=int((frame[factor].astype(str) == level).sum()),
                )
            )

    contrasts: list[PairwiseContrast] = []
    if pairwise:
        contrasts = _pairwise(
            fit, response, factors, order, bases, confidence_level, adjust, notes, frame
        )

    return MarginalMeansResult(
        formula=fit.formula,
        model=fit.kind,
        confidence_level=confidence_level,
        adjustment=adjust,
        marginal_means=estimates,
        pairwise=contrasts,
        notes=notes,
    )


def _pin(series: pd.Series, level: str) -> pd.Series:
    """Replace a column with one repeated level, keeping its original encoding."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(np.full(len(series), float(level)), index=series.index, name=series.name)
    categories = sorted(str(value) for value in series.dropna().unique())
    return pd.Series(
        pd.Categorical([level] * len(series), categories=categories),
        index=series.index,
        name=series.name,
    )


def _pairwise(
    fit: _Fit,
    response: str,
    factors: list[str],
    order: dict[str, list[str]],
    bases: dict[tuple[str, str], np.ndarray],
    confidence_level: float,
    adjust: str,
    notes: list[str],
    frame: pd.DataFrame,
) -> list[PairwiseContrast]:
    """Difference between every pair of levels, adjusted for multiplicity."""
    t_crit = _t_critical(confidence_level, fit.df_resid)
    contrasts: list[PairwiseContrast] = []

    for factor in factors:
        levels = order[factor]
        pairs = [
            (levels[i], levels[j]) for i in range(len(levels)) for j in range(i + 1, len(levels))
        ]
        if not pairs:
            continue
        raw = [
            _contrast(fit, factor, level_a, level_b, bases, t_crit) for level_a, level_b in pairs
        ]
        p_values = np.array([item["p_value"] for item in raw], dtype=float)
        method, adjusted = _adjust(
            p_values, adjust, frame, factor, levels, response, confidence_level, notes
        )
        for item, p_adj in zip(raw, adjusted, strict=True):
            item["p_adjusted"] = float(p_adj)
            item["significant"] = bool(p_adj < 1 - confidence_level)
            contrasts.append(PairwiseContrast(factor=factor, **item))
        if method != adjust:
            notes.append(f"Reported adjustment is '{method}', not '{adjust}'.")

    return contrasts


def _adjust(
    p_values: np.ndarray,
    adjust: str,
    frame: pd.DataFrame,
    factor: str,
    levels: list[str],
    response: str,
    confidence_level: float,
    notes: list[str],
) -> tuple[str, np.ndarray]:
    """Multiplicity adjustment, using a true Tukey test where one applies."""
    if adjust == "tukey":
        if pd.api.types.is_numeric_dtype(frame[factor]):
            notes.append("Tukey needs a categorical factor; used Holm instead.")
            adjust = "holm"
        else:
            tukey = pairwise_tukeyhsd(
                frame[response],
                frame[factor].astype(str),
                alpha=1 - confidence_level,
            )
            if list(tukey.groupsunique) == levels and tukey.pvalues.size == p_values.size:
                notes.append(
                    "Pairwise p-values use Tukey's studentized-range correction; the "
                    "confidence intervals come from the model's delta method and differ "
                    "slightly."
                )
                return "tukey", np.asarray(tukey.pvalues, dtype=float)
            notes.append("Tukey adjustment did not line up with the model; used Holm instead.")
            adjust = "holm"

    _, adjusted, _, _ = multipletests(p_values, method=adjust)
    return adjust, np.asarray(adjusted, dtype=float)


def _contrast(
    fit: _Fit,
    factor: str,
    level_a: str,
    level_b: str,
    bases: dict[tuple[str, str], np.ndarray],
    t_crit: float,
) -> dict[str, Any]:
    basis = bases[(factor, level_a)] - bases[(factor, level_b)]
    difference, std_error = _estimate(fit, basis)
    statistic = difference / std_error if std_error > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(statistic), fit.df_resid))
    return {
        "level_a": level_a,
        "level_b": level_b,
        "difference": round(difference, 4),
        "std_error": round(std_error, 4),
        "ci_low": round(difference - t_crit * std_error, 4),
        "ci_high": round(difference + t_crit * std_error, 4),
        "p_value": p_value,
        "p_adjusted": p_value,
        "significant": False,
    }


_ADJUSTMENTS = {"tukey", "holm", "bonferroni", "sidak", "fdr_bh", "hommel"}


def check_assumptions(
    response: str,
    fixed_effects: list[str],
    group: str | None = None,
    top_outliers: int = 5,
) -> AssumptionChecks:
    """Test the assumptions behind an OLS or mixed-model fit.

    Reports residual normality (Shapiro-Wilk, or D'Agostino-Pearson above 5000
    rows), skew and kurtosis, Breusch-Pagan heteroscedasticity, Levene's test
    per categorical factor, variance inflation factors for collinearity, the
    most influential rows by Cook's distance, and the arrays for a QQ or
    residual plot. Use it before trusting a p-value from anova_effect or
    fit_mixed_model.

    Collinearity and influence come from an OLS fit of the same fixed effects,
    because they describe the design. Residual-based tests use whichever model
    you asked for.
    """
    frame = _checked(response, *fixed_effects, *([group] if group else []))

    fit = _fit(frame, response, fixed_effects, group)
    design = _fit(frame, response, fixed_effects, None)
    residuals = np.asarray(fit.result.resid, dtype=float)
    fitted_values = np.asarray(fit.result.fittedvalues, dtype=float)
    notes: list[str] = []
    if not fit.healthy:
        notes.append(
            "The optimizer returned a non-finite likelihood; the residual tests may be meaningless."
        )

    normality = _normality(residuals)
    if normality.p_value < 0.05:
        notes.append(
            "Residuals are not consistent with normality, so p-values from this model "
            "are approximate; check the QQ plot for the shape of the deviation."
        )

    tests: list[DiagnosticTest] = []
    exog = np.asarray(design.result.model.exog, dtype=float)
    bp = het_breuschpagan(residuals, exog)
    tests.append(
        DiagnosticTest(
            name="breusch_pagan",
            statistic=round(float(bp[0]), 4),
            p_value=round(float(bp[1]), 4),
            verdict=(
                "residual variance looks constant"
                if float(bp[1]) >= 0.05
                else "residual variance changes with the fitted value"
            ),
        )
    )
    for term in fixed_effects:
        if pd.api.types.is_numeric_dtype(frame[term]):
            continue
        grouped = [
            residuals[frame[term].astype(str).to_numpy() == str(level)]
            for level in sorted(frame[term].dropna().unique())
        ]
        if len(grouped) < 2 or any(len(group) < 2 for group in grouped):
            continue
        levene = stats.levene(*grouped)
        tests.append(
            DiagnosticTest(
                name=f"levene:{term}",
                statistic=round(float(levene.statistic), 4),
                p_value=round(float(levene.pvalue), 4),
                verdict=(
                    f"spread of residuals is similar across {term} levels"
                    if float(levene.pvalue) >= 0.05
                    else f"spread of residuals differs across {term} levels"
                ),
            )
        )

    vif = _vif(exog, list(design.params.index), notes)
    max_vif = max(vif.values()) if vif else None
    if max_vif is not None and max_vif > 5:
        notes.append(
            f"Largest variance inflation factor is {max_vif:.1f}: some terms are "
            "correlated and their individual estimates are unstable."
        )

    influence = OLSInfluence(design.result)
    cook_d, _ = influence.cooks_distance
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    standardized = np.asarray(influence.resid_studentized_internal, dtype=float)
    labels = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])][:3]
    top = np.argsort(-np.asarray(cook_d, dtype=float))[: max(top_outliers, 0)]
    outliers = [
        OutlierRow(
            row=int(index),
            cook_d=round(float(cook_d[index]), 6),
            leverage=round(float(leverage[index]), 4),
            standardized_residual=round(float(standardized[index]), 4),
            detail={label: str(frame.at[frame.index[index], label]) for label in labels},
        )
        for index in top
    ]
    if outliers and outliers[0].cook_d > 1:
        notes.append(
            "At least one row has a Cook's distance above 1, which moves the fit "
            "noticeably; refit without it before reporting the effect."
        )

    if group:
        notes.append(
            f"Collinearity, heteroscedasticity and influence are computed on an OLS fit "
            f"of the same fixed effects; the random intercept for {group} absorbs "
            "between-group variation from the residuals."
        )

    return AssumptionChecks(
        formula=fit.formula,
        model=fit.kind,
        n_obs=len(frame),
        normality=normality,
        skew=round(float(stats.skew(residuals)), 4),
        excess_kurtosis=round(float(stats.kurtosis(residuals, fisher=True)), 4),
        tests=tests,
        vif=vif,
        max_vif=round(max_vif, 4) if max_vif is not None else None,
        outliers=outliers,
        qq=_qq(residuals),
        residuals=_residual_points(fitted_values, residuals),
        notes=notes,
    )


def _normality(residuals: np.ndarray) -> NormalityTest:
    if 3 <= residuals.size <= 5000:
        statistic, p_value = stats.shapiro(residuals)
        name = "shapiro_wilk"
    else:
        statistic, p_value = stats.normaltest(residuals)
        name = "dagostino_pearson"
    return NormalityTest(
        test=name,
        statistic=round(float(statistic), 4),
        p_value=round(float(p_value), 4),
        verdict=(
            "consistent with a normal residual distribution"
            if float(p_value) >= 0.05
            else "departs from normality"
        ),
    )


def _vif(exog: np.ndarray, names: list[str], notes: list[str]) -> dict[str, float]:
    """Variance inflation factors for the non-constant design terms."""
    terms = [(i, name) for i, name in enumerate(names) if name != "Intercept"]
    if len(terms) < 2:
        notes.append("Only one predictor, so collinearity does not apply.")
        return {}
    if len(terms) > MAX_VIF_TERMS:
        notes.append(f"VIF skipped: {len(terms)} terms exceeds the {MAX_VIF_TERMS} limit.")
        return {}
    return {name: round(float(variance_inflation_factor(exog, index)), 4) for index, name in terms}


def _sample(count: int) -> np.ndarray:
    if count <= MAX_POINTS:
        return np.arange(count)
    return np.linspace(0, count - 1, MAX_POINTS).astype(int)


def _qq(residuals: np.ndarray) -> list[QQPoint]:
    standardized = (residuals - residuals.mean()) / residuals.std(ddof=1)
    ordered = np.sort(standardized)
    theoretical = stats.norm.ppf((np.arange(1, ordered.size + 1) - 0.5) / ordered.size)
    picks = _sample(ordered.size)
    return [
        QQPoint(
            theoretical=round(float(theoretical[i]), 4), standardized=round(float(ordered[i]), 4)
        )
        for i in picks
    ]


def _residual_points(fitted_values: np.ndarray, residuals: np.ndarray) -> list[ResidualPoint]:
    order = np.argsort(fitted_values)
    picks = order[_sample(order.size)]
    return [
        ResidualPoint(
            fitted=round(float(fitted_values[i]), 4), residual=round(float(residuals[i]), 4)
        )
        for i in picks
    ]


def power_analysis(
    response: str,
    factor: str,
    alpha: float = 0.05,
    target_power: float = 0.8,
    target_effect: float | None = None,
) -> PowerResult:
    """Report the power of this design, and the sample size a target effect needs.

    Cohen's f comes from the observed level means against the pooled within-level
    standard deviation. Returns the power the current sample size achieves, the
    smallest effect it can detect at `target_power`, and the rows per level
    needed to detect the observed effect (or `target_effect`, in response units)
    at that power. Use it to answer "should we run more blocks?".
    """
    frame = _checked(response, factor)

    one_way = _one_way(frame, response, factor)
    pooled_sd = float(np.sqrt(one_way.ms_within))
    if pooled_sd <= 0:
        raise ToolError(f"Factor '{factor}' has no within-level variance: f is undefined.")

    cohens_f = float(np.sqrt(one_way.ss_between / (one_way.n * one_way.ms_within)))
    partial_eta_squared = float(one_way.ss_between / (one_way.ss_between + one_way.ss_within))

    power_model = FTestAnovaPower()
    achieved = float(
        power_model.power(effect_size=cohens_f, nobs=one_way.n, alpha=alpha, k_groups=one_way.k)
    )
    detectable = float(
        power_model.solve_power(
            effect_size=None,
            nobs=one_way.n,
            alpha=alpha,
            power=target_power,
            k_groups=one_way.k,
        )
    )

    notes: list[str] = []
    effect_for_size = cohens_f
    if target_effect is not None:
        effect_for_size = float(target_effect) / pooled_sd
        notes.append(
            f"Sample size targets {target_effect} response units on the pooled SD of "
            f"{pooled_sd:.3f}, which is Cohen's f = {effect_for_size:.3f}."
        )
    required_total = int(
        np.ceil(
            power_model.solve_power(
                effect_size=effect_for_size,
                nobs=None,
                alpha=alpha,
                power=target_power,
                k_groups=one_way.k,
            )
        )
    )
    required_per_group = int(np.ceil(required_total / one_way.k))
    if achieved < target_power:
        notes.append(
            f"The current design only reaches power {achieved:.2f}; "
            f"{required_per_group} rows per level would reach {target_power:.2f}."
        )

    return PowerResult(
        response=response,
        factor=factor,
        alpha=alpha,
        target_power=target_power,
        n_obs=one_way.n,
        n_per_group={str(level): int(count) for level, count in one_way.counts.items()},
        level_means={str(level): round(float(value), 4) for level, value in one_way.means.items()},
        pooled_sd=round(pooled_sd, 4),
        cohens_f=round(cohens_f, 4),
        partial_eta_squared=round(partial_eta_squared, 4),
        achieved_power=round(achieved, 4),
        minimum_detectable_f=round(detectable, 4),
        required_n_total=required_total,
        required_n_per_group=required_per_group,
        notes=notes,
    )


def dose_response(
    response: str,
    dose: str,
    model: str = "4pl",
    by: str | None = None,
    confidence_level: float = 0.95,
) -> DoseResponseResult:
    """Fit a dose-response curve and report the half-effective dose.

    `model='4pl'` fits a four-parameter logistic (bottom, top, EC50, Hill), which
    tolerates a zero dose. `model='quadratic'` fits a parabola and reports the
    peak, for the case where the response turns over within the tested range.
    With `by` the fit repeats per level of that factor, e.g. one curve per
    variety. Returns parameter estimates, EC50 or peak dose with a
    confidence interval, R-squared, AIC and the predicted curve.
    """
    frame = _checked(response, dose, *([by] if by else []))
    _require_numeric(frame, dose, "Dose")
    if model not in {"4pl", "quadratic"}:
        raise ToolError(f"Unknown model '{model}'. Use '4pl' or 'quadratic'.")
    if (frame[dose] < 0).any():
        raise ToolError(f"Dose '{dose}' has negative values; dose must be zero or positive.")

    if by:
        levels = sorted(str(level) for level in frame[by].dropna().unique())
        if len(levels) > MAX_BY_LEVELS:
            raise ToolError(
                f"'by' column '{by}' has {len(levels)} levels (limit {MAX_BY_LEVELS}). "
                "Fit one group at a time instead."
            )
        fits = [
            _dose_fit(
                frame[frame[by].astype(str) == level],
                response,
                dose,
                model,
                confidence_level,
                level,
            )
            for level in levels
        ]
    else:
        fits = [_dose_fit(frame, response, dose, model, confidence_level, None)]

    return DoseResponseResult(response=response, dose=dose, fits=fits)


def _failed_fit(group: str | None, model: str, n_obs: int, note: str) -> DoseResponseFit:
    """A non-converged fit: no parameters, no curve, just the reason."""
    return DoseResponseFit(
        group=group,
        n_obs=n_obs,
        model=model,
        converged=False,
        parameters=[],
        curve=[],
        notes=[note],
    )


def _curve_points(grid: np.ndarray, predicted: np.ndarray) -> list[CurvePoint]:
    return [
        CurvePoint(dose=round(float(d), 4), predicted=round(float(p), 4))
        for d, p in zip(grid, predicted, strict=True)
    ]


def _dose_fit(
    frame: pd.DataFrame,
    response: str,
    dose: str,
    model: str,
    confidence_level: float,
    group: str | None,
) -> DoseResponseFit:
    x = frame[dose].to_numpy(dtype=float)
    y = frame[response].to_numpy(dtype=float)
    grid = np.linspace(float(x.min()), float(x.max()), 50)
    if model == "quadratic":
        return _quadratic_fit(x, y, grid, confidence_level, group, response, dose)
    return _four_pl_fit(x, y, grid, confidence_level, group, response, dose)


def _quadratic_fit(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    confidence_level: float,
    group: str | None,
    response: str,
    dose: str,
) -> DoseResponseFit:
    design = pd.DataFrame({dose: x, response: y})
    try:
        fitted = smf.ols(f"{response} ~ {dose} + I({dose} ** 2)", design).fit()
    except Exception as exc:
        return _failed_fit(
            group, "quadratic", int(x.size), f"Quadratic fit failed: {type(exc).__name__}: {exc}"
        )

    params = fitted.params
    linear = float(params[dose])
    quadratic = float(params[f"I({dose} ** 2)"])
    notes: list[str] = []
    optimum = ci_low = ci_high = None
    if quadratic < 0:
        optimum = -linear / (2 * quadratic)
        gradient = np.zeros(len(params))
        gradient[list(params.index).index(dose)] = -1 / (2 * quadratic)
        gradient[list(params.index).index(f"I({dose} ** 2)")] = linear / (2 * quadratic**2)
        variance = float(gradient @ fitted.cov_params().to_numpy() @ gradient)
        half_width = _t_critical(confidence_level, fitted.df_resid) * float(np.sqrt(variance))
        ci_low, ci_high = optimum - half_width, optimum + half_width
        if not float(x.min()) <= optimum <= float(x.max()):
            notes.append(
                f"The peak ({optimum:.2f}) falls outside the tested dose range "
                f"({float(x.min()):.2f} to {float(x.max()):.2f}); treat it as an extrapolation."
            )
    else:
        notes.append(
            "The quadratic term is not negative, so the curve has no peak in this range: "
            "the response rises with dose."
        )

    predicted = _poly_predict(params, grid, dose)
    return DoseResponseFit(
        group=group,
        n_obs=int(x.size),
        model="quadratic",
        converged=True,
        parameters=[
            FitParameter(
                name=str(name),
                estimate=round(float(value), 6),
                std_error=round(float(np.sqrt(fitted.cov_params().loc[name, name])), 6),
                ci_low=round(float(fitted.conf_int().loc[name, 0]), 6),
                ci_high=round(float(fitted.conf_int().loc[name, 1]), 6),
            )
            for name, value in params.items()
        ],
        ec50=None,
        optimum_dose=round(optimum, 4) if optimum is not None else None,
        optimum_ci_low=round(ci_low, 4) if ci_low is not None else None,
        optimum_ci_high=round(ci_high, 4) if ci_high is not None else None,
        r_squared=round(float(fitted.rsquared), 4),
        aic=round(float(fitted.aic), 2),
        curve=_curve_points(grid, predicted),
        notes=notes,
    )


def _poly_predict(params: pd.Series, grid: np.ndarray, dose: str) -> np.ndarray:
    return (
        float(params["Intercept"])
        + float(params[dose]) * grid
        + float(params[f"I({dose} ** 2)"]) * grid**2
    )


def _four_pl(x: np.ndarray, bottom: float, top: float, ec50: float, hill: float) -> np.ndarray:
    """4PL in the algebraically safe form: bottom at dose 0, top as dose grows."""
    ec50 = max(float(ec50), 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = np.power(ec50 / np.maximum(x, 1e-9), hill)
        fraction = np.where(x > 0, 1.0 / (1.0 + ratio), 0.0)
    return bottom + (top - bottom) * fraction


def _finite(value: float) -> float | None:
    """JSON cannot carry inf or nan, so a non-finite estimate becomes null."""
    return float(value) if np.isfinite(value) else None


def _four_pl_fit(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    confidence_level: float,
    group: str | None,
    response: str,
    dose: str,
) -> DoseResponseFit:
    spread = float(np.ptp(y))
    if spread <= 0:
        return _failed_fit(
            group,
            "4pl",
            int(x.size),
            "Response has no variation in this group, so no curve can be fitted.",
        )

    low, high = float(y.min()) - 10 * spread, float(y.max()) + 10 * spread
    positive = x[x > 0]
    p0 = [
        float(y.min()),
        float(y.max()),
        float(np.median(positive)) if positive.size else float(max(x.max(), 1.0)),
        1.0,
    ]
    bounds = (
        [low, low, max(float(x.max()) / 1000, 1e-6), 0.1],
        [high, high, float(x.max()) * 10 + 1e-6, 10.0],
    )
    try:
        popt, pcov = curve_fit(_four_pl, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception as exc:
        return _failed_fit(
            group,
            "4pl",
            int(x.size),
            f"Four-parameter logistic fit did not converge ({type(exc).__name__}: {exc}). "
            "Try model='quadratic', or check that the dose range shows a clear plateau.",
        )

    names = ["bottom", "top", "ec50", "hill"]
    errors = np.sqrt(np.clip(np.diag(pcov), 0, None))
    df_resid = max(int(x.size) - len(popt), 1)
    t_crit = _t_critical(confidence_level, df_resid)
    predicted = _four_pl(grid, *popt)
    residual_sum = float(((y - _four_pl(x, *popt)) ** 2).sum())
    total_sum = float(((y - y.mean()) ** 2).sum())
    notes: list[str] = []
    if not 0.2 < float(popt[3]) < 5:
        notes.append(
            f"The Hill slope is {float(popt[3]):.2f}, near a bound of the search; "
            "the curve may be poorly identified by this dose range."
        )

    def half_width(index: int) -> float | None:
        error = _finite(float(errors[index]))
        return round(t_crit * error, 6) if error is not None else None

    ec50 = float(popt[2])
    ec50_half = half_width(2)
    return DoseResponseFit(
        group=group,
        n_obs=int(x.size),
        model="4pl",
        converged=True,
        parameters=[
            FitParameter(
                name=name,
                estimate=round(float(value), 6),
                std_error=_finite(float(error)),
                ci_low=_finite(float(value) - t_crit * float(error)),
                ci_high=_finite(float(value) + t_crit * float(error)),
            )
            for name, value, error in zip(names, popt, errors, strict=True)
        ],
        ec50=round(ec50, 4),
        ec50_ci_low=round(ec50 - ec50_half, 4) if ec50_half is not None else None,
        ec50_ci_high=round(ec50 + ec50_half, 4) if ec50_half is not None else None,
        r_squared=round(1 - residual_sum / total_sum, 4) if total_sum > 0 else None,
        aic=round(float(x.size) * np.log(residual_sum / x.size) + 2 * len(popt), 2),
        curve=_curve_points(grid, predicted),
        notes=notes,
    )
