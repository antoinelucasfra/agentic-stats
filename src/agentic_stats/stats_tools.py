"""Statistical tools over an R&D design-of-experiments dataset.

Pure functions: no MCP, no HTTP. The MCP server and the FastAPI app both call
them, so what the tests cover is what ships.

Errors are raised as `ToolError`, whose message is written for an LLM to act on
(available columns, expected ranges) rather than a bare KeyError.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pydantic import BaseModel, Field

from agentic_stats.data import DATA_PATH


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
    converged: bool
    group_variance: float = Field(description="Between-group variance component")
    residual_variance: float
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
    level_means: dict[str, float]


@lru_cache(maxsize=4)
def _load(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ToolError(f"Dataset at {path} is empty.")
    return frame


def dataset_path() -> Path:
    """Resolve the dataset path, overridable with AGENTIC_STATS_DATA."""
    return Path(os.environ.get("AGENTIC_STATS_DATA", DATA_PATH))


def _frame() -> pd.DataFrame:
    return _load(str(dataset_path()))


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    unknown = [c for c in columns if c not in frame.columns]
    if unknown:
        raise ToolError(f"Unknown column(s): {unknown}. Available columns: {list(frame.columns)}")


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
    group: str = "batch",
) -> MixedModelResult:
    """Fit a linear mixed model with a random intercept for `group`.

    Use this when repeated measurements are nested in a higher-level unit, e.g.
    assay readings replicated within pilot batches. `response` must be numeric,
    `fixed_effects` may mix categorical factors and numeric covariates, and
    `group` is the column defining the random-effect groups (a random intercept
    per level). Returns fixed-effect estimates with 95% confidence intervals
    plus the between-group and residual variances.

    Prefer this over plain OLS whenever observations repeat within `group`: it
    separates batch-to-batch variability from the effect you actually care about.
    """
    frame = _frame()
    _require(frame, [response, *fixed_effects, group])
    if not pd.api.types.is_numeric_dtype(frame[response]):
        raise ToolError(
            f"Response '{response}' is not numeric. Use one of {describe_dataset().numeric}."
        )

    formula = f"{response} ~ " + " + ".join(fixed_effects)
    notes: list[str] = []
    try:
        fitted = smf.mixedlm(formula, frame, groups=frame[group]).fit(reml=True, method="lbfgs")
    except Exception as exc:  # convergence / singular design / rank deficiency
        raise ToolError(
            f"Mixed model '{formula}' failed to fit ({type(exc).__name__}: {exc}). "
            "Check that groups have enough observations and fixed effects are not collinear."
        ) from exc

    conf = fitted.conf_int()
    effects = [
        FixedEffect(
            term=term,
            estimate=float(fitted.params[term]),
            std_error=float(fitted.bse[term]),
            p_value=float(fitted.pvalues[term]),
            ci_low=float(conf.loc[term, 0]),
            ci_high=float(conf.loc[term, 1]),
        )
        for term in fitted.params.index
    ]
    if not fitted.converged:
        notes.append("Model did not converge: treat inference as indicative only.")

    return MixedModelResult(
        formula=formula,
        response=response,
        group=group,
        n_obs=int(fitted.nobs),
        converged=bool(fitted.converged),
        group_variance=float(fitted.cov_re.iloc[0, 0]),
        residual_variance=float(fitted.scale),
        fixed_effects=effects,
        notes=notes,
    )


def anova_effect(response: str, factor: str) -> AnovaResult:
    """Test whether `factor` shifts the mean of `response`, with effect size.

    Use for a one-factor comparison across levels (formulation, dose band,
    operator). Reports the F-test p-value and omega-squared, which is the
    share of variance in `response` attributable to `factor` after correcting
    for sample size. Pair it with fit_mixed_model when replicates are nested
    within batches.
    """
    frame = _frame()
    _require(frame, [response, factor])
    if not pd.api.types.is_numeric_dtype(frame[response]):
        raise ToolError(
            f"Response '{response}' is not numeric. Use one of {describe_dataset().numeric}."
        )
    if frame[factor].nunique() < 2:
        raise ToolError(f"Factor '{factor}' has fewer than 2 distinct levels: nothing to compare.")

    formula = f"{response} ~ C({factor})"
    fitted = smf.ols(formula, frame).fit()
    table = sm.stats.anova_lm(fitted, typ=2)
    row = table.loc[f"C({factor})"]

    ss_between = float(row["sum_sq"])
    ss_error = float(table["sum_sq"].sum() - ss_between)
    df_between = int(row["df"])
    df_error = int(fitted.df_resid)
    ms_error = ss_error / df_error
    n = len(frame)
    omega_squared = (ss_between - df_between * ms_error) / (ss_between + ss_error + ms_error)

    return AnovaResult(
        formula=formula,
        factor=factor,
        n_obs=n,
        df_between=df_between,
        f_statistic=float(row["F"]),
        p_value=float(row["PR(>F)"]),
        omega_squared=round(omega_squared, 4),
        level_means={
            str(level): round(float(value), 4)
            for level, value in frame.groupby(factor)[response].mean().items()
        },
    )
