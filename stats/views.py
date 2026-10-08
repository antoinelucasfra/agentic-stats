"""Tool results rendered as Shiny tags.

One renderer per tool, mirroring the labels and number formats the browser app
used. `full_view` is the Playground result and `card_view` is an Overview card,
which drops the tables to keep the six cards scannable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from shiny import ui

from stats import charts
from stats.stats_tools import (
    AnovaResult,
    AssumptionChecks,
    DatasetDescription,
    DoseResponseResult,
    MarginalMeansResult,
    MixedModelResult,
    PowerResult,
)
from utils.formatting import fmt, fmt_ci, fmt_p, fmt_small

BlockKind = Literal["metrics", "chart", "table", "heading", "warning", "notes", "text"]

TABLE_CLASSES = "table table-sm table-striped align-middle"


@dataclass(frozen=True)
class Block:
    """One piece of a rendered result, tagged so a view can drop it."""

    kind: BlockKind
    tag: Any


# --- blocks -----------------------------------------------------------------


def metrics(items: list[tuple[str, str]]) -> Block:
    children: list[Any] = []
    for label, value in items:
        children.append(ui.tags.dt(label, class_="text-muted small fw-normal"))
        children.append(ui.tags.dd(value, class_="fw-semibold mb-2"))
    return Block("metrics", ui.tags.dl(*children, class_="row row-cols-2 row-cols-lg-4 g-2 mb-3"))


def table(headers: list[str], rows: list[list[Any]]) -> Block:
    # pandas escapes every cell, so result values never reach the DOM raw.
    frame = pd.DataFrame(rows, columns=headers)
    return Block("table", ui.HTML(frame.to_html(index=False, classes=TABLE_CLASSES, border=0)))


def chart_block(
    frame: pd.DataFrame,
    build: Callable[[pd.DataFrame], Any],
    alt: str,
    empty_message: str = "No data to plot.",
) -> Block:
    if frame.empty:
        return Block("text", ui.tags.p(empty_message, class_="text-muted"))
    return Block("chart", charts.figure_to_img(build(frame), alt))


def notes(items: list[str]) -> list[Block]:
    if not items:
        return []
    return [
        Block("notes", ui.tags.ul(*[ui.tags.li(item) for item in items], class_="small text-muted"))
    ]


# --- one renderer per tool --------------------------------------------------


def dataset_blocks(result: DatasetDescription) -> list[Block]:
    return [
        metrics(
            [
                ("Rows", str(result.n_rows)),
                ("Columns", str(result.n_columns)),
                ("Factors", ", ".join(result.factors)),
                ("Numeric", ", ".join(result.numeric)),
            ]
        ),
        table(
            ["Column", "Type", "Missing", "Unique", "Min", "Max", "Mean"],
            [
                [
                    column.name,
                    column.dtype,
                    column.missing,
                    column.unique,
                    fmt(column.minimum, 3),
                    fmt(column.maximum, 3),
                    fmt(column.mean, 3),
                ]
                for column in result.columns
            ],
        ),
        *notes(result.notes),
    ]


def anova_blocks(result: AnovaResult) -> list[Block]:
    bands = pd.DataFrame(
        [
            {"label": level, "value": mean, "low": mean, "high": mean}
            for level, mean in result.level_means.items()
        ]
    )
    return [
        metrics(
            [
                ("F", fmt(result.f_statistic, 2)),
                ("p", fmt_p(result.p_value)),
                ("Omega squared", fmt(result.omega_squared)),
                ("Partial eta squared", fmt(result.partial_eta_squared)),
                ("Levene p", fmt_p(result.levene_p_value)),
            ]
        ),
        chart_block(
            bands,
            lambda frame: charts.dot_with_ci(
                frame, f"Mean signal by {result.factor}", "Mean signal"
            ),
            alt=f"Mean assay signal for each level of {result.factor}",
        ),
    ]


def marginal_blocks(result: MarginalMeansResult) -> list[Block]:
    bands = pd.DataFrame(
        [
            {
                "label": mean.level,
                "value": mean.estimate,
                "low": mean.ci_low,
                "high": mean.ci_high,
            }
            for mean in result.marginal_means
        ]
    )
    confidence = round(result.confidence_level * 100)
    return [
        metrics(
            [
                ("Model", result.model),
                ("Adjustment", result.adjustment),
                ("Level of confidence", f"{confidence}%"),
            ]
        ),
        chart_block(
            bands,
            lambda frame: charts.dot_with_ci(
                frame, "Marginal means with confidence intervals", "Mean signal"
            ),
            alt=f"Marginal means with {confidence}% confidence intervals per level",
            empty_message="No estimates to plot.",
        ),
        table(
            ["A", "B", "Difference", "95% CI", "p (adjusted)", "Significant"],
            [
                [
                    contrast.level_a,
                    contrast.level_b,
                    fmt(contrast.difference, 3),
                    fmt_ci(contrast.ci_low, contrast.ci_high),
                    fmt_small(contrast.p_adjusted),
                    "yes" if contrast.significant else "no",
                ]
                for contrast in result.pairwise
            ],
        ),
        *notes(result.notes),
    ]


def mixed_blocks(result: MixedModelResult) -> list[Block]:
    variances = pd.DataFrame(
        [
            {
                "label": result.group,
                "value": result.group_variance,
                "low": result.group_variance,
                "high": result.group_variance,
            },
            {
                "label": "residual",
                "value": result.residual_variance,
                "low": result.residual_variance,
                "high": result.residual_variance,
            },
        ]
    )
    return [
        metrics(
            [
                ("Group", result.group),
                ("Groups", str(result.n_groups)),
                ("Between-group variance", fmt(result.group_variance)),
                ("Residual variance", fmt(result.residual_variance)),
                ("ICC", fmt(result.icc)),
                ("Converged", "yes" if result.converged else "no"),
            ]
        ),
        chart_block(
            variances,
            lambda frame: charts.dot_with_ci(frame, "Variance components", "Variance"),
            alt=(
                f"Between-group variance for {result.group} and residual variance, "
                f"with an intraclass correlation of {fmt(result.icc)}"
            ),
        ),
        table(
            ["Term", "Estimate", "Std error", "95% CI", "p"],
            [
                [
                    effect.term,
                    fmt(effect.estimate, 3),
                    fmt(effect.std_error, 3),
                    fmt_ci(effect.ci_low, effect.ci_high),
                    fmt_small(effect.p_value),
                ]
                for effect in result.fixed_effects
            ],
        ),
        *notes(result.notes),
    ]


def checks_blocks(result: AssumptionChecks) -> list[Block]:
    qq = pd.DataFrame(
        [
            {"theoretical": point.theoretical, "standardized": point.standardized}
            for point in result.qq
        ]
    )
    residuals = pd.DataFrame(
        [{"fitted": point.fitted, "residual": point.residual} for point in result.residuals]
    )
    blocks = [
        metrics(
            [
                ("Model", result.model),
                ("Normality test", result.normality.test),
                ("p", fmt_p(result.normality.p_value)),
                ("Verdict", result.normality.verdict),
                ("Skew", fmt(result.skew, 3)),
                ("Excess kurtosis", fmt(result.excess_kurtosis, 3)),
                ("Max VIF", "n/a" if result.max_vif is None else fmt(result.max_vif, 2)),
            ]
        ),
        table(
            ["Diagnostic", "Statistic", "p", "Reading"],
            [
                [test.name, fmt(test.statistic, 3), fmt_p(test.p_value), test.verdict]
                for test in result.tests
            ],
        ),
        chart_block(qq, charts.qq_plot, alt="Normal QQ plot of standardized residuals"),
        chart_block(
            residuals,
            charts.residual_plot,
            alt="Residuals against fitted values, centred on zero",
        ),
    ]
    if result.outliers:
        detail_columns = list(result.outliers[0].detail)
        blocks.append(
            table(
                ["Row", *detail_columns, "Cook's D", "Leverage", "Std residual"],
                [
                    [
                        outlier.row,
                        *[outlier.detail.get(column, "") for column in detail_columns],
                        fmt(outlier.cook_d, 5),
                        fmt(outlier.leverage, 4),
                        fmt(outlier.standardized_residual, 3),
                    ]
                    for outlier in result.outliers
                ],
            )
        )
    blocks.extend(notes(result.notes))
    return blocks


def power_blocks(result: PowerResult) -> list[Block]:
    blocks = [
        metrics(
            [
                ("Rows", str(result.n_obs)),
                ("Pooled SD", fmt(result.pooled_sd, 3)),
                ("Cohen's f", fmt(result.cohens_f, 3)),
                ("Partial eta squared", fmt(result.partial_eta_squared, 3)),
                ("Power achieved", fmt(result.achieved_power, 3)),
                ("Smallest detectable f", fmt(result.minimum_detectable_f, 3)),
                ("Rows needed per level", str(result.required_n_per_group)),
                ("Rows per level now", str(next(iter(result.n_per_group.values()), ""))),
            ]
        )
    ]
    blocks.extend(notes(result.notes))
    return blocks


def dose_blocks(result: DoseResponseResult) -> list[Block]:
    blocks: list[Block] = []
    for fit in result.fits:
        if fit.group:
            blocks.append(Block("heading", ui.tags.h4(fit.group)))
        if not fit.converged:
            blocks.append(
                Block("warning", ui.tags.p("This fit did not converge.", class_="text-danger"))
            )
            blocks.extend(notes(fit.notes))
            continue
        peak = fit.optimum_dose is not None
        ci_low = fit.optimum_ci_low if peak else fit.ec50_ci_low
        ci_high = fit.optimum_ci_high if peak else fit.ec50_ci_high
        curve = pd.DataFrame(
            [{"dose": point.dose, "predicted": point.predicted} for point in fit.curve]
        )
        blocks.append(
            metrics(
                [
                    ("Peak dose" if peak else "EC50", fmt(fit.optimum_dose or fit.ec50, 3)),
                    ("95% CI", fmt_ci(ci_low, ci_high)),
                    ("R squared", fmt(fit.r_squared, 4)),
                    ("AIC", fmt(fit.aic, 1)),
                    ("Model", fit.model),
                ]
            )
        )
        blocks.append(
            chart_block(
                curve,
                lambda frame: charts.scatter(
                    frame,
                    "dose",
                    "predicted",
                    f"Predicted response against {result.dose}",
                    result.dose,
                    result.response,
                ),
                alt=f"Predicted {result.response} against {result.dose} for the fitted curve",
            )
        )
        blocks.append(
            table(
                ["Parameter", "Estimate", "Std error", "95% CI"],
                [
                    [
                        parameter.name,
                        fmt(parameter.estimate, 4),
                        fmt(parameter.std_error, 4),
                        fmt_ci(parameter.ci_low, parameter.ci_high),
                    ]
                    for parameter in fit.parameters
                ],
            )
        )
        blocks.extend(notes(fit.notes))
    return blocks


def family_blocks(result: Any) -> list[Block]:
    blocks: list[Block] = []
    for comparison in result.comparisons:
        if comparison.group:
            blocks.append(Block("heading", ui.tags.h4(comparison.group)))
        blocks.append(
            table(
                ["Shape", "Converged", "R squared", "AIC", "ΔAIC"],
                [
                    [
                        candidate.model,
                        str(candidate.converged),
                        fmt(candidate.r_squared, 4),
                        fmt(candidate.aic, 1),
                        fmt(comparison.delta_aic.get(candidate.model, 0.0), 1)
                        if candidate.model != comparison.best
                        else "winner",
                    ]
                    for candidate in comparison.candidates
                ],
            )
        )
        blocks.extend(notes(comparison.notes))
    return blocks


def correlation_blocks(result: Any) -> list[Block]:
    columns = result.columns
    return [
        metrics([("Rows", str(result.n_obs)), ("Columns", ", ".join(columns))]),
        table(
            ["", *columns],
            [
                [row, *[fmt(result.pearson[row][column], 3) for column in columns]]
                for row in columns
            ],
        ),
        *notes([*result.notes, "Spearman (rank) values live in the raw JSON."]),
    ]


_VIEWS: dict[str, Callable[[Any], list[Block]]] = {
    "describe_dataset": dataset_blocks,
    "fit_mixed_model": mixed_blocks,
    "anova_effect": anova_blocks,
    "marginal_means": marginal_blocks,
    "check_assumptions": checks_blocks,
    "power_analysis": power_blocks,
    "dose_response": dose_blocks,
    "fit_curve_family": family_blocks,
    "correlate": correlation_blocks,
}


def result_blocks(tool: str, result: Any) -> list[Block]:
    """Render one tool result. An unknown tool falls back to pretty JSON."""
    view = _VIEWS.get(tool)
    if view is None:
        return [Block("text", ui.tags.pre(json.dumps(result.model_dump(), indent=2)))]
    return view(result)


def full_view(tool: str, result: Any) -> list[Any]:
    """Every block, for the Playground."""
    return [block.tag for block in result_blocks(tool, result)]


def card_view(tool: str, result: Any) -> list[Any]:
    """Overview cards: same renderer, tables dropped."""
    return [block.tag for block in result_blocks(tool, result) if block.kind != "table"]
