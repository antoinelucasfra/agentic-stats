"""Plotnine figures for tool results, encoded as inline PNGs.

`figure_to_img` is here because a result tree holds a variable number of charts
(dose response has one per group) and Shiny cannot declare a variable number of
plot outputs ahead of time. Static plotnine output loses nothing by going out
as an image: there is no hover or zoom to lose, and `alt` carries the same text
`@render.plot(alt=)` would.
"""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib
import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_abline,
    geom_hline,
    geom_point,
    geom_pointrange,
    ggplot,
    labs,
    theme,
    theme_minimal,
)
from shiny import ui

# A server has no display: force the file backend so matplotlib never probes
# for Tk, which fails on a machine whose Python has no usable Tcl.
matplotlib.use("Agg", force=True)

# One size for every chart, so Overview cards and Playground results line up.
WIDTH_INCHES = 7.0
HEIGHT_INCHES = 3.4
DPI = 110

_THEME = theme_minimal() + theme(text=element_text(size=9))


def dot_with_ci(frame: pd.DataFrame, title: str, y_label: str) -> ggplot:
    """Estimates per level with confidence whiskers. Needs label/value/low/high."""
    return (
        ggplot(frame, aes("label", "value"))
        + geom_pointrange(aes(ymin="low", ymax="high"))
        + labs(x="Level", y=y_label, title=title)
        + _THEME
    )


def scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str,
    y_label: str,
) -> ggplot:
    """Points with no reference line. Needs columns named `x` and `y`."""
    return (
        ggplot(frame, aes(x, y))
        + geom_point(alpha=0.6)
        + labs(x=x_label, y=y_label, title=title)
        + _THEME
    )


def qq_plot(frame: pd.DataFrame) -> ggplot:
    """Standardized residuals against normal quantiles, with the y=x line."""
    return (
        ggplot(frame, aes("theoretical", "standardized"))
        + geom_point(alpha=0.6)
        + geom_abline(slope=1, intercept=0, linetype="dashed")
        + labs(x="Theoretical quantile", y="Sample quantile", title="Normal QQ plot")
        + _THEME
    )


def residual_plot(frame: pd.DataFrame) -> ggplot:
    """Residuals against fitted values, centred on zero."""
    return (
        ggplot(frame, aes("fitted", "residual"))
        + geom_hline(yintercept=0, linetype="dashed")
        + geom_point(alpha=0.6)
        + labs(x="Fitted value", y="Residual", title="Residuals against fitted values")
        + _THEME
    )


def figure_to_img(plot: ggplot, alt: str) -> ui.Tag:
    """Encode a plotnine figure as an inline PNG, so it composes anywhere."""
    buffer = BytesIO()
    plot.save(
        buffer,
        format="png",
        width=WIDTH_INCHES,
        height=HEIGHT_INCHES,
        dpi=DPI,
        verbose=False,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return ui.img(src=f"data:image/png;base64,{encoded}", alt=alt, width="100%")
