"""Chart tests: a plotnine figure becomes an inline PNG with alt text."""

from __future__ import annotations

import pandas as pd

import charts


def bands() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"label": "A", "value": 21.4, "low": 21.1, "high": 21.7},
            {"label": "B", "value": 24.8, "low": 24.5, "high": 25.1},
        ]
    )


def test_figure_to_img_returns_a_png_data_uri() -> None:
    img = charts.figure_to_img(
        charts.dot_with_ci(bands(), "Means", "Mean signal"), alt="Mean signal per level"
    )
    html = str(img)

    assert "data:image/png;base64," in html
    assert 'alt="Mean signal per level"' in html
    assert len(html) > 2000  # a real PNG, not an empty canvas


def test_every_builder_draws_without_raising() -> None:
    points = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 5.0]})
    qq = pd.DataFrame({"theoretical": [-1.0, 0.0, 1.0], "standardized": [-1.1, 0.1, 0.9]})
    residuals = pd.DataFrame({"fitted": [1.0, 2.0], "residual": [0.1, -0.2]})

    for plot in (
        charts.dot_with_ci(bands(), "Means", "Mean signal"),
        charts.scatter(points, "x", "y", "Scatter", "x", "y"),
        charts.qq_plot(qq),
        charts.residual_plot(residuals),
    ):
        assert "data:image/png;base64," in str(charts.figure_to_img(plot, alt="chart"))
