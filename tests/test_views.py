"""Renderer tests: every tool's result becomes tags, charts carry alt text."""

from __future__ import annotations

import pandas as pd
import pytest

from stats import registry, views
from utils import formatting

# One call per tool, with the same arguments the Overview cards use.
CASES = [
    ("describe_dataset", {}, "Rows"),
    (
        "fit_mixed_model",
        {"response": "grain", "fixed_effects": ["variety", "nitrogen"]},
        "ICC",
    ),
    ("anova_effect", {"response": "grain", "factor": "nitrogen"}, "Omega squared"),
    (
        "marginal_means",
        {"response": "grain", "factors": ["variety"], "covariates": ["nitrogen"]},
        "Adjustment",
    ),
    (
        "check_assumptions",
        {"response": "grain", "fixed_effects": ["variety", "nitrogen"]},
        "Excess kurtosis",
    ),
    ("power_analysis", {"response": "grain", "factor": "nitrogen"}, "Power achieved"),
    ("dose_response", {"response": "grain", "dose": "nitrogen", "model": "4pl"}, "EC50"),
]


def run(tool: str, arguments: dict[str, object]) -> object:
    return registry.get(tool).handler(**arguments)


@pytest.mark.parametrize(("tool", "arguments", "expected"), CASES)
def test_full_view_renders_every_tool(
    tool: str, arguments: dict[str, object], expected: str
) -> None:
    tags = views.full_view(tool, run(tool, arguments))
    html = "".join(str(tag) for tag in tags)

    assert tags
    assert expected in html


@pytest.mark.parametrize(("tool", "arguments", "_expected"), CASES)
def test_charts_always_carry_alt_text(
    tool: str, arguments: dict[str, object], _expected: str
) -> None:
    html = "".join(str(tag) for tag in views.full_view(tool, run(tool, arguments)))

    if "<img" in html:
        assert 'alt="' in html
        assert 'alt=""' not in html


def test_card_view_drops_tables_but_keeps_the_chart() -> None:
    result = run("describe_dataset", {})
    card = "".join(str(tag) for tag in views.card_view("describe_dataset", result))
    full = "".join(str(tag) for tag in views.full_view("describe_dataset", result))

    assert "<table" in full
    assert "<table" not in card
    assert "Rows" in card

    result = run("anova_effect", {"response": "grain", "factor": "nitrogen"})
    card = "".join(str(tag) for tag in views.card_view("anova_effect", result))
    assert "<img" in card


def test_chart_block_handles_empty_data() -> None:
    block = views.chart_block(
        pd.DataFrame(columns=["label", "value", "low", "high"]),
        lambda frame: charts_identity(frame),
        alt="unused",
    )

    assert block.kind == "text"
    assert "No data to plot." in str(block.tag)


def charts_identity(frame: pd.DataFrame) -> pd.DataFrame:
    return frame


class _FakeResult:
    """Stands in for a tool result the renderer has no view for."""

    def model_dump(self) -> dict[str, object]:
        return {"answer": 42}


def test_unknown_tool_falls_back_to_json() -> None:
    html = "".join(str(tag) for tag in views.full_view("no_such_tool", _FakeResult()))

    assert '"answer": 42' in html


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "n/a"), (0.5, "0.5000"), (0.0004, "4.00e-04"), (float("nan"), "n/a")],
)
def test_number_formatting(value: float | None, expected: str) -> None:
    assert formatting.fmt(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "p = n/a"), (0.0004, "p < 0.001"), (0.0512, "p = 0.051")],
)
def test_p_value_formatting(value: float | None, expected: str) -> None:
    assert formatting.fmt_p(value) == expected
