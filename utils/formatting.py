"""Number formatting shared by the tool renderers and the forms."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["fmt", "fmt_ci", "fmt_p", "fmt_small", "number"]


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if value != 0 and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def fmt_p(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "p = n/a"
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def fmt_small(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return "< 0.001" if value < 0.001 else fmt(value, digits)


def fmt_ci(low: float | None, high: float | None, digits: int = 3) -> str:
    return f"{fmt(low, digits)} to {fmt(high, digits)}"


def number(value: Any) -> float | None:
    """A form control's value as a float, or None when it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
