"""Reading tools: the chart, and the interpretation of a stored result.

These are the tools that keep a narrative honest. `plot` renders a chart and
returns the features behind it, so "a clear upward trend" is a claim the tool
either supports or does not. `explain_result` turns a stored result into effects,
caveats and the list of what it cannot establish, with the figures frozen. The
model composes the prose from those parts. It cannot change a number, and it
cannot drop a caveat.

Nothing here chooses a conclusion. The reading is the model's; the evidence is
this module's, and every figure it emits traces to a stored result.

The gate follows the evidence class. A fit or a comparison needs the whole
chain, including sensitivity, before it can be interpreted. A query or a chart
is descriptive, so it needs the design and the quality stages only.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_boxplot,
    geom_col,
    geom_histogram,
    geom_line,
    geom_point,
    geom_tile,
    ggplot,
    labs,
    stat_ecdf,
    theme,
    theme_minimal,
)
from pydantic import BaseModel, Field

from stats import charts, session
from stats.errors import ToolError
from utils.formatting import fmt, fmt_ci

__all__ = ["Explanation", "PlotFeatures", "PlotResult", "explain_result", "plot"]

PlotKind = Literal["hist", "box", "scatter", "line", "bar", "heatmap", "ecdf"]

DESCRIPTIVE_KINDS = ("query", "table", "plot", "profile")

# Below this many rows, a share or a rate is too thin to read.
SMALL_GROUP = 5

_THEME = theme_minimal() + theme(text=element_text(size=9))


class ExplainedEffect(BaseModel):
    """One effect, decoded, with the sentence a reader could quote."""

    term: str
    contrast: str
    statement: str
    estimate: float
    ci_low: float
    ci_high: float
    scale: str = "response"


class PlotFeatures(BaseModel):
    """What is actually in the picture, so a description can be checked."""

    n_rows: int
    series: int
    x_range: list[float] = Field(default_factory=list)
    y_range: list[float] = Field(default_factory=list)
    trend_slope: float | None = None
    trend_direction: str = "no trend computed"
    largest_gap: float | None = None
    extremes: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PlotResult(BaseModel):
    """A rendered chart plus the numbers behind it."""

    kind: PlotKind
    table: str
    x: str
    y: str | None = None
    color: str | None = None
    facet: str | None = None
    image: str = Field(default="", description="PNG data URI")
    alt_text: str = ""
    features: PlotFeatures
    result_id: str = ""
    next: str = ""
    notes: list[str] = Field(default_factory=list)


class Explanation(BaseModel):
    """A stored result as parts the model composes prose from."""

    source_result_id: str
    kind: str
    audience: str
    effects: list[ExplainedEffect] = Field(default_factory=list)
    interactions: list[ExplainedEffect] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    cannot_say: list[str] = Field(default_factory=list)
    causal_claims_permitted: bool = False
    reading: list[str] = Field(default_factory=list)
    suggested_phrasing: str = ""
    result_id: str = ""
    next: str = ""
    notes: list[str] = Field(default_factory=list)


def _gate(store: session.SessionStore, kind: str) -> None:
    """A descriptive reading needs less than an inferential one."""
    if kind in DESCRIPTIVE_KINDS:
        store.state.require("quality")
    else:
        store.state.require("interpretation")


def _design_plan(store: session.SessionStore) -> Any | None:
    """The stored plan from the design stage, when there is one."""
    record = store.state.records.get("design")
    if record is None or not record.result_id:
        return None
    try:
        return store.get(record.result_id).payload
    except ToolError:
        return None


def _causal_check(store: session.SessionStore) -> tuple[bool, str]:
    """Causal language needs a target `frame_question` accepted as causal.

    One gate, in one place: `frame_question` refuses a causal target when the
    design cannot carry it, so a permitted flag here means the design was
    checked. A model that never asked for a causal target does not get causal
    verbs, which is the conservative direction.
    """
    plan = _design_plan(store)
    if plan is None:
        return False, (
            "No design stage is recorded, so no causal claim is permitted. "
            "Call frame_question() first."
        )
    target = str(getattr(plan, "target", ""))
    design = str(getattr(plan, "design", "independent"))
    if target == "causal":
        return True, ""
    return False, (
        f"This is a {design} design with an {target or 'unspecified'} target, so no causal "
        "claim is permitted. A causal target needs a treatment or a time structure; "
        "frame_question() names the column that would support it."
    )


def _as_effects(
    items: Any, audience: str, *, scale_default: str, link_suffix: bool
) -> list[ExplainedEffect]:
    """One loop for coefficient tables and response-scale effects.

    The payloads name the same fields differently and disagree on one rule: raw
    coefficients always carry the link-scale suffix, response-scale effects only
    for an analyst audience. `link_suffix` selects the rule; the loop is shared.
    """
    out: list[ExplainedEffect] = []
    for item in items or []:
        contrast = str(getattr(item, "contrast", "") or getattr(item, "term", ""))
        scale = str(getattr(item, "scale", scale_default))
        statement = f"{contrast}: {fmt(item.estimate)} ({fmt_ci(item.ci_low, item.ci_high)})"
        if scale == "link" and (link_suffix or audience == "analyst"):
            statement += " on the link scale"
        out.append(
            ExplainedEffect(
                term=str(getattr(item, "term", contrast)),
                contrast=contrast,
                statement=statement,
                estimate=float(item.estimate),
                ci_low=float(item.ci_low),
                ci_high=float(item.ci_high),
                scale=scale,
            )
        )
    return out


def _contrasts_as_effects(payload: Any) -> list[ExplainedEffect]:
    out: list[ExplainedEffect] = []
    for contrast in getattr(payload, "contrasts", []) or []:
        label = f"{contrast.level_a} vs {contrast.level_b}"
        out.append(
            ExplainedEffect(
                term=str(getattr(contrast, "factor", "")),
                contrast=label,
                statement=(
                    f"{label}: difference {fmt(contrast.difference)} "
                    f"({fmt_ci(contrast.ci_low, contrast.ci_high)})"
                    + (
                        f", adjusted p {fmt(contrast.adjusted_p_value, 3)}"
                        if getattr(contrast, "adjusted_p_value", None) is not None
                        else ""
                    )
                ),
                estimate=float(contrast.difference),
                ci_low=float(contrast.ci_low),
                ci_high=float(contrast.ci_high),
            )
        )
    return out


def _fit_explanation(payload: Any, audience: str) -> tuple[list, list, list, list]:
    coefficients = _as_effects(
        getattr(payload, "coefficients", None),
        audience,
        scale_default="link",
        link_suffix=True,
    )
    response_scale = _as_effects(
        getattr(payload, "effects", None),
        audience,
        scale_default="response",
        link_suffix=False,
    )
    caveats = [str(item) for item in getattr(payload, "caveats", []) or []]
    cannot = [str(item) for item in getattr(payload, "cannot_say", []) or []]
    diagnostics = getattr(payload, "diagnostics", []) or []
    for verdict in diagnostics:
        if str(getattr(verdict, "status", "")) == "blocked":
            caveats.append(f"{verdict.check}: {verdict.statement} Fix: {verdict.remedy}")
    return response_scale or coefficients, coefficients, caveats, cannot


def _comparison_explanation(payload: Any) -> tuple[list, list, list]:
    effects = _contrasts_as_effects(payload)
    effect_size = getattr(payload, "effect_size", None)
    if effect_size is not None:
        effects.insert(
            0,
            ExplainedEffect(
                term=str(payload.factor),
                contrast=str(effect_size.name),
                statement=(
                    f"{effect_size.name} {fmt(effect_size.estimate)} "
                    f"({fmt_ci(effect_size.ci_low, effect_size.ci_high)}): "
                    f"{effect_size.interpretation}"
                ),
                estimate=float(effect_size.estimate),
                ci_low=float(effect_size.ci_low),
                ci_high=float(effect_size.ci_high),
            ),
        )
    caveats = [f"Routed to {payload.test} because {payload.routing_reason}."]
    for verdict in getattr(payload, "diagnostics", []) or []:
        if str(getattr(verdict, "status", "")) != "ok":
            caveats.append(f"{verdict.check}: {verdict.statement} Fix: {verdict.remedy}")
    return effects, caveats, list(getattr(payload, "cannot_say", []) or [])


def _marginal_explanation(payload: Any) -> tuple[list, list, list]:
    """Average predictive margins: the contrasts, and the averaging that was done."""
    effects = _contrasts_as_effects(payload)
    caveats: list[str] = []
    adjust = str(getattr(payload, "adjust", ""))
    if adjust:
        caveats.append(
            f"Contrasts are adjusted with {adjust}, over the contrasts within this one call."
        )
    levels = getattr(payload, "reference_levels", None) or {}
    if levels:
        named = ", ".join(f"{key}={value}" for key, value in levels.items())
        caveats.append(f"Reference levels: {named}.")
    cannot_say = [
        "These are averages over the observed rows, not the effect for any single row.",
    ]
    return effects, caveats, cannot_say


def _table_reading(payload: Any) -> tuple[list[str], list[str]]:
    """Ranked contributions and shares from a query result, with its traps."""
    columns = [str(name) for name in getattr(payload, "columns", []) or []]
    rows = list(getattr(payload, "rows", []) or [])
    if len(columns) < 2 or not rows:
        return ["The query returned nothing to rank."], []

    frame = pd.DataFrame(rows, columns=columns)
    label, metric = columns[0], columns[-1]
    values = pd.to_numeric(frame[metric], errors="coerce")
    if values.isna().all():
        return [f"{metric} is not numeric, so nothing can be ranked."], []

    total = float(values.sum())
    reading = [f"{len(frame)} rows of {metric} totalling {fmt(total)}."]
    order = values.sort_values(ascending=False).index
    for index in order[:5]:
        share = f"{values[index] / total:.1%} of the total" if total else "share not computable"
        reading.append(f"{frame.loc[index, label]}: {fmt(values[index])}, {share}.")

    if total <= 0:
        reading.append(
            f"{metric} sums to {fmt(total)}, so shares are meaningless: the values are "
            "not a set of parts of a whole."
        )
    thin = frame[label].value_counts()
    if len(frame) and int(thin.min()) < SMALL_GROUP and thin.index.size > 1:
        reading.append(
            f"Some {label} values appear fewer than {SMALL_GROUP} times, so a rate or a "
            "mean over them would be unstable."
        )
    reading.append(
        "This is a single grouping. A direction that reverses between subgroups cannot "
        "be seen here; split by the suspected confounder before claiming a pattern."
    )
    return reading, []


def _plot_reading(payload: Any) -> tuple[list[str], list[str]]:
    features = payload.features
    reading = [payload.alt_text]
    reading.append(
        f"{features.n_rows} rows over {features.series} series. "
        f"x spans {features.x_range or 'nothing'}, y spans {features.y_range or 'nothing'}."
    )
    reading.append(features.trend_direction)
    if features.largest_gap is not None:
        reading.append(f"The largest gap along x is {fmt(features.largest_gap)}.")
    reading.extend(features.extremes)
    return reading, list(features.notes)


def _sensitivity_reading(payload: Any) -> list[str]:
    reading = [f"{check.check}: {check.status}. {check.statement}" for check in payload.checks]
    reading.append(
        f"{payload.stable_share:.0%} of the checks kept both the direction and the "
        f"significance. {payload.verdict}"
    )
    return reading


def explain_result(
    result_id: str,
    audience: Literal["analyst", "stakeholder"] = "analyst",
) -> Explanation:
    """Turn a stored result into effects, caveats and limits, with figures frozen.

    One tool for every stored kind, because three tools that differ only in
    payload would cost a weak model three wrong choices. Rewrite the suggested
    phrasing freely, keep every figure as returned, and keep every caveat.
    """
    store = session.current_session()
    stored = store.get(result_id)
    _gate(store, stored.kind)
    payload = stored.payload

    effects: list[ExplainedEffect] = []
    interactions: list[ExplainedEffect] = []
    caveats: list[str] = []
    cannot_say: list[str] = []
    reading: list[str] = []

    if stored.kind == "fit":
        effects, all_effects, caveats, cannot_say = _fit_explanation(payload, audience)
        interactions = [item for item in all_effects if ":" in item.term]
    elif stored.kind == "comparison":
        effects, caveats, cannot_say = _comparison_explanation(payload)
    elif stored.kind == "marginal":
        effects, caveats, cannot_say = _marginal_explanation(payload)
    elif stored.kind in ("query", "table"):
        reading, caveats = _table_reading(payload)
    elif stored.kind == "plot":
        reading, caveats = _plot_reading(payload)
    elif stored.kind == "sensitivity":
        reading = _sensitivity_reading(payload)
    else:
        raise ToolError(
            f"Nothing to interpret in a {stored.kind} result. Use explain_result on a fit, "
            "a comparison, a query or a plot."
        )

    permitted, refusal = _causal_check(store)
    if refusal:
        cannot_say.append(refusal)

    statements = [item.statement for item in effects] or reading
    phrasing = " ".join(statements[:3])
    if caveats:
        phrasing = f"{phrasing} {caveats[0]}".strip()

    recorded = store.put(
        Explanation(
            source_result_id=result_id,
            kind=stored.kind,
            audience=audience,
            effects=effects,
            interactions=interactions,
            caveats=caveats,
            cannot_say=cannot_say,
            causal_claims_permitted=permitted,
            reading=reading,
            suggested_phrasing=phrasing,
            next=(
                "Call analysis_status() to check what is still missing before writing "
                "the conclusion."
            ),
            notes=list(getattr(payload, "notes", []) or []),
        ),
        kind="explanation",
        evidence_class=stored.evidence_class,
        summary=f"Reading of {result_id} ({stored.kind})",
    )
    store.state.record(
        "interpretation",
        status="caution" if (caveats or cannot_say) else "ok",
        result_id=recorded.result_id,
        note=f"{len(effects)} effects, {len(caveats)} caveats, {len(cannot_say)} limits",
    )
    return recorded.payload


def _require_columns(frame: pd.DataFrame, names: list[str]) -> None:
    unknown = [name for name in names if name not in frame.columns]
    if unknown:
        raise ToolError(f"Unknown column(s): {unknown}. Available columns: {list(frame.columns)}.")


def _build_plot(frame: pd.DataFrame, spec: dict[str, Any]) -> ggplot:
    kind, x, y, color = spec["kind"], spec["x"], spec["y"], spec["color"]
    mapping: dict[str, Any] = {"x": x}
    if y is not None:
        mapping["y"] = y
    if color is not None:
        mapping["color"] = color
    base = ggplot(frame, aes(**mapping))
    match kind:
        case "hist":
            return base + geom_histogram(bins=30)
        case "ecdf":
            return base + stat_ecdf()
        case "box":
            return base + geom_boxplot()
        case "scatter":
            return base + geom_point(alpha=0.6)
        case "line":
            return base + geom_line()
        case "bar":
            return base + geom_col()
        case _:
            return base + geom_tile()


def _features(frame: pd.DataFrame, spec: dict[str, Any], notes: list[str]) -> PlotFeatures:
    """What the picture contains, computed rather than described."""
    x, y = spec["x"], spec["y"]
    x_values = frame[x]
    series = int(frame[spec["color"]].nunique()) if spec["color"] else 1
    features = PlotFeatures(
        n_rows=len(frame),
        series=series,
        x_range=[float(x_values.min()), float(x_values.max())]
        if pd.api.types.is_numeric_dtype(x_values)
        else [],
        y_range=[],
        notes=notes,
    )

    if y is not None and pd.api.types.is_numeric_dtype(frame[y]):
        values = frame[y].dropna()
        features.y_range = [float(values.min()), float(values.max())]
        top = values.nlargest(2)
        low = values.nsmallest(1)
        features.extremes = [
            f"highest {y} is {fmt(float(top.iloc[0]))} at {x} {frame.loc[top.index[0], x]}",
            f"lowest {y} is {fmt(float(low.iloc[0]))} at {x} {frame.loc[low.index[0], x]}",
        ]
        if pd.api.types.is_numeric_dtype(x_values):
            paired = frame[[x, y]].dropna()
            if len(paired) >= 3:
                slope = float(np.polyfit(paired[x].astype(float), paired[y].astype(float), 1)[0])
                features.trend_slope = slope
                word = "upward" if slope > 0 else "downward"
                features.trend_direction = (
                    f"Least-squares slope {fmt(slope)}, a {word} trend across the observed "
                    f"range of {x}."
                )
        if spec["kind"] == "line":
            ordered = pd.to_numeric(x_values, errors="coerce").dropna().sort_values()
            if len(ordered) >= 2:
                features.largest_gap = float(ordered.diff().dropna().max())
    return features


def _alt_text(features: PlotFeatures, spec: dict[str, Any]) -> str:
    described = spec["y"] or spec["x"]
    series = f", {features.series} series" if features.series > 1 else ""
    return (
        f"{spec['kind']} plot of {described} over {features.n_rows} rows{series}. "
        f"{features.trend_direction}"
    )


def plot(
    kind: PlotKind,
    x: str,
    y: str | None = None,
    color: str | None = None,
    facet: str | None = None,
    table: str = "primary",
    max_levels: int = 12,
) -> PlotResult:
    """Draw a chart and return the features behind it.

    A declarative spec, not code: name the columns and the kind. Needs `y` for
    box, scatter, line, bar and heatmap, and `color` for heatmap. Categorical
    axes above `max_levels` are refused rather than drawn unreadably.
    """
    store = session.current_session()
    store.state.require("quality")
    frame = store.frame(table)
    if len(frame) < 2:
        raise ToolError("A chart needs at least two rows.")
    needed = [name for name in (x, y, color, facet) if name]
    _require_columns(frame, needed)

    if kind in ("box", "scatter", "line", "bar", "heatmap") and y is None:
        raise ToolError(f"{kind} needs a y column. Numeric columns: {_numeric(frame)}.")
    if kind == "heatmap" and color is None:
        raise ToolError("heatmap needs a color column to fill the tiles with.")
    if kind in ("scatter", "line") and not pd.api.types.is_numeric_dtype(frame[y]):
        raise ToolError(f"{kind} needs a numeric y. Numeric columns: {_numeric(frame)}.")

    notes: list[str] = []
    for axis in (x, color):
        if axis is None:
            continue
        levels = int(frame[axis].nunique())
        if not pd.api.types.is_numeric_dtype(frame[axis]) and levels > max_levels:
            raise ToolError(
                f"{axis} has {levels} levels, above the {max_levels} a chart can show. "
                f"Filter to the top {max_levels} first, or aggregate with run_sql."
            )
    if facet is not None:
        notes.append(
            "The facet column is recorded but not drawn: plotnine facets need a layout the "
            "single-figure renderer does not provide yet."
        )

    spec = {"kind": kind, "x": x, "y": y, "color": color}
    features = _features(frame, spec, notes)
    alt = _alt_text(features, spec)
    figure = _build_plot(frame, spec) + labs(x=x, y=y or "count", title=alt) + _THEME

    stored = store.put(
        PlotResult(
            kind=kind,
            table=table,
            x=x,
            y=y,
            color=color,
            facet=facet,
            image=charts.figure_to_data_uri(figure),
            alt_text=alt,
            features=features,
            next=(
                "Call explain_result(result_id) to get the reading, or run_sql to rank the "
                "categories behind the chart."
            ),
            notes=notes,
        ),
        kind="plot",
        evidence_class="descriptive",
        summary=f"{kind} of {y or x}",
    )
    return stored.payload


def _numeric(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if pd.api.types.is_numeric_dtype(frame[name])]
