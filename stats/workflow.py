"""The analysis workflow tools: frame, profile and query, in a fixed order.

`stats.stats_tools` computes correctly but leaves the order to the caller. A
small local model cannot hold that order across a dozen turns, so these tools
carry it: `frame_question` states the estimand and writes the design stage,
`profile_table` reports schema and data quality and writes the quality stage,
`run_sql` answers lookups while refusing to let a descriptive number pass as an
estimate, and `analysis_status` reads the ledger back as a checklist.

Each result opens with `next`, one imperative sentence naming the call that
follows, so a model that follows instructions blindly still finishes the
analysis. Every refusal names the remedy as a tool call, never a dead end, and
`stats.session` enforces the stage order each tool claims to keep.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from stats import roles
from stats.errors import ToolError
from stats.session import AnalysisStatus, Verdict, current_session
from utils import config
from utils import data as dataset
from utils.duck import SqlGuardError, connect, guard_query, register_frame, run_select
from utils.fetch import FetchError, fetch_table
from utils.formatting import fmt, fmt_p

Target = Literal["description", "comparison", "association", "prediction", "causal"]

# Thresholds the quality verdicts share. Named so a reader can disagree with one
# number instead of hunting for it in a statement string.
MISSING_WARN = 0.05
MISSING_BLOCK = 0.40
DUPLICATE_BLOCK = 0.05
LEAKAGE_CORRELATION = 0.95
IMBALANCE_RATIO = 4.0

# Column names that mark an assigned exposure. A causal target is refused unless
# one of these, or a time column, is present: a grouping column is a stratum, not
# a treatment, and treating it as one is how an observational table gets read as
# an experiment.
TREATMENT_NAMES = (
    "treatment",
    "trt",
    "arm",
    "condition",
    "intervention",
    "assigned",
    "randomised",
    "randomized",
    "exposed",
    "exposure",
    "variant",
    "dose",
    "dosage",
)

# The exact text run_sql puts on every result, so the model reads the tag in the
# payload and not only in a prompt it has already forgotten.
DESCRIPTIVE_NOTE = (
    "Descriptive only. Cite compare_groups, fit_model or sensitivity for any claim about an effect."
)

_ROLES = ("response", "factor", "numeric", "time", "group", "dose", "identifier", "leakage")


class WorkflowResult(BaseModel):
    """Fields every workflow tool returns, whatever the payload behind them."""

    result_id: str = Field(default="", description="Handle for the stored result")
    next: str = Field(default="", description="The tool call that follows, as one sentence")
    notes: list[str] = Field(default_factory=list)


class AnalysisPlan(WorkflowResult):
    """A stated estimand, a design classification and the sequence of calls."""

    question: str
    estimand: str = Field(description="The quantity being estimated, as one sentence")
    target: Target
    design: str = Field(description="independent, paired, repeated or clustered")
    unit_of_analysis: str
    outcome: str
    outcome_type: str
    predictors: list[str] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list, description="Ordered tool calls as text")
    refusals: list[str] = Field(default_factory=list)


class ColumnProfile(BaseModel):
    """One column: its shape, and every role the profile gave it."""

    name: str
    dtype: str
    missing: int
    missing_rate: float
    distinct: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    roles: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class TableProfile(WorkflowResult):
    """Schema and quality in one read, with a verdict per quality check."""

    table: str
    source: str = ""
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile] = Field(default_factory=list)
    roles: dict[str, list[str]] = Field(default_factory=dict)
    suggested: dict[str, str] = Field(default_factory=dict)
    verdicts: list[Verdict] = Field(default_factory=list)


class QueryResult(WorkflowResult):
    """Rows from one guarded SELECT, tagged descriptive on the result itself."""

    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    statement_type: str = ""
    elapsed_s: float = 0.0
    note: str = DESCRIPTIVE_NOTE
    evidence_class: Literal["descriptive", "inferential"] = "descriptive"


class TableSummary(WorkflowResult):
    """A table that just entered the session, and what it can answer."""

    name: str
    source: str = ""
    sha256: str = ""
    n_rows: int = 0
    columns: list[str] = Field(default_factory=list)
    suggested: dict[str, str] = Field(default_factory=dict)
    profile_notes: list[str] = Field(default_factory=list)


def frame_question(
    question: str,
    outcome: str | None = None,
    predictors: list[str] | None = None,
    unit: str | None = None,
    target: Target | None = None,
) -> AnalysisPlan:
    """State the estimand and the design before any method is chosen.

    Infers the target from the question and the outcome from the role map when
    either is missing, classifies the design from the unit, time and grouping
    columns, and writes the `design` stage. Refuses a causal target when the
    design has no treatment or time structure, because an association from a
    cross-section is not an effect.
    """
    store = current_session()
    store.state.require("design")

    ref = store.table()
    frame = ref.frame
    profile = roles.infer_profile(frame, ref.roles or None)

    chosen_target: Target = target or _infer_target(question)
    chosen_outcome = outcome or profile.best("response")
    if chosen_outcome is None:
        raise ToolError(
            f"No outcome column could be inferred for table '{ref.name}'. "
            f"Pass outcome= one of {list(frame.columns)}."
        )
    if chosen_outcome not in frame.columns:
        raise ToolError(
            f"Unknown outcome column '{chosen_outcome}'. Available columns: {list(frame.columns)}."
        )
    if unit is not None and unit not in frame.columns:
        raise ToolError(f"Unknown unit column '{unit}'. Available columns: {list(frame.columns)}.")

    time_col = profile.best("time")
    group_col = profile.best("group")
    treatment_col = _treatment_column(frame)
    design, unit_of_analysis = _classify_design(frame, unit, time_col, group_col)

    if chosen_target == "causal" and not _supports_causal(design, time_col, treatment_col):
        raise ToolError(
            f"A causal target needs an assigned exposure or a time structure, and this "
            f"design is {design} with neither, so the data cannot carry a causal claim. "
            f"A column whose name marks an assignment, one of {list(TREATMENT_NAMES)}, or a "
            f"date column, would support it. Otherwise reframe the question as association. "
            f"Available columns: {list(frame.columns)}."
        )

    if predictors is not None:
        unknown = [name for name in predictors if name not in frame.columns]
        if unknown:
            raise ToolError(
                f"Unknown predictor(s) {unknown}. Available columns: {list(frame.columns)}."
            )
        terms = [name for name in predictors if name != chosen_outcome]
    else:
        terms = _select_predictors(profile, chosen_outcome, unit_of_analysis)

    estimand = _estimand(chosen_target, chosen_outcome, terms)
    plan_model = AnalysisPlan(
        question=question,
        estimand=estimand,
        target=chosen_target,
        design=design,
        unit_of_analysis=unit_of_analysis,
        outcome=chosen_outcome,
        outcome_type=_outcome_type(frame[chosen_outcome]),
        predictors=terms,
        required_checks=_required_checks(chosen_target, design, unit_of_analysis),
        plan=_plan(ref.name, chosen_target, chosen_outcome, terms, design, group_col, estimand),
        refusals=_refusals(design, unit_of_analysis),
        notes=list(profile.notes),
    )
    stored = store.put(
        plan_model,
        kind="plan",
        summary=f"{design} design; estimand: {estimand}",
    )
    plan_model.next = (
        f"Call profile_table(table='{ref.name}', outcome='{chosen_outcome}') "
        "to check schema and data quality."
    )
    store.state.record("design", result_id=stored.result_id, note=estimand)
    return plan_model


def profile_table(table: str = "primary", outcome: str | None = None) -> TableProfile:
    """Read the schema and run the quality checks, because one read pays for both.

    Writes the `quality` stage. The leakage check is the one that saves an
    analysis: a predictor that encodes the outcome turns every other coefficient
    into noise, so it blocks instead of warning.
    """
    store = current_session()
    store.state.require("quality")

    ref = store.table(table)
    frame = ref.frame
    profile = roles.infer_profile(frame, ref.roles or None)

    chosen_outcome = outcome or profile.best("response")
    if chosen_outcome is None:
        raise ToolError(
            f"No outcome column could be inferred for table '{ref.name}'. "
            f"Pass outcome= one of {list(frame.columns)}."
        )
    if chosen_outcome not in frame.columns:
        raise ToolError(
            f"Unknown outcome column '{chosen_outcome}'. Available columns: {list(frame.columns)}."
        )

    columns = [_column(frame[name], name, profile) for name in frame.columns]
    verdicts = _missingness_verdicts(frame)
    verdicts.extend(_missingness_outcome_verdicts(frame, chosen_outcome))
    verdicts.append(_duplicate_verdict(frame))
    verdicts.append(_zero_variance_verdict(frame))
    verdicts.append(_leakage_name_verdict(profile))
    verdicts.append(_leakage_association_verdict(frame, chosen_outcome))
    verdicts.extend(_imbalance_verdicts(frame, profile))

    blocking = [v for v in verdicts if v.status == "blocked"]
    cautions = [v for v in verdicts if v.status == "caution"]
    status: Literal["ok", "caution", "blocked"] = (
        "blocked" if blocking else "caution" if cautions else "ok"
    )
    roles_map = {role: profile.columns(role) for role in _ROLES}
    profile_model = TableProfile(
        table=ref.name,
        source=ref.source,
        n_rows=len(frame),
        n_columns=len(frame.columns),
        columns=columns,
        roles={role: names for role, names in roles_map.items() if names},
        suggested=dict(profile.suggested),
        verdicts=verdicts,
        notes=list(profile.notes),
    )
    stored = store.put(
        profile_model,
        kind="profile",
        rows_total=len(frame),
        rows_used=len(frame),
        summary=f"{len(frame)} rows x {len(frame.columns)} columns; "
        f"{len(blocking)} blocking, {len(cautions)} caution",
    )
    if blocking:
        profile_model.next = f"Resolve this before fitting: {blocking[0].remedy}"
    else:
        profile_model.next = (
            f"Call fit_model(response='{chosen_outcome}') to estimate the effect, "
            "or compare_groups() for a difference between group means."
        )
    store.state.record(
        "quality",
        status=status,
        result_id=stored.result_id,
        note=f"{len(columns)} columns profiled",
        missing=[f"{v.check}: {v.statement}" for v in cautions],
        verdicts=verdicts,
    )
    return profile_model


def run_sql(query: str, table: str = "primary", max_rows: int = 200) -> QueryResult:
    """Run one guarded SELECT over every loaded table, tagged descriptive.

    The result answers a lookup or a join. It is not an estimate, and the note
    on the result says which tools can support a claim about an effect. A table
    whose name is a SQL keyword must be quoted, as in `select * from "primary"`.
    """
    store = current_session()
    store.table(table)
    if max_rows < 1:
        raise ToolError("max_rows must be at least 1.")
    try:
        with connect() as conn:
            for ref in store.tables.values():
                register_frame(conn, ref.name, ref.frame)
            guard_query(query)
            outcome = run_select(conn, query, max_rows=max_rows)
    except SqlGuardError as exc:
        raise ToolError(
            f"{exc}. Only a single SELECT is allowed; it may not read files, "
            "open a socket or change the database."
        ) from exc

    rows = [[_jsonable(value) for value in row] for row in outcome.rows]
    result = QueryResult(
        columns=list(outcome.columns),
        rows=rows,
        row_count=len(rows),
        truncated=outcome.truncated,
        statement_type=outcome.statement_type,
        elapsed_s=round(outcome.elapsed_s, 4),
        note=DESCRIPTIVE_NOTE,
        evidence_class="descriptive",
    )
    store.put(
        result,
        kind="query",
        evidence_class="descriptive",
        summary=f"{len(rows)} row(s) returned" + (" (truncated)" if outcome.truncated else ""),
    )
    result.next = (
        "Call profile_table() to check these columns before you build an estimate "
        "on them, and cite fit_model or compare_groups for any effect."
    )
    return result


def _name_from_url(url: str) -> str:
    """A table name from a URL path, safe for duckdb and the registry."""
    stem = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    cleaned = "".join(character if character.isalnum() else "_" for character in stem.lower())
    return cleaned.strip("_") or "imported"


def _unique_name(store: Any, wanted: str) -> str:
    """A table name no loaded table uses yet."""
    if wanted not in store.tables:
        return wanted
    index = 2
    while f"{wanted}_{index}" in store.tables:
        index += 1
    return f"{wanted}_{index}"


def read_imported(path: Path, format: str) -> pd.DataFrame:
    """A downloaded file as a frame. Anything unreadable is the tool's error."""
    try:
        if format == "parquet":
            return dataset.read_parquet(path)
        if format == "json":
            return pd.read_json(path)
        return pd.read_csv(path)
    except Exception as exc:
        raise ToolError(f"Downloaded {format} but could not read it as a table: {exc}") from exc


def load_table(url: str, name: str | None = None) -> TableSummary:
    """Fetch a public CSV, parquet or JSON file and make it the primary table.

    The URL passes the same guard as every other fetch: https only, no private
    address even after redirects, a byte cap enforced while streaming, and a
    format check. A table above `MAX_IMPORT_ROWS` rows or `MAX_IMPORT_COLS`
    columns is refused rather than loaded into memory.

    A new table invalidates the old analysis, so the results and the ledger are
    cleared. Needs no stage, like `run_sql`: it is how data enters the session,
    not an inference.
    """
    store = current_session()
    try:
        fetched = fetch_table(url, dest_dir=config.cache_dir() / config.IMPORT_DIRNAME)
    except FetchError as exc:
        raise ToolError(str(exc)) from exc
    frame = read_imported(fetched.path, fetched.format)
    if frame.empty:
        raise ToolError(f"The file at {url} has no rows.")
    if len(frame) > config.MAX_IMPORT_ROWS or len(frame.columns) > config.MAX_IMPORT_COLS:
        raise ToolError(
            f"The file at {url} has {len(frame):,} rows and {len(frame.columns)} columns, "
            f"above the {config.MAX_IMPORT_ROWS:,} by {config.MAX_IMPORT_COLS} cap. "
            "Aggregate it before importing."
        )
    table_name = _unique_name(store, (name or _name_from_url(url)).strip() or "imported")
    store.add_table(frame, name=table_name, source=fetched.url, sha256=fetched.sha256, primary=True)
    store.reset_analysis()
    profile = roles.infer_profile(frame)
    summary = TableSummary(
        name=table_name,
        source=fetched.url,
        sha256=fetched.sha256,
        n_rows=len(frame),
        columns=list(frame.columns),
        suggested=dict(profile.suggested),
        profile_notes=list(profile.notes),
        next=(
            f"Call frame_question(question='...') to state the estimand, then "
            f"profile_table(table='{table_name}') to check schema and data quality."
        ),
    )
    store.put(
        summary,
        kind="table",
        evidence_class="descriptive",
        summary=f"Loaded {table_name}: {len(frame):,} rows from {fetched.url}",
        rows_total=len(frame),
        rows_used=len(frame),
    )
    return summary


def analysis_status() -> AnalysisStatus:
    """The ledger as a checklist. Call this before writing a conclusion.

    It returns the stage reached, the checks still missing, the blocking issues
    with their remedies and `recommended_next`, and writes no stage itself.
    """
    return current_session().state.status()


def _infer_target(question: str) -> Target:
    """The question's target from its wording, most specific keyword first."""
    lowered = question.lower()
    groups: list[tuple[Target, tuple[str, ...]]] = [
        (
            "causal",
            (
                "cause",
                "causes",
                "caused",
                "drives",
                "driven",
                "impact of",
                "impact on",
                "effect of",
                "effect on",
                "leads to",
                "lead to",
                "due to",
            ),
        ),
        (
            "comparison",
            ("differ", "difference", "compare", "comparison", "between", "versus", " vs "),
        ),
        ("prediction", ("predict", "forecast", "will ")),
        (
            "description",
            (
                "how many",
                "how much",
                "top ",
                "what is",
                "what are",
                "which is",
                "distribution",
                "count of",
            ),
        ),
    ]
    for target, keywords in groups:
        if any(keyword in lowered for keyword in keywords):
            return target
    return "association"


def _classify_design(
    frame: pd.DataFrame,
    unit: str | None,
    time_col: str | None,
    group_col: str | None,
) -> tuple[str, str]:
    """The design and the unit it is measured on, in the contract's order."""
    if unit is not None and bool(frame[unit].duplicated().any()):
        return "paired", unit
    if time_col is not None and unit is not None:
        return "repeated", unit
    if group_col is not None and int(frame[group_col].value_counts().max()) > 1:
        return "clustered", group_col
    return "independent", unit or "row"


def _supports_causal(
    design: str,
    time_col: str | None,
    treatment_col: str | None,
) -> bool:
    """A causal target needs an assigned exposure or a time structure.

    A grouping column is not enough. Clustered data with no treatment and no
    time is observational: "revenue by region" is a comparison, not an effect,
    and the contract refuses causal language there. The earlier version accepted
    any non-independent design, which let a group column license a causal claim.
    """
    if treatment_col is not None:
        return True
    return time_col is not None or design in ("paired", "repeated")


def _treatment_column(frame: pd.DataFrame) -> str | None:
    """A column whose name marks an assigned exposure rather than a measured one."""
    for name in frame.columns:
        lowered = str(name).lower()
        parts = lowered.split("_")
        for hint in TREATMENT_NAMES:
            if hint == lowered or hint in parts:
                return str(name)
    return None


def _select_predictors(
    profile: roles.DataProfile, outcome: str, unit_of_analysis: str
) -> list[str]:
    """The columns a model would use, in role order, minus the unit and the outcome."""
    excluded = {outcome, unit_of_analysis}
    excluded.update(profile.columns("identifier"))
    excluded.update(profile.columns("leakage"))
    ordered: list[str] = []
    for role in ("factor", "dose", "numeric"):
        for name in profile.columns(role):
            if name not in excluded and name not in ordered:
                ordered.append(name)
    return ordered


def _outcome_type(series: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(series):
        return "categorical"
    distinct = int(series.nunique(dropna=True))
    if distinct == 2:
        return "binary"
    if pd.api.types.is_integer_dtype(series):
        return "count"
    return "continuous"


def _estimand(target: Target, outcome: str, predictors: list[str]) -> str:
    """The quantity being estimated, as one sentence the plan can quote."""
    if target == "description":
        return f"the distribution of {outcome}"
    if target == "comparison":
        factor = predictors[0] if predictors else "the groups"
        return f"the difference in mean {outcome} between levels of {factor}"
    if target == "prediction":
        return f"the expected {outcome} given {_listing(predictors)}"
    if target == "causal":
        treatment = predictors[0] if predictors else "the exposure"
        return f"the causal effect of {treatment} on {outcome}"
    return f"the association between {outcome} and {_listing(predictors)}"


def _listing(names: list[str]) -> str:
    return ", ".join(names) if names else "the other columns"


def _required_checks(target: Target, design: str, unit_of_analysis: str) -> list[str]:
    checks = [
        "missingness rate per column",
        "duplicate rows at the unit of analysis",
        "zero-variance columns",
        "predictors associated with the outcome (leakage)",
    ]
    if target != "description":
        checks.append("assumption checks: residual structure, heteroscedasticity, influence")
    if design in {"paired", "repeated", "clustered"}:
        checks.append(f"within-{unit_of_analysis} correlation")
    if target == "causal":
        checks.append("unmeasured confounding (not verifiable from the data)")
    return checks


def _refusals(design: str, unit_of_analysis: str) -> list[str]:
    refusals: list[str] = []
    if design == "independent":
        refusals.append(
            "Causal language is not supported: the design is independent and there is "
            "no time column."
        )
    else:
        refusals.append(
            f"Rows within {unit_of_analysis} are not independent; do not read a row-level "
            "p-value as if they were."
        )
    return refusals


def _plan(
    table: str,
    target: Target,
    outcome: str,
    predictors: list[str],
    design: str,
    group_col: str | None,
    estimand: str,
) -> list[str]:
    """The ordered calls the UI renders as a checklist."""
    steps = [f"profile_table(table='{table}', outcome='{outcome}')"]
    group_arg = ""
    if group_col is not None and design in {"clustered", "repeated"}:
        group_arg = f", group='{group_col}'"
    if target == "description":
        return [*steps, "explain_result(result_id)", "analysis_status()"]
    if target == "comparison":
        factor = predictors[0] if predictors else outcome
        steps.append(f"compare_groups(response='{outcome}', factor='{factor}'{group_arg})")
        steps.append("marginal_effects(result_id)")
    else:
        terms = "[" + ", ".join(f"'{name}'" for name in predictors) + "]"
        steps.append(
            f"fit_model(estimand='{estimand}', response='{outcome}', terms={terms}{group_arg})"
        )
        if len(predictors) > 1:
            steps.append("marginal_effects(result_id)")
    return [*steps, "sensitivity(result_id)", "explain_result(result_id)", "analysis_status()"]


def _column(series: pd.Series, name: str, profile: roles.DataProfile) -> ColumnProfile:
    numeric = pd.api.types.is_numeric_dtype(series)
    present = series.dropna()
    guesses = [guess for guess in profile.guesses if guess.column == name]
    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        missing=int(series.isna().sum()),
        missing_rate=float(series.isna().mean()),
        distinct=int(series.nunique(dropna=True)),
        minimum=float(present.min()) if numeric and len(present) else None,
        maximum=float(present.max()) if numeric and len(present) else None,
        mean=float(present.mean()) if numeric and len(present) else None,
        roles=[guess.role for guess in guesses],
        reasons=[guess.reason for guess in guesses],
    )


def _missingness_verdicts(frame: pd.DataFrame) -> list[Verdict]:
    rates = {name: float(frame[name].isna().mean()) for name in frame.columns}
    over_warn = {name: rate for name, rate in rates.items() if rate > MISSING_WARN}
    if not over_warn:
        return [
            Verdict(
                check="missingness",
                status="ok",
                statement="No column is more than 5 percent missing.",
                severity="info",
            )
        ]
    listed = ", ".join(
        f"{name} {rate:.0%}" for name, rate in sorted(over_warn.items(), key=lambda kv: -kv[1])
    )
    over_block = {name: rate for name, rate in over_warn.items() if rate > MISSING_BLOCK}
    if over_block:
        worst = max(over_block, key=lambda name: over_block[name])
        return [
            Verdict(
                check="missingness",
                status="blocked",
                statement=f"Missingness over 40 percent: {listed}. Complete-case would "
                "drop most rows.",
                remedy=(
                    f"Choose between complete-case (drop rows missing {worst}) and "
                    f"imputation for {worst}, then call profile_table again."
                ),
                severity="blocking",
            )
        ]
    return [
        Verdict(
            check="missingness",
            status="caution",
            statement=f"Missing values above 5 percent: {listed}.",
            remedy="Decide whether to drop or impute the affected rows, then call "
            "profile_table again.",
            severity="warning",
        )
    ]


def _missingness_outcome_verdicts(frame: pd.DataFrame, outcome: str) -> list[Verdict]:
    """A missing rate is only worrying when the outcome differs across the gap."""
    series = frame[outcome]
    if not pd.api.types.is_numeric_dtype(series):
        return []
    verdicts: list[Verdict] = []
    for name in frame.columns:
        if name == outcome:
            continue
        column = frame[name]
        if not bool(column.isna().any()):
            continue
        present = series[column.notna()].dropna()
        absent = series[column.isna()].dropna()
        if len(present) < 2 or len(absent) < 2:
            continue
        try:
            _, p_value = stats.mannwhitneyu(present, absent, alternative="two-sided")
        except ValueError:
            continue
        if p_value < 0.05:
            verdicts.append(
                Verdict(
                    check=f"missingness associated with outcome: {name}",
                    status="caution",
                    statement=(
                        f"{name} is missing in {int(column.isna().sum())} rows and the "
                        f"outcome differs between present and absent rows ({fmt_p(p_value)}), "
                        "so the missingness is not random."
                    ),
                    remedy=(
                        f"Report or model the missingness in {name} rather than dropping "
                        f"those rows silently, then call profile_table(outcome='{outcome}') "
                        "again."
                    ),
                    severity="warning",
                )
            )
    return verdicts


def _duplicate_verdict(frame: pd.DataFrame) -> Verdict:
    duplicates = int(frame.duplicated().sum())
    rate = duplicates / len(frame) if len(frame) else 0.0
    if rate > DUPLICATE_BLOCK:
        return Verdict(
            check="duplicate rows",
            status="blocked",
            statement=f"{duplicates} of {len(frame)} rows ({rate:.0%}) are exact duplicates, "
            "which double-counts the same observation.",
            remedy="Drop the duplicates, or name the unit column that legitimately repeats "
            "and call profile_table again.",
            severity="blocking",
        )
    return Verdict(
        check="duplicate rows",
        status="ok",
        statement=f"{duplicates} duplicate rows ({rate:.0%}), at or below the 5 percent threshold.",
        severity="info",
    )


def _zero_variance_verdict(frame: pd.DataFrame) -> Verdict:
    constant = [name for name in frame.columns if frame[name].nunique(dropna=True) <= 1]
    if constant:
        return Verdict(
            check="zero-variance columns",
            status="caution",
            statement=f"Constant columns carry no information: {', '.join(constant)}.",
            remedy="Drop the constant columns from the terms, then call profile_table again.",
            severity="warning",
        )
    return Verdict(
        check="zero-variance columns",
        status="ok",
        statement="Every column varies.",
        severity="info",
    )


def _leakage_name_verdict(profile: roles.DataProfile) -> Verdict:
    flagged = [guess.column for guess in profile.guesses if guess.role == "leakage"]
    unique_flagged = list(dict.fromkeys(flagged))
    if unique_flagged:
        return Verdict(
            check="leakage name",
            status="caution",
            statement=f"{', '.join(unique_flagged)} may be recorded after the outcome.",
            remedy="Confirm each name was measured before the outcome, then call "
            "profile_table again.",
            severity="warning",
        )
    return Verdict(
        check="leakage name",
        status="ok",
        statement="No column name suggests a post-outcome field.",
        severity="info",
    )


def _leakage_association_verdict(frame: pd.DataFrame, outcome: str) -> Verdict:
    """Catch the predictor that encodes the outcome, which dominates any model."""
    flagged: list[tuple[str, str]] = []
    binary_outcome = int(frame[outcome].nunique(dropna=True)) == 2
    for name in frame.columns:
        if name == outcome:
            continue
        column = frame[name]
        numeric = pd.api.types.is_numeric_dtype(column)
        if numeric and pd.api.types.is_numeric_dtype(frame[outcome]):
            pair = frame[[name, outcome]].dropna()
            if len(pair) >= 3 and pair[name].std() > 0 and pair[outcome].std() > 0:
                correlation = float(pair[name].corr(pair[outcome]))
                if abs(correlation) > LEAKAGE_CORRELATION:
                    flagged.append((name, f"{name} has r={fmt(correlation, 3)} with {outcome}"))
        elif not numeric and binary_outcome:
            for level in column.dropna().unique():
                rows = frame[column == level]
                if len(rows) >= 2 and int(rows[outcome].nunique(dropna=True)) == 1:
                    value = rows[outcome].iloc[0]
                    flagged.append(
                        (
                            name,
                            f"{name}='{level}' holds only outcome {value} across {len(rows)} rows",
                        )
                    )
                    break
    if flagged:
        detail = "; ".join(description for _, description in flagged)
        return Verdict(
            check="leakage associated with outcome",
            status="blocked",
            statement="A predictor this close to the outcome is almost certainly recorded "
            "after it, so a model on it would predict the outcome from a copy of itself and "
            f"every other coefficient would be meaningless. Flagged: {detail}.",
            remedy=(
                f"Remove {flagged[0][0]} from the terms and refit; if it was genuinely "
                "measured before the outcome, record why and call profile_table again."
            ),
            severity="blocking",
        )
    return Verdict(
        check="leakage associated with outcome",
        status="ok",
        statement="No predictor is close enough to the outcome to encode it.",
        severity="info",
    )


def _imbalance_verdicts(frame: pd.DataFrame, profile: roles.DataProfile) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for name in profile.columns("factor"):
        counts = frame[name].value_counts(dropna=True)
        if len(counts) < 2:
            continue
        ratio = float(counts.max()) / float(counts.min())
        if ratio > IMBALANCE_RATIO:
            verdicts.append(
                Verdict(
                    check=f"imbalance: {name}",
                    status="caution",
                    statement=f"{name} levels are unbalanced {int(counts.max())} to "
                    f"{int(counts.min())} ({ratio:.1f} to 1).",
                    remedy=f"Report the level counts for {name} or weight the model, then call "
                    "profile_table again.",
                    severity="warning",
                )
            )
    if verdicts:
        return verdicts
    return [
        Verdict(
            check="imbalance",
            status="ok",
            statement="No factor level count is more than 4 to 1 against another.",
            severity="info",
        )
    ]


def _jsonable(value: Any) -> Any:
    """A row value as JSON-safe Python, because the store keeps JSON."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)
