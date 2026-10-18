"""Which column plays which part, inferred from names, dtypes and cardinality.

The tools are generic, the arguments are not. `compare_groups` needs a factor, a
mixed model needs a grouping column, a curve fit needs a non-negative exposure.
Roles are the translation layer between a table nobody has seen before and a
tool with typed arguments.

Every guess carries its reason and a confidence, so a reader can see why a
column was called a factor and overrule it. Nothing here decides an analysis; it
decides which questions the table can answer, and says so when the answer is
none.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

__all__ = ["DataProfile", "Role", "RoleGuess", "infer_profile"]

Role = Literal["response", "factor", "numeric", "time", "group", "dose", "identifier", "leakage"]

MAX_FACTOR_LEVELS = 50
MAX_GROUP_LEVELS = 20
MAX_DOSE_LEVELS = 12

# A candidate this sure is worth its own column when a role picks a default.
STRONG_CONFIDENCE = 0.6

TIME_NAMES = (
    "date",
    "time",
    "timestamp",
    "datetime",
    "year",
    "month",
    "quarter",
    "week",
    "day",
    "period",
    "created",
    "updated",
    "started",
    "ended",
)
GROUP_NAMES = (
    "block",
    "batch",
    "lot",
    "site",
    "cluster",
    "subject",
    "patient",
    "participant",
    "school",
    "store",
    "farm",
    "plot",
    "household",
    "session",
    "group",
    "centre",
    "center",
    "hospital",
    "cohort",
)
DOSE_NAMES = (
    "dose",
    "dosage",
    "concentration",
    "conc",
    "exposure",
    "amount",
    "spend",
    "budget",
    "level",
    "intensity",
    "duration",
    "age",
)
ID_NAMES = ("id", "index", "idx", "key", "uuid", "guid", "ref", "reference", "code")
# Columns that are plausibly recorded after the outcome, which makes them
# predictors that would leak the answer.
LEAKAGE_NAMES = (
    "result",
    "final",
    "post",
    "after",
    "treated",
    "converted",
    "churned",
    "resolved",
    "flag",
    "completed",
    "success",
)
RESPONSE_NAMES = (
    "yield",
    "response",
    "outcome",
    "score",
    "price",
    "sales",
    "revenue",
    "value",
    "count",
    "rating",
    "mass",
    "weight",
    "height",
    "length",
    "temperature",
    "speed",
    "area",
    "profit",
    "spend",
    "amount",
    "quality",
    "target",
    "label",
)


class RoleGuess(BaseModel):
    """One column's claim to one role, with the evidence for it."""

    role: Role
    column: str
    confidence: float = Field(description="0 to 1, how much the name and shape agree")
    reason: str = Field(description="Why this column got this role")


class DataProfile(BaseModel):
    """What the table can answer, and which columns would answer it."""

    n_rows: int
    n_columns: int
    guesses: list[RoleGuess]
    suggested: dict[str, str] = Field(
        default_factory=dict, description="role to the column that best fills it"
    )
    notes: list[str] = Field(default_factory=list)

    def columns(self, role: Role) -> list[str]:
        """Every column that could fill this role, best first."""
        ranked = sorted(
            (g for g in self.guesses if g.role == role),
            key=lambda g: g.confidence,
            reverse=True,
        )
        return [g.column for g in ranked]

    def best(self, role: Role) -> str | None:
        """The column the profile would use for this role, if any."""
        return self.suggested.get(role)


def _matches(name: str, hints: tuple[str, ...]) -> str | None:
    """The hint a column name contains, if any."""
    lowered = name.lower()
    parts = lowered.split("_")
    for hint in hints:
        if hint == lowered or hint in parts:
            return hint
    return None


def _looks_like_time(series: pd.Series, name: str) -> tuple[float, str]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return 0.95, "the dtype is a datetime"
    hint = _matches(name, TIME_NAMES)
    if hint:
        return 0.6, f"the name contains '{hint}'"
    # pandas 3 gives text columns a string dtype, so test for non-numeric rather
    # than for `object`.
    if not pd.api.types.is_numeric_dtype(series):
        sample = series.dropna().astype(str).head(200)
        if len(sample) >= 5:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if float(parsed.notna().mean()) > 0.9:
                return 0.7, "the values parse as dates"
    return 0.0, ""


def _looks_like_identifier(series: pd.Series, name: str, n_rows: int) -> tuple[float, str]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return 0.0, ""
    hint = _matches(name, ID_NAMES)
    unique = int(series.nunique(dropna=True))
    ratio = unique / n_rows if n_rows else 0.0
    if hint and ratio > 0.5:
        return 0.85, f"the name contains '{hint}' and {ratio:.0%} of rows are distinct"
    # Only a non-numeric all-distinct column is an identifier on shape alone. An
    # integer column whose values are all distinct is a measurement: order
    # counts, prices in cents, elapsed days. Guarding only against floats left a
    # unique integer outcome tagged as an identifier, so it got no response role
    # and no model could be fitted to a table that answers questions fine.
    if ratio >= 0.999 and not pd.api.types.is_numeric_dtype(series):
        return 0.7, "every row is distinct"
    if hint:
        return 0.5, f"the name contains '{hint}'"
    return 0.0, ""


def infer_profile(frame: pd.DataFrame, overrides: dict[str, str] | None = None) -> DataProfile:
    """Guess the role of every column, with reasons and confidences.

    `overrides` is a reader-confirmed role to column map. A confirmed role wins
    over the guess, and a column that does not exist is reported in the notes
    rather than applied silently.
    """
    n_rows = len(frame)
    guesses: list[RoleGuess] = []

    for name in frame.columns:
        series = frame[name]
        numeric = pd.api.types.is_numeric_dtype(series)
        unique = int(series.nunique(dropna=True))

        # Time first: a datetime column is a time column, not an identifier,
        # even though every one of its values is distinct.
        time_guess = _looks_like_time(series, name)
        if time_guess[0]:
            guesses.append(
                RoleGuess(role="time", column=name, confidence=time_guess[0], reason=time_guess[1])
            )
            continue

        identifier = _looks_like_identifier(series, name, n_rows)
        if identifier[0]:
            guesses.append(
                RoleGuess(
                    role="identifier",
                    column=name,
                    confidence=identifier[0],
                    reason=identifier[1],
                )
            )
            continue

        # A numeric column with a handful of distinct values behaves like a
        # factor for every tool that groups rows.
        if not numeric or (unique <= 10 and pd.api.types.is_integer_dtype(series)):
            rows_per_level = n_rows / unique if unique else 0.0
            group_hint = _matches(name, GROUP_NAMES)
            if group_hint and 2 <= unique <= MAX_GROUP_LEVELS and rows_per_level >= 2:
                guesses.append(
                    RoleGuess(
                        role="group",
                        column=name,
                        confidence=0.85,
                        reason=(
                            f"the name contains '{group_hint}' with {unique} levels "
                            f"and {rows_per_level:.1f} rows each"
                        ),
                    )
                )
            elif 2 <= unique <= MAX_GROUP_LEVELS and rows_per_level >= 3 and unique <= n_rows / 3:
                guesses.append(
                    RoleGuess(
                        role="group",
                        column=name,
                        confidence=0.4,
                        reason=f"{unique} levels with {rows_per_level:.1f} rows each",
                    )
                )
            if unique <= MAX_FACTOR_LEVELS:
                guesses.append(
                    RoleGuess(
                        role="factor",
                        column=name,
                        confidence=0.9 if not numeric else 0.4,
                        reason=f"{unique} distinct values"
                        + ("" if not numeric else ", numeric with few levels"),
                    )
                )
        elif numeric:
            if float(series.min()) >= 0 and unique <= MAX_DOSE_LEVELS:
                dose_hint = _matches(name, DOSE_NAMES)
                guesses.append(
                    RoleGuess(
                        role="dose",
                        column=name,
                        confidence=0.8 if dose_hint else 0.35,
                        reason=(
                            f"the name contains '{dose_hint}'"
                            if dose_hint
                            else "non-negative with few distinct values"
                        )
                        + f", {unique} distinct values",
                    )
                )
            guesses.append(
                RoleGuess(
                    role="numeric",
                    column=name,
                    confidence=0.9,
                    reason=f"{unique} distinct numeric values",
                )
            )

        # Only a numeric column can be an outcome. A two-level text column is a
        # factor, however much it looks like a yes or no.
        if numeric:
            response_hint = _matches(name, RESPONSE_NAMES)
            if response_hint:
                guesses.append(
                    RoleGuess(
                        role="response",
                        column=name,
                        confidence=0.85,
                        reason=f"the name contains '{response_hint}'",
                    )
                )
            elif unique == 2:
                guesses.append(
                    RoleGuess(
                        role="response",
                        column=name,
                        confidence=0.6,
                        reason="two distinct values, so it can be a binary outcome",
                    )
                )
            elif unique > 10:
                guesses.append(
                    RoleGuess(
                        role="response",
                        column=name,
                        confidence=0.4,
                        reason="continuous numeric column",
                    )
                )

        leakage_hint = _matches(name, LEAKAGE_NAMES)
        if leakage_hint:
            guesses.append(
                RoleGuess(
                    role="leakage",
                    column=name,
                    confidence=0.5,
                    reason=(
                        f"the name contains '{leakage_hint}' and may be recorded after the outcome"
                    ),
                )
            )

    order = {name: index for index, name in enumerate(frame.columns)}
    suggested: dict[str, str] = {}
    excluded: set[str] = set()
    # Group first, so the nuisance column of a design does not also win the
    # factor role. A role only shares a column when no confident candidate of
    # its own is left, which keeps a one-categorical-column table usable.
    for role in ("group", "response", "factor", "dose", "time"):
        pool = [g for g in guesses if g.role == role]
        if not pool:
            continue
        fresh = [g for g in pool if g.column not in excluded]
        strong = [g for g in fresh if g.confidence >= STRONG_CONFIDENCE]
        chosen = _best(strong or pool, order)
        if chosen is not None:
            suggested[role] = chosen
            excluded.add(chosen)

    override_notes: list[str] = []
    for role, column in (overrides or {}).items():
        if role not in ("response", "factor", "group", "dose", "time"):
            override_notes.append(f"Ignored override {role}={column}: not a role.")
        elif column not in frame.columns:
            override_notes.append(
                f"Ignored override {role}={column}: no such column. Columns: {list(frame.columns)}."
            )
        else:
            suggested[role] = column
            override_notes.append(f"{role} confirmed as {column} by the reader.")

    return DataProfile(
        n_rows=n_rows,
        n_columns=len(frame.columns),
        guesses=guesses,
        suggested=suggested,
        notes=[*_notes(frame, guesses, suggested), *override_notes],
    )


def _best(candidates: list[RoleGuess], order: dict[str, int]) -> str | None:
    """The most confident candidate, earliest column first on a tie."""
    if not candidates:
        return None
    best = max(candidates, key=lambda g: (g.confidence, -order[g.column]))
    return best.column


def _notes(frame: pd.DataFrame, guesses: list[RoleGuess], suggested: dict[str, str]) -> list[str]:
    """What the profile would tell a reader before they pick a question."""
    notes: list[str] = []
    if "response" not in suggested:
        notes.append("No numeric response column was found, so no model can be fitted.")
    if "group" not in suggested:
        notes.append(
            "No grouping column was found, so mixed models and repeated-measures "
            "checks are unavailable."
        )
    if "time" not in suggested:
        notes.append("No time column was found, so trend and seasonality cannot be read.")

    leakage = [g.column for g in guesses if g.role == "leakage"]
    if leakage:
        notes.append(
            f"Possible leakage: {', '.join(leakage)} may be recorded after the outcome. "
            "Check it before using it as a predictor."
        )

    constant = [name for name in frame.columns if frame[name].nunique(dropna=True) <= 1]
    if constant:
        notes.append(f"Constant columns carry no information: {', '.join(constant)}.")

    missing = int(frame.isna().sum().sum())
    if missing:
        columns = int((frame.isna().sum() > 0).sum())
        notes.append(f"{missing} missing values across {columns} columns.")
    return notes
