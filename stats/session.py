"""Per-session analysis state: tables, results and the ledger.

A tool reads the active session instead of a module-level dataset, so two
readers of the same app never share a frame, a result id or a stage. The tests
and the CLI scope a session with `use_session`; the Shiny app holds one per
connection.

The ledger is what makes an analysis complete rather than merely plausible. Six
stages, in order, each written by the tool that establishes it. Tools read the
ledger and refuse out of order, so a fit cannot be interpreted before its
assumptions were checked, and `analysis_status` reports what is still missing
instead of leaving it to a model to remember.

Results are stored behind an id and the model gets a summary. A twelve-level
marginal-means table does not belong in a context window, and a number a
narrative quotes can be traced to the call that produced it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

from stats.errors import ToolError
from utils.formatting import fmt

__all__ = [
    "STAGE_ORDER",
    "STAGE_TOOL",
    "AnalysisState",
    "AnalysisStatus",
    "EvidenceClass",
    "SessionStore",
    "Stage",
    "StageRecord",
    "StoredResult",
    "TableRef",
    "Verdict",
    "current_session",
    "default_session",
    "quotable_numbers",
    "reset_default_session",
    "set_default_session",
    "use_session",
]

Stage = Literal["design", "quality", "assumptions", "estimate", "sensitivity", "interpretation"]
Severity = Literal["ok", "caution", "blocked"]
Impact = Literal["info", "warning", "blocking"]
EvidenceClass = Literal["descriptive", "inferential"]

# Result kinds that make no inferential claim, so their figures are descriptive.
DESCRIPTIVE_KINDS = frozenset({"query", "table", "plot", "profile"})

# How many display strings one result may contribute to its quotable set. Large
# enough for a 200-row query, bounded so a big table cannot exhaust memory.
MAX_QUOTABLE = 2000

STAGE_ORDER: tuple[Stage, ...] = (
    "design",
    "quality",
    "assumptions",
    "estimate",
    "sensitivity",
    "interpretation",
)

# The tool that establishes each stage, so `recommended_next` names a call.
STAGE_TOOL: dict[Stage, str] = {
    "design": "frame_question",
    "quality": "profile_table",
    "assumptions": "fit_model",
    "estimate": "fit_model",
    "sensitivity": "sensitivity",
    "interpretation": "explain_result",
}

# Result ids stay readable: the prefix says what kind of thing was stored.
KIND_PREFIX: dict[str, str] = {
    "fit": "f",
    "comparison": "c",
    "marginal": "m",
    "table": "t",
    "plot": "p",
    "query": "q",
    "profile": "d",
    "explanation": "e",
    "sensitivity": "s",
    "plan": "a",
    "status": "x",
}


class Verdict(BaseModel):
    """One check's outcome, with the fix when it failed.

    `status` is what the check found. `severity` is how much it matters, and a
    `blocking` verdict stops an inference until the `remedy` is applied.
    """

    check: str = Field(description="What was checked")
    status: Severity = Field(description="Outcome of the check")
    statement: str = Field(description="One sentence a reader could quote")
    remedy: str = Field(default="", description="The tool call that resolves a failed check")
    severity: Impact = Field(default="info", description="How much the outcome matters")


class StageRecord(BaseModel):
    """One stage of the ledger: what established it, and what is still open."""

    stage: Stage
    status: Severity
    result_id: str | None = None
    note: str = ""
    missing: list[str] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)


class AnalysisStatus(BaseModel):
    """The ledger as a checklist, which is what `analysis_status` returns."""

    stages: list[StageRecord]
    complete: bool
    blocking: list[Verdict] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    recommended_next: str = ""


@dataclass
class AnalysisState:
    """The ledger: what has been established, and what is still open."""

    records: dict[str, StageRecord] = field(default_factory=dict)

    @property
    def verdicts(self) -> list[Verdict]:
        """Every verdict from the stages currently recorded.

        Verdicts live on their stage rather than in one growing list, so
        re-running a tool replaces its verdicts and a remedy can clear a block.
        An append-only list meant a `blocked` verdict outlived its own fix, and
        the ledger could never recover.
        """
        return [verdict for record in self.records.values() for verdict in record.verdicts]

    def record(
        self,
        stage: Stage,
        *,
        status: Severity = "ok",
        result_id: str | None = None,
        note: str = "",
        missing: list[str] | None = None,
        verdicts: list[Verdict] | None = None,
    ) -> StageRecord:
        """Write a stage, replacing whatever an earlier call wrote for it."""
        entry = StageRecord(
            stage=stage,
            status=status,
            result_id=result_id,
            note=note,
            missing=list(missing or []),
            verdicts=list(verdicts or []),
        )
        self.records[stage] = entry
        return entry

    def has(self, stage: Stage) -> bool:
        return stage in self.records

    def blocking(self) -> list[Verdict]:
        """Verdicts that stop an inference until their remedy is applied."""
        return [v for v in self.verdicts if v.status == "blocked" or v.severity == "blocking"]

    def missing(self) -> list[str]:
        """Every open item, stage by stage, in the order the stages run."""
        gaps = [f"{stage}: not started" for stage in STAGE_ORDER if stage not in self.records]
        for stage in STAGE_ORDER:
            entry = self.records.get(stage)
            if entry is not None:
                gaps.extend(f"{stage}: {item}" for item in entry.missing)
        return gaps

    def require(self, stage: Stage) -> None:
        """Refuse to run out of order, naming the call that unblocks."""
        for earlier in STAGE_ORDER:
            if earlier == stage:
                break
            if earlier not in self.records:
                raise ToolError(
                    f"{stage} needs the {earlier} stage first. "
                    f"Call {STAGE_TOOL[earlier]}() and retry."
                )
        blocked = self.blocking()
        if blocked:
            detail = "; ".join(
                f"{v.check}: {v.statement} Fix: {v.remedy or 'see the tool that reported it'}"
                for v in blocked
            )
            raise ToolError(f"Resolve this before continuing. {detail}")

    def recommended_next(self) -> str:
        """One tool call and why, for a model that has lost the thread."""
        for stage in STAGE_ORDER:
            if stage not in self.records:
                return f"{STAGE_TOOL[stage]}() to establish the {stage} stage."
        for stage in STAGE_ORDER:
            entry = self.records[stage]
            if entry.missing:
                return f"{STAGE_TOOL[stage]}() to close: {', '.join(entry.missing)}."
        return "Nothing blocking. Write the report from the stored results."

    def status(self) -> AnalysisStatus:
        stages = [
            self.records.get(stage, StageRecord(stage=stage, status="blocked", note="not started"))
            for stage in STAGE_ORDER
        ]
        blocking = self.blocking()
        missing = self.missing()
        return AnalysisStatus(
            stages=stages,
            complete=not missing and not blocking,
            blocking=blocking,
            missing=missing,
            recommended_next=self.recommended_next(),
        )


def _walk_numbers(value: Any) -> Iterator[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        yield float(value)
        return
    if isinstance(value, str):
        # Figures a tool wrote into a sentence are quotable too: a p-value that
        # lives only in a notes string is still the tool's number. Long strings
        # are skipped, because a base64 image is not prose, and digits glued to
        # letters are skipped, because a result id is not a figure.
        if len(value) > 200 or value.startswith("data:"):
            return
        for match in _FIGURE.finditer(value.replace(",", "")):
            yield float(match.group(0))
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_numbers(item)
        return
    if isinstance(value, list | tuple):
        for item in value:
            yield from _walk_numbers(item)


# A figure inside prose, for the string branch of `_walk_numbers`.
_FIGURE = re.compile(r"(?<![\w])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")


def quotable_numbers(payload: Any, *, limit: int | None = None) -> list[str]:
    """Display strings a narrative may quote, harvested from a result payload.

    Every number contributes several roundings, because a writer may reasonably
    say 104.5 or 104.50. That makes the list a superset, which is deliberate:
    the check it feeds exists to reject invented figures, not unusual rounding.
    A tool with a better list passes its own to `SessionStore.put`.

    `limit` defaults to no limit up to `MAX_QUOTABLE`. A 200-row query result
    holds far more figures than a small cap allows, and a figure missing from
    the list reads as invented to the verify pass.
    """
    if isinstance(payload, BaseModel):
        payload = payload.model_dump()
    cap = MAX_QUOTABLE if limit is None else min(limit, MAX_QUOTABLE)
    seen: list[str] = []
    for value in _walk_numbers(payload):
        for digits in (4, 3, 2, 1, 0):
            text = fmt(value, digits)
            if text != "n/a" and text not in seen:
                seen.append(text)
        if len(seen) >= cap:
            break
    return seen[:cap]


@dataclass(frozen=True)
class TableRef:
    """One loaded table, with where it came from."""

    name: str
    frame: pd.DataFrame
    source: str = ""
    path: Path | None = None
    sha256: str | None = None
    license_note: str = ""
    # Reader-confirmed roles, role to column. Empty means the profile's own
    # suggestion stands; a reader who knows the data overrules a guess here.
    roles: dict[str, str] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.frame)

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)

    def with_roles(self, roles: dict[str, str]) -> TableRef:
        """A copy of this reference with reader-confirmed roles."""
        return TableRef(
            name=self.name,
            frame=self.frame,
            source=self.source,
            path=self.path,
            sha256=self.sha256,
            license_note=self.license_note,
            roles=dict(roles),
        )


@dataclass
class StoredResult:
    """A tool's full output, kept behind an id so the model gets a summary."""

    result_id: str
    kind: str
    payload: BaseModel
    evidence_class: EvidenceClass = "inferential"
    summary: str = ""
    quotable: list[str] = field(default_factory=list)
    rows_total: int | None = None
    rows_used: int | None = None
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_payload(self) -> dict[str, Any]:
        """The stored payload as JSON, with its provenance beside it."""
        return {
            "result_id": self.result_id,
            "kind": self.kind,
            "evidence_class": self.evidence_class,
            "summary": self.summary,
            "created_at": self.created_at,
            "payload": self.payload.model_dump(),
        }


class SessionStore:
    """One reader's tables, results and ledger."""

    def __init__(self) -> None:
        self.tables: dict[str, TableRef] = {}
        self.primary: str | None = None
        self.results: dict[str, StoredResult] = {}
        self.state = AnalysisState()

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, **kwargs: Any) -> SessionStore:
        """A store holding one table: the tests, the CLI and an upload."""
        store = cls()
        store.add_table(frame, **kwargs)
        return store

    def add_table(
        self,
        frame: pd.DataFrame,
        *,
        name: str = "primary",
        source: str = "",
        path: Path | None = None,
        sha256: str | None = None,
        license_note: str = "",
        primary: bool = True,
    ) -> TableRef:
        """Load a table under a name. The first one loaded becomes primary."""
        if frame.empty:
            raise ToolError(f"Table '{name}' has no rows.")
        ref = TableRef(
            name=name,
            frame=frame,
            source=source,
            path=path,
            sha256=sha256,
            license_note=license_note,
        )
        self.tables[name] = ref
        if primary or self.primary is None:
            self.primary = name
        return ref

    def table(self, name: str | None = None) -> TableRef:
        """One loaded table, primary by default."""
        key = name or self.primary
        if key is None or key not in self.tables:
            loaded = ", ".join(self.tables) or "none"
            raise ToolError(f"No table named {key!r} is loaded. Loaded tables: {loaded}.")
        return self.tables[key]

    def frame(self, name: str | None = None) -> pd.DataFrame:
        """The rows of one loaded table, primary by default."""
        return self.table(name).frame

    def set_primary(self, name: str) -> TableRef:
        """Make a loaded table the primary one every tool runs against."""
        ref = self.table(name)  # raises, naming the loaded tables, when unknown
        self.primary = name
        return ref

    def refreshed(self) -> SessionStore:
        """A new object holding the same content, so reactive dependents re-read.

        `reactive.Value.set` skips an identical object, and the tools mutate the
        store in place, so a panel that needs the UI to notice a change replaces
        the object with this copy. Nothing is deep-copied except the table and
        result maps; the old object is discarded.
        """
        fresh = SessionStore()
        fresh.tables = dict(self.tables)
        fresh.primary = self.primary
        fresh.results = dict(self.results)
        fresh.state = self.state
        return fresh

    def reset_analysis(self) -> None:
        """Drop results and the ledger. New data invalidates the old analysis."""
        self.results = {}
        self.state = AnalysisState()

    def put(
        self,
        payload: BaseModel,
        *,
        kind: str,
        evidence_class: EvidenceClass | None = None,
        summary: str = "",
        quotable: list[str] | None = None,
        rows_total: int | None = None,
        rows_used: int | None = None,
        exclusions: list[dict[str, Any]] | None = None,
    ) -> StoredResult:
        """Store a result and return it with the id a narrative can cite.

        `evidence_class` follows from `kind` unless a caller overrides it, so a
        query cannot be stored as evidence by forgetting an argument.
        """
        stored = StoredResult(
            result_id=self.new_id(kind),
            kind=kind,
            payload=payload,
            evidence_class=evidence_class
            or ("descriptive" if kind in DESCRIPTIVE_KINDS else "inferential"),
            summary=summary,
            quotable=list(quotable) if quotable is not None else quotable_numbers(payload),
            rows_total=rows_total,
            rows_used=rows_used,
            exclusions=list(exclusions or []),
        )
        self.results[stored.result_id] = stored
        # Stamp the id onto the payload too: a tool returns `stored.payload`, and a
        # result the model cannot cite by id is a result it cannot use.
        if hasattr(payload, "result_id"):
            payload.result_id = stored.result_id
        return stored

    def get(self, result_id: str) -> StoredResult:
        """One stored result, with a message that lists the valid ids."""
        try:
            return self.results[result_id]
        except KeyError:
            raise ToolError(
                f"Unknown result id {result_id!r}. Stored ids: {self.ids() or 'none'}."
            ) from None

    def ids(self) -> list[str]:
        return list(self.results)

    def new_id(self, kind: str) -> str:
        """A fresh, readable id, retried so two results cannot collide."""
        prefix = KIND_PREFIX.get(kind, "r")
        while True:
            candidate = f"{prefix}_{uuid4().hex[:4]}"
            if candidate not in self.results:
                return candidate


_ACTIVE: ContextVar[SessionStore | None] = ContextVar("agentic_stats_session", default=None)
_DEFAULT: SessionStore | None = None
_FACTORY: Callable[[], SessionStore] | None = None
_KEY: Callable[[], str] | None = None
_KEY_VALUE: str | None = None


def set_default_session(
    factory: Callable[[], SessionStore],
    key: Callable[[], str] | None = None,
) -> None:
    """Tell the session layer how to build the environment dataset, once.

    `stats_tools` registers the loader at import, so this module never has to
    know about `utils.data` or the cache path. `key` names the dataset the
    cached store was built from; a different key rebuilds it, which is how an
    `AGENTIC_STATS_DATA` change or an edited file is picked up.
    """
    global _FACTORY, _KEY
    _FACTORY = factory
    _KEY = key


def reset_default_session() -> None:
    """Drop the process-wide store, so the next read rebuilds it."""
    global _DEFAULT, _KEY_VALUE
    _DEFAULT = None
    _KEY_VALUE = None


def default_session() -> SessionStore:
    """The store for the environment dataset, built on first use."""
    global _DEFAULT, _KEY_VALUE
    key = _KEY() if _KEY is not None else ""
    if _DEFAULT is None or key != _KEY_VALUE:
        _DEFAULT = _FACTORY() if _FACTORY is not None else SessionStore()
        _KEY_VALUE = key
    return _DEFAULT


def current_session() -> SessionStore:
    """The session in scope, or the environment one."""
    store = _ACTIVE.get()
    return store if store is not None else default_session()


@contextmanager
def use_session(store: SessionStore) -> Iterator[SessionStore]:
    """Make `store` the active session for the duration of the block."""
    token = _ACTIVE.set(store)
    try:
        yield store
    finally:
        _ACTIVE.reset(token)
