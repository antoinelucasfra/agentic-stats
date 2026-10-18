"""The check that keeps a narrative attached to its evidence.

`verify_answer` reads a finished answer and asks two questions. Is every figure
in it present in a stored result? And does it claim causation when the design
stage did not permit it? Both are deterministic, so both are testable, and
together they turn "no invented numbers" from a promise into a check.

The allowance list is explicit rather than hidden. Prose legitimately contains
figures nobody computed, such as an interval level or a year, and excusing them
silently would make the report useless. They are listed, counted separately, and
reported, so a reader can see the size of the loophole instead of trusting it.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from stats import session
from stats.errors import ToolError

__all__ = ["VerificationReport", "stored_numbers", "ungrounded_numbers", "verify_answer"]

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")

# Phrases that assert a cause. Kept to the strong ones: a marginal effect is
# standard statistical vocabulary, so "effect of" lands in SOFT_TERMS instead.
CAUSAL_TERMS = (
    "cause",
    "causes",
    "caused",
    "causing",
    "causal",
    "leads to",
    "led to",
    "drives",
    "drove",
    "driven by",
    "results in",
    "resulted in",
    "because of",
    "due to",
    "attributable to",
)

# Attributional phrasing that is worth rewording when the design is observational.
SOFT_TERMS = ("effect of", "impact of", "responsible for", "explains")

# Words that make a numeric claim about an effect rather than a description. When
# one of these appears, the figures behind it have to come from an inferential
# result: a difference of means computed in SQL is a descriptive number, and a
# claim that rests on it is the case the verify pass exists to catch.
EFFECT_TERMS = (
    "difference",
    "differs",
    "higher",
    "lower",
    "increases",
    "decreases",
    "raises",
    "lowers",
    "reduces",
    "improves",
    "associated",
    "correlated",
    "predicts",
    "significant",
    "effect",
    "impact",
)

# Structural figures that appear in prose without being computed: an alpha, an
# interval level, a power target, a scale bound. Everything here is a known
# loophole, so it is counted in the report rather than hidden.
ALLOWED = frozenset(
    {
        "0",
        "0.01",
        "0.05",
        "0.1",
        "0.2",
        "0.5",
        "0.8",
        "0.9",
        "0.95",
        "0.99",
        "0.001",
        "1",
        "1.5",
        "2",
        "2.5",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "15",
        "20",
        "25",
        "30",
        "40",
        "50",
        "60",
        "70",
        "75",
        "80",
        "90",
        "95",
        "99",
        "100",
        "1000",
    }
)

YEAR_LOW, YEAR_HIGH = 1900, 2100


class VerificationReport(BaseModel):
    """Whether an answer stayed attached to its evidence."""

    verdict: Literal["clean", "review"]
    statement: str
    grounded: list[str] = Field(default_factory=list)
    ungrounded: list[str] = Field(default_factory=list)
    misclassified: list[str] = Field(
        default_factory=list,
        description="Figures that exist, but only in a descriptive result, behind an effect claim",
    )
    excused: list[str] = Field(default_factory=list)
    causal_terms: list[str] = Field(default_factory=list)
    effect_terms: list[str] = Field(default_factory=list)
    soft_terms: list[str] = Field(default_factory=list)
    causal_claims_permitted: bool = False
    inferential_available: bool = False
    notes: list[str] = Field(default_factory=list)


def stored_numbers(store: session.SessionStore) -> set[str]:
    """Every figure a narrative is allowed to quote, across stored results."""
    allowed: set[str] = set()
    for result in store.results.values():
        allowed.update(result.quotable)
    return allowed


def _numbers_by_class(store: session.SessionStore) -> tuple[set[str], set[str]]:
    """Figures that resolve to an inferential result, and to a descriptive one."""
    inferential: set[str] = set()
    descriptive: set[str] = set()
    for result in store.results.values():
        target = inferential if result.evidence_class == "inferential" else descriptive
        target.update(result.quotable)
    return inferential, descriptive


def _normalize(token: str) -> str:
    """A written figure as the stored lists spell it."""
    text = token.replace(",", "")
    if text.startswith("+"):
        text = text[1:]
    if text.startswith("-0"):
        text = "-" + text.lstrip("-0")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _excused(token: str) -> bool:
    if token in ALLOWED:
        return True
    try:
        value = float(token.replace(",", ""))
    except ValueError:
        return False
    return float(YEAR_LOW) <= value <= float(YEAR_HIGH) and value.is_integer()


def ungrounded_numbers(
    text: str, store: session.SessionStore
) -> tuple[list[str], list[str], list[str]]:
    """Split the figures in `text` into grounded, ungrounded and excused."""
    allowed = stored_numbers(store)
    grounded: list[str] = []
    ungrounded: list[str] = []
    excused: list[str] = []
    for match in NUMBER.finditer(text):
        token = match.group(0)
        if token in allowed or _normalize(token) in allowed:
            grounded.append(token)
        elif _excused(token):
            excused.append(token)
        elif token not in ungrounded:
            ungrounded.append(token)
    return grounded, ungrounded, excused


def _find_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered)]


def _causal_permitted(store: session.SessionStore) -> bool:
    """Causal language needs a design stage that accepted a causal target."""
    record = store.state.records.get("design")
    if record is None or not record.result_id:
        return False
    try:
        payload = store.get(record.result_id).payload
    except ToolError:
        return False
    return str(getattr(payload, "target", "")) == "causal"


def verify_answer(text: str, store: session.SessionStore) -> VerificationReport:
    """Check an answer's figures and its language against the store.

    Call it on the finished text, before showing it. `verdict` is `review` when a
    figure appears in no stored result, when an effect claim rests only on a
    descriptive result, or when causal language appears in an answer whose design
    stage never permitted it.
    """
    grounded, ungrounded, excused = ungrounded_numbers(text, store)
    inferential, descriptive = _numbers_by_class(store)
    permitted = _causal_permitted(store)
    causal = _find_terms(text, CAUSAL_TERMS)
    effect = _find_terms(text, EFFECT_TERMS)
    soft = _find_terms(text, SOFT_TERMS)

    misclassified = (
        [token for token in grounded if token not in inferential and token in descriptive]
        if (effect or causal)
        else []
    )

    notes: list[str] = []
    if ungrounded:
        notes.append(
            f"Ungrounded figures: {ungrounded}. Quote a stored value or run the tool that "
            "computes it."
        )
    if misclassified:
        notes.append(
            f"These figures come only from a descriptive result: {misclassified}. A claim "
            "about an effect cites fit_model, compare_groups or sensitivity."
        )
    if (effect or causal) and not inferential:
        notes.append(
            "The answer claims an effect but no inferential result is stored. Run "
            "fit_model or compare_groups before making the claim."
        )
    if causal and not permitted:
        notes.append(
            f"Causal language without a causal design: {causal}. Call frame_question() with "
            "a causal target and let it refuse or name the design that would support it."
        )
    if soft and not permitted:
        notes.append(f"Attributional phrasing worth rewording: {soft}.")
    if excused:
        notes.append(f"Excused as structural: {excused}.")

    clean = (
        not ungrounded
        and not misclassified
        and (not causal or permitted)
        and (not (effect or causal) or bool(inferential))
    )
    statement = (
        f"{len(grounded)} figures traced to stored results"
        + (f", {len(ungrounded)} did not" if ungrounded else "")
        + (f", {len(misclassified)} came only from a descriptive result" if misclassified else "")
        + ("; causal language present" if causal and permitted else "")
    )
    return VerificationReport(
        verdict="clean" if clean else "review",
        statement=statement + ".",
        grounded=grounded,
        ungrounded=ungrounded,
        misclassified=misclassified,
        excused=excused,
        causal_terms=causal,
        effect_terms=effect,
        soft_terms=soft,
        causal_claims_permitted=permitted,
        inferential_available=bool(inferential),
        notes=notes,
    )
