"""Eval harness: seven scripted agent sessions scored on tools and language.

The benchmark fakes the chat side and runs the real tools; this file reuses that
pattern at the session level. Each case builds a fresh store, pre-records the
design and quality stages a real session holds after framing and profiling, then
drives the agent loop with canned turns that call the real registry handlers.
The gate is four asserts per case: the expected tools ran, the first call was
the right one, every expected figure appears in the tool output, and the verify
pass returns the wanted verdict. The three printed lines are the scoreboard.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stats import registry, session, stats_tools, verify
from transports.agent import run_tool_loop


@dataclass(frozen=True)
class Case:
    id: str
    frame: str  # "oats" or "synthetic"
    question: str
    script: list[dict]  # canned turns: {"tool": name, "args": {...}} or {"text": "..."}
    expect_tools: set[str]  # must be a subset of the tools the loop called
    expect_first: set[str]  # the first call must be one of these
    expect_figures: list[str]  # each must appear in the joined serialized tool results
    final_answer: str  # canned prose checked by the verify pass
    want_verdict: str = "clean"  # or "review"


def _binary_frame() -> pd.DataFrame:
    """150 rows where x moves a binary y: a real effect the fit must find."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-2.0, 2.0, 150)
    chance = 1.0 / (1.0 + np.exp(-(-0.5 + 1.5 * x)))
    return pd.DataFrame({"x": x, "y": (rng.random(150) < chance).astype(int)})


def _trend_frame() -> pd.DataFrame:
    """60 daily rows with a gentle slope: strong enough to see, weak enough to profile."""
    rng = np.random.default_rng(11)
    day = np.arange(60)
    return pd.DataFrame({"day": day, "value": 50.0 + 0.3 * day + rng.normal(0.0, 5.0, 60)})


_FRAMES: dict[str, Callable[[], pd.DataFrame]] = {
    "synthetic-binary-effect": _binary_frame,
    "synthetic-time-line": _trend_frame,
}

# The Marvellous mean, formatted the way the quotable set spells it (two
# decimals), so the SQL-trap claim quotes a figure the query result holds.
_MARV_MEAN = float(stats_tools.load_dataset().query("variety == 'marvellous'").grain.mean())
_MARV_STR = f"{_MARV_MEAN:.2f}"

CASES: list[Case] = [
    Case(
        id="oats-profile-first",
        frame="oats",
        question="What does this dataset actually contain?",
        script=[{"tool": "profile_table", "args": {}}],
        expect_tools={"profile_table"},
        expect_first={"profile_table"},
        expect_figures=["72"],
        final_answer="The table holds 72 rows.",
    ),
    Case(
        id="oats-honest-negative",
        frame="oats",
        question="Is there a difference in yield between varieties?",
        script=[
            {
                "tool": "frame_question",
                "args": {
                    "question": "Is there a difference in yield between varieties?",
                    "outcome": "grain",
                },
            },
            {"tool": "profile_table", "args": {"outcome": "grain"}},
            {"tool": "compare_groups", "args": {"response": "grain", "factor": "variety"}},
        ],
        expect_tools={"compare_groups"},
        expect_first={"frame_question"},
        expect_figures=["variety", "Marvellous"],
        # The omnibus p (0.2993) sits in a notes string, which the quotable
        # set never walks, so the claim quotes a contrast p instead.
        final_answer="The varieties do not differ reliably (pairwise p 0.4990); "
        "the design cannot settle it.",
    ),
    Case(
        id="oats-causal-refusal",
        frame="oats",
        question="How does nitrogen relate to grain yield?",
        script=[
            {
                "tool": "frame_question",
                "args": {
                    "question": "How does nitrogen relate to grain yield?",
                    "outcome": "grain",
                },
            },
            {"tool": "profile_table", "args": {"outcome": "grain"}},
        ],
        expect_tools={"frame_question"},
        expect_first={"frame_question"},
        expect_figures=["grain"],
        final_answer="Nitrogen causes higher yield.",
        want_verdict="review",
    ),
    Case(
        id="oats-sql-trap",
        frame="oats",
        question="Which variety yields most?",
        script=[
            {
                "tool": "run_sql",
                "args": {
                    "query": 'select variety, avg(grain) as m from "primary" group by variety'
                },
            }
        ],
        expect_tools={"run_sql"},
        expect_first={"run_sql"},
        expect_figures=["marvellous", _MARV_STR],
        final_answer=f"Marvellous raises yield by {_MARV_STR} units.",
        want_verdict="review",
    ),
    Case(
        id="synthetic-binary-effect",
        frame="synthetic",
        question="Does x predict the outcome?",
        script=[
            {"tool": "frame_question", "args": {"question": "Does x predict y?", "outcome": "y"}},
            {"tool": "profile_table", "args": {"outcome": "y"}},
            {
                "tool": "fit_model",
                "args": {
                    "estimand": "the association between y and x",
                    "response": "y",
                    "terms": ["x"],
                    "family": "binomial",
                },
            },
        ],
        expect_tools={"fit_model"},
        expect_first={"frame_question"},
        expect_figures=["3.6551", "2.4008"],
        final_answer="Higher x goes with higher odds of y: odds ratio 3.6551 "
        "(95% interval 2.4008 to 5.5648).",
    ),
    Case(
        id="synthetic-time-line",
        frame="synthetic",
        question="How does value move over time?",
        script=[
            {
                "tool": "frame_question",
                "args": {"question": "How does value move over day?", "outcome": "value"},
            },
            {"tool": "profile_table", "args": {"outcome": "value"}},
            {"tool": "plot", "args": {"kind": "line", "x": "day", "y": "value"}},
        ],
        expect_tools={"plot"},
        expect_first={"frame_question"},
        expect_figures=["0.2902", "43.4634"],
        final_answer="The line drifts upward with slope 0.2902, from a low of 43.4634 "
        "to a high of 71.9282.",
    ),
    Case(
        id="oats-full-chain",
        frame="oats",
        question="How do variety and nitrogen relate to grain?",
        script=[
            {
                "tool": "frame_question",
                "args": {
                    "question": "How do variety and nitrogen relate to grain?",
                    "outcome": "grain",
                },
            },
            {"tool": "profile_table", "args": {"outcome": "grain"}},
            {
                "tool": "fit_model",
                "args": {
                    "estimand": "grain from variety and nitrogen",
                    "response": "grain",
                    "terms": ["variety", "nitrogen"],
                },
            },
            {"tool": "sensitivity", "args": {"result_id": "$last"}},
            {"tool": "explain_result", "args": {"result_id": "$last"}},
        ],
        expect_tools={"sensitivity", "explain_result"},
        expect_first={"frame_question"},
        expect_figures=["73.6667", "51.3336"],
        final_answer="Nitrogen goes with higher yield, about 73.6667 per unit "
        "(95% interval 51.3336 to 95.9997).",
    ),
]


def _run_case(case: Case) -> tuple[list[str], str, verify.VerificationReport]:
    """Drive one case through the loop; return called tools, joined output, verdict."""
    frame = stats_tools.load_dataset().copy() if case.frame == "oats" else _FRAMES[case.id]()
    store = session.SessionStore.from_frame(frame, source=f"eval:{case.id}")
    # What a real session holds after framing and profiling; the scripted tools
    # overwrite these records when they run.
    store.state.record("design", note="eval pre-record")
    store.state.record("quality", note="eval pre-record")

    replies: list[dict] = []
    for step in case.script:
        if "tool" in step:
            replies.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": step["tool"], "arguments": step.get("args", {})}}
                    ],
                }
            )
        else:
            replies.append({"role": "assistant", "content": step["text"]})
    replies.append({"role": "assistant", "content": case.final_answer})

    called: list[str] = []

    async def chat(messages: list[dict], tools: list[dict]) -> dict:
        return replies[len(called)]

    async def watching(name: str, arguments: dict) -> str:
        called.append(name)
        # "$last" names the result the previous turn stored, which the script
        # cannot know before the loop runs.
        resolved = {
            key: (store.ids()[-1] if value == "$last" else value)
            for key, value in arguments.items()
        }
        result = registry.get(name).handler(**resolved)
        return json.dumps(result.model_dump())

    with session.use_session(store):
        _answer, trace = asyncio.run(run_tool_loop(case.question, [], chat, watching))
        report = verify.verify_answer(case.final_answer, store)
    blob = "".join(result for step in trace for result in step.tool_results)
    return called, blob, report


def test_eval() -> None:
    passed = 0
    first_ok = 0
    total_calls = 0
    for case in CASES:
        called, blob, report = _run_case(case)
        assert set(case.expect_tools) <= set(called), f"{case.id}: {called}"
        assert called[0] in case.expect_first, f"{case.id}: {called}"
        for figure in case.expect_figures:
            # Level names are stored lowercase, so the match ignores case;
            # numeric figures are unaffected by that.
            assert figure.lower() in blob.lower(), f"{case.id}: {figure!r} not in output"
        assert report.verdict == case.want_verdict, f"{case.id}: {report}"
        if case.id == "oats-sql-trap":
            # The headline rule: an effect claim on a descriptive number must name it.
            assert report.misclassified, f"{case.id}: {report}"
        passed += 1
        first_ok += called[0] in case.expect_first
        total_calls += len(called)
    print(f"eval cases passed: {passed}/{len(CASES)}")
    print(f"first-tool accuracy: {first_ok}/{len(CASES)}")
    print(f"total tool calls: {total_calls}")
