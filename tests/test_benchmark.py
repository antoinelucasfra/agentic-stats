"""Benchmark: eight scripted questions asserting the minimum tool path.

The chat side is faked with pre-scripted turns, so no model and no network are
involved; the tool side runs the real registry handlers against the oats fixture
(conftest points AGENTIC_STATS_DATA at it). What is asserted is the teaching
contract: the right tool gets called, its output carries the guidance a small
model needs, and a tool error recovers instead of ending the answer.

Real-model variant: set AGENTIC_STATS_BENCH_MODEL to run Q1-Q8 against a live
endpoint. That path is opt-in and informational only.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from stats import registry
from transports.agent import run_tool_loop

REAL_MODEL = os.environ.get("AGENTIC_STATS_BENCH_MODEL", "")


def tool_call(name: str, **arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }


async def call_tool(name: str, arguments: dict) -> str:
    """The real tool, serialized the way the MCP server would send it."""
    result = registry.get(name).handler(**arguments)
    return json.dumps(result.model_dump())


def run_scripted(question: str, replies: list[dict]):
    """Drive the loop with canned assistant turns; return answer, trace, calls."""
    calls: list[tuple[str, dict]] = []

    async def chat(messages, tools):
        return replies[len(calls)]

    async def watching(name: str, arguments: dict) -> str:
        calls.append((name, arguments))
        return await call_tool(name, arguments)

    answer, trace = asyncio.run(run_tool_loop(question, [], chat, watching))
    return answer, trace, [name for name, _ in calls]


def results(trace) -> list[str]:
    return [result for step in trace for result in step.tool_results]


def test_q1_profile_first():
    _answer, trace, names = run_scripted(
        "What does this dataset actually contain?",
        [tool_call("describe_dataset"), {"content": "72 rows, block and variety."}],
    )

    assert names == ["describe_dataset"]
    assert '"n_rows": 72' in trace[0].tool_results[0]
    assert "Suggested flow" in trace[0].tool_results[0]


def test_q2_variety_difference_is_not_significant():
    _answer, trace, names = run_scripted(
        "Is there a difference in yield between varieties?",
        [
            tool_call("describe_dataset"),
            tool_call("anova_effect", response="grain", factor="variety"),
            {"content": "No: p 0.30, omega squared 0.006."},
        ],
    )

    assert names == ["describe_dataset", "anova_effect"]
    payload = json.loads(trace[1].tool_results[0])
    assert payload["p_value"] > 0.05
    assert payload["omega_squared"] < 0.01


def test_q3_split_plot_needs_the_mixed_model():
    _answer, trace, names = run_scripted(
        "Does variety matter once the blocks are accounted for?",
        [
            tool_call("describe_dataset"),
            tool_call(
                "fit_mixed_model",
                response="grain",
                fixed_effects=["variety"],
                group="block",
            ),
            {"content": "Still not significant; the trial is underpowered for variety."},
        ],
    )

    assert names == ["describe_dataset", "fit_mixed_model"]
    payload = json.loads(trace[1].tool_results[0])
    assert payload["icc"] > 0.2
    assert any("power_analysis" in note for note in payload["notes"])


def test_q4_nitrogen_effect_with_confidence_interval():
    _answer, trace, names = run_scripted(
        "Does nitrogen increase yield?",
        [
            tool_call("describe_dataset"),
            tool_call(
                "fit_mixed_model",
                response="grain",
                fixed_effects=["nitrogen"],
                group="block",
            ),
            {"content": "Yes: +73.7 per unit nitrogen, CI inside the tool output."},
        ],
    )

    assert names == ["describe_dataset", "fit_mixed_model"]
    payload = json.loads(trace[1].tool_results[0])
    nitrogen = next(e for e in payload["fixed_effects"] if e["term"] == "nitrogen")
    assert nitrogen["p_value"] < 1e-10
    assert nitrogen["ci_low"] < nitrogen["estimate"] < nitrogen["ci_high"]


def test_q5_dose_response_flags_extrapolation():
    _answer, trace, names = run_scripted(
        "Does the yield saturate with nitrogen?",
        [
            tool_call("dose_response", response="grain", dose="nitrogen", by="variety"),
            {"content": "Cannot tell: the EC50 is outside the tested range."},
        ],
    )

    assert names == ["dose_response"]
    assert any("extrapolation" in result for result in results(trace))


def test_q6_power_shortfall_is_named_underpowered():
    _answer, trace, names = run_scripted(
        "How big should the trial be to settle variety?",
        [
            tool_call("power_analysis", response="grain", factor="variety"),
            {"content": "Underpowered: need 96 per variety, have 24."},
        ],
    )

    assert names == ["power_analysis"]
    assert any("Underpowered" in result for result in results(trace))


def test_q7_typo_recovers_through_the_tool_error():
    _answer, trace, names = run_scripted(
        "Fit the model on grain by varety.",
        [
            tool_call(
                "fit_mixed_model",
                response="grain",
                fixed_effects=["varety"],
                group="block",
            ),
            tool_call(
                "fit_mixed_model",
                response="grain",
                fixed_effects=["variety"],
                group="block",
            ),
            {"content": "Typo fixed: variety, not varety."},
        ],
    )

    assert names == ["fit_mixed_model", "fit_mixed_model"]
    first, second = trace[0].tool_results[0], trace[1].tool_results[0]
    assert first.startswith("ERROR")
    assert "Available columns" in first
    assert not second.startswith("ERROR")


def test_q8_synthesis_combines_means_and_power():
    _answer, trace, names = run_scripted(
        "Should we scale up Golden Rain?",
        [
            tool_call("describe_dataset"),
            tool_call(
                "marginal_means",
                response="grain",
                factors=["variety"],
                covariates=["nitrogen"],
            ),
            tool_call("power_analysis", response="grain", factor="variety"),
            {"content": "Means overlap and power is 0.26: no basis to pick a winner."},
        ],
    )

    assert names == ["describe_dataset", "marginal_means", "power_analysis"]
    assert all(not result.startswith("ERROR") for result in results(trace))


def test_new_tools_are_on_the_registry_surface():
    assert registry.get("fit_curve_family").name == "fit_curve_family"
    assert registry.get("correlate").name == "correlate"


@pytest.mark.skipif(not REAL_MODEL, reason="needs AGENTIC_STATS_BENCH_MODEL")
def test_live_model_reaches_a_tool():
    from transports import agent as agent_module

    _answer, trace = asyncio.run(agent_module.ask("What columns exist?", model=REAL_MODEL))
    assert any(step.tool_calls for step in trace)
