"""Tests for the agent loop: tool calling, tool errors, and turn limits.

Fakes only, so no LLM and no MCP process are needed here. The MCP wiring itself
is covered by test_mcp_server.py.
"""

from __future__ import annotations

import asyncio

import pytest

from transports.agent import Step, run_tool_loop

TOOLS = [{"name": "describe_dataset", "description": "profile", "parameters": {}}]


def tool_call(name: str, **arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
    }


def test_loop_calls_tool_then_returns_answer():
    replies = [tool_call("describe_dataset"), {"content": "Marvellous is the strongest variety."}]
    calls: list[tuple[str, dict]] = []

    async def chat(messages, tools):
        assert tools == TOOLS
        return replies[len(calls)]

    async def call_tool(name, arguments):
        calls.append((name, arguments))
        return '{"n_rows": 576}'

    answer, trace = asyncio.run(run_tool_loop("Which variety wins?", TOOLS, chat, call_tool))

    assert calls == [("describe_dataset", {})]
    assert answer == "Marvellous is the strongest variety."
    assert len(trace) == 2
    assert trace[0].tool_results == ['{"n_rows": 576}']


def test_tool_failure_is_fed_back_to_the_model_instead_of_raising():
    replies = [
        tool_call("fit_mixed_model", response="nope"),
        {"content": "That column does not exist; grain does."},
    ]
    seen: list[str] = []

    async def chat(messages, tools):
        return replies[len(seen)]

    async def call_tool(name, arguments):
        raise KeyError(arguments["response"])

    async def chat_and_watch(messages, tools):
        reply = await chat(messages, tools)
        seen.extend(m["content"] for m in messages if m["role"] == "tool")
        return reply

    answer, trace = asyncio.run(run_tool_loop("fit it", TOOLS, chat_and_watch, call_tool))

    assert "ERROR" in trace[0].tool_results[0]
    assert "does not exist" in answer
    assert any("fit_mixed_model" in content for content in seen)


def test_loop_gives_up_after_max_turns():
    async def chat(messages, tools):
        return tool_call("describe_dataset")

    async def call_tool(name, arguments):
        return "{}"

    with pytest.raises(RuntimeError, match="exceeded 2 turns"):
        asyncio.run(run_tool_loop("loop forever", TOOLS, chat, call_tool, max_turns=2))


def test_step_dataclass_defaults_are_independent():
    assert Step().tool_calls == [] and Step().tool_results == []
