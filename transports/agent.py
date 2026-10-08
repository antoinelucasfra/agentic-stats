"""Minimal tool-calling agent over any OpenAI-compatible chat endpoint.

Deliberately dependency-light. The agent loop is `run_tool_loop`, which takes
callables for "talk to the model" and "run a tool", so tests drive it with fakes
and no network. `ask()` wires it to a real MCP session and a chat endpoint.

Ollama is the default, but vLLM, llama.cpp server, LM Studio, Groq and OpenAI
all speak the same request shape, so one backend covers local and hosted models:

    uv run agentic-stats-agent "Does variety matter once block is accounted for?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"

Chat = Callable[[list[dict], list[dict]], Awaitable[dict]]
CallTool = Callable[[str, dict], Awaitable[str]]


@dataclass
class Step:
    """One turn of the loop, for printing and for tests."""

    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)


async def run_tool_loop(
    question: str,
    tools: list[dict],
    chat: Chat,
    call_tool: CallTool,
    max_turns: int = 6,
) -> tuple[str, list[Step]]:
    """Drive tool calling until the model answers without asking for more tools.

    `chat(messages, tools)` returns an assistant message that may carry a
    "tool_calls" list; `call_tool(name, arguments)` returns the tool output as
    text. Tool failures are fed back to the model instead of raising, so it can
    correct itself. Returns the final answer and the trace of turns.
    """
    messages: list[dict] = [{"role": "user", "content": question}]
    trace: list[Step] = []

    for _ in range(max_turns):
        message = await chat(messages, tools)
        messages.append(message)
        requested = [
            (call["function"]["name"], call["function"].get("arguments") or {})
            for call in message.get("tool_calls") or []
        ]
        step = Step(tool_calls=requested)
        for name, arguments in requested:
            try:
                result = await call_tool(name, arguments)
            except Exception as exc:  # shown to the model so it can recover
                result = f"ERROR: {type(exc).__name__}: {exc}"
            step.tool_results.append(result)
            messages.append({"role": "tool", "content": f"{name}: {result}"})
        trace.append(step)
        if not requested:
            return message.get("content", ""), trace

    raise RuntimeError(f"Agent exceeded {max_turns} turns without answering.")


async def openai_chat(
    messages: list[dict],
    tools: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
) -> dict:
    """Chat against an OpenAI-compatible endpoint, returning the assistant message."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "tools": [{"type": "function", "function": tool} for tool in tools],
                "stream": False,
            },
        )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


async def _call_mcp_tool(session, name: str, arguments: dict) -> str:
    result = await session.call_tool(name, arguments)
    payload = "|".join(getattr(item, "text", "") for item in result.content)
    return f"ERROR: {payload}" if result.is_error else payload


async def ask(
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    max_turns: int = 6,
) -> tuple[str, list[Step]]:
    """Run one question end to end against the MCP server over stdio."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # cwd: `mcp_server` is a top-level module here, so it must be importable.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "transports.mcp_server"],
        cwd=str(Path(__file__).resolve().parent),
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listing = await session.list_tools()
        tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            }
            for tool in listing.tools
        ]

        async def call_tool(name: str, arguments: dict) -> str:
            return await _call_mcp_tool(session, name, arguments)

        async def chat(messages: list[dict], specs: list[dict]) -> dict:
            return await openai_chat(
                messages, specs, model=model, base_url=base_url, api_key=api_key
            )

        return await run_tool_loop(question, tools, chat, call_tool, max_turns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the oats trial")
    parser.add_argument(
        "question",
        help="e.g. 'Does variety matter once block is accounted for?'",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.environ.get("AGENTIC_STATS_API_KEY"))
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    try:
        answer, trace = asyncio.run(
            ask(
                args.question,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                max_turns=args.max_turns,
            )
        )
    except httpx.HTTPError as exc:
        sys.exit(
            f"Could not reach {args.base_url}: {exc}. For Ollama, start it with `ollama serve`."
        )

    for step in trace:
        for (name, arguments), result in zip(step.tool_calls, step.tool_results, strict=True):
            print(f"[tool] {name}({json.dumps(arguments)}) -> {result[:200]}")
    print(f"\n[answer] {answer}")


if __name__ == "__main__":
    main()
