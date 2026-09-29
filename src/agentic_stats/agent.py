"""Minimal tool-calling agent: Ollama (local LLM) + MCP tools over stdio.

Deliberately dependency-light. The agent loop is `run_tool_loop`, which takes
callables for "talk to the model" and "run a tool", so tests drive it with fakes
and no network. `ask()` wires it to a real MCP session and a local Ollama server.

    uv run agentic-stats-agent "Does formulation matter, once batch is accounted for?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
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


async def ollama_chat(
    messages: list[dict],
    tools: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_URL,
) -> dict:
    """Chat backend against a local Ollama server, in OpenAI-style tool format."""
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "tools": [{"type": "function", "function": tool} for tool in tools],
                "stream": False,
            },
        )
    response.raise_for_status()
    return response.json()["message"]


async def _call_mcp_tool(session, name: str, arguments: dict) -> str:
    result = await session.call_tool(name, arguments)
    payload = "|".join(getattr(item, "text", "") for item in result.content)
    return f"ERROR: {payload}" if result.is_error else payload


async def ask(
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_URL,
    max_turns: int = 6,
) -> tuple[str, list[Step]]:
    """Run one question end to end against the MCP server over stdio."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "agentic_stats.mcp_server"])

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listing = await session.list_tools()
        tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            }
            for tool in listing.tools
        ]

        async def call_tool(name: str, arguments: dict) -> str:
            return await _call_mcp_tool(session, name, arguments)

        async def chat(messages: list[dict], specs: list[dict]) -> dict:
            return await ollama_chat(messages, specs, model=model, base_url=base_url)

        return await run_tool_loop(question, tools, chat, call_tool, max_turns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the DOE dataset")
    parser.add_argument(
        "question",
        help="e.g. 'Does formulation matter once batch is accounted for?'",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=OLLAMA_URL)
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    try:
        answer, trace = asyncio.run(
            ask(args.question, model=args.model, base_url=args.base_url, max_turns=args.max_turns)
        )
    except httpx.HTTPError as exc:
        sys.exit(f"Could not reach Ollama at {args.base_url}: {exc}. Start it with `ollama serve`.")

    for step in trace:
        for (name, arguments), result in zip(step.tool_calls, step.tool_results, strict=True):
            print(f"[tool] {name}({json.dumps(arguments)}) -> {result[:200]}")
    print(f"\n[answer] {answer}")


if __name__ == "__main__":
    main()
