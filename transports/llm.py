"""chatlas client and the registry tools, for the Agent panel.

`ChatOpenAICompletions`, not `ChatOpenAI`: the latter targets OpenAI's Responses
API, which Ollama, vLLM, LM Studio and Groq do not implement. Same defaults as
`agent.py`, so the CLI and the app talk to the same endpoint.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import chatlas
from chatlas import ContentToolResult

from stats import registry

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You answer questions about an agronomy trial: oat varieties in the whole "
    "plots, nitrogen rates in the subplots, replicated over blocks. Every number "
    "you report must come from a tool call: profile the data with describe_dataset "
    "first, then fit the model the question needs. Quote estimates with their "
    "confidence intervals and name the tool that produced them. When a tool "
    "returns an error, read the message and retry with corrected arguments."
)


def _as_tool(handler: Any) -> Any:
    """Wrap a stats tool so the model gets JSON and every result has one shape."""

    @functools.wraps(handler)
    def call(**kwargs: Any) -> ContentToolResult:
        result = handler(**kwargs)
        return ContentToolResult(value=result.model_dump(), model_format="json")

    return call


def build_client(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> chatlas.ChatOpenAICompletions:
    """An OpenAI-compatible client with every registered tool attached."""
    client = chatlas.ChatOpenAICompletions(
        model=model,
        base_url=base_url,
        api_key=api_key or "not-needed",
        system_prompt=system_prompt,
    )
    for spec in registry.TOOLS:
        # name= is required: with model=, chatlas names the tool after the model,
        # and registry builds those as "<tool>Request".
        client.register_tool(_as_tool(spec.handler), name=spec.name, model=spec.input_model)
    return client


def settings() -> tuple[str, str, str]:
    """Model, base URL and key from the environment, read once at startup."""
    return (
        os.environ.get("AGENTIC_STATS_LLM_MODEL", DEFAULT_MODEL),
        os.environ.get("AGENTIC_STATS_LLM_BASE_URL", DEFAULT_BASE_URL),
        os.environ.get("AGENTIC_STATS_LLM_API_KEY", ""),
    )
