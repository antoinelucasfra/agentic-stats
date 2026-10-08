"""chatlas client and the registry tools, for the Agent panel.

`ChatOpenAICompletions`, not `ChatOpenAI`: the latter targets OpenAI's Responses
API, which Ollama, vLLM, LM Studio and Groq do not implement. Same defaults as
`agent.py`, so the CLI and the app talk to the same endpoint.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any, Literal

import chatlas
from chatlas import ContentToolResult

from stats import registry

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class Preset:
    """One sidebar choice: which model, where, and whose key pays for it."""

    id: str
    label: str
    model: str
    base_url: str
    # "server": the host's AGENTIC_STATS_LLM_API_KEY, never shown in the page.
    # "own": the visitor types a key. "none": local endpoints take anything.
    key: Literal["server", "own", "none"]


PRESETS: tuple[Preset, ...] = (
    Preset(
        "demo",
        "Demo (no key needed)",
        "llama-3.1-8b-instant",
        GROQ_BASE_URL,
        "server",
    ),
    Preset("ollama-qwen", "Ollama: qwen2.5:7b", "qwen2.5:7b", DEFAULT_BASE_URL, "none"),
    Preset("ollama-llama", "Ollama: llama3.1:8b", "llama3.1:8b", DEFAULT_BASE_URL, "none"),
    Preset("groq", "Groq (your key)", "llama-3.1-8b-instant", GROQ_BASE_URL, "own"),
    Preset("openai", "OpenAI (your key)", "gpt-4o-mini", OPENAI_BASE_URL, "own"),
    Preset("custom", "Custom endpoint", "", "", "own"),
)

SYSTEM_PROMPT = (
    "You answer questions about an agronomy trial: oat varieties in the whole "
    "plots, nitrogen rates in the subplots, replicated over blocks. Every number "
    "you report must come from a tool call: profile the data with describe_dataset "
    "first, then fit the model the question needs. Quote estimates with their "
    "confidence intervals and name the tool that produced them. When a tool "
    "returns an error, read the message and retry with corrected arguments. "
    "Repeat a tool's notes verbatim when they warn you: an EC50 or peak outside "
    "the tested dose range is an extrapolation, not a finding. A non-significant "
    "p-value means the trial could not resolve the effect; check power_analysis "
    "before saying two varieties are equal. A low achieved power means the "
    "design is underpowered, not that there is no effect."
)


@dataclass(frozen=True)
class TraceEvent:
    """One tool call, as the Agent panel shows it after the turn."""

    name: str
    arguments: dict[str, Any]
    summary: str
    is_error: bool = False


def preset_by_id(preset_id: str) -> Preset:
    """Look a preset up, falling back to the custom endpoint."""
    for preset in PRESETS:
        if preset.id == preset_id:
            return preset
    return PRESETS[-1]


def preset_default() -> str:
    """The demo preset when the host holds a key, else local Ollama."""
    return "demo" if settings()[2] else "ollama-qwen"


def resolve(
    preset_id: str, model: str = "", base_url: str = "", api_key: str = ""
) -> tuple[str, str, str]:
    """Preset defaults plus typed-in overrides, as a concrete endpoint triple."""
    preset = preset_by_id(preset_id)
    key = api_key.strip()
    if preset.key == "server" and not key:
        key = settings()[2]
        if not key:
            raise ValueError(
                "The demo preset needs AGENTIC_STATS_LLM_API_KEY set where the app "
                "runs. Locally, pick an Ollama preset instead."
            )
    return (model.strip() or preset.model, base_url.strip() or preset.base_url, key)


def _record_request(trace, pending, event) -> None:
    """Remember a tool call; its result arrives in `_record_result`."""
    key = getattr(event, "id", None) or id(event)
    pending[key] = (event.name, dict(event.arguments or {}))


def _summarize(event) -> tuple[str, bool]:
    if getattr(event, "error", None) is not None:
        return f"ERROR: {event.error}", True
    try:
        text = event.get_model_value()
    except Exception:  # pragma: no cover - defensive: value shapes vary
        text = str(getattr(event, "value", ""))
    flat = " ".join(str(text).split())
    return (flat[:160] + "\u2026") if len(flat) > 160 else flat, False


def _record_result(trace, pending, event) -> None:
    """Pair a tool result with its request and append one trace line."""
    request = getattr(event, "request", None)
    key = getattr(request, "id", None) if request is not None else None
    if key in pending:
        name, arguments = pending.pop(key)
    elif pending:
        name, arguments = pending.pop(next(iter(pending)))
    else:
        name, arguments = "?", {}
    summary, is_error = _summarize(event)
    trace.append(TraceEvent(name, arguments, summary, is_error))


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
    trace: list[TraceEvent] | None = None,
) -> chatlas.ChatOpenAICompletions:
    """An OpenAI-compatible client with every registered tool attached.

    Pass a `trace` list and each tool call appends one `TraceEvent`, which the
    Agent panel renders after the turn. Tool errors already reach the model:
    chatlas converts a raising tool into an error result and feeds it back.
    """
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
    if trace is not None:
        pending: dict[Any, tuple[str, dict[str, Any]]] = {}
        client.on_tool_request(lambda event: _record_request(trace, pending, event))
        client.on_tool_result(lambda event: _record_result(trace, pending, event))
    return client


def settings() -> tuple[str, str, str]:
    """Model, base URL and key from the environment, read once at startup."""
    return (
        os.environ.get("AGENTIC_STATS_LLM_MODEL", DEFAULT_MODEL),
        os.environ.get("AGENTIC_STATS_LLM_BASE_URL", DEFAULT_BASE_URL),
        os.environ.get("AGENTIC_STATS_LLM_API_KEY", ""),
    )
