"""chatlas client and the registry tools, for the Agent panel.

`ChatOpenAICompletions`, not `ChatOpenAI`: the latter targets OpenAI's Responses
API, which Ollama, vLLM, LM Studio and Groq do not implement. Same defaults as
`agent.py`, so the CLI and the app talk to the same endpoint.
"""

from __future__ import annotations

import asyncio
import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import chatlas
from chatlas import ContentToolResult

from stats import registry

if TYPE_CHECKING:
    from shinychat.types import ToolResultDisplay

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Every provider below publishes an OpenAI-compatible chat completions endpoint,
# checked against its own documentation. The base paths are not uniform: Gemini
# carries the version inside the compatibility prefix and the others do not.
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
        "openai/gpt-oss-20b",
        GROQ_BASE_URL,
        "server",
    ),
    Preset("ollama-qwen", "Ollama: qwen2.5:7b", "qwen2.5:7b", DEFAULT_BASE_URL, "none"),
    Preset("ollama-llama", "Ollama: llama3.1:8b", "llama3.1:8b", DEFAULT_BASE_URL, "none"),
    Preset("groq", "Groq (your key)", "openai/gpt-oss-20b", GROQ_BASE_URL, "own"),
    Preset("openai", "OpenAI (your key)", "gpt-4o-mini", OPENAI_BASE_URL, "own"),
    # No model name is shipped for these four: the names change faster than this
    # file, and a stale one fails with a message from the provider instead of
    # from us. Type one, or read the list at <base_url>/models.
    Preset("anthropic", "Anthropic (your key)", "", ANTHROPIC_BASE_URL, "own"),
    Preset("gemini", "Google Gemini (your key)", "", GEMINI_BASE_URL, "own"),
    Preset("mistral", "Mistral (your key)", "", MISTRAL_BASE_URL, "own"),
    Preset("openrouter", "OpenRouter (your key)", "", OPENROUTER_BASE_URL, "own"),
    Preset("custom", "Custom endpoint", "", "", "own"),
)

SYSTEM_PROMPT = (
    "You analyse a table the reader loaded, and you never compute a number yourself. "
    "Every figure you report comes from a tool result, and you name the result id beside "
    "it. Work in the order the ledger enforces: frame_question to state the estimand, "
    "profile_table to learn the columns and the quality problems, then fit_model or "
    "compare_groups, then sensitivity, then explain_result. Call analysis_status whenever "
    "you are unsure what comes next, and follow its recommended_next. When a tool errors, "
    "read the message and retry: it names the columns that exist. Repeat a tool's caveats "
    "and cannot_say verbatim, because they are the limits of the answer. A wide interval "
    "is not evidence of no difference, and a column recorded after the outcome is leakage, "
    "not a predictor. Do not call a result causal unless the design stage permitted a "
    "causal target. run_sql is for lookups and aggregates only: a claim about an effect "
    "cites fit_model, compare_groups or sensitivity."
)

# The copy the Agent panel shows beside the key box. It has to be true.
KEY_POLICY = (
    "Your key is sent to this server, used for your session's requests, and held in "
    "memory only: it is never written to disk and never logged. The values the model "
    "reads from your table are sent to the provider you chose."
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
    endpoint = base_url.strip() or preset.base_url
    chosen = model.strip() or preset.model
    if not chosen:
        raise ValueError(
            f"The {preset.label} preset needs a model name. Type one in the Model box, "
            f"or read the list at {endpoint.rstrip('/')}/models."
        )
    return (chosen, endpoint, key)


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


def _tool_display(name: str, value: dict[str, Any]) -> ToolResultDisplay | None:
    """A card for the tools whose results read well as one, else None.

    Every preview derives from the result dict with `.get()`: a missing field
    degrades to a plainer card, never an exception.
    """
    try:
        from shinychat.types import ToolResultDisplay
    except Exception:  # pragma: no cover - defensive: shinychat may be absent
        return None

    if name == "sensitivity":
        checks = value.get("checks") or []
        display = ToolResultDisplay(title="Sensitivity check", open_style="framed")
        if checks:
            stable = sum(1 for check in checks if check.get("status") == "stable")
            display.value_preview = f"{stable}/{len(checks)} checks stable"
        return display
    if name == "plot":
        spec = ", ".join(
            str(value.get(field))
            for field in ("kind", "x", "y", "color", "facet")
            if value.get(field)
        )
        return ToolResultDisplay(title="Plot", label=spec, open_style="framed")
    if name == "fit_model":
        coefficients = value.get("coefficients") or []
        if not coefficients:
            return ToolResultDisplay(title="Fitted model")
        first = coefficients[0]
        return ToolResultDisplay(
            title="Fitted model",
            value_preview=(
                f"{first.get('term')}: {first.get('estimate')} "
                f"[{first.get('ci_low')}, {first.get('ci_high')}]"
            ),
        )
    return ToolResultDisplay(title=name.replace("_", " ").capitalize())


def _as_tool(handler: Any) -> Any:
    """Wrap a stats tool so the model gets JSON and every result has one shape.

    The wrapper is async and runs the sync handler in a worker thread: chatlas
    awaits async tools directly on the event loop, and without the thread a
    slow fit_model would stall every concurrent session.
    """

    @functools.wraps(handler)
    async def call(**kwargs: Any) -> ContentToolResult:
        result = await asyncio.to_thread(handler, **kwargs)
        value = result.model_dump()
        display = _tool_display(getattr(handler, "__name__", ""), value)
        extra: dict[str, Any] = {"display": display} if display is not None else {}
        return ContentToolResult(value=value, model_format="json", extra=extra)

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
    for spec in registry.agent_tools():
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
