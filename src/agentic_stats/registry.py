"""One definition per statistical tool, shared by every transport.

`stats_tools` holds the functions. This module derives, from each function's
signature, docstring and return annotation, everything the other transports
need: the JSON Schema an agent reads, the input model a form is built from, and
the description. `mcp_server` and the Shiny app both iterate over `TOOLS`, so a
new tool shows up in the MCP surface and in the app's Playground form without a
second edit.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

from agentic_stats import stats_tools

_EMPTY = inspect.Parameter.empty


@dataclass(frozen=True)
class ToolSpec:
    """A tool's identity and its derived schema, in one place."""

    name: str
    description: str
    handler: Callable[..., BaseModel]
    input_model: type[BaseModel]
    output_model: type[BaseModel]

    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the flat keyword arguments `handler` accepts."""
        return self.input_model.model_json_schema()


def spec(handler: Callable[..., BaseModel]) -> ToolSpec:
    """Derive a `ToolSpec` from a function's signature, docstring and return type."""
    hints = get_type_hints(handler)
    fields: dict[str, Any] = {}
    for parameter in inspect.signature(handler).parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        default = ... if parameter.default is _EMPTY else parameter.default
        fields[parameter.name] = (hints[parameter.name], default)

    name = handler.__name__
    return ToolSpec(
        name=name,
        description=inspect.getdoc(handler) or "",
        handler=handler,
        input_model=create_model(f"{name}Request", **fields),
        output_model=hints["return"],
    )


TOOLS: list[ToolSpec] = [
    spec(stats_tools.describe_dataset),
    spec(stats_tools.fit_mixed_model),
    spec(stats_tools.anova_effect),
    spec(stats_tools.marginal_means),
    spec(stats_tools.check_assumptions),
    spec(stats_tools.power_analysis),
    spec(stats_tools.dose_response),
]


def get(name: str) -> ToolSpec:
    """Look up a tool by name, with a message that lists the valid names."""
    for tool in TOOLS:
        if tool.name == name:
            return tool
    raise KeyError(f"Unknown tool '{name}'. Available tools: {[tool.name for tool in TOOLS]}")
