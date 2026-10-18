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

from rag import tools as rag_tools
from stats import estimates, reading, stats_tools, workflow

_EMPTY = inspect.Parameter.empty


@dataclass(frozen=True)
class ToolSpec:
    """A tool's identity and its derived schema, in one place."""

    name: str
    description: str
    handler: Callable[..., BaseModel]
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    # Where the Playground picker files it, so the workflow tools read first.
    group: str = "Other"
    # Whether the agent is offered it. The domain pack keeps working over MCP and
    # in the Playground while the agent gets the smaller, ledger-aware surface.
    agent: bool = True

    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the flat keyword arguments `handler` accepts."""
        return self.input_model.model_json_schema()


def spec(
    handler: Callable[..., BaseModel],
    group: str = "Other",
    agent: bool = True,
) -> ToolSpec:
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
        group=group,
        agent=agent,
    )


# The workflow tools come first: a weak model reads the list top down, and the
# ledger refuses anything that skips these anyway.
WORKFLOW = "1. Frame and profile"
ESTIMATE = "2. Estimate"
READING = "3. Read the result"
DESIGN = "4. Design size"
DOMAIN = "5. Domain pack"
DOCS = "6. Ask the docs"

TOOLS: list[ToolSpec] = [
    spec(workflow.frame_question, WORKFLOW),
    spec(workflow.profile_table, WORKFLOW),
    spec(workflow.run_sql, WORKFLOW),
    spec(workflow.load_table, WORKFLOW),
    spec(workflow.analysis_status, WORKFLOW),
    spec(estimates.fit_model, ESTIMATE),
    spec(estimates.compare_groups, ESTIMATE),
    spec(estimates.marginal_effects, ESTIMATE),
    spec(estimates.sensitivity, ESTIMATE),
    spec(reading.plot, READING),
    spec(reading.explain_result, READING),
    spec(stats_tools.power_analysis, DESIGN),
    spec(stats_tools.describe_dataset, DOMAIN, agent=False),
    spec(stats_tools.fit_mixed_model, DOMAIN, agent=False),
    spec(stats_tools.anova_effect, DOMAIN, agent=False),
    spec(stats_tools.marginal_means, DOMAIN, agent=False),
    spec(stats_tools.check_assumptions, DOMAIN, agent=False),
    spec(stats_tools.dose_response, DOMAIN),
    spec(stats_tools.fit_curve_family, DOMAIN, agent=False),
    spec(stats_tools.correlate, DOMAIN, agent=False),
    spec(rag_tools.search_docs, DOCS),
    spec(rag_tools.ask_docs, DOCS),
    spec(rag_tools.index_documents, DOCS),
]


def grouped() -> dict[str, dict[str, str]]:
    """Tool names by group, in declaration order, for the Playground picker."""
    groups: dict[str, dict[str, str]] = {}
    for tool in TOOLS:
        groups.setdefault(tool.group, {})[tool.name] = tool.name
    return groups


def agent_tools() -> list[ToolSpec]:
    """The surface the chat agent is offered: the ledger-aware tools only."""
    return [tool for tool in TOOLS if tool.agent]


def get(name: str) -> ToolSpec:
    """Look up a tool by name, with a message that lists the valid names."""
    for tool in TOOLS:
        if tool.name == name:
            return tool
    raise KeyError(f"Unknown tool '{name}'. Available tools: {[tool.name for tool in TOOLS]}")
