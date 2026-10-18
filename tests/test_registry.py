"""The registry is the single definition the MCP server and the app share.

The MCP side is checked through a real stdio session rather than by inspecting
our own generator, so the assertion is about what a client actually receives.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from stats import registry


def published_tools():
    """Tool definitions as an MCP client sees them."""

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "transports.mcp_server"],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            return {tool.name: tool for tool in listing.tools}

    return asyncio.run(run())


def test_every_tool_declares_a_model_and_a_description():
    for tool in registry.TOOLS:
        assert tool.description.strip(), f"{tool.name} needs a docstring"
        assert issubclass(tool.output_model, BaseModel), f"{tool.name} needs a result model"
        assert issubclass(tool.input_model, BaseModel), f"{tool.name} needs a request model"
        assert tool.input_schema()["type"] == "object"


def test_an_optional_literal_becomes_a_dropdown():
    """`Literal[...] | None` nests the enum, so the form must unwrap it."""
    from stats import forms

    spec = registry.get("frame_question")
    model_fields = spec.input_model.model_fields
    fields = {field.name: field for field in forms.fields_for(spec.input_schema(), model_fields)}

    assert fields["target"].kind == "enum"
    assert "causal" in fields["target"].choices


def test_the_agent_surface_is_smaller_than_the_registry():
    """The domain pack stays callable, but the agent is not offered two ways to
    profile a table."""
    agent = {tool.name for tool in registry.agent_tools()}

    assert "profile_table" in agent
    assert "describe_dataset" not in agent
    assert agent < {tool.name for tool in registry.TOOLS}


def test_registry_names_are_unique():
    names = [tool.name for tool in registry.TOOLS]
    assert len(names) == len(set(names))


def test_args_schema_matches_what_mcp_publishes():
    published = published_tools()
    assert set(published) == {tool.name for tool in registry.TOOLS}

    for tool in registry.TOOLS:
        mcp_schema = published[tool.name].input_schema
        ours = tool.input_schema()
        assert published[tool.name].description == tool.description
        assert set(mcp_schema.get("properties", {})) == set(ours["properties"])
        assert set(mcp_schema.get("required", [])) == set(ours.get("required", []))


def test_get_lists_valid_names_on_a_typo():
    with pytest.raises(KeyError, match="describe_dataset"):
        registry.get("describe_datasets")
