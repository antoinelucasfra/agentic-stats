"""Integration test: the MCP server really publishes the tools over stdio."""

from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_mcp_server_lists_and_calls_tools():
    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "agentic_stats.mcp_server"]
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            names = {tool.name for tool in listing.tools}
            assert names == {"describe_dataset", "fit_mixed_model", "anova_effect"}

            result = await session.call_tool("describe_dataset", {})
            assert not result.is_error
            payload = "".join(getattr(item, "text", "") for item in result.content)
            assert "assay_signal" in payload

    asyncio.run(run())
