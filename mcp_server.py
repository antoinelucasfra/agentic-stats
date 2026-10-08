"""MCP server: the statistical tools, exposed to any MCP-capable agent.

Transport is stdio by default, which is what editors and agent harnesses spawn.
The tool surface comes from `registry.TOOLS`, so signatures and docstrings are
declared once, on the functions in `stats_tools`. Run it directly to inspect it:

    uv run agentic-stats-mcp
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

import registry

INSTRUCTIONS = """
Statistical analysis of a real agronomy split-plot trial: three oat varieties in the
whole plots, four nitrogen rates in the subplots, replicated over six blocks.

Typical flow:
1. describe_dataset() to learn the columns, which are factors, which are numeric.
2. anova_effect(response, factor) for a quick level comparison and effect size.
3. fit_mixed_model(response, fixed_effects, group) when readings are replicated
   within a higher-level unit such as a block: it separates block variability
   from the effect of interest.

Call describe_dataset() first rather than guessing column names.
""".strip()

server = MCPServer(name="agentic-stats", instructions=INSTRUCTIONS)

for _tool in registry.TOOLS:
    server.add_tool(_tool.handler, name=_tool.name, description=_tool.description)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
