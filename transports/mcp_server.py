"""MCP server: the statistical tools, exposed to any MCP-capable agent.

Transport is stdio by default, which is what editors and agent harnesses spawn.
The tool surface comes from `registry.TOOLS`, so signatures and docstrings are
declared once, on the functions in `stats_tools`. Run it directly to inspect it:

    uv run agentic-stats-mcp
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from stats import registry

INSTRUCTIONS = """
Statistical analysis of a real agronomy split-plot trial: three oat varieties in the
whole plots, four nitrogen rates in the subplots, replicated over six blocks.

Typical flow:
1. describe_dataset() to learn the columns, which are factors, which are numeric.
2. anova_effect(response, factor) for a quick level comparison and effect size.
3. fit_mixed_model(response, fixed_effects, group) when readings are replicated
   within a higher-level unit such as a block: it separates block variability
   from the effect of interest.

Which tool answers which question:
- Readings repeated within a block: fit_mixed_model(..., group="block"), and
  quote the ICC, not just the fixed effects. Plain anova_effect on a grouping
  factor double-counts the subplot replication.
- Before quoting an effect, ask power_analysis whether the design can resolve it
  at all: a non-significant p-value with low achieved power is underpowered,
  not a finding of no effect.
- Any dose question: dose_response for one curve, fit_curve_family to compare
  shapes on the same rows; repeat their notes about range
  extrapolation verbatim. An EC50 outside the tested range is not a finding.

Call describe_dataset() first rather than guessing column names.
""".strip()

server = MCPServer(name="agentic-stats", instructions=INSTRUCTIONS)

for _tool in registry.TOOLS:
    server.add_tool(_tool.handler, name=_tool.name, description=_tool.description)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
