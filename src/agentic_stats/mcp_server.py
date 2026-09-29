"""MCP server: the statistical tools, exposed to any MCP-capable agent.

Transport is stdio by default, which is what editors and agent harnesses spawn.
Run it directly to inspect the tool surface:

    uv run agentic-stats-mcp
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from agentic_stats import stats_tools

INSTRUCTIONS = """
Statistical analysis of an R&D design-of-experiments dataset (formulation x dose x
operator, replicated within pilot batches).

Typical flow:
1. describe_dataset() to learn the columns, which are factors, which are numeric.
2. anova_effect(response, factor) for a quick level comparison and effect size.
3. fit_mixed_model(response, fixed_effects, group) when readings are replicated
   within a higher-level unit such as a batch: it separates batch variability
   from the effect of interest.

Call describe_dataset() first rather than guessing column names.
""".strip()

server = MCPServer(name="agentic-stats", instructions=INSTRUCTIONS)


@server.tool()
def describe_dataset() -> stats_tools.DatasetDescription:
    """Profile the experiment dataset: columns, types, missing values, ranges."""
    return stats_tools.describe_dataset()


@server.tool()
def fit_mixed_model(
    response: str,
    fixed_effects: list[str],
    group: str = "batch",
) -> stats_tools.MixedModelResult:
    """Fit a linear mixed model with a random intercept per group level.

    Use when observations repeat within a higher-level unit (e.g. assay readings
    within pilot batches). Returns fixed-effect estimates with 95% CIs and the
    between-group / residual variances.
    """
    return stats_tools.fit_mixed_model(response, fixed_effects, group)


@server.tool()
def anova_effect(response: str, factor: str) -> stats_tools.AnovaResult:
    """Test whether a factor shifts the mean of a numeric response, with omega-squared."""
    return stats_tools.anova_effect(response, factor)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
