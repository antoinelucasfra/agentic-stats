"""FastAPI surface over the same statistical tools.

Two audiences: a human with curl, and the demo's landing page. It calls
stats_tools directly (no MCP round-trip), which keeps the HTTP layer thin and
means both transports share one tested implementation.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentic_stats import stats_tools

app = FastAPI(
    title="agentic-stats",
    description="Statistical analysis of an R&D design-of-experiments dataset, "
    "exposed over HTTP (here) and MCP (stdio).",
    version="0.1.0",
)

TOOLS = [
    {
        "name": "describe_dataset",
        "description": stats_tools.describe_dataset.__doc__,
    },
    {
        "name": "fit_mixed_model",
        "description": stats_tools.fit_mixed_model.__doc__,
    },
    {
        "name": "anova_effect",
        "description": stats_tools.anova_effect.__doc__,
    },
]


class FitRequest(BaseModel):
    response: str = Field(examples=["assay_signal"])
    fixed_effects: list[str] = Field(examples=[["formulation", "dose"]])
    group: str = Field(default="batch", examples=["batch"])


class AnovaRequest(BaseModel):
    response: str = Field(examples=["assay_signal"])
    factor: str = Field(examples=["formulation"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tools")
def tools() -> list[dict[str, str]]:
    """Tool surface, same three tools the MCP server publishes."""
    return TOOLS


@app.get("/dataset")
def dataset() -> stats_tools.DatasetDescription:
    """Column profile of the experiment dataset."""
    return _guard(stats_tools.describe_dataset)


@app.post("/mixed-model")
def mixed_model(request: FitRequest) -> stats_tools.MixedModelResult:
    """Linear mixed model with a random intercept per `group` level."""
    return _guard(
        lambda: stats_tools.fit_mixed_model(request.response, request.fixed_effects, request.group)
    )


@app.post("/anova")
def anova(request: AnovaRequest) -> stats_tools.AnovaResult:
    """One-factor ANOVA with a bias-adjusted effect size."""
    return _guard(lambda: stats_tools.anova_effect(request.response, request.factor))


def _guard(fn):
    try:
        return fn()
    except stats_tools.ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
