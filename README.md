# agentic-stats

Statistical analysis of an R&D design-of-experiments dataset, exposed to LLM agents
through **MCP** and to humans through a small **FastAPI** app.

An agent is handed three well-described statistical tools instead of a Python
sandbox. It profiles the data, picks a model, and gets typed results with
confidence intervals — with no code execution and no shell in the loop.

> *"Does formulation B matter, once we account for batch-to-batch variation?"*
> The agent calls `describe_dataset`, then `fit_mixed_model`, then answers with the
> mixed-model estimate, its 95% CI, and the batch variance component.

## Why this shape

- **Tools, not a code interpreter.** `fit_mixed_model(response, fixed_effects, group)`
  is a bounded, typed operation. The agent cannot write arbitrary code, cannot touch
  the filesystem, and gets a validation error it can act on instead of a stack trace.
- **One implementation, two transports.** `stats_tools.py` holds pure functions.
  The MCP server and the HTTP API both call it, so the tests cover shipped code.
- **Statistics that survives review.** Linear mixed models with random intercepts,
  REML estimates, confidence intervals, omega-squared. Not a re-implementation of
  what `statsmodels` already does well.
- **No keys, no network, no data.** Synthetic dataset, local LLM (Ollama), everything
  runs offline in one container.

## Architecture

```
                    ┌───────────────────────────┐
   user question ──▶│  agent.py                 │
                    │  tool-calling loop        │
                    └──────────┬────────────────┘
                               │ MCP (stdio, JSON-RPC)
                               ▼
                    ┌───────────────────────────┐
                    │  mcp_server.py            │   3 tools, JSON Schema
                    └──────────┬────────────────┘
                               ▼
                    ┌───────────────────────────┐        ┌────────────────────┐
                    │  stats_tools.py           │◀──────▶│  data/doe_…csv     │
                    │  describe / mixedlm /    │        │  576 rows, seed 42 │
                    │  anova (pydantic models) │        └────────────────────┘
                    └──────────┬────────────────┘
                               ▲
                    ┌──────────┴────────────────┐
                    │  api.py (FastAPI)         │  same functions over HTTP
                    └───────────────────────────┘
```

## Stack

Python 3.12 · [`mcp`](https://py.sdk.modelcontextprotocol.io) 2.x (`MCPServer`, stdio) · FastAPI ·
pydantic · pandas · statsmodels · [uv](https://docs.astral.sh/uv/) (lockfile, `uv run`) · pytest ·
ruff · Docker / Compose · GitHub Actions.

## Quickstart

```bash
uv sync                                   # Python 3.12+, deps resolved and locked
uv run python -m agentic_stats.data       # regenerate the synthetic dataset (seeded)
uv run pytest -q                          # 16 tests, no network needed

uv run agentic-stats-api                  # http://127.0.0.1:8000/docs
```

Docker, one service:

```bash
docker compose up --build                 # http://127.0.0.1:8000
```

Ask a question with a local LLM (needs [Ollama](https://ollama.com) running):

```bash
ollama pull qwen2.5:7b
uv run agentic-stats-agent "Which formulation should we scale up, and why?"
```

Example trace:

```
[tool] describe_dataset({}) -> {"n_rows": 1728, "factors": ["batch", "formulation", "operator"], ...}
[tool] fit_mixed_model({"response": "assay_signal", "fixed_effects": ["formulation", "dose"]}) -> ...
[answer] Formulation B leads A by 3.34 units (95% CI 3.16–3.51, p ≈ 1e-310) once batch
        variation is accounted for; batch variance is 1.59 against 2.26 residual, so
        pilot batches matter as much as the formulation effect itself.
```

## Tools

| Tool | Signature | Answers |
| --- | --- | --- |
| `describe_dataset` | `()` | Columns, dtypes, missing values, ranges; which are factors, which numeric |
| `fit_mixed_model` | `(response, fixed_effects, group="batch")` | REML fixed effects with 95% CIs, between-group and residual variance, convergence flag |
| `anova_effect` | `(response, factor)` | F-test, p-value, omega-squared, per-level means |

Tool errors are written for a model to recover from:

```
Unknown column(s): ['formulation_id']. Available columns: ['batch', 'formulation', 'dose', ...]
```

Over HTTP they surface as `400` with the same message — one behaviour, two transports.

## The dataset

`data/doe_experiment.csv` is generated, not collected: 12 pilot batches x 3 formulations
x 4 dose levels x 3 operators x 4 replicates = **1728 rows** of continuous `assay_signal`,
with a real batch random effect (SD 2.0) on top of residual noise (SD 1.5). Tests assert
the model recovers that structure, so the demo cannot silently rot.

Sanity numbers, so you can check the demo against something: `anova_effect` on
`formulation` gives omega-squared 0.388 and per-level means A 21.44 / B 24.78 / C 19.71;
`fit_mixed_model` recovers batch variance 1.59 against 2.26 residual and a B-vs-A effect of
+3.34 (95% CI 3.16–3.51).

## Layout

```
src/agentic_stats/
  stats_tools.py   pure statistical functions + pydantic result models
  mcp_server.py    FastMCP server (stdio)
  api.py           FastAPI app
  agent.py         tool-calling loop + Ollama backend
  data.py          synthetic DOE generator (seeded)
tests/             tools, API, agent loop, MCP wiring
```

## Limitations (deliberate)

- Synthetic data — the statistics are real, the biology is not.
- No auth, no rate limiting, no streaming: a demo, not a service.
- Single-host statistics: no streaming statistics, no distributed fitting.
- The agent loop is intentionally minimal (no planning, memory, or retries beyond
  feeding errors back). Anything smarter belongs in a framework.

## License

MIT
