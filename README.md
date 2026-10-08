# agentic-stats

Statistical analysis of an R&D design-of-experiments dataset, exposed to LLM agents
through **MCP**, to scripts through a **CLI agent**, and to anyone with a browser
through a **Shiny for Python** app.

An agent is handed seven well-described statistical tools instead of a Python
sandbox. It profiles the data, picks a model, and gets typed results with
confidence intervals: no code execution and no shell in the loop.

> *"Does formulation B matter, once we account for batch-to-batch variation?"*
> The agent calls `describe_dataset`, then `fit_mixed_model`, then answers with the
> mixed-model estimate, its 95% CI, and the batch variance component.

## Why this shape

- **Tools, not a code interpreter.** `fit_mixed_model(response, fixed_effects, group)`
  is a bounded, typed operation. The agent cannot write arbitrary code, cannot touch
  the filesystem, and gets a validation error it can act on instead of a stack trace.
- **One registry, three transports.** `registry.py` derives each tool's JSON Schema
  from the function's signature and docstring. The MCP server, the CLI agent and the
  Shiny app all read that one list, so they cannot drift apart.
- **Statistics that survives review.** Mixed models, marginal means, assumption checks,
  power, and dose response, computed by `statsmodels` and `scipy`, not reimplemented.
- **The app is the package.** The Shiny app imports `stats_tools` directly, so the
  numbers on the page are the numbers the tests cover. There is no second
  implementation and no statistics in JavaScript.
- **Synthetic data, real statistics.** The dataset is generated and seeded, and the
  models are `statsmodels`. The Agent tab is the only part that needs an endpoint.

## Architecture

```
                        ┌──────────────────────────────┐
                        │  stats_tools.py              │
                        │  pure functions              │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │  registry.py                 │
                        │  name · JSON Schema · output │
                        └───┬──────────┬───────────┬───┘
                            │          │           │
              ┌─────────────▼──┐  ┌────▼──────┐  ┌─▼──────────────────────┐
              │ mcp_server.py  │  │ agent.py  │  │ app.py (Shiny)         │
              │ stdio, 7 tools │  │ CLI loop  │  │ Overview · Playground  │
              └────────────────┘  └───────────┘  │ · Agent                │
                                                 └────────────────────────┘
```

## What the tools answer

| Tool | Arguments | Answers |
| --- | --- | --- |
| `describe_dataset` | `()` | Columns, dtypes, missing values, ranges; which are factors, which numeric |
| `fit_mixed_model` | `(response, fixed_effects, group="batch")` | REML fixed effects with 95% CIs, variance components, ICC |
| `anova_effect` | `(response, factor)` | F-test, p-value, omega-squared, partial eta-squared, Levene |
| `marginal_means` | `(response, factors, covariates=None, group=None, …)` | Average predicted mean per level with CIs, plus pairwise contrasts |
| `check_assumptions` | `(response, fixed_effects, group=None, …)` | Residual normality, Breusch-Pagan, Levene, VIF, Cook's distance, QQ data |
| `power_analysis` | `(response, factor, alpha=0.05, target_power=0.8, …)` | Power achieved, smallest detectable effect, rows needed per level |
| `dose_response` | `(response, dose, model="4pl", by=None)` | Four-parameter logistic or quadratic fit, EC50 or peak with a CI, predicted curve |

Tool errors are written for a model to recover from:

```
Unknown column(s): ['formulation_id']. Available columns: ['batch', 'formulation', 'dose', ...]
```

The Shiny Playground shows the same message inline, and in the Agent tab the model
reads the error and retries with corrected arguments. One behaviour, three transports.

## The dataset

`data/doe_experiment.csv` is generated, not collected: 12 pilot batches x 3 formulations
x 4 dose levels x 3 operators x 4 replicates = **1728 rows** of continuous `assay_signal`,
with a real batch random effect (SD 2.0) on top of residual noise (SD 1.5). Tests assert
the model recovers that structure, so the demo cannot silently rot.

Sanity numbers, so you can check the demo against something:

| Call | Result |
| --- | --- |
| `anova_effect("assay_signal", "formulation")` | F 549.0, omega-squared 0.388, Levene p 0.20 |
| `marginal_means("assay_signal", ["formulation"], covariates=["dose"])` | A 21.44, B 24.78, C 19.71; B - A = +3.34 (95% CI 3.06 to 3.61) |
| `fit_mixed_model("assay_signal", ["formulation", "dose"])` | batch variance 3.60, residual 2.26, ICC 0.61 |
| `power_analysis("assay_signal", "formulation")` | Cohen's f 0.80, power 1.00, 7 rows per level would suffice |
| `check_assumptions(...)` | OLS residuals depart from normality (batch variation is unmodelled); the mixed model's do not (p 0.50) |

That last row is the point of the fourth tool: the OLS residual test fails precisely
because the batch effect is missing, and adding the random intercept fixes it.

## Run it

```bash
uv sync --all-extras --all-groups      # Python 3.12+, deps resolved and locked
uv run python -m agentic_stats.data    # regenerate the synthetic dataset (seeded)
uv run pytest -q                       # 100 tests, no network needed

uv run shiny run --host 127.0.0.1 --port 8766 app.py   # the Shiny app
uv run agentic-stats-app               # same app, host and port from
                                       # AGENTIC_STATS_HOST / AGENTIC_STATS_PORT
```

Docker, one service:

```bash
docker compose up --build              # http://127.0.0.1:8000
```

The image serves the same app through `shiny run`. The Agent tab needs a reachable
OpenAI-compatible endpoint; Overview and Playground do not.

Ask a question with any OpenAI-compatible endpoint, Ollama by default:

```bash
ollama pull qwen2.5:7b
uv run agentic-stats-agent "Which formulation should we scale up, and why?"
```

## The Shiny app

```bash
uv run shiny run --host 127.0.0.1 --port 8766 app.py
```

Three panels, one page:

- **Overview** answers six questions from the shipped data, computed once when the
  process starts. Every card has an *Open in playground* button that carries the tool
  and its arguments across.
- **Playground** builds its form from `registry`, so a tool added in Python appears
  with no UI change. It runs the tool in the server process against the shipped CSV
  or one you upload, and renders results with the same functions the Overview cards
  use. Uploads are read from the temp file Shiny writes, capped at 5 MB, and never
  written to disk.
- **Agent** is a chat over the same seven tools: `chatlas` with
  `ChatOpenAICompletions`, so Ollama, vLLM, LM Studio, Groq and OpenAI all work.
  Model, base URL and key are sidebar inputs; the key stays in server memory for that
  session and falls back to `AGENTIC_STATS_LLM_API_KEY`. Tool calls, their arguments
  and their errors are visible in the transcript.

Charts are plotnine figures encoded as inline PNGs, so tables and charts come from the
same result objects on every surface. An uploaded CSV is scoped to its session through
a `contextvars` override in `stats_tools.use_frame`, so one reader's file never
changes another reader's numbers.

## Hosting on Posit Connect Cloud

The free plan runs the app on Posit's servers from a public GitHub repository: 4 GB RAM,
20 usage credits a month, five applications, public content only. No container, so the
GHCR image is not involved; the image is still published for Cloud Run, Azure Container
Apps, or a VM you own.

Connect Cloud installs from `requirements.txt` and reads nothing else, and that file is
generated from `uv.lock`:

```bash
uv export --format requirements-txt --no-hashes --no-header --extra app -o requirements.txt
```

CI regenerates it and diffs it against the committed copy, so a dependency change that
was not exported fails the build. The first line is `-e .`, which installs this package
from the repository and is what makes the `src/` layout importable in Connect Cloud's
environment.

To deploy: **Publish** in Connect Cloud, framework **Shiny for Python**, the repository
and branch, primary file **app.py**, Python **3.12**. Add the Agent tab's endpoint as
secret variables rather than in the repo:

| Variable | Value |
| --- | --- |
| `AGENTIC_STATS_LLM_BASE_URL` | an OpenAI-compatible endpoint, for example `https://api.openai.com/v1` |
| `AGENTIC_STATS_LLM_MODEL` | the model name |
| `AGENTIC_STATS_LLM_API_KEY` | the key |

Republish on push is on by default, which replaces the GitHub Pages job that used to
deploy this. Check two things on the first deploy: that Connect Cloud accepts the
editable `-e .` line, and that Overview and Playground answer without an endpoint
configured, since only the Agent tab needs one.

Usage is metered while the app is active, roughly 11 hours a month at 1 CPU and 4 GB
from the 20 free credits. Free content is public, so a reader either uses the key you
set as a secret or types their own into the Agent tab's key field, which stays in server
memory for that session.

## Releases and hosting

Commits follow [Conventional Commits](https://www.conventionalcommits.org). Pushing to
`main` makes [release-please](https://github.com/googleapis/release-please) open a
release PR; merging it bumps the version, writes `CHANGELOG.md`, and creates the tag and
GitHub Release. `pyproject.toml` is the only place a version is written.

Publishing a release runs `.github/workflows/release.yml`, which:

- attaches the wheel and sdist to the Release and pushes
  `ghcr.io/antoinelucasfra/agentic-stats` to GHCR, which is how the app is hosted now:
  `docker run -p 8000:8000 ghcr.io/antoinelucasfra/agentic-stats`.

CI runs one job: ruff, then pytest. The suite covers the statistics, the registry, the
MCP surface, the CLI agent loop, and the app itself through Shiny's in-memory test
server, so no browser and no network are needed. The same job regenerates
`requirements.txt` from `uv.lock` and fails if the two disagree.

## Layout

```
src/agentic_stats/
  stats_tools.py   pure statistical functions + pydantic result models
  registry.py      derives each tool's schema from its function signature
  mcp_server.py    MCPServer over stdio, built from the registry
  agent.py         CLI tool-calling loop + OpenAI-compatible backend
  data.py          synthetic DOE generator (seeded)
  app.py           Shiny app: Overview, Playground, Agent
  views.py         one renderer per tool: metrics, charts, tables, notes
  charts.py        plotnine figures, encoded as inline PNGs
  forms.py         JSON Schema to form fields, and field values back to arguments
  llm.py           chatlas client with the registry tools attached
app.py             `shiny run app.py` entry point
tests/             tools, registry, MCP wiring, agent loop, app, views, forms
```

## Limitations (deliberate)

- Synthetic data. The statistics are real, the biology is not.
- No auth, no rate limiting, no queueing. The demo is a demo, and the Agent tab can
  be pointed at any OpenAI-compatible endpoint from the sidebar, so do not host it
  publicly without gating that.
- The Agent tab needs a reachable endpoint. Bigger models are better at choosing
  tools, but they are still choosing tools, not doing statistics.
- An uploaded CSV runs against whatever columns it has; the tools report unknown
  columns the same way they do for the shipped dataset.
- The dose response does not turn over inside the tested range, so `dose_response`
  reports the quadratic peak as an extrapolation rather than a finding.

## License

MIT. See `LICENSE`.
