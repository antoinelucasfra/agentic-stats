# agentic-stats

Statistical analysis of a real agronomy split-plot trial, exposed to LLM agents
through **MCP**, to scripts through a **CLI agent**, and to anyone with a browser
through a **Shiny for Python** app.

An agent is handed seven well-described statistical tools instead of a Python
sandbox. It profiles the data, picks a model, and gets typed results with
confidence intervals: no code execution and no shell in the loop.

> *"Does the oat variety matter, once we account for block-to-block variation?"*
> The agent calls `describe_dataset`, then `fit_mixed_model`, then answers with the
> mixed-model estimate, its 95% CI, and the block variance component.

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
- **Real data, fetched not committed.** `data.py` pulls the trial from Rdatasets
  over HTTPS and writes a parquet cache with `duckdb`; the repository carries no
  dataset. The Agent tab is the only part that needs an endpoint.

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
              │ stdio, 7 tools │  │ CLI loop  │  │ modules/overview       │
              └────────────────┘  └───────────┘  │ modules/playground     │
                                                 │ modules/agent          │
                                                 └────────────────────────┘
```

`app.py` wires `app_ui.py` and `server.py` into an `App`. Each panel is a Shiny
module under `modules/`, so its inputs are namespaced (`playground-tool`) and two
panels cannot collide. The only cross-panel state is a `pending` value created in
`server.py`: the Overview writes a card's tool and arguments into it, the
Playground selects that tool, and `server.py` switches the navbar, because a module
cannot address another module's inputs and the navbar belongs to the root.

## What the tools answer

| Tool | Arguments | Answers |
| --- | --- | --- |
| `describe_dataset` | `()` | Columns, dtypes, missing values, ranges; which are factors, which numeric |
| `fit_mixed_model` | `(response, fixed_effects, group="block")` | REML fixed effects with 95% CIs, variance components, ICC |
| `anova_effect` | `(response, factor)` | F-test, p-value, omega-squared, partial eta-squared, Levene |
| `marginal_means` | `(response, factors, covariates=None, group=None, …)` | Average predicted mean per level with CIs, plus pairwise contrasts |
| `check_assumptions` | `(response, fixed_effects, group=None, …)` | Residual normality, Breusch-Pagan, Levene, VIF, Cook's distance, QQ data |
| `power_analysis` | `(response, factor, alpha=0.05, target_power=0.8, …)` | Power achieved, smallest detectable effect, rows needed per level |
| `dose_response` | `(response, dose, model="4pl", by=None)` | Four-parameter logistic or quadratic fit, EC50 or peak with a CI, predicted curve |

Tool errors are written for a model to recover from:

```
Unknown column(s): ['nitrogen_rate']. Available columns: ['block', 'variety', 'grain', 'nitrogen']
```

The Shiny Playground shows the same message inline, and in the Agent tab the model
reads the error and retries with corrected arguments. One behaviour, three transports.

## The dataset

Nothing is committed. `data.py` downloads the oats split-plot trial from
Rdatasets and converts it to parquet with `duckdb` on first use, so a fresh
clone, a wheel, and a deployed app all carry the same 72 rows without a dataset
in the repository.

The trial is the one Yates (1935) used to introduce the split-plot design: six
blocks, three oat varieties sown in the whole plots, four nitrogen rates applied
to the subplots. Grain yield in grams per subplot is the response. Pinheiro and
Bates (2000) analyse it in *Mixed-Effects Models in S and S-PLUS*.

| Column | Role |
| --- | --- |
| `block` | random-effect group, six levels |
| `variety` | whole-plot factor, three levels, contrasts |
| `grain` | response, numeric |
| `nitrogen` | subplot covariate and dose, four levels |

`yield` is the name in the source CSV; it becomes `grain` because `yield` is a
Python keyword and patsy parses formulas as Python. Keyword columns still work
if you upload one: the tools quote them with patsy's `Q()`.

```bash
uv run python -m utils.data            # fetch into the cache
uv run python -m utils.data --force    # refetch
```

The cache is `<tempdir>/agentic_stats/oats.parquet`, with an
`oats.parquet.meta.json` recording the source URL, fetch time, row count and
sha256. Two environment variables move it: `AGENTIC_STATS_CACHE` for the
directory, `AGENTIC_STATS_DATA` for an explicit CSV or parquet path (that is how
the test suite runs offline against a committed fixture).

Sanity numbers, so you can check the demo against something:

| Call | Result |
| --- | --- |
| `anova_effect("grain", "nitrogen")` | F 14.2, omega-squared 0.355, Levene p 0.98 |
| `marginal_means("grain", ["variety"], covariates=["nitrogen"])` | Golden Rain 104.50, Marvellous 109.79, Victory 97.63; no pair significant |
| `fit_mixed_model("grain", ["variety", "nitrogen"])` | block variance 245.0, residual 234.7, ICC 0.511 |
| `power_analysis("grain", "nitrogen")` | Cohen's f 0.77, power 1.00, 6 rows per level would suffice |
| `check_assumptions(...)` | OLS residuals depart from normality (p 0.025, blocks unmodelled); the mixed model's do not (p 0.18) |

That last row is the point of the fourth tool: the OLS residual test fails
precisely because the block effect is missing, and adding the random intercept
fixes it. Variety is the honest negative result: F 1.23, p 0.30, which is why the
power card and the variety card both say the design cannot settle it.

## Run it

```bash
uv sync --all-extras --all-groups      # Python 3.12+, deps resolved and locked
uv run python -m utils.data            # fetch the trial into the parquet cache
uv run --all-extras pytest -q          # 91 tests, no network needed

uv run shiny run --host 127.0.0.1 --port 8766 app.py   # the Shiny app
uv run python app.py                   # same app, host and port from
                                       # AGENTIC_STATS_HOST / AGENTIC_STATS_PORT
```

The stats core needs no web server, so Shiny and the chat client live in extras.
`--all-extras` is what makes the app, the MCP server and the CLI runnable; a bare
`uv run` syncs the core and the dev group only.

The Agent tab needs a reachable
OpenAI-compatible endpoint; Overview and Playground do not.

Ask a question with any OpenAI-compatible endpoint, Ollama by default:

```bash
ollama pull qwen2.5:7b
uv run python agent.py "Which oat variety should we scale up, and why?"
uv run python mcp_server.py            # the same tools over stdio, for any MCP client
```

## The Shiny app

```bash
uv run shiny run --host 127.0.0.1 --port 8766 app.py
```

Three panels, one page:

- **Overview** answers six questions from the fetched trial, computed once when the
  process starts. Every card has an *Open in playground* button that carries the tool
  and its arguments across.
- **Playground** builds its form from `registry`, so a tool added in Python appears
  with no UI change. It runs the tool in the server process against the cached
  parquet or one you upload, and renders results with the same functions the
  Overview cards use. Uploads are read from the temp file Shiny writes, capped at
  5 MB, and never written to disk; CSV and parquet both work.
- **Agent** is a chat over the same seven tools: `chatlas` with
  `ChatOpenAICompletions`, so Ollama, vLLM, LM Studio, Groq and OpenAI all work.
  Model, base URL and key are sidebar inputs; the key stays in server memory for that
  session and falls back to `AGENTIC_STATS_LLM_API_KEY`. Tool calls, their arguments
  and their errors are visible in the transcript.

Charts are plotnine figures encoded as inline PNGs, so tables and charts come from the
same result objects on every surface. An uploaded table is scoped to its session through
a `contextvars` override in `stats_tools.use_frame`, so one reader's file never
changes another reader's numbers.

## Hosting on Posit Connect Cloud

The free plan runs the app on Posit's servers from a public GitHub repository: 4 GB RAM,
20 usage credits a month, five applications, public content only.

Connect Cloud installs from `requirements.txt` and reads nothing else, and that file is
generated from `uv.lock`:

```bash
uv export --no-dev --format requirements-txt --no-hashes --no-header --extra app -o requirements.txt
```

`--no-dev` keeps the dev group out, so Connect Cloud does not install pytest and ruff.

CI regenerates it and diffs it against the committed copy, so a dependency change that
was not exported fails the build. The file lists dependencies only: there is no `.`
line, because the app is not a package and Connect Cloud runs `app.py` from the
checkout, where `modules/` and `utils/` sit next to it.

The dataset is not shipped. On the platform the app downloads the CSV from Rdatasets
and writes the parquet cache on first use, so the deploy needs outbound HTTPS and a
writable temp directory. Warm the cache before a demo by opening the Overview once, or
set `AGENTIC_STATS_DATA` to a path you ship yourself.

To deploy: **Publish** in Connect Cloud, framework **Shiny for Python**, the repository
and branch, primary file **app.py**, Python **3.12**. Add the Agent tab's endpoint as
secret variables rather than in the repo:

| Variable | Value |
| --- | --- |
| `AGENTIC_STATS_LLM_BASE_URL` | an OpenAI-compatible endpoint, for example `https://api.openai.com/v1` |
| `AGENTIC_STATS_LLM_MODEL` | the model name |
| `AGENTIC_STATS_LLM_API_KEY` | the key |

Republish on push is on by default, which replaces the GitHub Pages job that used to
deploy this. On the first deploy, check that Overview and Playground answer without an
endpoint configured, since only the Agent tab needs one.

Usage is metered while the app is active, roughly 11 hours a month at 1 CPU and 4 GB
from the 20 free credits. Free content is public, so a reader either uses the key you
set as a secret or types their own into the Agent tab's key field, which stays in server
memory for that session.

## Releases

Commits follow [Conventional Commits](https://www.conventionalcommits.org), and
`pyproject.toml` is the only place a version is written. Bump it and tag the release by
hand: nothing is published automatically and no container image is built.

CI runs one job: ruff, then pytest. The suite covers the statistics, the registry, the
MCP surface, the CLI agent loop, and the app itself through Shiny's in-memory test
server, so no browser and no network are needed: `tests/conftest.py` points every tool
at the 72-row fixture in `tests/fixtures/`. The same job pre-warms the parquet cache,
regenerates `requirements.txt` from `uv.lock`, fails if the two disagree, then
pip-installs it into a clean venv and imports the app, which is the path a Connect Cloud
deploy takes.

## Layout

```
app.py            entry point: App(app_ui, server), and `python app.py`
app_ui.py         the navbar, assembling one module per panel
server.py         the root server, and the only shared state between panels

modules/
  overview/       ui.py, server.py: the six cards, computed at import
  playground/     ui.py, server.py, helpers.py: tool picker, form, upload
  agent/          ui.py, server.py: chatlas client with the tools attached

utils/
  config.py       constants, paths, the card definitions, panel ids
  data.py         fetch + parquet cache for the oats split-plot trial
  formatting.py   number and p-value formatting

stats_tools.py    pure statistical functions + pydantic result models
registry.py       derives each tool's schema from its function signature
mcp_server.py     MCPServer over stdio, built from the registry
agent.py          CLI tool-calling loop + OpenAI-compatible backend
views.py          one renderer per tool: metrics, charts, tables, notes
charts.py         plotnine figures, encoded as inline PNGs
forms.py          JSON Schema to form fields, and field values back to arguments
llm.py            chatlas client with the registry tools attached
tests/            tools, registry, MCP wiring, agent loop, app, views, forms
```

This is an application, not a distributable package: `app.py` runs from the
checkout, uv manages the environment (`package = false`), and `requirements.txt`
lists dependencies only.

## Limitations (deliberate)

- One dataset, one design. Swap it with `AGENTIC_STATS_DATA`, but the tools assume a
  response column, a categorical factor, a numeric dose and a grouping column.
- Real data, small trial. 72 rows and six blocks is what Yates ran in 1935, so the
  variety effect is genuinely underpowered here; that is the demo, not a defect.
- The app fetches the dataset on first use. Offline, without `AGENTIC_STATS_DATA`,
  `describe_dataset` fails with a message naming the fetch command.
- No auth, no rate limiting, no queueing. The demo is a demo, and the Agent tab can
  be pointed at any OpenAI-compatible endpoint from the sidebar, so do not host it
  publicly without gating that.
- The Agent tab needs a reachable endpoint. Bigger models are better at choosing
  tools, but they are still choosing tools, not doing statistics.
- An uploaded table runs against whatever columns it has; the tools report unknown
  columns the same way they do for the fetched trial.
- Four nitrogen levels are the whole dose range. The 4PL fits, but the EC50 it prints
  for Marvellous (6.0) sits far outside 0 to 0.6, and the quadratic peak is reported
  as an extrapolation: with four doses, treat both as shape, not as potency.

## License

MIT. See `LICENSE`.
