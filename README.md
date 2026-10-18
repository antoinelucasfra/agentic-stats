# agentic-stats

Statistical analysis of a table nobody has seen before, exposed to LLM agents
through **MCP**, to scripts through a **CLI agent**, and to anyone with a browser
through a **Shiny for Python** app.

An agent is handed a small set of typed statistical tools instead of a Python
sandbox. It profiles the data, states an estimand, fits a model, tries to break
its own conclusion, and reads the result back with the caveats attached. Every
figure it reports has to trace to a stored result, and a deterministic check
enforces that before the answer is shown.

> *"Does the oat variety matter, once we account for block-to-block variation?"*
> The agent calls `frame_question`, then `profile_table`, then `fit_model`, then
> `sensitivity`, and answers with the mixed-model estimate, its 95% CI, the block
> variance component, and the share of robustness checks that survived.

## Why this shape

- **Tools, not a code interpreter.** `fit_model(response, terms, group=...)` is a
  bounded, typed operation. The agent cannot write arbitrary code, cannot touch
  the filesystem, and gets a validation error it can act on instead of a stack
  trace.
- **One registry, three transports.** `stats/registry.py` derives each tool's
  JSON Schema from the function's signature and docstring. The MCP server, the
  CLI agent and the Shiny app all read that one list, so they cannot drift apart.
  23 tools are registered; the chat agent is offered 16, because the domain pack
  and the ledger-aware surface should not compete for a small model's attention.
- **A ledger, not a prompt.** Six stages, in order, each written by the tool that
  establishes it. A fit cannot be interpreted before its assumptions were
  checked, and `analysis_status` reports what is still missing. The sequencing
  lives in Python because a 7B model cannot be trusted to remember it.
- **Statistics that survives review.** Mixed models, marginal means, GLM,
  assumption verdicts with named remedies, and a sensitivity stage, computed by
  `statsmodels` and `scipy`, not reimplemented.
- **One implementation.** The Shiny app imports the same tools, so the numbers on
  the page are the numbers the tests cover. There is no second implementation and
  no statistics in JavaScript.
- **Honest about its own approximations.** Where a weaker method stands in for a
  stronger one, the result says so. See *Known approximations* in
  [docs/analysis-tools.md](docs/analysis-tools.md).

## The ledger

| Stage | Established by | Blocks |
| --- | --- | --- |
| design | `frame_question` | Everything except `run_sql` |
| quality | `profile_table` | Inference when a blocking issue stands |
| assumptions | `fit_model`, `compare_groups` | Interpretation |
| estimate | `fit_model`, `compare_groups`, `marginal_effects` | Interpretation |
| sensitivity | `sensitivity` | Publication of a conclusion |
| interpretation | `explain_result` | The report |

A descriptive reading, a query or a chart, needs the design and quality stages
only. An inferential one needs the whole chain, including sensitivity.

## What the tools answer

Sixteen tools on the agent surface, in ledger order: `frame_question`,
`profile_table`, `run_sql`, `load_table`, `analysis_status`, `fit_model`,
`compare_groups`, `marginal_effects`, `sensitivity`, `plot`, `explain_result`,
plus `power_analysis` and `dose_response`. Arguments, guarantees and refusal
rules for each: [docs/analysis-tools.md](docs/analysis-tools.md).

Seven more tools, the domain pack (`describe_dataset`, `fit_mixed_model`,
`anova_effect`, `marginal_means`, `check_assumptions`, `fit_curve_family`,
`correlate`), stay callable over MCP and in the Playground but are not offered to
the chat agent.

Three docs tools, the "Ask the docs" group (`search_docs`, `ask_docs`,
`index_documents`), are offered to the chat agent too. They are non-inferential:
they write no ledger stage and require none.

Tool errors are written for a model to recover from:

```
Unknown column(s): ['nitrogen_rate']. Available columns: ['block', 'variety', 'grain', 'nitrogen']
```

The Playground shows the same message inline, and in the Agent tab the model
reads the error and retries with corrected arguments. One behaviour, three
transports.

## The dataset

Nothing is committed. `utils/data.py` downloads the oats split-plot trial from
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
| `variety` | whole-plot factor, three levels |
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
uv run --all-extras pytest -q          # 306 tests, no network needed

uv run shiny run --host 127.0.0.1 --port 8766 app.py   # the Shiny app
uv run python app.py                   # same app, host and port from
                                       # AGENTIC_STATS_HOST / AGENTIC_STATS_PORT
```

The stats core needs no web server, so Shiny and the chat client live in extras.
`--all-extras` is what makes the app, the MCP server and the CLI runnable; a bare
`uv run` syncs the core and the dev group only.

The Agent tab needs a reachable OpenAI-compatible endpoint; Overview and
Playground do not.

Ask a question with any OpenAI-compatible endpoint, Ollama by default:

```bash
ollama pull qwen2.5:7b
uv run python -m transports.agent "Which oat variety should we scale up, and why?"
uv run python -m transports.mcp_server   # the same tools over stdio, for any MCP client
```

## The Shiny app

```bash
uv run shiny run --host 127.0.0.1 --port 8766 app.py
```

Three panels, one page, plus the data that feeds them:

- **Data** loads a table three ways: a public URL through the fetch guard, the
  catalog search over a curated list plus the cached Rdatasets index, and the
  shipped trial. Many tables can live in one session with a primary switch, and
  the panel shows what the profile guessed for each column, with its reason, so
  a reader who knows the data can confirm the roles once instead of correcting
  every tool call.
- **Overview** answers six questions from the fetched trial, computed once when the
  process starts. Every card has an *Open in playground* button that carries the tool
  and its arguments across.
- **Playground** builds its form from `registry`, so a tool added in Python appears
  with no UI change. It runs the tool in the server process against the session's
  primary table, renders results with the renderers the Overview cards
  use, and shows the ledger as a banner above every result: what is missing, and
  which call closes it. Uploads are read from the temp file Shiny writes, capped at
  5 MB, and never written to disk; CSV and parquet both work. A download button
  exports the session, tables, ledger and findings, as one self-contained HTML file.
- **Agent** is a chat over the sixteen ledger-aware tools: `chatlas` with
  `ChatOpenAICompletions`, so Ollama, vLLM, LM Studio, Groq, OpenAI, Anthropic,
  Gemini, Mistral and OpenRouter all work through one client. A preset select fills
  model and endpoint; the four newest providers ship no model name, because those
  names move faster than this repository, and the tool says so and points at the
  provider's model list. Your key stays in server memory for that session, is never
  written to disk and never logged, and the values the model reads from your table
  are sent to the provider you chose. Every tool call lands in the transcript as
  `tool(args)` with a one-line result, so you watch the model work. The finished
  text of every turn goes through the verify pass: a figure with no stored result,
  an effect claim resting on a descriptive one, or causal language without a causal
  design lands in the transcript as a warning. Twenty turns is the session budget,
  and three slow streams is the instance budget: the twenty-first turn and the
  fourth concurrent one are refused with a message, not queued.

Charts are plotnine figures encoded as inline PNGs, so tables and charts come from the
same result objects on every surface. Each connection gets its own session store, so
one reader's file, result ids and ledger never touch another's.

## What is not done yet

The rework is largely landed, and [docs/analysis-tools.md](docs/analysis-tools.md)
carries the status table. What remains is polish, not structure:

- No visible plan checklist. `frame_question` returns the ordered calls and the
  ledger banner names the next one, but there is no checklist panel yet.
- The agent's memory of past findings is the chat history. A findings panel that
  survives across turns would make long analyses easier to follow.
- The eval harness runs scripted cases only. A live-model variant, env-gated like
  the benchmark's, would measure the system end to end.
- No per-IP rate limiting. The turn budget and the concurrency cap protect a demo,
  and anything stricter wants an auth story first.

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
checkout, where `web/`, `stats/`, `transports/` and `utils/` sit next to it.

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
| `AGENTIC_STATS_LLM_API_KEY` | the key; a free Groq key makes the Demo preset work with no typing |

Republish on push is on by default. On the first deploy, check that Overview and
Playground answer without an endpoint configured, since only the Agent tab needs one.

Usage is metered while the app is active, roughly 11 hours a month at 1 CPU and 4 GB
from the 20 free credits.

## Releases

Commits follow [Conventional Commits](https://www.conventionalcommits.org), and
`pyproject.toml` is the only place a version is written. Bump it and tag the release by
hand: nothing is published automatically and no container image is built.

CI runs one job: ruff, then pytest. The suite covers the statistics, the registry, the
MCP surface, the CLI agent loop, a scripted eight-question agent benchmark, the session
store and the ledger, the role inference, the fetch and SQL guards, the reading tools,
the verify pass, and the app itself through Shiny's in-memory test server, so no browser
and no network are needed: `tests/conftest.py` points every tool at the 72-row fixture in
`tests/fixtures/`. The same job pre-warms the parquet cache, regenerates
`requirements.txt` from `uv.lock`, fails if the two disagree, then pip-installs it into a
clean venv and imports the app, which is the path a Connect Cloud deploy takes.

## Layout

```
app.py                 entry point: App(app_ui, server), and `python app.py`

docs/
  analysis-tools.md    the tool contract: what each tool guarantees, and its limits

web/                   everything Shiny
  app_ui.py            the navbar, assembling one module per panel
  server.py            the root server, and the only shared state between panels
  panels/
    overview/          ui.py, server.py: the six cards, computed at import
    data/              ui.py, server.py: import by URL or catalog, tables, roles
    playground/        ui.py, server.py, helpers.py: tool picker, form, upload, ledger
    agent/             ui.py, server.py: chatlas client, shared store, verify, budgets

stats/                 the statistics and the workflow, transport-agnostic
  session.py           tables, results, the six-stage ledger, quotable numbers
  errors.py            ToolError, shared by the store and the tools
  roles.py             column roles, guessed with a reason and a confidence
  workflow.py          frame_question, profile_table, run_sql, load_table, analysis_status
  estimates.py         fit_model, compare_groups, marginal_effects, sensitivity
  reading.py           plot, explain_result
  verify.py            the check that a figure exists in a stored result
  report.py            the session as self-contained Markdown or HTML
  stats_tools.py       the domain pack: pure statistical functions and result models
  registry.py          derives each tool's schema from its signature; groups and the agent surface
  views.py             one renderer per tool: metrics, charts, tables, verdicts
  charts.py            plotnine figures, encoded as inline PNGs
  forms.py             JSON Schema to form fields, and field values back to arguments

transports/            one module per way of calling the tools
  mcp_server.py        MCPServer over stdio, built from the registry
  agent.py             CLI tool-calling loop + OpenAI-compatible backend
  llm.py               chatlas client, provider presets, the system prompt

utils/
  config.py            constants, paths, the card definitions, panel ids
  data.py              fetch + parquet cache for the oats split-plot trial
  fetch.py             safe fetch for a public URL: SSRF guard, byte cap, format check
  catalog.py           the curated list and the cached Rdatasets index
  duck.py              locked-down duckdb: external access off, SELECT only, row and time caps
  formatting.py        number and p-value formatting

tests/                 tools, registry, MCP wiring, agent loop, app, session, ledger,
                       roles, reading, verify, report, fetch and SQL guards, catalog,
                       eval harness
```

Only `app.py` sits at the root, because `shiny run app.py` needs it there.

This is an application, not a distributable package: `app.py` runs from the
checkout, uv manages the environment (`package = false`), and `requirements.txt`
lists dependencies only.

## Limitations (deliberate)

- The shipped demo is one dataset and one design. The tools no longer assume it,
  but the Overview cards and the sanity numbers do.
- Real data, small trial. 72 rows and six blocks is what Yates ran in 1935, so the
  variety effect is genuinely underpowered here; that is the demo, not a defect.
- `run_sql` can compute a difference of means, and that number is descriptive.
  The verify pass refuses an effect claim that rests on one, but a reader who
  ignores the tag can still read it as a finding.
- Four nitrogen levels are the whole dose range. The 4PL fits, but the EC50 it prints
  for Marvellous (6.0) sits far outside 0 to 0.6, and the quadratic peak is reported
  as an extrapolation: with four doses, treat both as shape, not as potency.
- `compare_groups` is the weakest tool in the set: no rank-based post-hoc for more
  than two groups, and the contrasts beside a Kruskal-Wallis test are mean-based.
  The result says so. See *Known approximations*.

## License

MIT. See `LICENSE`.
