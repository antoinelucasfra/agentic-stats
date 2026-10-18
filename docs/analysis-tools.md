# Analysis tool contract

What the tools do, what they guarantee, and what stays with the model.

Not implemented yet. `stats/registry.py` derives each tool's input schema from
its signature as it does today, so the Playground form and the MCP surface keep
working for every tool below without a second edit.

## Decisions this contract rests on

| Decision | Choice |
| --- | --- |
| Target model | Small local models, 7B to 20B. The enforcement below exists because a weak model cannot be trusted to sequence an analysis |
| `run_sql` | Exposed to the model, every result tagged `evidence_class: descriptive`, and the verify pass rejects an inferential claim that cites one |
| Causal language | Refused on observational data. `explain_result` returns `cannot_say` naming the design that would support the claim |
| v1 cut | `design_power` and `fit_curve` deferred. They are gated on columns many public datasets lack |

## Implementation notes

- `status` is what a check found; `severity` (`info`, `warning`, `blocking`) is how much it matters.
- A descriptive reading needs the design and quality stages only; anything inferential needs the whole chain including sensitivity.
- The verify allowance list is explicit and counted; see `stats/verify.py`.
- `put` stamps the new `result_id` onto the payload as well as the store entry.
- Role suggestion picks the group before the factor, sharing a column only when no confident candidate is left.

One portability fact worth carrying forward: pandas 3 gives text columns a
`StringDtype`, not `object`, so a `dtype == object` test is dead code. The role
inference tests caught that, and a two-level text column was being offered as a
binary outcome because of it.

## Status

| Part | State |
| --- | --- |
| Phase 0, safe fetch and duckdb lockdown | Implemented, 62 tests. `utils/fetch.py`, `utils/duck.py` |
| Phase 1, session store, ledger, result ids, role inference | Implemented. `stats/session.py`, `stats/roles.py`, `stats/errors.py` |
| Phase 2, the eleven tools | Implemented. `stats/workflow.py`, `stats/estimates.py`, `stats/reading.py` |
| The verify pass | Implemented in `stats/verify.py`, and the Agent panel calls it on every turn's final text. Ungrounded figures, effect claims resting on a descriptive result, and unpermitted causal language land in the transcript as warnings |
| Phase 3, agent loop | Landed: one shared store per connection, trace, a turn budget of twenty, and a concurrency cap of three that refuses instead of queueing. Missing: a visible plan checklist, and agent memory of past findings beyond the chat history |
| Phase 4, import UI, catalog, eval harness, export | Landed: the Data panel imports by URL, searches a curated list plus the cached Rdatasets index, holds many tables with a primary switch, and takes reader-confirmed roles. The eval harness runs seven scripted cases with a scoreboard. Findings export as self-contained HTML from the Playground |
| Phase 5, hardening and docs | Landed: turn budget, concurrency cap, key policy copy, README and this contract. No new dependencies, so `requirements.txt` needs no regen |

Two contract rules were tightened during implementation, because the code could
not honour the looser version:

- **A grouping column does not license a causal claim.** `_supports_causal`
  accepted any non-independent design, and `clustered` comes from a group column
  alone, so "revenue by region" was being read as causal. A causal target now
  needs a column whose name marks an assigned exposure, or a time structure.
- **`run_sql` is exempt from the design stage.** It answers lookups, and a reader
  may want to see the data before stating an estimand. Every result it returns is
  tagged `descriptive`, and the verify pass refuses an effect claim that rests on
  one, which is the guarantee that matters. The stage table below says design
  blocks everything; `run_sql` is the stated exception.

## Split of responsibility

| Layer | Owns |
| --- | --- |
| Model | Framing the question in words, choosing among methods the data permits, writing the narrative, deciding what to check next |
| Tools | Reading the data, computing with validated libraries, deriving assumption verdicts, row accounting, caveat derivation, the analysis ledger, the set of numbers that may be quoted |
| Neither | Numbers that appear in no result, exclusions nobody counted, causal claims from a design that cannot support them |

Interpretation tools return components plus a candidate phrasing, with the
numbers frozen. The model composes the prose, may improve the wording, may not
change a figure, and may not drop a caveat.

## Small-model contract

Ten tools is near the practical ceiling for a 7B model, so sequencing is not the
model's job. Every rule below exists to keep a weak model on the path.

- The plan lives in the UI, not in the context window. `frame_question` returns a
  `plan` list that the app renders as a checklist. The model reads the checklist
  when it needs to, instead of remembering it across twelve turns.
- Every tool result opens with `next`, one imperative sentence naming the tool
  call that follows. A model that follows instructions blindly still completes
  the workflow.
- Every refusal names the remedy as a tool call with arguments. `blocked` is an
  instruction, never a dead end.
- Tool results are compact. Full payloads stay in the store behind `result_id`,
  and the model receives a summary plus the ids. A twelve-level marginal-means
  table does not belong in the context window.
- Descriptions are imperative and short. If a tool needs a paragraph to explain
  when to use it, it is two tools or it is wrong.
- `analysis_status` is the reset button. Whatever the model has forgotten, one
  call returns the stage, the gaps, and the next action.

Commitment: if the eval harness shows first-tool accuracy below 85 percent on
the golden questions, merge tools further rather than adding prompt text.

## Global conventions

Every tool obeys these, and tests assert them.

- Every computation returns a `result_id`. Interpretation tools take a
  `result_id`, never raw numbers.
- Every result carries `evidence_class`, either `descriptive` or `inferential`.
  A claim about an effect may cite only `inferential` results.
- Row accounting on every tool that filters rows: `rows_total`, `rows_used`,
  `exclusions: list[{reason, n}]`. Fluid analysis hides deletions. Expert
  analysis reports them.
- Every estimate carries `estimate`, `ci_low`, `ci_high`, `ci_level`, `n`. No
  point estimate without an interval.
- Severity vocabulary for every verdict: `ok`, `caution`, `blocked`. `blocked`
  means no inference passes through until the named remedy is applied.
- Verdict shape: `{check, status, statement, remedy, severity}`.
- Numbers come back twice: `raw`, the float, and `display`, the rounded string.
  Rounding happens once, at the boundary, and `display` is what a narrative may
  quote verbatim.
- `quotable_numbers` lists the figures a narrative may use. The final-answer
  check accepts a figure only if it matches an entry in some stored result.

## The ledger

`AnalysisState` lives in the session store next to the loaded tables. Six
stages, in order.

| Stage | Established by | Blocks |
| --- | --- | --- |
| design | `frame_question` | Everything. No estimate before a stated estimand |
| quality | `profile_table` | Inference when a `blocked` issue stands |
| assumptions | `fit_model`, `compare_groups` | Interpretation |
| estimate | `fit_model`, `compare_groups`, `marginal_effects` | Interpretation |
| sensitivity | `sensitivity` | Publication of a conclusion |
| interpretation | `explain_result` | Report |

Tools read the ledger and refuse out of order. `explain_result` on a fit whose
assumptions stage is empty returns `status: caution` and names the remedy
instead of producing an interpretation.

## Stage 1, frame the analysis

### `frame_question`

```python
frame_question(
    question: str,
    outcome: str | None = None,
    predictors: list[str] | None = None,
    unit: str | None = None,
    target: Literal["description", "comparison", "association", "prediction", "causal"] | None = None,
) -> AnalysisPlan
```

Turns a plain-language question plus the data profile into a stated estimand, a
design classification, and the sequence the answer requires. Writes the design
stage.

Returns `estimand` as one sentence, `question_type`, `unit_of_analysis`,
`outcome_type` inferred from the column, `design` classified as independent,
paired, repeated, nested or clustered, `required_checks`, `smallest_effect`
when the caller supplied one, and `plan`, the ordered tool calls with arguments
where they are already determined. `plan` is what the UI renders as the
checklist.

Refuses when the outcome column does not exist, when the target is causal and
the design has no treatment or time structure, or when the unit is ambiguous
because identifier columns suggest duplicate measurements.

Principle enforced: the estimand is stated before a method is chosen. Most bad
analyses on public data are a method looking for a question.

## Stage 2, know what is in the table

### `profile_table`

```python
profile_table(table: str = "primary", outcome: str | None = None) -> TableProfile
```

Schema and data quality in one call, because they are the same act of looking
and because every session pays for both.

Schema section: columns, dtypes, ranges, level lists, cardinality, identifier
candidates, time candidates, candidate roles with the evidence for each.

Quality section: missingness counts and rates per column, columns grouped by
identical missingness pattern, a test of whether missingness in each column is
associated with `outcome`, duplicate rows at the unit level, zero-variance and
near-duplicate columns, imbalance across factor combinations, and suspected
leakage. Leakage means a column associated with the outcome so strongly that it
plausibly encodes it, plus name and dtype heuristics for post-treatment or
post-outcome fields.

Every finding carries a verdict with a recommended default, the alternatives,
and what changes if the reader chooses differently. Any recommendation that
drops rows reports how many.

The leakage check is the one that saves analyses. A column recorded after the
outcome will dominate any model and make the rest of the results meaningless.

## Stage 3, estimate

### `run_sql`

```python
run_sql(
    query: str,
    table: str = "primary",
    max_rows: int = 200,
) -> QueryResult
```

Read-only duckdb over session views. Statement allow-list, row cap, wall-clock
cap, and the fetch guard from phase 0 standing between the model and the
network.

Answers lookups, joins and aggregates. Tagged `descriptive`, and the note on the
result says so in words the model can act on:

```
Descriptive only. Cite compare_groups, fit_model or sensitivity
for any claim about an effect.
```

This is the tool that makes arbitrary datasets usable and the tool that most
threatens the statistical story, which is why the tag is a first-class field
rather than a paragraph in a prompt. A correlation or a difference in means
computed in SQL is a descriptive number, and the verify pass treats it that way.

### `fit_model`

```python
fit_model(
    estimand: str,
    response: str,
    terms: list[str],
    group: str | None = None,
    family: Literal["gaussian", "binomial", "poisson", "negbin"] = "gaussian",
    weights: str | None = None,
    confidence_level: float = 0.95,
) -> FitResult
```

One entry point for OLS, GLM and random-intercept mixed models. The engine
follows from `family` and `group`, so the model never chooses between three
tools that differ only in the estimator.

Returns coefficients on the link scale, effects back-transformed to the response
scale, diagnostics verdicts rather than raw p-value dumps, a caveat list, and
row accounting. Diagnostics cover residual structure, heteroscedasticity,
normality where it matters, influence via Cook's distance, collinearity,
convergence, separation for binomial fits, and cells with fewer than two
observations. Every failed check arrives with a remedy.

Refuses when the response is constant, when a term is an exact linear
combination of the others, when a fixed effect has a single level, and when a
binomial response is not two-valued.

### `compare_groups`

```python
compare_groups(
    response: str,
    factor: str,
    group: str | None = None,
    paired: str | None = None,
    smallest_effect: float | None = None,
    confidence_level: float = 0.95,
) -> ComparisonResult
```

Routes on the design and the observed violations, not on a preference. Equal
variance and roughly symmetric residuals give the standard test. Unequal
variance gives Welch. A violated distributional assumption with small cells
gives Kruskal-Wallis or Mann-Whitney. Paired data gives the paired test on
differences. The routing decision and the reason are both returned, so the
reader can disagree with it.

Always returns an effect size with an interval, Cohen's d, rank-biserial or
eta-squared depending on the route, plus adjusted pairwise contrasts with the
multiplicity method named.

When `smallest_effect` is supplied and the interval excludes it, the result says
so in equivalence language. When it is absent, the result reports that the
interval includes zero and states what it does not exclude. A wide interval is
never evidence of no difference.

### `marginal_effects`

```python
marginal_effects(
    result_id: str,
    at: dict[str, list[Any]] | None = None,
    covariates: list[str] | None = None,
    contrasts: Literal["pairwise", "vs_reference", "none"] = "pairwise",
    adjust: Literal["tukey", "holm", "bonferroni", "fdr_bh"] = "tukey",
) -> MarginalEffectsResult
```

Average predictive margins from a fitted model, at specified covariate settings.
This is the tool that stops the model reading a coefficient as an effect.

Returns per-cell estimates with intervals, contrasts with intervals and adjusted
p-values, the adjustment method, and a note naming the reference level and the
covariate values the averages were taken at. With interactions present, the
contrasts are the interaction contrasts, expressed as a difference of
differences.

## Stage 4, try to break the conclusion

### `sensitivity`

```python
sensitivity(
    result_id: str,
    checks: list[Literal["influence", "robust_se", "alternative_family", "bootstrap_ci", "adjustment_set"]] | None = None,
) -> SensitivityReport
```

Re-runs the estimate under conditions that should not change the answer if the
answer is real: influential points removed, heteroscedasticity-robust or
cluster-robust standard errors, a different family or a rank-based equivalent, a
bootstrap interval instead of the analytic one, and a perturbed adjustment set
for confounder choices.

Each check reports the recomputed estimate with its interval and a stability
verdict: stable, attenuated, flipped or inconclusive. The report ends with the
share of checks that preserved direction and significance, and what the
instability would mean if it is real.

This stage separates a textbook answer from an expert one, and it is the one an
LLM never volunteers.

## Stage 5, turn results into reading material

### `explain_result`

```python
explain_result(
    result_id: str,
    audience: Literal["analyst", "stakeholder"] = "analyst",
) -> Explanation
```

One tool for every stored result. Dispatches on the kind: a fit, a comparison, a
table, a plot. Three tools that differ only in payload would cost a weak model
three wrong choices.

From a fit:

- `effects`, one entry per term, contrast decoded, effect on the response scale,
  interval, reference level named.
- `interactions`, each as a difference of differences with an interval.
- `caveats`: extrapolation, aliasing, separation, singular fit, wide interval,
  unbalanced cells, influential points, design limits.
- `cannot_say`: what the fit does not establish. Direction of an effect the
  model has no power to detect, causality without a design, prediction outside
  the observed range.
- `suggested_phrasing`, a draft paragraph the model may rewrite.

From a table: ranked contributions, shares, deltas against a named baseline,
extremes, and what would mislead a reader. Small denominators, sums over
incommensurable categories, and splits where the direction reverses between
subgroups, which is the Simpson's-paradox warning a plain `group by` misses.

From a plot: axes, units inferred from column names, series and counts, observed
ranges, fitted trend with its slope, extremes, gaps, clipping, and alt text. The
model narrates from these numbers, so "a clear upward trend" is a claim the tool
either supports or does not.

`audience` controls register only. Stakeholder mode drops link-scale
coefficients and keeps response-scale effects and intervals.

### `plot`

```python
plot(
    kind: Literal["hist", "box", "scatter", "line", "bar", "heatmap", "ecdf"],
    x: str,
    y: str | None = None,
    color: str | None = None,
    facet: str | None = None,
    table: str = "primary",
    max_levels: int = 12,
) -> PlotResult
```

Declarative spec, rendered through the existing plotnine path, returning a
`result_id` and the reading above. The model picks columns. It never writes
drawing code.

## Stage 6, know whether the analysis is finished

### `analysis_status`

```python
analysis_status() -> AnalysisStatus
```

The ledger as a checklist: each stage with its status, the results that
established it, the checks still missing, blocking issues with remedies, and
`recommended_next`, one tool call and why.

The model calls this before writing a conclusion. In the app it drives the
progress panel, which is the visible form of "a complete analysis" and the part
a reviewer reads as the difference between a chatbot and an analysis assistant.

## Resolved defaults

Settled so they stop being open questions. Overrule any of them and the contract
changes in one place.

| Question | Default |
| --- | --- |
| Who sets the smallest effect of interest | The user, in a UI field, pre-filled with a data-derived suggestion. Recorded in the ledger. Absent means no equivalence language |
| Multiplicity family | The contrasts within one `marginal_effects` call. Tukey when the design is a balanced one-way comparison, Holm otherwise. The method is always named in the result |
| Table retention | Memory for the session. Above 25 MB or 200k rows, spill to a session temp parquet deleted on session end. User data is never cached across sessions |
| Joins | The model proposes SQL. A join across tables runs only after the user approves the key in the UI. No silent joins |
| Bootstrap | 2000 resamples, seed recorded in the result |
| Audience register | `analyst`. Stakeholder mode is opt-in per explanation |
| Eval fixtures | penguins, gapminder, Titanic, one TidyTuesday dataset with a time column |

## Worked sequence

Question: does body mass differ by species, once sex is accounted for.

| Step | Call | Establishes |
| --- | --- | --- |
| 1 | `frame_question("...", outcome="body_mass_g", predictors=["species", "sex"])` | estimand, design, checklist |
| 2 | `profile_table(outcome="body_mass_g")` | schema, missingness in sex, duplicate units, leakage screen |
| 3 | `fit_model("difference in mean body mass by species adjusted for sex", "body_mass_g", ["species", "sex"])` | estimate, diagnostics verdicts |
| 4 | `marginal_effects(fit_id, covariates=["sex"])` | adjusted species means, contrasts, intervals |
| 5 | `sensitivity(fit_id)` | stability under influence, robust SE, rank-based alternative |
| 6 | `explain_result(fit_id)` | effects, caveats, cannot_say, quotable numbers |
| 7 | `analysis_status()` | nothing blocking, then the report |

Seven calls. The model can skip any of them, and the ledger reports what is
missing.

## What the tools must never do

- Never choose the conclusion. They compute and disclose. The reading is the
  model's, and the ledger records which result it rested on.
- Never report an effect without an interval, or a comparison without an effect
  size.
- Never drop rows silently. Every exclusion is counted and named.
- Never let an assumption failure pass into interpretation as a footnote. It is
  a verdict with a remedy, and it appears in `cannot_say` when unresolved.
- Never present an adjusted observational association as a causal effect. The
  target from `frame_question` controls the language, and `explain_result`
  refuses the causal form when the design does not support it.
- Never let a `descriptive` result carry an inferential claim.

## Tests each tool owes

| Tool | Fixture pair | Assertion |
| --- | --- | --- |
| `frame_question` | clear question, causal target on cross-sectional data | estimand stated, checklist returned; causal target refused with a remedy |
| `profile_table` | clean table, table with an outcome-encoded column and a 40 percent missing subgroup | leakage `blocked`; missingness reported with row counts |
| `run_sql` | allowed select, blocked statement and a fetch attempt | allow-list holds, result tagged `descriptive`, note present |
| `fit_model` | well-behaved fit, separated binomial fit | diagnostics `ok`; separation returns `blocked`, not a coefficient claim |
| `compare_groups` | equal variance, heteroscedastic with unequal cells | route switches to Welch with the reason; effect size present in both |
| `marginal_effects` | fit with interaction, fit without | interaction contrast is a difference of differences |
| `sensitivity` | real effect, effect driven by two points | stable in the first, flipped in the second |
| `explain_result` | fit with failed assumptions, clean fit, table with a reversing split | caveats and `cannot_say` differ; reversal flagged; quotable numbers match the stored result |
| `plot` | chart with a real trend, flat chart | trend slope and its sign match the data |
| `analysis_status` | fresh session, completed sequence | blocking stages listed first; `recommended_next` after each step |
| verify pass | answer citing a descriptive result for an effect | claim rejected |

## Phases

| Phase | Content | Gate |
| --- | --- | --- |
| 0 | Fetch guard, duckdb lockdown, SSRF tests | `read_csv_auto('/etc/passwd')` and a link-local URL both fail closed |
| 1 | Session store, `AnalysisState` ledger, `result_id` handles, role map | two concurrent sessions with two tables never share numbers or ledger state |
| 2 | `frame_question`, `profile_table`, `run_sql`, `fit_model`, `compare_groups`, `marginal_effects`, `sensitivity` | per-tool fixture pairs above |
| 3 | `explain_result`, `plot`, `analysis_status`, ledger gating, verify pass | claim from a descriptive result is rejected; gating refuses out of order |
| 4 | Import UI, checklist and progress panel, findings export, eval harness | first-tool accuracy and ungrounded-number count reported |
| 5 | Rate limits, BYOK provider surface, README, requirements regen | CI green, quota behaviour documented |

Deferred to v1.1: `design_power`, `fit_curve`, and the multi-table join approval
flow beyond the single-key case.

## Known approximations

Disclosed by the implementation rather than hidden, because a statistical tool
that quietly substitutes a weaker method is worse than one that says so. Each
line names where it lives and what stands in.

| Tool | Approximation | What stands in |
| --- | --- | --- |
| `compare_groups` | No rank-based post-hoc for more than two groups | The effect size is epsilon-squared, and the pairwise contrasts are mean-based with a note saying they do not inherit the rank route. This is the weakest guarantee in the set |
| `compare_groups` | Cohen's d, d_z, rank-biserial and epsilon-squared intervals | Large-sample normal approximation for d, percentile bootstrap with a recorded seed for the rank measures |
| `compare_groups` | Eta-squared interval | Noncentral-F inversion by `brentq`, transformed as lambda over lambda plus N |
| `compare_groups` | Welch ANOVA eta-squared | Same transform with fractional Welch df |
| `fit_model` | Separation detection | Fitted probabilities within 1e-8 of 0 or 1 plus a coefficient above 10. Mild or partial separation can pass unflagged |
| `fit_model` | Breusch-Pagan for non-gaussian families | Run on Pearson residuals, where the statistic is not a proper chi-square test. The verdict says which residuals it used |
| `fit_model` | Residual normality for non-gaussian families | Skipped by design and reported as a verdict that says the family does not assume it |
| `fit_model` | A separated logit effect | Can be `inf` in the payload rather than an error |
| `marginal_effects` | Non-identity link | First-order delta method with a numerical gradient. No bias correction, no simulation interval |
| `sensitivity` | Robust standard errors for binomial, poisson and negbin | `GLMResults` has no robust covariance in the installed statsmodels. Reported `inconclusive` rather than hand-rolled |
| `sensitivity` | Influence check | Cook's distance from an OLS design of the same fixed effects, which is a proxy for a mixed or GLM source |
| `sensitivity` | Bootstrap intervals for a mixed source | Resamples refit without the random intercept, and significance is read from whether the percentile interval excludes zero |
| `sensitivity` | Alternative family for gaussian | Rank-transformed response, so only the sign and the significance are compared |

`fit_model` and `marginal_effects` carry the contract's guarantees with only the
disclosed approximations. `compare_groups` is the one to read with care, and
`sensitivity` is weaker in the narrow bootstrap case above.
