# Local models for coding agents

Coding agents (Claude Code, Codex, OpenCode…) accept custom endpoints. They can drive **your** local model, keeping code and prompts on-premise. The wiring is two environment variables; some stacks also ship a shortcut:

```bash
# generic: point the agent at any local server that speaks its dialect
unsloth start claude                            # Unsloth Studio shortcut (also: codex | hermes | opencode)
ollama run <model>                              # Ollama serves an OpenAI-compatible API on its default port
llama serve -hf <repo>:<quant>                   # llama.app: OpenAI-compatible API + web UI

The manual version works with **any** OpenAI- or Anthropic-compatible server (llama-server, Ollama, vLLM, LM Studio, a studio's API). Only the base URL and the model name change:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8888"     # whatever port your server prints
# only if the server enforces keys (Ollama and llama-server do not, by default):
export ANTHROPIC_AUTH_TOKEN="<local-api-key>"
export ANTHROPIC_API_KEY=""
claude --model <model-name-as-the-server-reports-it>

The endpoint needs to speak both dialects on one port to be maximally useful: Anthropic `/v1/messages` and OpenAI `/v1/chat/completions` + `/v1/responses`, with streaming, tool calling and vision. Most local servers expose the OpenAI dialect; Anthropic-compatible `/v1/messages` is the rarer half (Unsloth Studio and LM Studio both provide it[^ch10-dialects]). Create an API key in the server's settings if it offers one. Shown once, stored hashed, revocable.

A critical performance fix for Claude Code. Its attribution header (`x-anthropic-billing-header:`) changes every request and invalidates the local model's KV cache. Up to 90 % slower inference. Disable it:

```bash
claude --settings '{"env":{"CLAUDE_CODE_ATTRIBUTION_HEADER":"0","CLAUDE_CODE_ENABLE_TELEMETRY":"0","CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC":"1"}}'

Per the AA Coding Agent Index, expect open 26–30B-class models to handle routine refactors, tests and reviews well, while the largest closed models keep an edge on very long multi-file missions. Route accordingly.

### What local models do well

| Task | Local fit | Why |
|---|---|---|
| Routine refactors, renames, boilerplate | ✓✓ | pattern-following, small blast radius |
| Writing and fixing unit tests | ✓✓ | verifiable output. The agent runs the tests |
| Code review comments, docstrings | ✓✓ | short-context, style-driven |
| Feature work in 2–5 files | ✓ | fine with ≥ 64k context |
| Long multi-file missions (30 min+) | △ | context pressure + compounding tool errors |
| Novel architecture decisions | ✗ | route to your strongest model, local or not |

### Budgeting the context window

Agent loops burn tokens fast. Every tool result, file reads, shell output, diffs, lands in context and stays there. Practical guidance:

- **32k** works for single-file edits and Q&A; agents start forgetting earlier instructions past that.
- **64–128k** is the comfortable agentic range; prefer models rated for it rather than forcing a chat model beyond its training.
- Watch for **KV-cache invalidation**: anything that rewrites the conversation prefix (the attribution header above, system-prompt churn) re-processes the whole window. The silent killer of local-agent latency.
- Keep the harness disciplined: scoped file reads, truncated shell output. A 128k window fills in a dozen careless `cat`s.

### Routing strategy

Local-first escalation beats purism:

1. Default endpoint = your local model. Code, prompts and diffs stay on-premise.
2. Explicit escalation trigger for hard tasks: a per-session flag or second config that points at the cloud provider.
3. Measure the split: log which tasks needed escalation; when the local pass-rate on routine work climbs (re-test with your golden set, [§ Evaluate](16-evaluer.qmd#evaluate)), lower the threshold again.

**Our take.** The privacy win here is real and immediate. We run reviews of sensitive client code through a local model without ever asking permission from anyone's compliance team. But there is a productivity tax on the hardest 10 % of tasks. Pretending otherwise is how local-AI pilots get quietly abandoned. Hybrid routing is not a compromise. It is the honest architecture.
