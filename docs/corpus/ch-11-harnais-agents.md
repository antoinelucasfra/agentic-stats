# Coding agent harnesses

An **agent harness** is the loop wrapped around a language model: system prompt + tool definitions (read files, edit files, run shell commands), permission gates before destructive actions, context assembly between turns, and the interface you type into. The model is the engine. The harness is the chassis and cockpit. The same model performs visibly differently across harnesses. The Coding Agent Index (Chap. 2) measures *harness+model pairs*, not models alone.

### Why local models change the calculus

- **Cloud harnesses assume cloud models**: huge context windows, flawless tool-call discipline, server-side retries. A local Qwen or GLM at Q4 needs a harness that survives imperfect function calling and tighter contexts.
- **Token burn becomes free**: agentic coding easily consumes millions of tokens per day. Locally that costs electricity, not budget. That changes how aggressively you let the agent iterate.
- **Code sovereignty holds even in agent mode**: repository contents, secrets in dotfiles, shell history. None of it leaves the machine.

### Our recommendation: pi

[**pi**](https://pi.dev) is our pick for local-first agent work:

- **Fully open source**, minimal by design: four tools (`read`, `write`, `edit`, `bash`), no sub-agent or plan-mode bloat you cannot remove. You adapt pi to your workflow instead of fighting it.
- **Provider-agnostic out of the box**: any OpenAI-compatible or Anthropic-compatible endpoint goes in `~/.pi/agent/models.json`. Examples: llama-server, Ollama, vLLM, LM Studio, or the local server your chat UI or studio exposes. `compat` flags handle servers that reject `developer` roles or `reasoning_effort`.
- **Extensible without forking**: TypeScript extensions, skills, prompt templates, themes, shareable packages; plus print/JSON, RPC and an SDK mode to embed it in scripts or CI.
- **Honest trade-off**: minimal means you assemble your own workflow. That is the point. Nothing is hidden from you.

Quick start against your local stack:

```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent

# ~/.pi/agent/models.json — any local OpenAI-compatible server (Chap. 4) as provider:
{
  "providers": {
    "local": {
      "baseUrl": "http://127.0.0.1:8888/v1",
      "api": "openai-completions",
      "apiKey": "<key-if-the-server-wants-one>",
      "models": [{ "id": "<model-id-as-the-server-reports-it>" }]
    }
  }
}

pi          # then /model → pick your local entry

### The fully open field

| Harness | Licence | Shape | Local-model story |
|---|---|---|---|
| **pi** | open source | TUI · print · RPC · SDK | any endpoint via `models.json`; built for adaptation |
| **OpenAI Codex CLI** | Apache-2.0 | Rust TUI | OSS providers configurable; strongest with GPT-family habits |
| **Aider** | Apache-2.0 | Python CLI | `--openai-api-base` / native Ollama; git-native pair programming, repo map |
| **OpenCode** | open source | TUI + LSP, client/server | multi-provider incl. local; LSP-aware context |
| **Crush** | open source | TUI (Charm) | multi-provider; polished terminal UX |
| **Goose** (Block) | Apache-2.0 | CLI + desktop | MCP-first extensions; recipe automation |
| **Cline / Roo Code** | Apache-2.0 | VS Code extension | plan/act modes; OpenAI-compatible providers |
| **Continue.dev** | Apache-2.0 | IDE assistant | autocomplete + chat + agent; Ollama first-class |
| **OpenHands** | MIT | sandboxed platform | full autonomous dev loop; heavy but complete |
| **Qwen Code** | Apache-2.0 | CLI (gemini-cli fork) | tuned for Qwen coder models. Natural local pairing |

Honourable mention: **Claude Code**. Closed source, but the reference UX. It drives your local stack once you point it at your endpoint (Chap. 10). Adopting its *workflow* on top of an open harness like pi gets you most of the experience with full sovereignty.

### Choosing in one line each

- Terminal purist who wants control → **pi**
- Git-commit-centric workflows → **Aider**
- Live inside VS Code → **Cline / Continue.dev**
- Autonomous pipelines & recipes → **Goose / OpenHands**
- Standardised on Qwen locally → **Qwen Code**

### Local reality checklist

![The agent loop: the harness exchanges prompts and tool calls with a localhost model endpoint; tools (files, shell, git, tests) run on-premise, and results feed back into context until done.]

- **Measure the harness tax first.** System prompt + tool schemas can eat 10–20 k tokens before your code is even attached. On a 32 k-context model that is the whole budget: pick harnesses that trim, and check Chap. 5's KV-cache math.
- **Tool-calling format matters more than size.** Use models trained for function calling (Qwen-Coder, GLM, Kimi) and serve them with `llama-server --jinja`. Some studios also repair malformed tool calls before they break the turn (Chap. 4).
- **Mind the KV-cache attribution trap** when pointing Claude-protocol clients at local servers. The fix is in Chap. 10.
- **Gate permissions anyway.** Free tokens invite long autonomous runs; run agents against untrusted repos inside a container.
- **Calibrate expectations with pass@1**: best fully-open pair ≈ 0.63 (Kimi Code CLI + Kimi K3) vs 0.68 closed frontier on the early-2026 index. A gap small enough that free unlimited local retries close it in practice (Chap. 2).
- **Harness rankings move weekly too.** The Coding Agent Index tracks specific harness+model pairs; pi itself is not separately ranked yet. Its edge is adaptability and provider-agnosticism, not a leaderboard spot. Benchmark the *pair* you intend to run on your own tasks (Chap. 16), and expect to swap models underneath as new releases land. With pi's `models.json` it is one config line.
