# Serving a team: from desktop to shared gateway

One workstation serves one researcher. A **lab server** serves thirty. The jump needs three additions: parallel serving, a gateway, and governance.

### Parallel serving

- **llama-server** (`llama.cpp`): `--parallel N --ctx-size …` splits the context budget across concurrent slots. Remember KV-cache math (Chap. 5): 8 B model, 32 k context, 4 slots ⇒ ~16 GB on top of weights.
- **vLLM**: continuous batching, much higher throughput for many simultaneous users; needs bigger GPUs and speaks OpenAI API natively.
- Rule: llama.cpp for ≤ ~10 light users per GPU; vLLM beyond.

### The gateway: LiteLLM

Put **LiteLLM** (open source) in front of every backend:

```yaml
model_list:
  - model_name: fast
    litellm_params: {model: openai/fast, api_base: http://gpu1:8080/v1}
  - model_name: big
    litellm_params: {model: openai/big, api_base: http://a100:8000/v1}
router_settings: {routing_strategy: usage-based-routing}
general_settings: {max_budget: 100, budget_duration: 30d}

You get: one stable OpenAI-compatible URL for the whole lab, **API keys per user/team**, quotas and budgets enforced, automatic fallback between GPUs, and a usage dashboard. Every tool in this book, chat UIs, pi's `models.json`, Aider, Cline, points at the same gateway.

### Governance hooks

- **Auth**: keep models unreachable except through the gateway behind your SSO/reverse proxy (OAuth2/OIDC). Local AI must not mean shadow IT.
- **Routing table.** Publish which tasks go to `fast` vs `big`. Most traffic belongs on small models; reserve frontier-size ones for review.
- **Privacy by default**: gateways can log prompts; *don't*. Aggregate token counts only. If an audit ever requires logging, make it opt-in per project with retention limits (Chap. 17).
- **Cost allocation** without cloud bills = tokens per team from the dashboard → feeds the TCO story of Chap. 18 and the scenarios of Chap. 19.

### The four numbers users feel

Four metrics govern whether a shared server feels fast or broken:[^ch15-latency]

- **Time to first token (TTFT)**: prefill plus the first decode step. The wait before anything appears.
- **Time per output token**: the steady-state reading speed.
- **Throughput**: tokens per second across *all* concurrent users. The number to size against; 30–40 t/s is feasible on one GPU.
- **End-to-end latency**: prompt in to answer complete. What your SLA records.

Decode speed is what people feel; prefill is what they wait for once. Under parallel serving, throughput is what you buy: batching trades each user's time-per-token for the room's aggregate tokens/s.

### Sizing sanity check

Concurrent users × average context ≈ KV memory (Chap. 5). A single RTX 4090 comfortably absorbs a 15-person lab on an 8–9 B Q4 model with 16 k contexts; a used A100/H100 or 2×4090 handles 27–32 B class for the same crowd. Start smaller than you think: utilisation data from the gateway tells you exactly when to buy.

**Our take.** Put the gateway in on day one, even for three users. Retrofitting auth and quotas after people have built habits is a political project, not an IT one. And resist the urge to give everyone the biggest model: in our lab, moving 90 % of traffic to a small fast model felt like an upgrade, because latency dropped and nobody missed the parameters.
