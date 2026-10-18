# The rise of local AI: why now

For three years, frontier AI meant closed APIs: GPT, Claude, Gemini. Enterprises traded data sovereignty for capability. Every prompt went through a third party, priced per token, subject to foreign jurisdictions and vendor retention policies.

That trade-off is dissolving. Artificial Analysis ([artificialanalysis.ai](https://artificialanalysis.ai)), which benchmarks models continuously, shows open-weight models converging on proprietary ones:

- **Intelligence**: on the AA Intelligence Index v4.1.1 (early 2026), the best open models, Kimi K3 (59.7) and GLM-5.3 (59.5), sit within ~3 points of the best closed models (Claude Opus 5 at 63.1, GPT-5.6 Sol and Grok 4.6 at ~60.9). Two years ago the gap was twice that.
- **Coding agents**: on the AA Coding Agent Index v1.4, the open Kimi K3 driven by a CLI harness scores 0.63 vs 0.68 for Claude Code + Opus 5. Near-parity on real software-engineering tasks.
- **Economics**: AA's efficiency trends show the price of a given intelligence level falling by orders of magnitude year over year, while output speed rises. Intelligence is becoming a commodity.
- **Architecture**: the open ecosystem has consolidated around Mixture-of-Experts (MoE) designs. Huge total parameter counts with small active-per-token counts make strong models runnable on a single workstation.

Meanwhile, hyperscalers pour hundreds of billions of dollars a year into AI infrastructure. AA tracks the spending. That money concentrates compute in a handful of US and Chinese labs. For European organisations, that concentration is itself a governance risk.

**Why local AI wins on governance and sovereignty:**

- **No third-party processor**: prompts, documents and fine-tuning corpora never leave your perimeter. No Art. 28 processor agreements, no international transfer analysis (Ch. V GDPR), no vendor training on your data.
- **Auditability**: you know exactly which weights, which version, which quantisation answered a question. Weights can be hashed, versioned and archived like any other IT asset.
- **No arbitrary deprecation**: API models are retired; your fine-tuned model keeps working.
- **Exit from lock-in**: open weights run on llama.cpp, vLLM, Ollama, LM Studio. Swap runtimes without changing models.
- **European option**: open models from EU labs (e.g. Mistral, France) reduce exposure to extra-EU jurisdiction.

The old objection was quality. For domain-specific work it no longer holds: a fine-tuned mid-size open model beats a generic frontier API on narrow tasks, at zero marginal cost, offline.

**Our take.** We are not missionaries: we still send the hardest, least sensitive reasoning tasks to frontier APIs when the gap matters. What changed our practice is the middle of the distribution. Drafting, summarising, extraction, first-pass code quietly migrated local in our own workflow and never went back. Go hybrid on purpose, not out of dogma in either direction.
