# The open-model landscape, ranked

### A method, not a ranking

**Models are perishable. Methods are not.** New state-of-the-art models, open and closed, now ship weekly, and leaderboard positions reshuffle just as fast. Every model name in this book is an *illustration from an early-2026 snapshot*, not a recommendation frozen in time. What stays valid is the selection method:

1. **Define the use first.** Task, context size, language mix, privacy level, latency, hardware (Chap. 5). The context decides, never the hype.
2. **Shortlist on trusted, independent sources**, checked *at decision time*:
   - [LMArena](https://lmarena.ai). Crowdsourced blind head-to-head battles (text, webdev, vision arenas). Good for perceived quality, immune to benchmark gaming;
   - [ArtificialAnalysis.ai](https://artificialanalysis.ai). Independent composite indices (intelligence, coding agents), price/speed per band, open-vs-proprietary tracking;
   - vendor benchmarks only as a starting hint. They select their own tests.
3. **Validate on your own golden set** before committing (Chap. 16): 50–200 real tasks scored pass@1. This final gate is yours. No leaderboard replaces it.
4. **Re-check at every re-decision.** A model pinned in a policy document ages like milk; pin the *method* instead.

Keep the list of named models in your infrastructure **short and swappable**: one small/fast, one strong generalist, one specialist if the domain demands it. Behind standard interfaces (Chap. 9, 15), swapping is a config line, not a project.
Snapshot from [artificialanalysis.ai](https://artificialanalysis.ai) (early 2026, AA Intelligence Index v4.1.1, higher is better):

| Rank | Model | License | AA Index |
|---|---|---|---|
| 1 | Claude Opus 5 (max) | Proprietary | 63.1 |
| 2 | Claude Fable 5 | Proprietary | 62.1 |
| 3 | GPT-5.6 Sol (max) | Proprietary | 60.9 |
| 3 | Grok 4.6 (high) | Proprietary | 60.9 |
| 5 | **Kimi K3 (max)** | **Open weights** | **59.7** |
| 6 | **GLM-5.3 (max)** | **Open weights** | **59.5** |
| 7 | Muse Spark 1.2 (xhigh) | Open weights | 56.8 |
| 8 | Gemini 3.7 Flash (high) | Proprietary | 56.0 |
| 9 | **DeepSeek V4 Pro** | **Open weights** | **53.2** |

And for agentic coding (AA Coding Agent Index v1.4, composite of DeepSWE, Terminal-Bench v2.1, SWE-Atlas-QnA; 0–1):

| Harness + model | Type | Index |
|---|---|---|
| Claude Code + Opus 5 | Closed | 0.68 |
| Claude Code + Fable 5 | Closed | 0.67 |
| Codex + GPT-5.6 Sol | Closed | 0.65 |
| Grok Build + Grok 4.5 | Closed | 0.64 |
| **Kimi Code CLI + Kimi K3** | **Open** | **0.63** |
| Opencode + Gemini 3.7 Flash | Closed | 0.60 |
| Codex + DeepSeek V4 Flash | **Open** | 0.50 |
| Claude Code + GLM-5.2 | **Open** | 0.43 |

Rankings move monthly. Re-check [artificialanalysis.ai/models](https://artificialanalysis.ai/models) and [/agents/coding-agents](https://artificialanalysis.ai/agents/coding-agents) before committing to a model. The *method* below outlives any ranking.

![The same snapshot seen as a map. Composite intelligence vs blended API price. Open weights now cluster in the value zone: near-frontier quality at roughly an order of magnitude lower cost. Illustrative: prices are typical blended rates, indices from the tables above.]
## Which model for which use case
Illustrative pairings **from the early-2026 snapshot**. Treat the *shape* of each answer (size class, capability, hardware) as durable and the names as replaceable. Cross live rankings with day-zero GGUF support in your runtime and your actual hardware (4-bit sizes below):

| Use case | Recommended open models | Why | Hardware (inference, 4-bit) |
|---|---|---|---|
| Best possible on-prem reasoning/agentic | Kimi K3, GLM-5.3 | Top of the open rankings, near-closed parity | Multi-GPU server or 128 GB+ unified memory |
| Coding assistant / local coding agent | Qwen3.5/3.8 Coder, Gemma 4 26B-A4B, GLM-5.x | Strong tool-calling, trained for code; Gemma 4 26B-A4B runs on a 24 GB machine | 24 GB GPU / Apple silicon |
| General-purpose departmental assistant | Qwen3.8 27B, Gemma 4, Muse Glimmer 30B | Balanced quality/speed, good multilingual (French included) | 1× 24 GB GPU |
| High-volume / edge / laptop | Qwen3.5 4B class, small Gemma, LFM2.5 230M[^ch02-lfm] | Sub-second latency, tiny footprint; 230M models run on a Raspberry Pi or a phone | CPU or 8 GB GPU |
| Documents & RAG-heavy workflows | Long-context Qwen/GLM variants + embedding models | Large context windows now exceed 100k tokens at the median (AA architecture trends) | 24 GB GPU |
| Vision (image understanding) | Gemma / Qwen vision variants | Multimodal fine-tuning (Unsloth, LLaMA-Factory) | 16–24 GB GPU |
| Image/video generation | FLUX, Z-Image, LTX, Wan, MiniMax-H3 | Runnable locally; LoRA training via the diffusers/kohya toolchain | 16–24 GB GPU |
| Voice | TTS/STT models (Whisper and similar) | Runnable locally; voices fine-tunable on-prem | 8–16 GB GPU |
| French/EU preference | Mistral open models | EU lab, permissive licenses, strong French | Size-dependent |

Rules of thumb:

- **Prefer recent MoE "AxB" models** (e.g. 26B-A4B: 26B total, ~4B active). Frontier-adjacent quality at small-model speed.
- **Instruct over base** for fine-tuning: chat templates work out of the box, less data needed.
- **Quantisation naming**: `UD-Q4_K_XL` (Unsloth *Dynamic*) recovers most of the accuracy a plain 4-bit quant loses; other runtimes ship their own 4-bit variants (MLX 4-bit, `IQ4_XS`). Compare on your golden set, not on the label.
- **Train in the precision you serve**: 4-bit serving pairs with QLoRA 4-bit training.

**Our take.** We stopped chasing the top of the rankings months ago: the crown changes hands weekly and re-plumbing a fleet costs more than the two index points gained. In practice we standardise on one 24-GB-class workhorse per year, run Chapter 16 evals on our own tasks, and only revisit when those numbers, not AA's, say so. Boring, cheap, and it has never let us down.
