# Inference parameters, precisely

A language model generates one token at a time. At each step it outputs a probability for every token in its vocabulary; the sampling parameters decide how the software picks the next token from that distribution. Everything below reshapes that distribution, or refuses to.

### The sampling knobs

| Parameter | Mechanical effect | Raise it → | Lower it → | Sensible range |
|---|---|---|---|---|
| **Temperature** | Divides the logits by *T* before softmax | Flatter distribution: more variety, more surprise, more factual errors | Sharper: near-deterministic, repetitive, safest for facts | 0–0.2 code/extraction · 0.3–0.5 factual Q&A · 0.7–1.0 writing |
| **Top-p** (nucleus) | Keep the smallest set of tokens whose cumulative probability ≥ p, renormalise, sample | Wider candidate pool | Only high-confidence candidates survive; near-greedy at p→0 | 0.9–0.95; tune temperature *or* top-p, not both blindly |
| **Top-k** | Hard cutoff: only the k most likely tokens can be sampled | More diversity | Fewer candidates; k=1 equals greedy decoding | 20–60 when used at all; min-p is usually better |
| **Min-p** | Dynamic cutoff relative to the top token's probability | Keeps tails alive when temperature is high | Prunes implausible tokens even at high temperature | 0.05–0.1; pairs well with temp 0.7–1.2 |
| **Repeat / frequency penalty** | Down-weights logits of already-generated tokens (frequency penalty scales with count) | Less repetition but degraded grammar and broken code if extreme | Natural text, possible loops on weak models | ≤ 1.1–1.2; fix looping with better models, not huge penalties |
| **Seed** | Fixes the random generator | n/a | Same seed + same params + same binary ⇒ reproducible output | Set it whenever you need to reproduce a result |
| **Max tokens** | Hard stop on generation length | Longer answers possible | Truncates long reasoning mid-step; does **not** make anything faster | Task-dependent; leave headroom for thinking modes |
| **Stop sequences** | Generation halts when the string appears | n/a | Clean machine-readable output for parsing | Use for structured pipelines |

The system prompt is a parameter too: it is prepended to every conversation and steers tone, format and constraints. Because engines cache the computation of an unchanged prefix (the **KV cache**), a stable system prompt also makes responses faster and cheaper.

**Our take.** Most people over-tune these knobs and under-write their system prompt. After two years of daily use, our entire production setup is temperature 0.2 for anything factual, min-p 0.05 when writing, everything else stock. The hours we once spent chasing sampling magic now go into better instructions, which move the needle ten times more.

### The context window: the most expensive parameter

The context window (`n_ctx`) is the total token budget for one request: system prompt + attached documents + conversation history + generated answer. Everything must fit; exceed it and the engine either errors or silently truncates the oldest content.

**Memory cost.** During generation the model keeps a key/value tensor per layer per past token. The KV cache. For a Llama-style 8B model (32 layers, grouped-query attention[^ch05-gqa]) it costs ≈128 KB *per token* at 16-bit: ≈1 GB at 8k tokens, **4 GB at 32k, 16 GB at 128k, on top of the weights**. Two levers cut that: KV-cache quantisation (`q8_0` halves it with negligible quality loss, `q4_0` quarters it) and flash attention (shrinks activation memory and speeds up long prompts).

**Speed cost.** Generation has two phases:

- *Prefill* (reading your prompt): parallel, compute-bound. Hundreds to thousands of tokens/s. A 10k-token document takes seconds on a GPU, minutes on CPU.
- *Decode* (producing the answer): strictly sequential and **memory-bandwidth-bound**[^ch05-bw]. Each new token requires reading every weight once. Approximate ceiling: `tokens/s ≈ memory bandwidth / model size in bytes`. A Q4 8B model (~4.5 GB) on a card with 1 TB/s bandwidth tops out around ~200 t/s; on CPU DDR5 (~60 GB/s shared) expect 10–20 t/s.

Context length barely affects decode speed but inflates prefill time super-linearly (attention is O(n²) in sequence length). Practical consequence: don't stuff whole books into context "just in case." Retrieval of the right 2k tokens beats 100k tokens of padding on speed *and* accuracy.

**Quality cost.** Beyond a model's trained window, providers apply RoPE/YaRN extrapolation. Usable but degraded. Even inside the window, recall is not uniform ("lost in the middle": models read best what sits early or late). Long contexts also dilute instructions. Bigger `n_ctx` never improves a task that fits in less.

A chat UI's auto-context manages this budget for you; when you drive llama-server or the API yourself, you own these numbers.

### What tokens/s to expect (orders of magnitude)

| Hardware class | Decode speed, 8B model @ Q4 |
|---|---|
| Laptop CPU | 5–15 t/s |
| Apple Silicon (M-series, unified RAM) | 20–60 t/s |
| Consumer GPU 24 GB (RTX 3090/4090) | 100–140 t/s |
| Data-center GPU (A100/H100) | 130–200+ t/s |

For MoE models, speed tracks the *active* parameters, not total. A 26B-A4B behaves like a 4B dense model at similar quality. Quantisation also changes speed: smaller weights = fewer bytes to move = faster decode, which is why Q4 beats Q8 beats FP16 on throughput.

Measure your own numbers instead of trusting anyone's table:

```bash
llama bench -hf ggml-org/gemma-4-e4b-it-GGUF:Q4_0 -p 128 -n 64 -r 3
# pp128 = prefill throughput at 128-token prompts, tg64 = decode of 64 tokens, 3 runs

On an M-series Mac this prints, for a 1B Q4_K model, ~2 184 t/s prefill and ~115 t/s decode. The table above is orders of magnitude; `llama bench` is your machine's truth.[^ch05-spec]

### Speculative decoding: free speed, paid in memory

Quantisation shrinks the weights; speculative decoding attacks the other bottleneck, the weight read per generated token. Pair the big model with a small drafter: the drafter proposes several tokens in one shot, the big model verifies all of them in a single forward pass and accepts the ones it would have produced itself; where they disagree it corrects. The output distribution is unchanged, so nothing is lost, unlike quantisation.[^ch05-spec]

The arithmetic works because decode is bandwidth-bound: eight drafted tokens cost eight reads of the small weights plus one read of the big ones, instead of eight reads of the big ones. But a drafter that agrees 70 % of the time still decays fast over a chain — acceptance survival drops from 70 % for one token to 34 % over three and 6 % over eight. Keep drafts short; rejection is sequential, so when token 3 of 8 is refused, tokens 4–8 are wasted.

- The drafter has to be good, or most drafts are rejected and you pay memory for nothing. Prefer drafters shipped by the model owners, often in the same Hub repo.
- The price is the drafter's size plus its KV cache, resident next to the main model.
- llama.cpp ships three techniques as of September 2026 — MTP, DFlash and EAGLE3:

```bash
# MTP: drafter ships in the main repo, no extra download
llama serve -hf ggml-org/Qwen3.8-27B-GGUF:Q8_0 --spec-type draft-mtp --spec-draft-n-max 3

# DFlash: separate drafter repo, block-diffusion drafting
llama serve -hf ggml-org/Qwen3.8-27B-GGUF:Q4_K_M \
  -hfd z-lab/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M \
  --spec-type draft-dflash --spec-draft-n-max 7
