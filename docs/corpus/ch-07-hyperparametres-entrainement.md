# Training hyperparameters, explained

The GUI of your trainer pre-fills sane defaults (rank 16 / alpha 32 / dropout 0.05, LR 2e-4, 3 epochs, batch 4 × grad-accum 8, AdamW 8-bit, linear schedule, warmup 5, seed 3407). The sample config of every CLI trainer does the same. Here is what each actually does, so you know when to deviate.

### Capacity: what the adapter can learn

- **LoRA rank**. The update to every frozen weight matrix is approximated by a product of two small matrices of rank *r*[^ch07-lora]. Rank is the capacity dial: it bounds *how much new behaviour* the adapter can encode. Style, tone, output format → 8–16 is plenty. Dense domain knowledge, many new behaviours at once → 32–64. Above ~64 you mostly buy overfitting risk and slower convergence, not quality. VRAM cost grows slowly with rank (adapters are ~1 % of params).
- **Alpha**. The adapter's contribution is scaled by α/r before being added to the base weights; with fixed rank, alpha is effectively an adapter learning-rate multiplier. Convention α = 2·r keeps that scale constant as you change rank. If you raise rank without touching alpha, you silently weaken each direction. Keep the ratio unless you have a reason.
- **Dropout**. Randomly zeroes adapter activations during training (0.05 default): cheap insurance against memorising small datasets. Raise to 0.1 when eval loss climbs while train loss falls; set 0 only for quick throwaway runs.
- **Target modules**. Which weight matrices get adapters. All attention + MLP projections (the common default) strictly dominates attention-only: knowledge lives mostly in the MLP blocks.

### Speed of learning: getting to good weights safely

- **Learning rate**. The single most sensitive knob. It sets the step size of gradient descent. Too high: loss spikes, gradient-norm explosions, catastrophic forgetting of general ability; too low: hundreds of wasted steps on a flat plateau. QLoRA/LoRA sweet spot ≈ 1e-4–3e-4 (2e-4 default)[^ch07-lr]; full fine-tuning needs ~10× less (1e-5–2e-5) because you move *all* weights.
- **Warmup** (steps or ratio). Starts training with tiny LR and ramps up. Needed because the Adam optimizer's momentum statistics are cold and wrong in the first steps; without warmup those bad steps can do real damage. 3–10 % of total steps is standard.
- **LR scheduler** (linear/cosine decay). Big steps early when far from optimum, small steps late for fine settling. The LR chart in your trainer's UI lets you verify it fired correctly.
- **Batch size × gradient accumulation**. Effective batch = product of both (default 4 × 8 = 32). Bigger batches smooth the gradient estimate (more stable, slightly more generic results); smaller batches are noisier but sometimes generalise better. Memory scales with physical batch × sequence length. Grad accumulation is how you simulate large batches on small GPUs. Zero quality cost, only wall-clock cost.
- **Epochs**. Full passes over the dataset. 1–3 for SFT; beyond that the model memorises rather than learns (train loss keeps dropping, eval loss turns upward. The U-turn is your stop signal). When in doubt, train 1 epoch and look.
- **Weight decay**. L2 penalty pulling unused weights toward zero; leave at default for LoRA (adapters are tiny; decay fights capacity you paid for).

### The remaining defaults

| Parameter | Default | Why this value |
|---|---|---|
| Max sequence length | 2048 | Must cover your longest example *including* prompt template; longer examples get silently truncated. The model never sees their answer end, and learns from broken pairs. Raising it costs memory quadratically (attention) unless flash attention is on, which most trainers and runtimes enable by default. |
| Optimizer | AdamW 8-bit | AdamW = per-parameter adaptive step sizes; the workhorse of transformer training. 8-bit state shrinks optimizer memory ~75 % vs a standard 32-bit state with no measurable quality loss[^ch07-8bit]. That is why QLoRA fits in small GPUs. |
| Gradient checkpointing | on | Recomputes activations instead of storing them: big VRAM savings for ~20–30 % slower steps. Good implementations (Unsloth's among them) cut that penalty substantially. |
| Seed | 3407 | Any fixed number makes runs reproducible; 3407 is a popular default (from Picard's study "torch.manual_seed(3407) is all you need"[^ch07-seed]). |
| Eval split & steps | optional | Enables the Eval-Loss chart, the only trustworthy overfitting signal. Always keep one for real projects. |

### Reading the training curves like an engineer

- **Train loss** (EMA-smoothed view exists precisely because raw loss is noisy): healthy SFT lands around **0.5–1.0**. Loss ≈ 0 means memorisation; a flat line near start value means LR too low or data misformatted.
- **Eval loss**: should track train loss downward and flatten. Divergence between the two = overfitting onset = time to Stop & Save.
- **Gradient norm**: occasional spikes are normal; repeated spikes mean LR too high or dirty data (a malformed example).
- **GPU monitor**: utilisation pegged near 100 % means compute-bound (good). Low utilisation points at a data-loading bottleneck. Keep memory under the card's limit with headroom for context spikes; watch temperature. Sustained >85 °C throttles your tokens/s.
