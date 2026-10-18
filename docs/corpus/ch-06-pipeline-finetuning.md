# Fine-tuning without code: the training pipeline

Every training GUI walks through the same four areas. LLaMA-Factory's WebUI, Unsloth Studio and TextGen's Training tab all ask for the same six answers before they can start.[^ch06-trainers] Underlying concepts:

**Methods.** What changes the model's weights:

| Method | Mechanism | VRAM | When |
|---|---|---|---|
| **SFT** (supervised fine-tuning) | Train on instruction→output pairs | n/a | Teach format, tone, domain knowledge |
| **QLoRA** | Freeze 4-bit base, train low-rank adapters (~1 % of params) | Lowest | Default choice; matches LoRA quality since dynamic 4-bit[^ch06-qlora] |
| **LoRA** | Same adapters on a 16-bit base | Medium | When serving will be 16-bit |
| **Full FT** | Update all weights | Highest | Rarely needed; if LoRA fails, FFT won't save it |
| **RL: GRPO / DPO** | Learn from rewards or preference pairs instead of labels | n/a | Tool-calling behaviour, reasoning, alignment to taste |

Fine-tuning can inject knowledge *and* behaviour. It can replicate what RAG does, not vice versa. Loss around 0.5–1.0 is typically healthy. Loss ≈ 0 means overfitting, so check eval loss.

![The sovereign fine-tuning loop: documents become a dataset via a builder, QLoRA produces checkpoints, export ships GGUF, and the golden-set evaluation gate feeds failures back into the dataset.]

**Our take.** Fine-tuning is the most fun part of this book and the one we recommend least often. In our experience, 8 out of 10 "we need to fine-tune" requests dissolve once the team writes a real system prompt, a few good few-shot examples, or a small RAG loop. Train when you need *behaviour* the base model refuses to adopt. Format discipline, domain tone, tool-calling style. Not when you merely need it to know things.

**The four areas of any trainer GUI:**

1. **Model & method.** Pick modality (text/vision/audio/embeddings), choose the method (QLoRA/LoRA/Full), then type any HF model name. Gated models (Llama, Gemma) need an HF token, validated inline. Defaults are pre-filled automatically. GGUF files are inference-only. Training needs safetensors. CLI trainers (Axolotl, torchtune) ask the same questions as flags in a YAML file.[^ch06-trainers]
2. **Dataset.** Pull from Hugging Face Hub or drop local `PDF / DOCX / JSONL / JSON / CSV / Parquet`. Format detection: `auto`, `alpaca` (instruction/input/output), `chatml` (messages array), `sharegpt`. Choose train/eval splits (an eval split enables the Eval-Loss chart) and optionally slice rows for quick experiments. A column-mapping dialog opens when roles aren't obvious.
3. **Hyperparameters.** Sound defaults (context 2048, LR 2e-4, LoRA rank 16 / alpha 32 / dropout 0.05, all attention+MLP projections targeted, 3 epochs, batch 4 × grad-accum 8, AdamW 8-bit, linear scheduler, warmup 5, gradient checkpointing, seed 3407). Each one explained in [§ Training hyperparameters](07-hyperparametres-entrainement.qmd#hyper). Optional W&B / TensorBoard logging.
4. **Training & config.** Save/Upload/Reset the whole config as **YAML** (`{model}_{method}_{dataset}_{timestamp}.yaml`). Reproducible and versionable, and directly reusable by a CLI trainer. Start Training validates inline (e.g. eval steps without eval split).

**Observability during training:** live loss (raw + EMA-smoothed + average), LR schedule, gradient norm, eval loss; GPU monitor (util%, °C, VRAM, watts) polled every few seconds; ETA and tokens/s; viewable from your phone. Stop & Save checkpoints at any time.
