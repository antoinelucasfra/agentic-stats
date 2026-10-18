# Fine-tuning in Python: the QLoRA loop

When the GUI is not enough, the same loop is scriptable. The listing below is **Unsloth Core**, the most widely used Python trainer. Axolotl, torchtune and LLaMA-Factory express the same five steps as a YAML config or a recipe[^ch12-trainers]:
Every trainer has a headless equivalent for automation. `unsloth train|inference|export`, `axolotl train config.yml`, `llamafactory-cli train …`, `tune run …`. The YAML a GUI saves replays as-is in CI.

### Installing

Linux or WSL2 with a CUDA 12.x driver (Unsloth is the listing's example; the same loop installs as Axolotl, torchtune or LLaMA-Factory):

```bash
pip install unsloth

Pin the version in your project (`pip freeze | grep -i unsloth`). The stack moves fast and reproducibility beats novelty. No GPU on the machine that writes the code? The same script runs unchanged on a rented GPU box; only data must be copied there, never the other way round if the corpus is sensitive.

### From snippet to production run

The listing above is the skeleton. A production run adds four things:

1. **A real dataset file**. JSONL with one instruction/output pair per line, produced by [§ Building training data](08-data-recipes.qmd#data) or hand-curated. Hash it and record the hash next to the config: `sha256sum internal_corpus.jsonl > corpus.sha256`.
2. **Deliberate hyperparameters**. `r=16` is a starting point, not a decision; learning rate, epochs and rank rationale live in [§ Hyperparameters](07-hyperparametres-entrainement.qmd#hyper).
3. **Evaluation before promotion**. Score pass@1 against your golden set before the adapter goes anywhere near production ([§ Evaluate](16-evaluer.qmd#evaluate)):

```python
FastLanguageModel.for_inference(model)
for task in load_dataset("json", data_files="golden_set.jsonl")["train"]:
    out = generate(task["instruction"])   # score against task["reference"]

4. **Versioned export**. `save_pretrained_gguf` output plus the YAML config and dataset hash in one archive ([§ Export](09-export-deploiement.qmd#export)).

### Common failures

| Symptom | Usual cause | Fix |
|---|---|---|
| CUDA out of memory | sequence length × batch too big for VRAM | lower `max_seq_length`, batch to 1 + gradient accumulation |
| Loss is 0 or NaN | learning rate too high, or broken chat template | drop LR an order of magnitude; verify tokenizer template |
| Gibberish after merge/export | template mismatch between train and inference | train and serve with the *same* template; test before deploying |
| Training crawls | gradient checkpointing / flash attention off | enable both; check layer offloading (`-ngl` on llama.cpp) at inference |

**Our take.** Stay on a trainer GUI until you retrain weekly. Then the Python path pays for itself in one afternoon of automation. The moment training becomes a script, it also becomes CI: dataset hash in, evaluated adapter out, no human clicking through screens at midnight before a deadline. That last part is where sovereign AI actually gets boring. Which is the compliment.
