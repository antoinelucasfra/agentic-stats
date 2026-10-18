# Evaluate before you trust

Never deploy a model on institutional work without a measurement. Three layers:

1. **Public benchmarks for shortlisting**. [AA Intelligence Index](https://artificialanalysis.ai/models) and [Coding Agent Index](https://artificialanalysis.ai/agents/coding-agents) rank candidates; runtime projects publish perplexity and KL-divergence per quantisation (llama.cpp's quantisation reference, Unsloth's GGUF benchmark pages). Shortlist, never conclude: public scores are averages, your domain is not.
2. **Your own golden set**. Collect 50–200 real tasks from the target workflow (with reference answers where possible), keep them out of any training data, and score pass@1 before and after every model, quantisation or fine-tune change. This is the acceptance gate referenced in the strategy paper (Chap. 19).
3. **Side-by-side inspection**. Two chats side by side (Unsloth Studio's Model Arena, or two windows): same prompt to base and candidate model; a 15-minute structured read of divergences catches failure modes no aggregate metric shows.

Quantisation rule: prefer the dynamic quants (`UD-*` from Unsloth, MLX 4-bit). They recover most of the accuracy standard 4-bit formats lose. When in doubt, compare the candidate quant against a higher precision on your golden set rather than trusting the label.

### Reading the quant names

Two families exist: post-training quantization (most labels you will meet) and quantization-aware training, where the model trains with the precision loss in the loop. Everything below is PTQ. And only repos with a `-GGUF` suffix are runnable; on the Hub, filter by the llama.cpp app, sort by downloads, and read the quantisation column rather than the model name.[^ch16-quant]

Hub listings are a wall of acronyms. They decode mechanically:[^ch16-quant]

| Label | Meaning |
|---|---|
| `Q4_0`, `Q8_0` | Legacy: symmetric, 0-centred scale. Runs everywhere, superseded by k-quants. |
| `Q4_K_M` | 4-bit k-quant, medium variant. K-quants quantize the scale too (weights 4-bit, scales 8-bit) and nest bins within bins to contain rounding errors. Mid-sized models save a few GB that would otherwise hold scaling constants. |
| `IQ4_XS` | 4-bit i-quant, extra-small. Weights are matched against a fixed codebook of lattice points; an importance matrix computed from sample text decides which weights deserve higher precision. Higher quality than k-quants at equal bits, slightly more compute per block. |
| `UD-*` | Unsloth Dynamic: quantizes the bulk low, upcasts information-dense and sensitive layers to higher precision. The default choice (rule above). |
| `MXFP4`, `NVFP4` | Modern microblock formats: Blackwell GPUs only. GGUF quants run everywhere. |

Rules of thumb: 4, 6 or 8-bit for medium models; avoid 1- and 2-bit unless the model is already large and strong — a weak model has no spare precision to donate.

You can quantize your own fine-tune:

```bash
# k-quant, 4-bit
llama quantize -hf ggml-org/gemma-4-e4b-it-GGUF Q4_K_M

# i-quant: importance matrix first, then quantize with it
llama-imatrix -m -hf ggml-org/gemma-4-e4b-it-GGUF -f calibration.txt -o imatrix.gguf -ngl 99
llama quantize --imatrix imatrix.gguf -hf ggml-org/gemma-4-e4b-it-GGUF model-IQ4_XS.gguf IQ4_XS

### Building the golden set

1. **Harvest real tasks**. Past tickets, real emails, actual reports from the target workflow. Synthetic tasks evaluate the model's ability to do synthetic tasks.
2. **Reference answers where cheap**. Exact expected output for extraction/classification; key-facts checklists for prose; unit tests for code. Where no ground truth exists, leave the row for human scoring instead of inventing one.
3. **Freeze and version**. `golden.jsonl` + a hash, stored next to the training configs. It must never appear in any training corpus. Contamination turns the gate into a mirror.
4. **Refresh quarterly**. Retire rows the models have all mastered; add rows from new failure modes you observed in production.

### Scoring without fooling yourself

- **Deterministic first**: exact/regex match for extraction, pass@1 with executed tests for code, checklist coverage for summaries. Cheap, reproducible, impossible to argue with.
- **LLM-as-judge only without ground truth**[^ch16-judge], with a written rubric (criteria, scale, refusal handling) and *one fixed judge model* recorded alongside results. Judges inherit biases. Calibrate yours against ~30 human-scored examples and report agreement before trusting it.
- **Set the gate before running**: "promoted if ≥ baseline − 2 points" decided in advance beats post-hoc rationalisation every time.

### The regression gate

Every change to model, quantisation or fine-tune re-runs the same script:

```bash
./eval.sh --golden golden.jsonl --candidate <model-or-adapter> \
          --baseline results/baseline.json --threshold 2

Exit non-zero blocks promotion. Wire it into the same CI that rebuilds the adapter ([§ Fine-tuning in Python](12-voie-python.qmd#core)). Keep every `results/*.json`: the score history is your evidence file for [§ GDPR accountability](17-gdpr-durcissement.qmd#gdpr) and the business case ([§ Economics](18-dossier-economique.qmd#bizcase)).

### Memorisation probe

Fine-tuned models can parrot their training corpus. Before any deployment that touches personal data:

- Plant or pick 10–20 canary strings (names, reference numbers) present in the training data.
- Prompt for them directly and via paraphrase; expect zero recall.
- Any hit → the corpus was not cleaned; go back to [§ Building training data](08-data-recipes.qmd#data), not to a disclaimer.

A minimal report reads: date · candidate (name, quant, config hash) · golden-set hash · scores per metric · delta vs baseline · decision. One page, versioned. Enough to reconstruct the decision two years later.

**Our take.** Evaluation is the cheapest insurance sold in this book: one weekend building a golden set protects every future model decision, and it is the only chapter of this stack your auditors will actually read. We have watched organisations debate benchmarks for months while a 60-row spreadsheet of their own real tasks would have settled the question in an afternoon.
