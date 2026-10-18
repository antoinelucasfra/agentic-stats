# Building training data from your documents

Getting a clean dataset is the hardest part of fine-tuning, and the least glamorous: 80 % of a fine-tune's quality is decided here, before any GPU spins up. Three ways to get there, all valid[^ch08-builders]:

- **Graph builders**. A node editor where you wire *seed data* → *generation* → *filter* blocks and preview rows as you go (Unsloth Studio's **Data Recipes**, built on NVIDIA NeMo Data Designer, is the most developed example).
- **Config-file builders**. A YAML that maps your files and columns to the training schema and runs generation as a batch job (LLaMA-Factory's dataset configs, or Axolotl's `datasets:` block). Reproducible, CI-friendly, no clicking.
- **Your own script**. 50 lines of Python calling a local model over its OpenAI-compatible API. Slowest to write, easiest to audit, and often enough.

Whichever you pick, the pipeline is the same six steps:

1. Point it at source documents; create a builder, config or script.
2. Add generation steps: synthesise QA pairs, classifications, conversations in your target schema.
3. **Validate** the setup to catch configuration errors early (schema mismatches, empty fields, missing columns).
4. **Preview** sample rows, iterate on prompts until the rows look like work you would have written yourself.
5. **Run** the full build and monitor progress.
6. Feed the resulting dataset (JSONL + its hash) straight into training and fine-tune ([§ Fine-tuning pipeline](06-pipeline-finetuning.qmd#finetune)).

Typical sovereign-AI pattern: feed internal documentation (procedures, tickets, contracts). Generate instruction/output pairs locally. Fine-tune a model that answers in your organisation's vocabulary. No document ever leaves the building.

### A worked example

Goal: a model that answers staff questions about internal IT procedures (a folder of 60 PDFs).

1. **Seed step.** Upload the PDF folder; the builder extracts text per document.
2. **Generation step 1.** "write one realistic question an employee would ask about this document".
3. **Generation step 2.** "answer strictly using the source extract; refuse anything not grounded in it".
4. **Filter step.** Drop duplicates and rows where the answer is a verbatim copy of the question.
5. **Validate → Preview → Run.** Read 20 random preview rows *before* the full run; fix prompt wording while it is cheap.

One afternoon of builder work typically yields 1–3k usable pairs from a corpus that size. Enough for a first LoRA ([§ Fine-tuning pipeline](06-pipeline-finetuning.qmd#finetune)).

### Failure modes

| Failure | Symptom | Counter |
|---|---|---|
| Hallucinated pairs | model invents confident answers at inference time | grounding instruction in the generation block + manual sample review before training |
| Personal data in seeds | PII memorised by the fine-tuned model | anonymise upstream. [§ GDPR mapping](17-gdpr-durcissement.qmd#gdpr) treats the corpus as a processing activity |
| Near-duplicates | inflated dataset, overfit answers | dedup filter block; check with exact-match on normalised question text |
| Schema drift between runs | retraining on inconsistent columns | version the recipe/config; export the dataset file and hash it alongside the training config |

**Our take.** A dataset builder gets you 80 % of a good dataset fast; the last 20 % is reading samples with your own eyes, and no metric substitutes for it. Budget one hour per thousand generated rows to skim them. We have caught leaked personal data, copy-pasted questions and confidently wrong answers in every synthetic corpus we have ever inspected, including our own[^ch08-collapse].
