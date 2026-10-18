# Local AI for Your Research. A 30-Minute Primer

## What is "local AI"?

A large language model (LLM) is the technology behind ChatGPT-style assistants. **Open-weight** versions of these models can be downloaded and run entirely on your own computer. No subscription, no internet connection needed after download, and no third party sees anything you type.

Quality is no longer the obstacle: independent benchmarks ([artificialanalysis.ai](https://artificialanalysis.ai)) show the best open models within ~3 points of the best closed ones (59.7 vs 63.1 on their intelligence index), and small models handle everyday research tasks well on an ordinary laptop.
## Why it matters for *your* research

- **Unpublished results stay yours**: manuscripts in progress, review responses, grant drafts never transit a commercial API.
- **Sensitive data becomes usable**: patient records, genomic data or personal data that legally cannot be sent to cloud providers can be analysed with models running inside your institution.
- **No meter running.** Experiment freely: summarise 200 papers, translate, refactor code, with zero per-token cost and no rate limits.
- **Reproducibility**: the exact model version you used can be recorded, archived and cited like any other software dependency.
## Five things to try first
| Task | Prompt pattern | Model size needed |
|---|---|---|
| Summarise a paper or report | Attach the PDF: "Summarise methods and limitations" | Any |
| Literature triage | Paste 20 abstracts: "Group by theme, flag gaps" | Any |
| Code help | "This R/Python script fails with X — fix and explain" | Small is fine |
| FR ⇄ EN translation of your own text | "Translate for an academic audience" | Any; strong French |
| Data wrangling | Attach CSV: "Profile this table, spot inconsistencies" | Small–medium |
[Important:]{fg="#b22222"} always verify factual claims, citations and statistics against the sources. Models sometimes invent plausible-sounding references. Treat output as a very fast draft assistant, not an oracle.
## What it cannot do

- Match the largest closed models on very long, multi-step expert work. Route those tasks elsewhere if you have access.
- Guarantee correctness: hallucinations remain possible; keep a human decision-maker.
- Replace statistical software: use R/Python for inference; LLMs help you *write* that code.
## Try it today

With admin rights, one install covers a runtime *and* a chat UI. The shortest CLI path (macOS/Linux):

```bash
curl -fsSL https://ollama.com/install.sh | sh    # runtime + model registry
ollama run <model>                               # chat in the terminal

Prefer clicking? Download a desktop app instead (LM Studio, Jan) or llama.app, which ships both a terminal chat and a web UI. Then pick a small model (e.g. a Qwen3.5-class 4B) and start chatting. No GPU required. Your IT team can also deploy a shared server. Point them at the chapters below.
## Going further

- **Chapters 1–15 (Parts I–IV)**. Setup, running models, inference parameters, fine-tuning and deployment
- **Chapter 18**. Costs, risks and a 90-day pilot plan for lab directors
- **Chapter 19**. Sovereignty, regulation and investment scenarios for institute leadership
