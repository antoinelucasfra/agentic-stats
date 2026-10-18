# Running models: chat interfaces

A chat UI is the shortest path from "model on disk" to "answer I can use", and it runs 100 % offline. Your pick in [§ The local AI stack](03-ecosysteme-installation.qmd#ecosystem) does not change the first ten minutes. LM Studio, Jan, llama.app's web UI, KoboldCpp, TextGen, Open WebUI, and the studios that also train all start the same way:

1. **Model picker** → search Hugging Face or pick a local file. Choose the quantisation matching your RAM/VRAM: a 4-bit quant is the usual default (`Q4_K_M` for classic GGUFs, `UD-Q4_K_XL` for Unsloth Dynamic quants, 4-bit MLX on Apple silicon).
2. Download happens in the background; GGUFs you already have (Hugging Face cache, another app's model folder) are usually auto-detected.
3. Chat. Most UIs accept images, audio, PDFs, DOCX and code files as context, and expose thinking and web-search toggles per message.

What the good ones add on top, all optional per message:

- **Sane inference defaults**: temperature/top-p/top-k pre-tuned per model family, context sized automatically by the runtime. What those knobs actually do → [§ Inference parameters](05-parametres-inference.qmd#params).
- **Tool calling and sandboxed code execution** (Bash, Python): the model can run real code to check its own answer. Some studios go further and *repair* malformed tool calls instead of ending the turn with an error[^ch04-tools].
- **Web search / deep research**: search plus page reading, so answers cite sources. Privacy depends on the search backend. Only a private or self-hosted backend keeps "offline" true.
- **Side-by-side comparison**: two models, one prompt, same screen (`Model Arena` in Unsloth Studio, a second chat tab or two app windows elsewhere). It is the cheapest qualitative evaluation you will ever run.
- **Providers**: connect OpenAI/Anthropic/Ollama/vLLM endpoints in the same interface. Useful for migrating from cloud to local model by model.

### Getting good answers

- **System prompt**: set role, output format and refusal boundaries once per conversation; it persists across messages and models. Save recurring ones as presets.
- **Thinking toggle**: leave on for maths, logic and code; switch off for short factual or stylistic requests. Reasoning tokens cost latency even locally.
- **Attachments**: images and audio need a vision/voice-capable model; PDFs, DOCX and code files are parsed to text and injected into context. On a 40-page document, ask section-by-section rather than "summarise everything".
- **Web search off = sovereign mode**: toggle off, not a byte leaves the machine. This is the configuration you describe in the DPIA ([§ GDPR mapping](17-gdpr-durcissement.qmd#gdpr)).

### Practical recipes

| Task | Setup | Tip |
|---|---|---|
| Summarise a long report | attach PDF, thinking on | summarise section-by-section then a global synthesis. Better than one giant pass |
| Translate with house style | paste glossary into system prompt | the model fine-tuned from your documents ([§ Building training data](08-data-recipes.qmd#data)) already speaks your vocabulary |
| Review a patch | paste diff, sandbox execution on | ask it to *run* the tests, not just read them |
| Transcribe meeting audio | attach recording + voice model | demand speaker turns and action items in one prompt |
| Compare two candidates | two chats side by side, same prompt ([Unsloth Studio's Model Arena](https://unsloth.ai/docs), or two windows) | fastest qualitative [evaluation](16-evaluer.qmd#evaluate) you can run |

**Chat UI vs API server**: the chat UI is for one human iterating alone; the moment an application or another team member needs the model, serve it. [§ Export](09-export-deploiement.qmd#export) and [§ Team serving](15-serving.qmd#serving).

**Our take.** The chat pane is the most underrated evaluation bench in the whole stack. Before writing any benchmark script, we spend fifteen minutes throwing our ugliest real prompts at the candidate side by side with the incumbent. The failures that surface, rambling, ignored constraints, timid hedging, predict benchmark results surprisingly well, and they are legible to non-technical stakeholders, which benchmarks never are.
