# A field guide to local-AI projects

The ecosystem moves fast; these projects are maintained, popular and genuinely useful. All run offline.

### Front-ends & assistants

| Project | What it does | Why it's interesting |
|---|---|---|
| **Open WebUI** | polished ChatGPT-style web UI over Ollama/llama.cpp | multi-user, RAG uploads, prompt library. The natural lab portal |
| **LibreChat** | multi-provider chat front-end | one UI for local + remote endpoints, agents & code interpreter |
| **Jan** | desktop ChatGPT alternative (local API server included) | clean UX, OpenAI-compatible server on localhost |
| **AnythingLLM** | document-chat workspaces | turnkey RAG on your PDFs, per-workspace isolation |

### Knowledge & productivity

| Project | What it does | Why it's interesting |
|---|---|---|
| **khoj** | personal AI search over your notes/docs | self-hosted second brain with web+Org/PDF connectors |
| **Paperless-ngx** (+AI) | document management | OCR + LLM tagging of scanned paperwork. Pairs with whisper/RAG |
| **Onyx (ex-Danswer)** | enterprise search + assistant | connectors for wikis/drives, hybrid retrieval |

### Development

| Project | What it does | Why it's interesting |
|---|---|---|
| **Tabby** | self-hosted code completion | GitHub-Copilot-style autocomplete on your own GPU |
| **Continue.dev** | IDE assistant (VS Code/JetBrains) | chat/edit/autocomplete wired to Ollama or any endpoint |
| **Aider / pi** | terminal coding agents | see Chapter 11 for the full harness comparison |

### Creative & home

| Project | What it does | Why it's interesting |
|---|---|---|
| **ComfyUI** | node-based diffusion studio | FLUX/SDXL pipelines, LoRA training, batch image ops |
| **whisper.cpp / Piper** | STT / TTS | see Chapter 14. Meeting minutes and spoken answers |
| **Home Assistant** (Wyoming) | voice assistant at home | full offline "hey Jarvis" with Piper + whisper + an LLM |

Selection criteria we applied: open source, active maintenance (check before adopting), no phone-home telemetry by default, French-community support. Anything here can be swapped later. Prefer standard interfaces (OpenAI-compatible APIs) so tools stay interchangeable.
