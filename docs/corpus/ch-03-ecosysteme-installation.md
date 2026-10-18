# The local AI stack: runtimes, studios, fine-tuners

Three jobs, three layers, usually three different projects:

| Layer | Job | Examples |
|---|---|---|
| **Runtime** | execute weights (GGUF, MLX, safetensors) | llama.cpp, llama.app, Ollama, vLLM, SGLang, MLX |
| **Studio / chat UI** | put a face on a runtime | LM Studio, Jan, llama.app, KoboldCpp, TextGen, Open WebUI |
| **Trainer** | adapt a model to your data | Unsloth, LLaMA-Factory, Axolotl, torchtune, mlx-lm |

None of the three is a product category you *must* choose once: the layers talk to each other over two de-facto standards (GGUF files, OpenAI-compatible HTTP), so you can swap any layer without touching the others. This chapter is the map; the rest of the book stays on the method, not on a vendor.
### Two shapes of tool

| Shape | How you drive it | Examples |
|---|---|---|
| **Library / CLI** | terminal, script, CI pipeline | llama.cpp, llama.app (`llama cli`), Ollama, vLLM, SGLang, torchtune, Axolotl, mlx-lm |
| **GUI / web UI** | browser or desktop app | LM Studio, Jan, KoboldCpp, TextGen, Open WebUI, LLaMA-Factory (WebUI), Unsloth Studio/Desktop |

Two rules of thumb that save weeks:

1. **Most of the ecosystem is CLI-only.** Fine-tuning in particular is dominated by config-file CLIs (Axolotl, torchtune, mlx-lm) and libraries (Unsloth); the no-code WebUIs (LLaMA-Factory, Unsloth Studio, TextGen's Training tab) are thin wrappers over the same recipes.
2. **The GUI layer is interchangeable and thinner than it looks.** Local chat front-ends are 90 % chat template + sampling parameters + a model picker, and they all speak to the same runtimes. Anything you can click, you can also script. Anything you script, you can later click.

If you only remember one thing from this chapter: pick **one** runtime and **one** chat UI, script them, and stop shopping. We have lost more hours juggling stacks than we ever gained by switching between them.
### Who does what

One line per tool, its own documentation as the source. Shapes and capabilities only, no benchmark claims:

**Runtimes (execute models)**

- **llama.cpp**. The C/C++ GGUF engine nearly everything else embeds; library + CLI, no GUI.[^tools-llamacpp]
- **llama.app**. A packaged distribution of the above: `llama cli` for the terminal, `llama serve` for an OpenAI-compatible API *plus* a built-in web UI, model download by Hugging Face repo id.[^tools-llamaapp]
- **Ollama**. CLI with a menu-bar/tray app, its own model registry (`pull`/`run`), GGUF models, OpenAI-compatible API; a runtime, not a trainer.[^tools-ollama]
- **vLLM**. Serving engine for throughput: continuous batching, OpenAI-compatible API, plus Anthropic Messages and gRPC; multi-LoRA adapters.[^tools-vllm]
- **SGLang**. High-throughput serving runtime for LLMs and multimodal models, OpenAI-compatible API, Apache-2.0.[^tools-sglang]
- **MLX / mlx-lm**. Apple silicon only: a Python package for inference **and** LoRA fine-tuning on the unified memory of M-series Macs.[^tools-mlxlm]

**Studios and chat UIs (put a face on a runtime)**

- **LM Studio**. Desktop app (macOS/Windows/Linux): model browser, chat, and a local server with OpenAI- and Anthropic-compatible endpoints plus TypeScript/Python SDKs; no training.[^tools-lmstudio]
- **Jan**. Desktop app: chat, model management, MCP connectors, and a local API server; CLI included.[^tools-jan]
- **KoboldCpp**. A single-file executable that starts a built-in web UI and an OpenAI-compatible API on port 5001; popular for writing/storytelling setups and retro hardware.[^tools-koboldcpp]
- **TextGen** (formerly text-generation-webui). desktop app for local LLMs, "UI + API", no telemetry, with a LoRA **Training** tab.[^tools-textgen]
- **Open WebUI**. A self-hosted chat front-end that connects to Ollama or any OpenAI-compatible endpoint; the usual way to put a team-friendly UI (multi-user, RAG) in front of a server you already run.[^tools-openwebui]

**Trainers (adapt a model)**

- **Unsloth**. Python library (Core) with a web UI (Studio) and a desktop app (Desktop) on top; LoRA/QLoRA/full fine-tuning, RL (GRPO, DPO), continued pretraining, and image/audio/embedding models. Vendor-reported ~2× faster training with ~70 % less VRAM[^tools-unsloth]; per-model figures vary from 1.5× to 3×, so verify on your workload.
- **LLaMA-Factory**. Zero-code WebUI (`llamafactory-cli webui`) over 100+ models: (continuous) pretraining, multimodal SFT, reward modelling, PPO/DPO/KTO/ORPO, with LoRA/QLoRA/full/freeze/Galore/BAdam modes.[^tools-llamafactory]
- **Axolotl**. CLI driven by a single YAML config (`axolotl train config.yml`), local file or URL; the config-as-artifact choice for CI.[^tools-axolotl]
- **torchtune**. PyTorch-native recipes, no wrapper framework: `tune run lora_finetune_single_device --config …`, multi-node support.[^tools-torchtune]

What is *not* in this table: nothing here is required. A GGUF file, `llama.cpp`, and your terminal are enough to run everything in Chapters 4 to 5, and a YAML config plus one of the trainers above is enough to fine-tune (Chapters 6 to 8).
### Setup and installation

Pick one runtime, one chat UI, and one trainer if you intend to train. Three install paths cover almost every case; the WebLLM widget further down needs none of them.

**Path 1. CLI runtime (recommended if you script anything).** llama.app gives you terminal chat, a server, and a web UI from one binary:

```bash
# llama cli   → interactive chat in the terminal (downloads the model once)
# llama serve → OpenAI-compatible API + built-in web UI
llama cli   -hf ggml-org/gemma-4-e4b-it-GGUF:Q4_0
llama serve -hf ggml-org/gemma-4-e4b-it-GGUF:Q4_0

# already have a GGUF file? point at it
llama cli -m my-model.gguf

`llama.cpp` itself (the same engine, built from source or a package) exposes `llama-cli`/`llama-server` with the same model format. Prefer it when you need a specific build, CUDA/Vulkan/ROCm backend, or a smaller binary.[^tools-llamacpp]

**Path 2. Package-managed runtime.** Ollama is the shortest path from zero to a running model, and it registers itself as a local API server:

```bash
curl -fsSL https://ollama.com/install.sh | sh    # macOS/Linux; installer on ollama.com for Windows
ollama pull <model>     # any GGUF model from the registry
ollama run <model>      # chat in the terminal; API on the default local port

**Path 3. GUI-first.** Download the app and skip the terminal entirely: **LM Studio** or **Jan** for a full desktop studio, **KoboldCpp** for a single-file executable, **llama.app** if you want the CLI *and* the web UI in one install. For training without code, install **Unsloth** (Studio/Desktop) or **LLaMA-Factory** (`llamafactory-cli webui`).

Whatever you pick, verify three things before trusting it with real data:

- **Telemetry.** Does it phone home? TextGen advertises "no telemetry"; check the setting, don't assume.[^tools-textgen]
- **Where models and chats live on disk.** You need to know what to back up and what to wipe.
- **Authentication.** A server bound to `0.0.0.0` without a password is an open endpoint on your LAN. Prefer binding to `127.0.0.1` and using an SSH tunnel (below).

### Launch and remote access

Start the runtime, then open the UI it prints at startup (usually `http://127.0.0.1:<port>`). Two access patterns that stay safe:

```bash
# expose on the LAN only if the tool supports auth — most do not by default
# safer: keep it on loopback and forward the port over SSH
ssh -L 8080:localhost:8080 user@server

For a public HTTPS URL, put a reverse proxy or mesh VPN in front (Cloudflare Tunnel, Tailscale, Caddy) rather than exposing the model server directly. Never publish an unauthenticated endpoint, however small the model.

### Hardware reality check

**Chat, embedding and dataset work run on CPU alone.** No GPU needed to follow this book's inference chapters. A 12–16 GB unified-memory laptop runs 8–12 B models at usable speed. **Training** is where the GPU matters: NVIDIA (CUDA) has the widest tool coverage, AMD (ROCm) and Intel (oneAPI/SYCL) are viable with more friction, and Apple silicon trains through MLX (`mlx-lm`) rather than through CUDA-based trainers. No GPU on the machine where you write code? The same scripts run unchanged on a rented GPU box. Copy the data there, never the reverse if the corpus is sensitive.
### Try a model in your browser, zero install

Impatient? The widget below runs a small language model **inside this web page** using [WebLLM](https://github.com/mlc-ai/web-llm) and your GPU via WebGPU. No account, no server, no install. First load downloads ~500 MB once (then cached); works in Chrome/Edge/Safari on any recent machine.

```
<div id="webllm-demo" style="border:1px solid #8888;border-radius:8px;padding:1em;max-width:44rem">
  <p><strong>Zero-install demo.</strong> Loads a 0.5B model (~500 MB, cached in your browser) via WebGPU. Nothing is sent anywhere — the model runs <em>in this page</em>.</p>
  <button id="wl-load" style="padding:.4em 1em">Load model</button>
  <progress id="wl-prog" value="0" max="100" style="width:100%;display:none"></progress>
  <div id="wl-status" style="opacity:.7;font-size:.9em"></div>
  <div id="wl-chat" style="display:none;margin-top:.5em">
    <input id="wl-in" placeholder="Ask something…" style="width:70%;padding:.4em"/>
    <button id="wl-send" style="padding:.4em 1em">Send</button>
    <pre id="wl-out" style="white-space:pre-wrap;background:#8881;padding:.6em;border-radius:4px;min-height:3em"></pre>
  </div>
</div>
<script type="module">
const btn=document.getElementById('wl-load'), prog=document.getElementById('wl-prog'),
      st=document.getElementById('wl-status'), chat=document.getElementById('wl-chat');
btn.onclick=async()=>{
  btn.disabled=true; prog.style.display='inline';
  const {CreateMLCEngine}=await import("https://esm.run/@mlc-ai/web-llm");
  const eng=await CreateMLCEngine("Qwen2.5-0.5B-Instruct-q4f16_1-MLC",{initProgressCallback:p=>{
    prog.value=p.progress*100; st.textContent=p.text;}});
  st.textContent="Ready (offline from cache)."; prog.style.display='none'; chat.style.display='block';
  document.getElementById('wl-send').onclick=async()=>{
    const q=document.getElementById('wl-in'); if(!q.value)return;
    document.getElementById('wl-out').textContent="…";
    const r=await eng.chat.completions.create({messages:[{role:"user",content:q.value}]});
    document.getElementById('wl-out').textContent=r.choices[0].message.content;
  };
};
</script>
It is of course a tiny model. The point is to *feel* that weights are just files executing locally. For real work, follow one of the three install paths above.
