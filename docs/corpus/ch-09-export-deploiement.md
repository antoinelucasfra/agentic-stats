# Export and deployment

Export screen in your trainer (or its CLI equivalent): pick a training run → pick a checkpoint → choose the target:

| Export type | Produces | Use with |
|---|---|---|
| **Merged model** | 16-bit weights with adapter merged | vLLM, further training, sharing |
| **LoRA only** | Adapter weights alone | Reusable on the same base, tiny files |
| **GGUF** | Quantised llama.cpp format | any GGUF runtime: llama.cpp, llama.app, Ollama, LM Studio, KoboldCpp |

Save locally or push to the Hugging Face Hub (write token needed). Deployment options from there:

- **Chat inside the same GUI**. Instant, nothing to deploy.
- **Ollama**: `ollama create assistant -f Modelfile` from the exported GGUF, then `ollama serve`.
- **llama-server / vLLM / SGLang**: OpenAI-compatible HTTP endpoints for apps.
- **Your studio as API server**: most studios and trainers ship a serve command that loads the export with configurable sampling, context size and GPU layers (llama.app's `llama serve`, Unsloth's `unsloth run`, LM Studio's Developer tab, …).
- **Remote access**: bind to `0.0.0.0` when the tool supports authentication; otherwise keep it on loopback and forward the port over SSH (a tunnel such as Cloudflare or Tailscale for public HTTPS).

### Quantisation cheat-sheet

Not all GGUF quantisations are equal. Sizes are approximate fractions of the 16-bit original[^ch09-quants]:

| Quant | Size | Quality | When |
|---|---|---|---|
| `Q2_K` | ~20 % | noticeably degraded | emergency CPU-only fits; avoid for production |
| `Q3_K_M` | ~25 % | usable, lossy | tight VRAM ceilings |
| `Q4_K_M` | ~31 % | sweet spot | the classic default |
| `UD-Q4_K_XL` | ~32 % | recovers most 4-bit loss | Unsloth Dynamic quants; runtimes also publish their own 4-bit variants (MLX 4-bit, `IQ4_XS`) |
| `Q5_K_M` | ~36 % | near-transparent | quality-critical single-user serving |
| `Q6_K` | ~41 % | very close to fp16 | reference comparisons |
| `Q8_0` | ~53 % | effectively lossless | A/B baseline to judge lower quants against |

Rule: pick the largest quant your hardware serves at acceptable speed, then verify on your golden set ([§ Evaluate](16-evaluer.qmd#evaluate)). Labels lie less than averages do.

### Serving from the GGUF

Ollama wrapper with a baked-in persona and sampling:

```bash
cat > Modelfile <<'EOF'
FROM ./assistant-UD-Q4_K_XL.gguf
PARAMETER temperature 0.7
SYSTEM """You answer internal IT questions. Cite the procedure name. Refuse anything else."""
EOF
ollama create assistant -f Modelfile && ollama serve

Raw llama.cpp for full control:

```bash
llama-server -m assistant-UD-Q4_K_XL.gguf -c 32768 -ngl 99 --port 8080

`-ngl 99` pushes every layer to GPU; `-c` is the context window. Size it to your worst-case prompt, not your average one ([§ Inference parameters](05-parametres-inference.qmd#params)).

### Choosing the target

| Situation | Target |
|---|---|
| You alone, iterating | chat UI, nothing to deploy |
| Team of ≤ 10 on one machine | Ollama or llama-server on the GPU box |
| Application backend | any OpenAI-compatible server: llama-server, vLLM, Ollama, or your studio's API |
| Many concurrent users, high throughput | vLLM on merged 16-bit weights ([§ Team serving](15-serving.qmd#serving)) |
| Air-gapped distribution | GGUF file + Modelfile in a versioned archive |

**Our take.** The GGUF is the portable currency of this whole stack: it runs everywhere, survives tool churn, and drops into Ollama, llama.cpp or any GUI unchanged. We keep merged 16-bit masters archived like source code and treat exported GGUFs as build artefacts. Rebuildable, disposable, versioned by dataset hash and config version, never hand-edited.
