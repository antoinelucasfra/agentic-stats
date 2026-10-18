# Hardware: what to buy, in what order

Speed expectations per tier were given in Chapter 5; this annex is the buying guide. Prices are rough early-2026 street prices.

### The ladder

| Tier | Hardware | Runs | Price | Verdict |
|---|---|---|---|---|
| 0 | Any laptop CPU (16 GB RAM) | 8 B Q4 chat at 5–15 t/s | €0 | enough to *start today* |
| 1 | Mac M-series Pro/Max (32–64 GB unified) | up to ~32 B Q4 comfortably | €2–4.5k | best all-round dev machine; silent |
| 2 | Desktop + used **RTX 3090** (24 GB) | 8–32 B Q4 fast, fine-tunes via QLoRA | €1–1.5k total | the price/perf sweet spot |
| 3 | RTX 4090 / 5090 (24/32 GB) | same, 1.5–3× faster; bigger contexts | €2–3k+ | for daily heavy use |
| 4 | Used A100/H100 80 GB or 2×4090 server | 70 B class, team serving (Chap. 15) | €10–25k | the lab server stage |

### Reading the table

- **Unified memory rule** on Apple silicon: a model needs roughly its file size free — a 32 GB MacBook runs 27–32 B Q4 GGUFs well because the GPU can use ~75% of unified memory.
- **The used 3090 is the community's workhorse**: 24 GB VRAM, cheap NVLink option, proven for QLoRA fine-tunes of up to ~14 B and comfortable inference to 32 B. Buy from refurbishers with warranty; stress-test VRAM (`gpu-burn`, `memtest_vulkan`) on arrival.
- **VRAM beats everything else.** A slower card with more memory outruns a faster small one the moment you exceed context or model size (Chap. 5).
- **Dense wants bandwidth, big MoEs want capacity.** A dense model reads every weight for every token, so it wants the fastest memory that fits it: VRAM. A MoE only reads its active experts — a 90B-A10B reads ~10 B per token — so it runs well on slower unified memory that would starve a dense model of the same total size. On one machine, small MoEs are happy in VRAM; large ones are the case for 64 GB+ Macs.[^hw-moe]
- **Disk**: models are 5–60 GB each; one NVMe for hot models, cheap SATA for the archive. Swapping models from SSD takes seconds.
- **Power & noise**: a 3090 draws ~350 W peak (undervolting cuts ~20% for ~5% speed); a lab server belongs in a machine room, not an office.
- **Clusters don't help decode**: splitting one LLM across Ethernet kills token speed (interconnect latency ≫ bandwidth rule of Chap. 5). One big GPU > four small ones for inference; clusters matter for *training* only.
- **Don't forget the humans**: one shared server (Tier 4) + Tier 0–1 client machines covers most labs better than everyone buying Tier 3 desktops — that's exactly the TCO argument of Chap. 18.

**Our take.** Buy VRAM, not benchmarks: a used 3090 has outlived two generations of "faster" cards we skipped, and the money we did not spend on a flagship bought the second one. Skip the CPU-only detour entirely. It teaches you the concepts and nothing else. If budget allows exactly one machine, make it 24 GB and shared.

![Decode speed by tier, from the ladder above. The gap to internalize: hardware buys roughly an order of magnitude, quantization buys back most of the model size at ~95% quality (Q4).]

![A 32 B model on disk across GGUF quantization levels — size (bars) vs retained task quality (line). Q4_K is the community default because it halves memory twice over FP16 while keeping ~95% of quality; below Q3 the curve falls off a cliff. Illustrative averages; UD-Dynamic quants recover part of the gap.]

### You already have an institute cluster? Use it

Most public labs already own the hardware. A classic HPC setup: **login nodes** (internet, no GPU), **GPU compute nodes** behind a scheduler (SLURM), a **shared filesystem** (`/home`, `/scratch` or Lustre), **compute nodes with blocked internet egress**, and Apptainer for containers. That is everything local AI needs. Three recipes:

**1. Bring weights in (once, from the login node).** Compute nodes can't download; the login node can:

```bash
export HF_HOME=/scratch/$USER/hf          # models land on shared storage
pip install -U huggingface_hub
hf download ggml-org/gemma-4-e4b-it-GGUF \
   Qwen3-8B-UD-Q4_K_XL.gguf --local-dir /scratch/$USER/llm/

Put models in a **group-writable directory** so the whole lab shares one copy.

**2. Serve it.** Two modes:

- **Dedicated serving node** (best daily experience): ask IT to pin one GPU node outside (or under a long reservation of) the scheduler, then run `llama-server` from a prebuilt llama.cpp binary. No root needed. Exactly as in Chap. 15. Put the LiteLLM gateway on top. The whole Part II/IV tooling then points at one stable `base_url`.
- **Per-job serving** (zero privileges):

```bash
#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 --mem=16G --time=08:00:00
cd /scratch/$USER/llm
./llama-server -m Qwen3-8B-UD-Q4_K_XL.gguf -ngl 99 -c 16384 \
               --host 0.0.0.0 --port 8080 &
until curl -sf localhost:8080/health >/dev/null; do sleep 2; done
python run_eval.py --base-url http://localhost:8080/v1   # your workload

Job ends → server dies. Ideal for batch pipelines and golden-set evaluations (Chap. 16); for interactive chat prefer the dedicated node.

**3. Train without drama.** A QLoRA fine-tune is a *single-GPU* job: `#SBATCH --gres=gpu:1` on any partition with 24 GB+ cards. Never request multi-node for LLM fine-tuning. Clusters help training throughput, not single-model quality (Chap. 5). Keep datasets + adapter checkpoints on `/scratch`; every trainer in [§ The local AI stack](03-ecosysteme-installation.qmd#ecosystem) runs unchanged under SLURM.

### Daily use once an endpoint exists

Everything in this book speaks the OpenAI-compatible protocol. Point each tool at the cluster endpoint and you're done:

| Tool | Where to set it |
|---|---|
| Chat UI | Settings → provider connection (Chap. 4) |
| pi / coding agents | `models.json` → `baseUrl` (Chap. 11) |
| Python scripts | `base_url` argument (Chap. 12) |

Cluster etiquette keeps everyone happy: one model per GPU (serve copies, don't tensor-split small models), heavy eval batches off-hours, cap context requests, and route team access through the gateway's per-user keys (Chap. 15).
