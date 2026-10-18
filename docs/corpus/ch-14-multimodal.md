# Voice, vision and images

Text is the default. Local stacks now cover ears and eyes too, all offline.

### Speech-to-text: whisper.cpp / faster-whisper

- **whisper.cpp** runs OpenAI's Whisper on CPU (and Metal/Vulkan): real-time transcription of a meeting on a laptop.
- **faster-whisper** (CTranslate2) is a further 2–4× faster inside Python pipelines; `large-v3` for accuracy, `small`/`medium` for speed.
- French works well out of the box; add **pyannote-audio** locally for speaker diarization ("who said what").
- Typical lab pipeline: recorder → whisper.cpp → LLM summary with action items → minutes in your notes system. Everything stays on-premises, which is precisely why legal lets you do it (Chap. 17).

### Text-to-speech: Piper

**Piper** gives fast, small neural voices that run on a Raspberry Pi. French voices exist (`fr_FR-siwis-medium`, `fr_FR-upmc-medium`). Use it to read summaries aloud, voice assistants, or accessibility features. For expressive studio voices, heavier options (XTTS-style cloning) exist but check consent/ethics before cloning any voice.

### Vision

Your chat UI accepts images with vision-capable models (Qwen-VL family, Gemma vision variants). Screenshots, scanned figures, whiteboards, fully offline (Chap. 4). Practical wins: extracting tables from PDFs into CSV, describing plots, reading handwritten lab notes.

In llama.cpp, vision and audio live in the same CLI and server — one model handles text, images and audio:[^ch14-multimodal]

```bash
llama cli -hf ggml-org/gemma-4-e4b-it-GGUF:Q4_0 --image photo.png -p "Describe this image."
# interactive
> /image chart.png
> what is the trend here?
> /audio meeting.mp3
> transcribe this audio

### Image generation

Diffusion models run locally via **ComfyUI** or `stable-diffusion.cpp`: FLUX.2, Z-Image, SDXL checkpoints in GGUF/safetensors. A 24 GB GPU generates 1024px images in seconds; figures for slides and papers (check publisher policy), mockups, dataset augmentation for vision fine-tunes. LoRA training for a house style runs in Studio/Desktop (Chap. 6).

### Sizing rule of thumb

Each modality adds its own model and its own memory budget: Whisper large ≈ 3 GB VRAM, a vision-LLM ≈ weights + image tokens (Chap. 5 KV math applies), diffusion ≈ 6–12 GB. On one 24 GB card, serve *either* an LLM *or* heavy diffusion comfortably. Not both at once. Annex D maps hardware tiers.
