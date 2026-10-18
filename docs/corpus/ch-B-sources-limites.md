# Sources

- Local-stack documentation, one page per tool (all checked 2026-10-04): [llama.app](https://llama.app/docs/introduction), [llama.cpp](https://github.com/ggml-org/llama.cpp), [Ollama](https://ollama.com/docs), [LM Studio](https://lmstudio.ai/docs), [Jan](https://jan.ai/docs), [KoboldCpp](https://github.com/LostRuins/koboldcpp), [TextGen](https://github.com/oobabooga/text-generation-webui), [Open WebUI](https://docs.openwebui.com), [LLaMA-Factory](https://llamafactory.readthedocs.io), [Axolotl](https://docs.axolotl.ai), [torchtune](https://pytorch.org/torchtune), [mlx-lm](https://github.com/ml-explore/mlx-lm), [vLLM](https://docs.vllm.ai), [SGLang](https://github.com/sgl-project/sglang).
- Unsloth documentation (one trainer among the above): [studio overview](https://unsloth.ai/docs/new/studio), [getting started](https://unsloth.ai/docs/new/studio/start), [installation](https://unsloth.ai/docs/new/studio/install), [chat](https://unsloth.ai/docs/new/studio/chat), [Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe), [export](https://unsloth.ai/docs/new/studio/export), [API endpoint](https://unsloth.ai/docs/basics/api), [Claude Code](https://unsloth.ai/docs/basics/claude-code), [Codex](https://unsloth.ai/docs/basics/codex), [fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [Desktop](https://unsloth.ai/docs/desktop)
- LMArena : [classement](https://lmarena.ai) (duels à l'aveugle arbitrés par la foule : texte, webdev, vision). À consulter au moment de décider.
- Artificial Analysis: [models leaderboard](https://artificialanalysis.ai/models), [coding agents](https://artificialanalysis.ai/agents/coding-agents), [AI trends](https://artificialanalysis.ai/trends) (Intelligence Index v4.1.1, Coding Agent Index v1.4; snapshot early 2026)
- Philip Kiely & Baseten, [Inference Engineering](https://baseten.co/inference-engineering/) — the inference-engine internals behind the Chapter 5 speed math.
- Hugging Face, *Local AI Ecosystem with llama.cpp* (talk deck, September 2026): [source deck](https://www.canva.com/design/DAHXJy7MerE/IkE4Ri9hWWNOnxz6azJTKg/view). The llama.cpp ecosystem walkthrough digested into Chapters 3, 5, 14–16 and Annex D.
## Scientific references
Peer-reviewed papers and primary technical sources backing the quantitative claims in this book (footnotes in chapters give context):

- **LoRA**. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021 ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685))
- **QLoRA**. Dettmers, Pagnoni, Holtzman, Zettlemoyer, *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023 ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314))
- **8-bit optimizers**. Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, ICLR 2022 ([arXiv:2110.02861](https://arxiv.org/abs/2110.02861))
- **Grouped-Query Attention**. Ainslie et al., 2023 ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245))
- **Inference memory/bandwidth analysis**. Kim et al., *Full Stack Optimization of Transformer Inference*, 2023 ([arXiv:2303.14005](https://arxiv.org/abs/2303.14005)); Sheng et al., *FlexGen*, 2023 ([arXiv:2303.06965](https://arxiv.org/abs/2303.06965))
- **RAG**. Lewis et al., NeurIPS 2020 ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401)); dense retrieval: Karpukhin et al., EMNLP 2020 ([arXiv:2004.04906](https://arxiv.org/abs/2004.04906))
- **Multilingual embeddings**. Chen et al., *BGE M3-Embedding*, 2024 ([arXiv:2402.03216](https://arxiv.org/abs/2402.03216))
- **LLM-as-judge**. Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023 ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685))
- **Synthetic-data degradation**. Shumailov et al., *AI models collapse when trained on recursively generated data*, Nature 631, 2024
- **Random seeds**. Picard, *"torch.manual_seed(3407) is all you need"*, 2021 ([arXiv:2109.08203](https://arxiv.org/abs/2109.08203))
- **GGUF quantisation sizes/perplexity**. Llama.cpp quantization reference ([tools/quantize/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)); k-quants: [PR #1684](https://github.com/ggml-org/llama.cpp/pull/1684)
- **GDPR**. Regulation (EU) 2016/679, official text ([EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj)), in particular Art. 22 (automated decisions) and Art. 28 (processors)
- **Unsloth performance figures**. Vendor benchmarks from the [Unsloth repository](https://github.com/unslothai/unsloth) and [blog/changelog](https://unsloth.ai/blog); per-model results vary (1.5×–3×)
## Limits
- Model leadership shifts **weekly**: every ranking here is a dated snapshot of a moving field. Before any decision, re-check live sources (artificialanalysis.ai, lmarena.ai) and apply the selection method of Chapter 2 — adapted to your use case and context, validated on your own golden set.
- Same for harness+model pairs in coding agents: benchmark the pair you plan to run on your own tasks rather than trusting any published table.
- A fine-tuned mid-size model is near-SOTA **on its domain**, not a frontier generalist; route hard general questions accordingly.
- Hallucinations remain possible. Keep a human in the loop for decisions affecting individuals (Art. 22 restricts fully automated decisions).
- Quantisation (4-bit) still costs some accuracy; benchmark UD quants vs higher precision before choosing.
- The GUI layer moves fast (several studios are beta): pin versions in production, and expect menu names to differ between tools.
