# The Business Case for Local AI

## Executive summary

Open-weight language models now sit within ~3 points of the best closed models on independent benchmarks ([Artificial Analysis](https://artificialanalysis.ai), early 2026). And they run on hardware you can buy. For a research organisation this flips the default. Sensitive workloads that could never use cloud APIs (unpublished results, personal data, grant material) become automatable **inside your perimeter**, at a cost structure you own.

This chapter gives you the cost model, the risk picture, and a 90-day pilot design with explicit go/no-go criteria. It sits in Part V alongside the strategy paper (Chap. 19); researchers new to the topic can start with Appendix A.
## The economics in one table

API pricing is a perpetual rental; local hardware is a purchase that amortises. Fill in your own numbers:

| Cost driver | Cloud API path | Local path |
|---|---|---|
| Upfront | €0 | Workstation €3–5k (24 GB GPU / Mac 64 GB+) or shared server €15–40k |
| Ongoing | Per-token fees × usage, forever; price per unit falls but usage grows faster | Electricity (~0.3–0.8 kW under load), occasional hardware refresh every 4–5 yrs |
| Scaling behaviour | Linear with adoption. Success increases the bill | Step function. Buy once, serve the whole lab |
| Sensitive-data workloads | Not available without compliance exceptions | Included |
| Rate limits / outages | Provider-controlled | Yours to size |

Two structural trends favour acting now rather than later ([AA trends](https://artificialanalysis.ai/trends)):

1. **Price per unit of intelligence keeps collapsing**. But so does the value of paying it, because open weights close the quality gap every quarter.
2. **Hardware capability per euro keeps rising**. MoE architectures mean a 26B-total/4B-active model delivers frontier-adjacent quality on a single 24 GB card.

![Cumulative cost: API rental vs buying local. The local line starts at the purchase (≈ €4k) and grows slowly (electricity); the API curve starts at zero but compounds with usage. Break-even lands around month 14 for a 5-seat team. Every month after that widens the gap in the local box's favour. Illustrative; substitute your own numbers.]
[Note:]{fg="#b22222"} budget bands above are indicative procurement ranges, not quotes. Build the real figure with your IT team using the formula rows; the point of the table is the *shape* of each curve, not any single number.
## Risk picture

| Risk | Cloud APIs | Local deployment | Mitigation either way |
|---|---|---|---|
| Data leakage to third party | Inherent to the model | None by design (offline capable) | locked down by contract / network egress rules |
| Vendor lock-in & deprecation | Model retired on provider's schedule | Weights archived forever | Open formats (GGUF/safetensors) |
| Compliance (GDPR Art. 28/35, Ch. V) | Processor agreements, transfer analyses | No processor; DPIA still required | Document corpus as processing activity |
| Hallucinated output | Yes | Yes | Human-in-the-loop; domain evaluation gate |
| Model memorising training data | n/a | Possible | Treat corpora as personal data; access control |
| Skills dependency | Low (vendor does ops) | Moderate (IT owns it) | Shared central service, not per-lab ops |

Local removes an entire class of *legal* risk while introducing ordinary *operational* risks. The kind institutions already know how to manage.
## A 90-day pilot that cannot fail expensively

| Phase | Weeks | What happens | Exit criterion |
|---|---|---|---|
| Setup | 1–2 | One workstation or server installed; 5 volunteer researchers trained | Interface reachable, users onboarded |
| Use-case selection | 3–4 | Pick **two** bounded tasks (e.g. literature triage + code help); define what "better than today" means | Success metrics written down before any use |
| Operation | 5–10 | Daily use; log time saved, quality issues, incidents | ≥ 1 metric clearly improved; zero data incidents |
| Evaluation | 11–12 | Compare against baseline; survey users; total the costs | Go/no-go against pre-written criteria |

Total pilot exposure: one workstation and ~10 person-days. The pilot is designed so its worst case is "we learned where local AI doesn't help us yet".
## KPIs to report upward

- Tasks automated per month; median time saved per task
- Cost per completed task (trend should fall toward €0 marginal)
- Data incidents: target zero
- Researcher satisfaction (simple quarterly pulse)
- Share of sensitive workloads moved off external tools
## Decision checklist

- [ ] Two candidate use cases identified with measurable baselines?
- [ ] Hardware path chosen (workstation vs shared server)?
- [ ] Data classification done for the corpora involved?
- [ ] Named owner for the pilot and for the eventual service?
- [ ] Eval gate defined (what quality bar must a model pass)?
