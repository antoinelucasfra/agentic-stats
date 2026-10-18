# Local AI as Institutional Strategy

## Strategic context: capability is concentrating

Frontier AI capability is concentrating. Artificial Analysis tracks hyperscaler AI capital expenditure in the hundreds of billions of dollars per year, and its country analysis shows frontier-intelligence production dominated by US and Chinese labs, with Europe represented mainly through open-weight releases (e.g. Mistral, France). Two consequences for a research institute:

1. **Dependency risk**: routing institutional knowledge work through a small set of foreign commercial platforms places core research processes under foreign jurisdiction, pricing power and product decisions.
2. **The capability hedge exists**: open-weight models are within ~3 points of the closed frontier (AA Intelligence Index v4.1.1: Kimi K3 59.7 / GLM-5.3 59.5 vs Claude Opus 5 63.1) and run on commodity hardware. Sovereignty no longer requires sacrificing capability.
## Regulatory horizon

| Date | Regulation | What it demands |
|---|---|---|
| Feb 2025 | EU AI Act — prohibitions apply | No unacceptable-risk practices; inventory of AI uses begins |
| Aug 2025 | EU AI Act — GPAI obligations | Transparency for general-purpose model providers; documentation flows downstream to deployers (us) |
| Ongoing | GDPR Arts. 5, 22, 25, 32, 35, Ch. V | Lawful basis, minimisation, DPIA for large-scale sensitive processing, security, transfer rules |
| Aug 2026 | EU AI Act — high-risk obligations | Risk management, data governance, logging, human oversight for high-risk systems |

A local-first posture is regulatory *arbitrage*: on-premise inference removes international-transfer analysis (GDPR Ch. V) and simplifies Art. 25 privacy-by-design arguments, while open weights give auditable, versioned artefacts that satisfy emerging AI Act documentation expectations better than opaque API endpoints ever will.
## Governance framework (the part that makes it real)

Technology without governance creates risk; governance without technology creates nothing. Institute-level instruments needed:

- **Model registry.** Every deployed model recorded. Version hash, source, licence, quantisation, intended purpose, evaluation report. Weights are assets; treat them as such.
- **Acceptable-use policy**: permitted data classes per deployment zone; mandatory human review for outputs affecting individuals (GDPR Art. 22); prohibition on feeding personal data into models outside approved zones.
- **Evaluation gate**: no model enters service without passing a domain-specific test set built from real institutional tasks (the practitioner guide's "Evaluate before you trust" section defines the protocol).
- **Incident procedure**: hallucination-caused harm, data exposure suspicion and prompt-injection abuse all route to the existing IT-security incident process.
- **Skills plan**: a small central platform team serves all labs; researchers get training, not ops duties.
## Decision framework: local, cloud or hybrid

| Situation | Posture |
|---|---|
| Personal data, unpublished results, grant drafts, clinical/genomic material | **Local only** |
| Public-domain tasks, non-sensitive text, where frontier quality is decisive and contracts allow | Cloud acceptable |
| High-volume batch jobs at predictable scale | Local (no rate limits, zero marginal cost) |
| Exploratory experiments by individuals | Local small models on laptops |
| Mission-critical long agentic runs needing top-1 global model | Cloud if policy allows; otherwise best local + human review |

The default question shifts from "why not use the cloud?" to "what legally and strategically *requires* the cloud?".
## Investment scenarios

| Scenario | Scope | Indicative budget band | Outcome |
|---|---|---|---|
| Minimal | 2–3 workstations, pilot in two labs, central registry spreadsheet | Low five figures (€) | Proof points and skills seed |
| Recommended | Shared GPU server (multi-GPU), platform engineer at ~0.5 FTE, training program, full governance pack | Mid six figures over 3 yrs incl. staff | Institute-scale capability, compliance-ready |
| Ambitious | Cluster + fine-tuning programme for domain models (French-language science, institutional corpora), published model cards | Seven figures, multi-year | Regional reference capability; possible research output on domain adaptation itself |

Sequencing matters more than size: each scenario starts with the same 90-day pilot (see business case) and scales only on evidence.
## Risks of acting vs not acting

**Acting too cautiously**: some engineering effort, hardware refresh cycles, need to hire/retain one platform profile. All ordinary.

**Not acting**: sensitive-data workloads remain unautomated or drift into shadow usage of consumer tools (the worst of both worlds); regulatory documentation debt accumulates against 2026 AI Act milestones; dependency on a shrinking set of foreign platforms deepens; competitors' institutes build capability and talent attraction.

The asymmetric risk is clearly on the side of inaction.
## Twelve-month roadmap

1. **Q1**. Governance instruments adopted (registry, AUP, eval gate); 90-day pilot launched; platform hire identified.
2. **Q2**. Pilot evaluated against pre-written criteria; shared server procured on go decision; first training cohort.
3. **Q3**. Two additional labs onboarded; French-language model evaluation published internally; hybrid routing policy live.
4. **Q4**. Board review: KPIs, incidents, cost per task; decision on domain fine-tuning programme (scenario tier).
