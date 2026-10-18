# GDPR mapping

How a local, self-hosted stack maps to core GDPR principles:

![Where personal data lives: users, auth gateway, GPU host and encrypted storage all sit inside the on-premise boundary; the only optional crossing is web search, which can be disabled entirely.]

| GDPR principle | Article | Local deployment answer |
|---|---|---|
| Data minimisation | Art. 5(1)(c) | Fine-tune on the minimum necessary corpus; a local dataset builder generates synthetic/anonymised pairs |
| Purpose limitation | Art. 5(1)(b) | One model per documented purpose; YAML configs make each purpose explicit and reproducible |
| Storage limitation | Art. 5(1)(e) | Adapters, datasets and checkpoints versioned and purged on a schedule |
| Security of processing | Art. 32 | Full-disk encryption, access control, VLAN-isolated GPU host; most studios/studies add password + token auth and hashed, revocable API keys |
| Accountability | Arts. 5(2), 30 | Records = dataset provenance, YAML run configs, training metrics history, model cards |
| Data protection by design | Art. 25 | On-premise inference and training form a privacy-by-design architecture; no usage telemetry leaves the box |
| DPIA | Art. 35 | Required for large-scale sensitive-data processing; local hosting lowers but does not cancel the risk |
| Transfers outside EU | Ch. V | None: compute, storage, telemetry stay under your control; web search is the only optional egress. Disable it for strict regimes |

local execution removes the *processor* risk, not your responsibilities as *controller*. Models can memorise personal data present in the training corpus. Treat every corpus as a processing activity and document it.

**Our take.** Do the DPIA before the pilot, not after the demo impresses someone. We have watched more local-AI projects die of retroactive compliance work than ever died of technical failure, and the checklist above is a weekend of work if done early. Local hosting is the strongest privacy argument you will ever put in that document. Write it down while it is still an architecture decision, not a fait accompli.
## Hardening checklist
