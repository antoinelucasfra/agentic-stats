# DPIA template for local AI

A data protection impact assessment (GDPR art. 35) is required when processing is likely to result in high risk: large-scale personal-data processing, or any AI-based decisions about people. Copy this template per use case. It follows the CNIL/PIA structure. Local deployment makes several answers *easy*. The template makes them *explicit*.

### 1. Processing description

- Controller / department / contact:
- Use case (one paragraph, plain language):
- Model(s) used, version and source (`model-id`, quantisation):
- Hosting: hardware location, who has physical access:
- Data flow diagram attached? yes/no

### 2. Data categories

- Personal data processed (types; special categories under art. 9?):
- Volume (number of people concerned):
- Sources (provided by users / imported corpora / generated):
- Retention: prompts kept? conversation history? where, how long?

### 3. Necessity & proportionality

- Why is a language model necessary at all?
- Why local rather than an API provider (data volume, secrecy, cost)?
- Why this model size? Could a smaller model do the job?
- Is the data strictly necessary for the purpose?

### 4. Risks and measures

| Risk | Likelihood | Severity | Measure | Residual |
|---|---|---|---|---|
| Re-identification from prompts in logs | | | no prompt logging by default (Chap. 15) | |
| Model output leaks training data | | | golden-set + canary-string tests (Chap. 16) | |
| Unauthorised access to the server | | | SSO gateway, disk encryption, no direct ports | |
| Telemetry/phone-home in tools | | | offline verification checklist (Chap. 17) | |
| Wrong or biased output affecting people | | | human review before external effect; citations shown (Chap. 13) | |

### 5. Residual risk & sign-off

- Overall residual risk (low/medium/high) and justification:
- Consulted: DPO opinion, date:
- Approved by (name/role/date):

*Local-first note*: items that are nightmares with cloud APIs (data leaving the EU, sub-processors, retention you don't control) collapse into verifiable one-liners here. That is a genuine administrative advantage of the architecture this book recommends.
