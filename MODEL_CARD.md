# Model card: Governed AI Service (AML/CTF compliance assistant)

> **This is decision support that requires human review, not an automated decision system.**
> It never lodges reports, contacts customers, or changes records by itself. The one action it can
> propose (flagging a reference for compliance review) is queued for a second person's approval,
> and a person remains accountable for every decision made with its output.

## Intended use

- Answering AML/CTF operations staff's questions about reporting obligations (TTR, SMR, IFTI,
  cross-border movements), customer due diligence, money-laundering typologies, and the internal
  policies in the corpus, with citations to the source passages.
- Proposing that a named transaction or customer reference be flagged for compliance review,
  subject to human approval.

## Out-of-scope uses

- Legal advice, or determining whether a specific transaction *is* suspicious. Only a qualified
  person can form that suspicion.
- Making or automating decisions about customers (onboarding, exits, transaction blocking).
- Lodging, drafting or submitting regulatory reports to AUSTRAC.
- Any question outside AML/CTF compliance (the scope guardrail refuses these).
- Processing real customer data outside approved environments. The service redacts common PII,
  but redaction is regex-based and not exhaustive (see Limitations).

## System components

| Component | What | Notes |
|---|---|---|
| Generator | Anthropic **Claude Opus 5.5** (`claude-opus-5-5`) via the official Python SDK | Adaptive thinking at `effort=medium` (configurable). Server-side refusal fallback is enabled (`fallbacks="default"`); the model that actually answered is recorded in each audit record. |
| Offline baseline | `extractive-baseline-v1`: deterministic sentence selection, rule-based tool choice | Used in CI and available as a degraded mode. Makes no network calls. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, run locally | Question text is embedded locally; only the redacted question and retrieved passages are sent to the model provider. |
| Vector store | Chroma (embedded), cosine similarity | Similarity score = max(0, cosine similarity). |
| Prompt | `PROMPT_VERSION = "v1"` (`src/agent/loop.py`) | Stamped on every audit record. |

**Data handling.** The question is PII-redacted before it reaches the embedding model, the model
provider, logs or storage. The model provider receives the redacted question, up to five retrieved
passages, and tool results. A deployment inside a regulated bank would route this through an
approved channel (for example Claude via Amazon Bedrock or Google Cloud Vertex AI in an Australian
region) under the organisation's data-handling agreements.

## Corpus and provenance

Five documents (about 2,000 words), all public-derived or synthetic, listed with SHA-256
fingerprints in [`data/corpus/MANIFEST.md`](data/corpus/MANIFEST.md):

- three plain-English summaries written for this project, paraphrasing public AUSTRAC and FATF
  guidance (reporting obligations, customer due diligence, typologies and red flags);
- two synthetic internal policies for *Kestrel FX*, a fictional remittance/FX business.

Reference data for the `lookup_threshold` tool (`data/reference/thresholds.yaml`) summarises public
regulator guidance for AU, NZ, US, CA and GB. **No real customer data is used anywhere.** The
summaries are simplified, are not legal advice, and predate full verification against the AML/CTF
Amendment Act 2024 changes that commenced on 31 March 2026.

## Evaluation

Golden set: 15 cases (`eval/golden_set.yaml`): 8 answerable (including two with PII in the
question, one needing the `lookup_threshold` tool, and one high-impact action that must be queued
for approval), 3 out-of-scope, 2 prompt-injection (one base64-encoded), and 2 in-scope but
unanswerable from the corpus. Release gate: `behaviour_match >= 0.8` and `safety_violations == 0`.

### Results

| Metric | Extractive baseline (CI gate) | Claude Opus 5.5 |
|---|---|---|
| Behaviour match | **1.00** (15/15) | _pending: run with an API key_ |
| Keyword recall | **1.00** | _pending_ |
| Safety violations | **0** | _pending_ |
| Approval match | **1.00** | _pending_ |
| Refusal rate | 0.33 (5/15, as designed) | _pending_ |
| Abstain rate | 0.13 (2/15, as designed) | _pending_ |
| Fallback rate | 0.00 | _pending_ |
| Mean / p95 latency | 4 ms / 9 ms | _pending_ |

Source: [`eval/results/extractive-baseline.json`](eval/results/extractive-baseline.json)
(commit `231333f`, prompt `v1`). Claude results are produced by
`python -m eval.run_eval --output eval/results/claude-opus-5-5.json` with `ANTHROPIC_API_KEY` set.

**Reading these numbers honestly.** Fifteen cases is a smoke test of the controls, not a measure of
answer quality. The baseline's perfect score shows the *controls* behave as specified (refusals,
abstentions, approval gating, PII handling). It does not show that answers are good. The golden set
was written by the same person who wrote the corpus, which inflates scores. A production evaluation
would need hundreds of cases written independently by compliance SMEs, graded answers rather than
keyword matching, and adversarial red-teaming.

## Known limitations

- **Heuristic grounding check.** Grounding is lexical overlap: the share of the answer's content
  words found in the evidence. It misses faithful paraphrase (false fail), can be fooled by
  answers that recombine source words into claims the source does not make (false pass), and is
  blind to negation. It is a cheap gate, not proof of faithfulness.
- **Regex PII detection.** It covers emails, AU phone numbers, TFN- and ABN-shaped numbers,
  Luhn-valid card numbers and simple name patterns ("Mr Smith", "my name is ..."). It will miss
  names without those cues, addresses, dates of birth, account numbers in unusual formats, and PII
  in images or attachments. It errs towards over-redacting nine- and eleven-digit numbers.
- **Keyword scope check.** Off-topic questions that mention an allowed keyword pass the scope
  check (retrieval confidence and grounding usually catch them later), and legitimate questions
  phrased without any listed keyword are refused.
- **Pattern-based injection detection.** It catches common override, role-hijack, prompt-extraction,
  delimiter and base64-encoded attempts, but novel phrasings, other languages, and indirect
  injection through documents are not reliably detected. Defence in depth (allow-listed and
  validated tools, the approval gate, grounding checks) limits the impact.
- **English only.** Tokenisation, stopwords, stemming and all regexes assume English.
- **Small, static corpus.** Five short documents. Answers are only as current as the corpus, which
  has not been reconciled with the 2024 AML/CTF reforms.
- **Confidence is retrieval similarity**, not a calibrated probability of correctness.

## Human oversight

- All output is labelled decision support. Fallback responses say "a qualified person must review
  before acting".
- High-impact tool calls are never executed by the agent. They wait in an approval queue; the
  requester cannot approve their own request; decisions are immutable and recorded with who decided
  and when.
- Every request, including refused and fallback ones, has an audit record reconstructing the
  inputs (redacted), the evidence, the prompt version, the model, the tool calls and the output.
