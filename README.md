# Governed AI Service

**A RAG + agent service for AML/CTF compliance questions, where the controls are the product.**

Most AI demos stop at "it answers questions". This one is built the way a regulated business
needs an AI service to behave:

- Every answer is grounded and cited.
- Personal information is redacted on the way in and on the way out.
- Prompt injection and off-topic use are refused.
- The model can only use allow-listed, validated tools.
- A high-impact action waits for a second person to approve it.
- The service degrades to deterministic, sourced answers when the model is unavailable.
- Every request, including refused ones, leaves an audit record that reconstructs exactly what
  happened.
- An evaluation suite gates every build in CI.

The subject matter is Australian AML/CTF compliance (AUSTRAC reporting obligations, customer due
diligence, typologies) plus the internal policies of *Kestrel FX*, a fictional remittance and FX
business. The model is Claude (Anthropic), behind a swappable interface.

> Decision support only: a person remains accountable for every decision. See
> [MODEL_CARD.md](MODEL_CARD.md).

## Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │                FastAPI SERVICE (app.py)          │
                    │  /ask  /approvals  /audit  /health  /metrics      │
                    └────────────────────────┬─────────────────────────┘
                                             │  every request gets a request_id
            ┌────────────────────────────────┼────────────────────────────────┐
            ▼                                ▼                                ▼
   ┌──────────────────┐           ┌────────────────────┐           ┌──────────────────┐
   │   GUARDRAILS IN   │           │    ORCHESTRATOR     │           │  GUARDRAILS OUT   │
   │ PII redaction     │──────────▶│  retrieve → agent   │──────────▶│ citations resolve │
   │ length limit      │           │  bounded tool loop  │           │ grounding check   │
   │ injection check   │           │  (src/agent)        │           │ confidence check  │
   │ scope check       │           │                     │           │ PII redaction     │
   └──────────────────┘           └─────────┬──────────┘           └─────────┬────────┘
                                            │                                │ fail
                    ┌───────────────────────┼──────────────────────┐         ▼
                    ▼                       ▼                      ▼   ┌───────────────┐
             ┌─────────────┐      ┌─────────────────┐     ┌──────────┐ │   FALLBACK     │
             │  RETRIEVER   │      │  TOOL REGISTRY   │     │  CLAUDE   │ │ deterministic  │
             │ local embed  │      │ allow-list       │     │ Opus 5.5  │ │ sourced        │
             │ + Chroma     │      │ arg validation   │     │ (swappable│ │ passages, no   │
             │              │      │ per-call limits  │     │  LLM)     │ │ generated text │
             └─────────────┘      └────────┬────────┘     └──────────┘ └───────────────┘
                                           │ high impact?
                                           ▼
                                  ┌──────────────────┐   a different person approves
                                  │  APPROVAL QUEUE   │──▶ only then does the action run
                                  └──────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │     AUDIT STORE (SQLite, append-only)          │
                    │  every request, every path, reconstructable    │
                    └───────────────────────────────────────────────┘
```

The `/ask` flow, in strict order ([`src/pipeline.py`](src/pipeline.py)):

1. Assign a `request_id`.
2. Redact PII from the question.
3. Run the input guardrails.
4. Retrieve, then run the agent: the model plus a controlled tool loop.
5. Check the output: citations resolve, the answer is grounded, confidence is high enough.
6. Redact PII from the answer.
7. Queue any high-impact action for approval.
8. Write the audit record. This happens on every path; if the write fails, the request fails
   closed with HTTP 503.
9. Respond.

## Requirement → implementation map

| What a regulated AI service needs | How this project delivers it | Where |
|---|---|---|
| **RAG** | Paragraph-aware chunking with overlap, local embeddings, Chroma retrieval, numbered citations | `src/ingest.py`, `src/retriever.py` |
| **Agents** | Bounded tool-calling loop (step budget) using Claude's native tool use | `src/agent/loop.py` |
| **Evaluation** | 15-case golden set scoring behaviour, keyword recall and safety; runs in CI and gates the build | `eval/` |
| **Tool controls** | Allow-list, pydantic argument validation, per-request call limits, impact classification | `src/agent/tools.py` |
| **Monitoring** | JSON logs keyed by `request_id`; Prometheus metrics for outcomes, fallbacks, refusals, tool calls, tokens, latency | `src/observability.py` |
| **Fallback** | Deterministic sourced-passage responses when the model is down, refuses, is ungrounded or unconfident | `src/fallback.py` |
| **Traceability & evidence** | Per-request audit record: redacted input, guardrail decisions, chunk ids and scores, prompt version, model, tool calls, output, tokens | `src/audit.py` |
| **Privacy** | AU-specific PII redaction (email, phone, TFN, ABN, Luhn-valid cards, names) on input *and* output; raw PII never logged or stored | `src/guardrails/pii.py` |
| **Human approval** | High-impact actions queued, not executed; segregation of duties; immutable decisions | `src/approval.py` |
| **Responsible AI** | Model card with intended use, limitations and eval results; refusal fallback; abstention over guessing | `MODEL_CARD.md` |
| **Secure coding, CI/CD** | gitleaks, ruff with bandit rules, pip-audit with documented risk acceptances, pinned deps, Dependabot, non-root image | `.github/`, `SECURITY.md` |
| **Production support** | Runbook with health-field meanings, alert expressions, incident playbooks and a worked incident | `RUNBOOK.md` |
| **Identity & state** | Requester and approver identities recorded; approval state machine enforced atomically | `src/approval.py`, `app.py` |
| **Provenance** | Corpus manifest with SHA-256 per document, verified in CI; hashes stored per chunk | `data/corpus/MANIFEST.md` |

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env              # then set ANTHROPIC_API_KEY (optional; see below)
python -m src.ingest              # build the vector index from data/corpus
uvicorn app:app --port 8000
```

Without an API key the service still runs: `/health` reports `degraded`, and `/ask` answers with
sourced passages via the deterministic fallback. To run fully offline, set
`LLM_BACKEND=extractive`.

### Docker

```bash
docker build -t governed-ai-service .
docker run -p 8000:8000 --env-file .env governed-ai-service
```

## Using the API

```bash
# Ask a question
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How long do we have to lodge a suspicious matter report?", "user_id": "analyst1"}'

# The model proposes a high-impact action: it is queued, not executed
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Please flag TXN-48213 for review: four cash deposits of AUD 9,500 this week.", "user_id": "analyst1"}'
curl -s localhost:8000/approvals
curl -s -X POST localhost:8000/approvals/<request_id> -H 'Content-Type: application/json' \
  -d '{"approved": true, "decided_by": "supervisor1"}'   # the requester cannot self-approve (403)

# Reconstruct any request after the fact
curl -s localhost:8000/audit/<request_id>
curl -s "localhost:8000/audit?fallback_only=true"

# Operations
curl -s localhost:8000/health
curl -s localhost:8000/metrics
```

Interactive API docs: `http://localhost:8000/docs`.

## Evaluation (release evidence)

```bash
python -m eval.run_eval                                      # configured backend (Claude)
python -m eval.run_eval --backend extractive --build-index   # what CI runs
```

| Metric | Extractive baseline (CI) | Claude Opus 5.5 |
|---|---|---|
| Behaviour match | 1.00 (15/15) | _run with an API key_ |
| Keyword recall | 1.00 | _run with an API key_ |
| Safety violations | 0 | _run with an API key_ |
| Approval match | 1.00 | _run with an API key_ |

CI fails the build if behaviour match drops below 0.8 or any safety violation appears, and uploads
`eval/results.json` as a build artifact on every run. Committed evidence lives in
[`eval/results/`](eval/results/). See [MODEL_CARD.md](MODEL_CARD.md) for what these numbers do and
do not show.

## Tests

```bash
python -m pytest -q     # unit and integration tests for every control
ruff check . && ruff format --check .
```

The end-to-end tests in [`tests/test_pipeline.py`](tests/test_pipeline.py) encode the definition of
done: grounded answers with citations; useful fallback when the model is down; injection refused
and audited; high-impact actions waiting for approval; no raw PII in model input, output, the audit
database file or logs; the service failing closed when auditing fails.

## Design decisions worth knowing

- **Controls do not trust the model.** Claude receives tool definitions, but every call it
  requests goes through the registry's allow-list, validation, limits and impact gate. Retrieved
  text is treated as data, not instructions.
- **Two backends, one interface.** `ClaudeLLM` is the production backend. `ExtractiveLLM` is a
  deterministic, offline baseline. It lets CI test every control reproducibly with no API key or
  network, and it doubles as a degraded mode.
- **Audit is append-only; decisions are joined, not overwritten.** An approval decided later does
  not rewrite the original audit record. `/audit/{id}` shows both.
- **Abstain over guess.** Low retrieval confidence, failed grounding or unresolvable citations all
  lead to a fallback made of source passages, never to unverified text.
- **Honest limits.** The grounding check is a lexical-overlap heuristic and PII detection is regex.
  Both are documented as such, with their failure modes, in the model card.

## Repository layout

```
app.py                  FastAPI entry point (thin)
src/config.py           settings (env / .env)
src/llm.py              LLM protocol + Claude and extractive implementations
src/ingest.py           load, chunk, embed the corpus
src/retriever.py        similarity search with traceable chunks
src/guardrails/         pii.py, injection.py (injection + scope), grounding.py
src/agent/              tools.py (controlled registry), loop.py (agent)
src/pipeline.py         the governed /ask flow
src/approval.py         human approval queue + review case log
src/fallback.py         deterministic degraded responses
src/audit.py            append-only audit store
src/observability.py    JSON logging + Prometheus metrics
eval/                   golden set, harness, committed results
tests/                  unit + end-to-end tests
data/corpus/            synthetic corpus + provenance manifest
data/reference/         reference data for the lookup_threshold tool
```
