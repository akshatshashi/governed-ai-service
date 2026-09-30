# Runbook: Governed AI Service

Operational guide for running and supporting the service. Audience: the engineer on call.

## 1. Start the service

### Docker (recommended)

```bash
docker build -t governed-ai-service .
# With Claude (normal operation). The key is read from the environment, never baked into the image.
docker run -p 8000:8000 --env-file .env governed-ai-service
# Without a key the service still starts and serves deterministic fallback answers (degraded mode).
docker run -p 8000:8000 governed-ai-service
```

The image contains the corpus, the vector index and the embedding model, so it needs no downloads
at runtime. The audit database is written to `/app/var/audit.db`; mount a volume there
(`-v gas-data:/app/var`) to keep the audit trail across container restarts.

### Local

```bash
python -m src.ingest            # (re)build the index from data/corpus
uvicorn app:app --port 8000
```

### Configuration (environment variables or `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Claude API key. Missing means degraded (fallback-only) mode. |
| `LLM_BACKEND` | `claude` | `claude`, or `extractive` (deterministic offline baseline). |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model ID. Change it only together with a full eval run. |
| `CLAUDE_EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` / `max`. Trades latency and cost against depth. |
| `CONFIDENCE_THRESHOLD` | `0.35` | Minimum mean similarity of the cited evidence before an answer is returned. |
| `GROUNDING_THRESHOLD` | `0.25` | Minimum lexical overlap between the answer and its evidence. |
| `AUDIT_DB_PATH` | `audit.db` | SQLite file holding audit records, approvals and review cases. |
| `LOG_LEVEL` | `INFO` | Log verbosity. Logs are JSON lines on stdout. |

## 2. Health checks: what each `/health` field means

```json
{"status": "ok", "llm_reachable": true, "index_chunks": 22, "audit_store_ok": true,
 "llm_backend": "claude", "model": "claude-opus-5-5", "prompt_version": "v1", "pending_approvals": 0}
```

| Field | Meaning | If it is wrong |
|---|---|---|
| `status` | `ok` when the model is reachable, the index is non-empty and the audit store answers; otherwise `degraded`. | `degraded` still serves traffic via fallback. Find the failing field below. |
| `llm_reachable` | Result of a free Models API call (credentials + network + model ID). Cached for 30 seconds. | See incident 4.1. |
| `index_chunks` | Number of chunks in the vector index. | `0` means every answer will abstain. Rebuild with `python -m src.ingest` or rebuild the image. |
| `audit_store_ok` | The SQLite audit store answers a trivial query. | **Critical.** `/ask` fails closed (HTTP 503) while the audit store is down. See incident 4.5. |
| `prompt_version` | The prompt version stamped on every audit record. | Should match the version that passed the eval gate for this release. |
| `pending_approvals` | High-impact actions waiting for a human decision. | See incident 4.3. |

The container `HEALTHCHECK` calls `/health`. It checks liveness (HTTP 200), not `status: ok`, so a
degraded-but-serving container is not restarted in a loop.

## 3. Signals and alerts

`GET /metrics` exposes Prometheus text. Suggested alerts:

| Alert | Expression (PromQL) | Why it matters |
|---|---|---|
| Fallback rate spike | `sum(rate(gas_fallback_count{reason!="guardrail_blocked"}[10m])) / sum(rate(gas_request_count[10m])) > 0.2` | Users are getting passages instead of answers. |
| Model unavailable | `rate(gas_fallback_count{reason="llm_unavailable"}[5m]) > 0` for 5 minutes | Provider outage, bad credentials or rate limiting. |
| Grounding failures | `rate(gas_fallback_count{reason="grounding_failed"}[1h]) > 0.05` | Possible model or prompt regression, or a corpus change. |
| Injection attempts | `increase(gas_refusal_count{guard="prompt_injection"}[1h]) > 10` | Possible probing; review the audit records. |
| Approval backlog | `gas_pending_approvals > 20` | High-impact actions are waiting on humans. |
| Latency | `histogram_quantile(0.95, rate(gas_latency_ms_bucket[5m])) > 30000` | Slow model calls; check the effort level and the provider status. |
| Token burn | `rate(gas_token_usage[1h])` above budget | Cost control. |

Every log line and every response carries a `request_id`. To investigate one request:
`GET /audit/{request_id}` returns the redacted question, the guardrail decisions, the chunks
retrieved (ids and scores), the prompt version, the model that answered, every tool call and its
control outcome, the redacted answer, latency and tokens, plus any later approval decision.

## 4. Incident playbooks

### 4.1 Model unreachable (`llm_reachable: false`, `fallback_reason=llm_unavailable`)

Users still get deterministic answers made of source passages. No action is taken without the model.

1. Find the cause in the audit trail: `GET /audit?fallback_only=true&limit=5`, then read
   `guardrails_out.llm_error` on the latest record:
   - `no_credentials` / `authentication_failed`: the key is missing, revoked or wrong. Check how the
     secret is injected (`--env-file`, orchestrator secret). Rotate the key if it may have leaked.
   - `rate_limited`: the account is over its rate limit. Reduce traffic, lower `CLAUDE_EFFORT`, or
     raise the limit.
   - `timeout` / `connection_error` / `api_error_5xx`: a network or provider problem. Check egress
     from the host and the Anthropic status page.
   - `model_not_found`: `CLAUDE_MODEL` is wrong or the model has been retired.
2. The SDK already retries twice with backoff. Do not add retry loops during an incident.
3. Once fixed, `/health` shows `llm_reachable: true` within 30 seconds (cache TTL).

### 4.2 Fallback rate spikes (with the model reachable)

1. Break it down by reason: `gas_fallback_count` by `reason` label.
2. `grounding_failed`: pull recent records with `GET /audit?fallback_only=true` and compare
   `guardrails_out.grounding` scores. Common causes: a prompt change (check `prompt_version`), a
   model change (check `model`; a refusal fallback can route to another model), or a corpus change
   (check the index and MANIFEST).
3. `low_confidence` (outcome `abstained`): questions are outside the corpus, or retrieval has
   degraded. Check `index_chunks` and whether the embedding model changed.
4. `model_refusal`: Claude's safety classifiers declined. Check `guardrails_out.refusal_category`.
   Server-side fallback is already enabled, so a refusal here means the whole chain declined.
5. If a release caused it, roll back to the previous image tag, whose eval evidence is known.

### 4.3 Approval queue backs up (`gas_pending_approvals` climbing)

Pending actions do nothing until a human decides, so a backlog delays actions but is never unsafe.

1. `GET /approvals` lists the pending requests, oldest first.
2. Page the compliance approvers on the rota. Approvals need a different person from the requester
   (segregation of duties is enforced, and self-approval returns HTTP 403).
3. Decide with `POST /approvals/{request_id}` `{"approved": true|false, "decided_by": "<id>"}`.
   Decisions are final (a second decision returns 409). To change your mind, raise a new request.
4. If the volume is abnormal, check whether a prompt or model change made the agent over-eager to
   call `flag_for_review` (look at `tool_calls` in the audit records).

### 4.4 Eval gate fails in CI

The eval gate is the release control: **do not merge or deploy while it is red.**

1. Download the `eval-results-<sha>` artifact from the failed run and open `cases`.
2. `safety_violations > 0`: treat as a potential data-protection or prompt-leak defect. Find which
   `must_not_mention` term appeared and in which case. Fix the control; do not edit the golden set
   to make it pass.
3. `behaviour_match < 0.8`: find the cases where `behaviour_match` is false and compare `outcome`
   with `expect_type`. Reproduce locally with
   `python -m eval.run_eval --backend extractive --build-index`.
4. A golden-set change needs the same review as a code change, because it changes what "safe to
   ship" means.

### 4.5 Audit store unavailable (`audit_store_ok: false`, `/ask` returns 503)

The service fails closed by design: no answer leaves without an audit record.

1. Check disk space and permissions on the volume holding `AUDIT_DB_PATH`.
2. Check for a stuck process holding a lock on the SQLite file.
3. Restore from the latest backup if the file is corrupt. Record the outage window: requests
   during it were rejected, not served unaudited.

## 5. Worked incident: "answers stopped, users see passages only"

**09:12** An alert fires: `gas_fallback_count{reason="llm_unavailable"}` has been increasing for
5 minutes. `/health` shows `status: degraded, llm_reachable: false`, and all other fields are fine.

**09:14** The on-call engineer pulls the most recent fallback record:

```bash
curl -s "localhost:8000/audit?fallback_only=true&limit=1" | jq '.[0].guardrails_out'
# {"llm_error": "authentication_failed", "pii_redaction": {"found": {}, "passed": true}}
```

`authentication_failed` rules out a provider outage. The key is being rejected.

**09:16** The deploy log shows a secret rotation at 09:05. The new key was stored under the wrong
variable name, so the container started with the old, revoked key.

**09:21** Fix the secret mapping and restart the container. `/health` returns `llm_reachable: true`.

**09:25** Verify:
`gas_request_count{outcome="answered"}` is increasing again, and fallbacks have returned to baseline.

**Impact:** 16 minutes of degraded service. Every request in the window was answered with sourced
passages via fallback, and every one has an audit record showing `fallback_reason=llm_unavailable`.
No ungrounded or unaudited answers were served.

**Follow-ups:** (1) a deploy-time check that calls `/health` and blocks the rollout if
`llm_reachable` is false; (2) alert on `authentication_failed` specifically, paging immediately
rather than after 5 minutes.
