# Changelog

All notable changes to this project. Versions are git tags; each release's CI run uploads its eval
results as release evidence.

## v1.0.0 (2026-09-30)

- The Claude backend is tested end to end on the real SDK wire format (mock HTTP transport).
- CI actions moved to current major versions.
- Release: every item in the definition of done is verified locally, in Docker, and in CI.

## v0.4.0 (2026-09-30)

- README (architecture, requirement-to-control map, API walkthrough), MODEL_CARD, RUNBOOK
  (health fields, alerts, incident playbooks, worked incident), `.env.example`.

## v0.3.0 (2026-09-30)

- Eval harness and 15-case golden set; CI release gate (behaviour ≥ 0.8, zero safety violations).
- Dockerfile (non-root, index and embedding model baked in) with a no-API-key smoke test in CI.
- CI: gitleaks, ruff with bandit rules, pytest, pip-audit, Docker. Dependabot.
- SECURITY.md with documented acceptance of chromadb server-mode CVEs.

## v0.2.0 (2026-09-30)

- Test suite covering every control, including end-to-end definition-of-done tests.

## v0.1.0 (2026-09-30)

- Core pipeline: Claude and extractive backends, ingest and retrieval, PII, injection, scope and
  grounding guardrails, controlled tool registry, approval queue, audit store, fallback, FastAPI
  service, observability, and a synthetic corpus with provenance manifest.
