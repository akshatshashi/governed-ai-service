# Security

## Controls in the delivery pipeline

| Control | Where |
|---|---|
| Secret scanning of the full git history (gitleaks) | `.github/workflows/ci.yml` → `secret-scan` |
| Static security lint (ruff `S` / flake8-bandit rules) | `pyproject.toml`, CI `lint` job |
| Dependency vulnerability audit (pip-audit) on exactly pinned runtime deps | CI `dependency-audit` job |
| Automated dependency update PRs | `.github/dependabot.yml` |
| Secrets only via environment / `.env` (git-ignored); held as `SecretStr` | `src/config.py`, `.gitignore` |
| Non-root container, no runtime downloads, `.env` excluded from the image | `Dockerfile`, `.dockerignore` |
| Parameterised SQL only | `src/audit.py`, `src/approval.py` |

## Data protection

- PII is redacted from the question **before** it reaches retrieval, the model provider, logs or
  the audit store, and again from the answer before it is returned (`src/guardrails/pii.py`).
- Logs carry identifiers, counts and PII *kinds* only — never user text.
- The only outbound call is to the Anthropic API. The embedding model and vector index run
  locally inside the service.

## Accepted risks

Every accepted risk has an owner, a rationale, and a review date. The CI audit ignores exactly
these IDs and nothing else.

| ID | Package | Summary | Why it does not apply here | Review by |
|---|---|---|---|---|
| PYSEC-2026-311 (CVE-2026-45829) | chromadb 1.5.9 | Pre-auth code injection via the Chroma **HTTP server** collections endpoint with `trust_remote_code` | Chroma runs embedded (`PersistentClient`, in-process). No Chroma server is started or exposed, and the service never passes `trust_remote_code`. | 2026-12-31 or first fixed release |
| PYSEC-2026-3814 (CVE-2026-45833) | chromadb 1.5.9 | Authenticated code injection via the **HTTP server** collection-update endpoint | As above: no Chroma server endpoint exists. | 2026-12-31 or first fixed release |
| PYSEC-2026-3815 (CVE-2026-45831) | chromadb 1.5.9 | `SimpleRBACAuthorizationProvider` ignores tenant scope (**server** auth) | No Chroma server and no Chroma auth provider in use. | 2026-12-31 or first fixed release |
| PYSEC-2026-3813 (CVE-2026-45830) | chromadb 1.5.9 | Cross-tenant read/write for authenticated **server** users | Single embedded collection; no multi-tenant Chroma server. | 2026-12-31 or first fixed release |

No fixed chromadb version was available when these were accepted (2026-09-30). When one ships,
Dependabot will propose the upgrade; remove the corresponding `--ignore-vuln` flag in the same PR.

## Reporting a vulnerability

Please open a private security advisory on this repository rather than a public issue.
