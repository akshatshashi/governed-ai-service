# Corpus manifest

This manifest is provenance evidence: it records what the service answers from, where each
document came from, and a SHA-256 fingerprint of the exact version indexed. The same fingerprint is
stored in the vector index metadata (`doc_sha256`) for every chunk, and
`tests/test_manifest.py` fails CI if a committed document changes without this manifest being
updated.

**Data policy:** everything in the corpus is either public or synthetic. No real customer data, no
personal information, and no confidential material is ever loaded.

## Committed documents (`data/corpus/synthetic/`)

| File | Type | Basis / source | SHA-256 |
|---|---|---|---|
| `austrac-reporting-obligations.md` | Plain-English summary written for this project | Paraphrases public AUSTRAC guidance on TTRs, SMRs, IFTIs, cross-border movements, compliance reports, tipping off and structuring ([austrac.gov.au](https://www.austrac.gov.au/business)) | `1cfc5dfa2983ea1e44bfe72432a7983e4b9f98f483140359c0eb16eec1062f70` |
| `austrac-customer-due-diligence.md` | Plain-English summary written for this project | Paraphrases public AUSTRAC customer due diligence guidance and DFAT sanctions guidance | `23aac10cfa000f3f65ec7291c71e5dd90c7240527c2615794dfe79bb3ba5855a` |
| `aml-typologies-and-red-flags.md` | Plain-English summary written for this project | Themes from public AUSTRAC and FATF typology publications (structuring, cuckoo smurfing, money mules, trade-based laundering) | `7298607cd99001de7672fd5eb8bf9ae8275f00711015b1ee7e48b08b3e8466aa` |
| `kestrel-fx-transaction-monitoring-policy.md` | Synthetic internal policy | Written for this project; Kestrel FX is fictional | `b94de2e873bb7049ca4804b164f42381183d7b7b0533dc3541dde836f6c52634` |
| `kestrel-fx-records-and-ai-use-standard.md` | Synthetic internal standard | Written for this project; Kestrel FX is fictional | `58bd21e2c4ef04075ac22caa411f73122258a4fc772d24f10c8ca7baccbf9260` |

## Reference data (`data/reference/`)

| File | Used by | Basis | SHA-256 |
|---|---|---|---|
| `thresholds.yaml` | `lookup_threshold` tool | Simplified summary of public regulator guidance (AUSTRAC, NZ Police FIU, FinCEN, FINTRAC, UK NCA) | `d3c2e3c7db1c0f0d458d55567e5723a788d2d210c1003ee3e944a7096861f704` |

## Local-only documents (`data/corpus/public/`, git-ignored)

Public source PDFs (for example AUSTRAC guidance notes) can be placed in `data/corpus/public/` and
are indexed alongside the synthetic documents. They are not committed: they are third-party
publications, and the authoritative copy is the regulator's website. Record each one here with its
URL and download date when you add it.

| File | Source URL | Downloaded | SHA-256 |
|---|---|---|---|
| _(none yet)_ | | | |

## Caveats

- The summaries are simplified and are **not legal advice**. The AML/CTF Amendment Act 2024
  changed several obligations from 31 March 2026; verify against current AUSTRAC guidance.
- Reviewed: 2026-09-30.
