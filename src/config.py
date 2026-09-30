"""Runtime configuration.

Values come from environment variables (case-insensitive, e.g. ``LLM_BACKEND``) or an optional
``.env`` file in the working directory. Secrets are held as ``SecretStr`` so they never appear in
logs or ``repr()`` output.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Keywords that put a question inside this service's remit (AML/CTF compliance operations).
# Used by the scope guardrail — see src/guardrails/injection.py for how it is applied.
DEFAULT_ALLOWED_TOPICS: list[str] = [
    "aml",
    "ctf",
    "anti-money laundering",
    "money laundering",
    "counter-terrorism financing",
    "terrorism financing",
    "austrac",
    "reporting entity",
    "report",
    "reporting",
    "threshold",
    "cash",
    "transaction",
    "transfer",
    "remittance",
    "ifti",
    "smr",
    "ttr",
    "suspicious",
    "structuring",
    "smurfing",
    "kyc",
    "customer due diligence",
    "cdd",
    "identification",
    "verify",
    "beneficial owner",
    "pep",
    "politically exposed",
    "sanctions",
    "compliance",
    "record keeping",
    "records",
    "enrolment",
    "registration",
    "cross-border",
    "currency",
    "foreign exchange",
    "fx",
    "alert",
    "escalate",
    "escalation",
    "flag",
    "review",
    "tipping off",
    "risk assessment",
    "typology",
    "policy",
]


class Settings(BaseSettings):
    """All tunable parameters for the service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Language model -------------------------------------------------------------------
    llm_backend: Literal["claude", "extractive"] = "claude"
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    claude_max_tokens: int = 16000
    llm_timeout_seconds: float = 60.0
    anthropic_api_key: SecretStr | None = None

    # --- Retrieval ------------------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    corpus_dir: str = "data/corpus"
    index_dir: str = "data/index"
    reference_dir: str = "data/reference"

    # --- Controls -------------------------------------------------------------------------
    audit_db_path: str = "audit.db"
    confidence_threshold: float = 0.35
    grounding_threshold: float = 0.25
    max_question_chars: int = 2000
    allowed_topics: list[str] = DEFAULT_ALLOWED_TOPICS

    # --- Operations -----------------------------------------------------------------------
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (read once, then cached)."""
    return Settings()
