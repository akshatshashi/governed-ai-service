"""The audit store: every request, fully reconstructable after the fact.

One ``AuditRecord`` is written per request on **every** exit path — answered, refused, abstained,
fallback and internal error. A record captures what was asked (redacted), which guardrails ran and
what they decided, which chunks were retrieved (ids and scores, not text — the text is recoverable
from the versioned index), the prompt version and model, every tool call and its control outcome,
what was returned (redacted), and cost/latency.

Rules enforced here:
- Only redacted text is stored. Raw PII never reaches this module.
- Records are append-only: there is no update or delete method. Later events (e.g. an approval
  decision) are recorded in their own tables and joined at read time.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from pydantic import BaseModel, Field

_JSON_FIELDS = ("guardrails_in", "retrieved", "tool_calls", "guardrails_out", "token_usage")


class AuditRecord(BaseModel):
    request_id: str
    timestamp: str  # ISO 8601 UTC
    user_input_redacted: str
    guardrails_in: dict  # check name -> {passed, reason, ...}
    retrieved: list[dict]  # doc_id, chunk_index, score (NOT full text)
    prompt_version: str
    model: str
    tool_calls: list[dict]  # tool name, validated args, status, result summary
    output_redacted: str
    guardrails_out: dict
    confidence: float
    fallback_used: bool
    approval_required: bool
    approval_status: str  # "not_required" | "pending" | "approved" | "rejected"
    latency_ms: float
    token_usage: dict
    outcome: str = "answered"  # "answered" | "refused" | "abstained" | "fallback"
    fallback_reason: str | None = None
    user_id: str = Field(default="anonymous")


class AuditWriteError(RuntimeError):
    """Raised when an audit record cannot be persisted. The service fails closed on this."""


_COLUMNS = list(AuditRecord.model_fields)


class AuditStore:
    """SQLite-backed, append-only audit log."""

    def __init__(self, db_path: str) -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_records (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_input_redacted TEXT NOT NULL,
                    guardrails_in TEXT NOT NULL,
                    retrieved TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tool_calls TEXT NOT NULL,
                    output_redacted TEXT NOT NULL,
                    guardrails_out TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    fallback_used INTEGER NOT NULL,
                    approval_required INTEGER NOT NULL,
                    approval_status TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    token_usage TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    fallback_reason TEXT,
                    user_id TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_records(timestamp)"
            )

    def write(self, record: AuditRecord) -> None:
        """Insert a record. A duplicate ``request_id`` is rejected — records are immutable."""
        row = record.model_dump()
        values = [
            json.dumps(row[c], sort_keys=True) if c in _JSON_FIELDS else row[c] for c in _COLUMNS
        ]
        placeholders = ", ".join("?" for _ in _COLUMNS)
        sql = f"INSERT INTO audit_records ({', '.join(_COLUMNS)}) VALUES ({placeholders})"  # noqa: S608 — column names are constants
        try:
            with self._lock, self._conn:
                self._conn.execute(sql, values)
        except sqlite3.Error as exc:
            raise AuditWriteError(str(exc)) from exc

    def get(self, request_id: str) -> AuditRecord | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM audit_records WHERE request_id = ?", (request_id,)
            )
            row = cur.fetchone()
            names = [d[0] for d in cur.description]
        return self._to_record(names, row) if row else None

    def query(self, since: str | None = None, fallback_only: bool = False) -> list[AuditRecord]:
        """Records in time order, optionally only those at/after ``since`` or that used fallback."""
        clauses: list[str] = []
        params: list[object] = []
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if fallback_only:
            clauses.append("fallback_used = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM audit_records {where} ORDER BY timestamp",  # noqa: S608 — clauses are constants
                params,
            )
            rows = cur.fetchall()
            names = [d[0] for d in cur.description]
        return [self._to_record(names, row) for row in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM audit_records").fetchone()[0])

    def ping(self) -> bool:
        """True if the database answers a trivial query (used by /health)."""
        try:
            with self._lock:
                self._conn.execute("SELECT 1").fetchone()
        except sqlite3.Error:
            return False
        return True

    @staticmethod
    def _to_record(names: list[str], row: tuple) -> AuditRecord:
        data = dict(zip(names, row, strict=True))
        for field in _JSON_FIELDS:
            data[field] = json.loads(data[field])
        data["fallback_used"] = bool(data["fallback_used"])
        data["approval_required"] = bool(data["approval_required"])
        return AuditRecord(**data)
