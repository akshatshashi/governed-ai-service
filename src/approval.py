"""Human-in-the-loop approval queue for high-impact tool calls.

A high-impact action requested by the agent is never executed inline. It is written here as a
``pending`` request, and only a named human decision moves it to ``approved`` or ``rejected``.

Auditability properties:
- **Decisions are immutable.** ``decide()`` refuses to change a request that has already been
  decided (raises ``ValueError``). A decision, once recorded, is evidence; it cannot be quietly
  reversed. A change of mind is a new request.
- **Segregation of duties.** The person who raised a request cannot approve it.
- Every decision records who made it and when.

Approved actions are applied to ``ReviewCaseLog`` — a stand-in for a case-management system.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ApprovalRequest(BaseModel):
    request_id: str
    tool_name: str
    args: dict
    rationale: str
    status: str  # "pending" | "approved" | "rejected"
    created_at: str
    decided_at: str | None = None
    decided_by: str | None = None
    requested_by: str = "anonymous"


class SelfApprovalError(PermissionError):
    """Raised when someone tries to approve a request they raised themselves."""


def _connect(db_path: str) -> sqlite3.Connection:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, check_same_thread=False)


class ApprovalQueue:
    """SQLite-backed queue of approval requests, keyed by the originating ``request_id``."""

    def __init__(self, db_path: str) -> None:
        self._conn = _connect(db_path)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    request_id TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    args TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
                    decided_by TEXT,
                    requested_by TEXT NOT NULL
                )
                """
            )

    def submit(self, req: ApprovalRequest) -> None:
        if req.status != "pending":
            raise ValueError("new approval requests must be pending")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO approvals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    req.request_id,
                    req.tool_name,
                    json.dumps(req.args, sort_keys=True),
                    req.rationale,
                    req.status,
                    req.created_at,
                    req.decided_at,
                    req.decided_by,
                    req.requested_by,
                ),
            )

    def get(self, request_id: str) -> ApprovalRequest | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM approvals WHERE request_id = ?", (request_id,)
            ).fetchone()
        return self._to_request(row) if row else None

    def pending(self) -> list[ApprovalRequest]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM approvals WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
        return [self._to_request(r) for r in rows]

    def decide(self, request_id: str, approved: bool, decided_by: str) -> ApprovalRequest:
        """Record a human decision.

        Raises ``KeyError`` if the request does not exist, ``ValueError`` if it has already been
        decided (decisions are immutable — see module docstring), and ``SelfApprovalError`` if
        the decider is the person who raised the request.
        """
        if not decided_by.strip():
            raise ValueError("decided_by is required")
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM approvals WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(request_id)
            current = self._to_request(row)
            if current.status != "pending":
                raise ValueError(
                    f"request {request_id} was already {current.status} by {current.decided_by}"
                )
            if decided_by.strip().lower() == current.requested_by.strip().lower():
                raise SelfApprovalError("the requester cannot approve their own request")
            status = "approved" if approved else "rejected"
            decided_at = utc_now()
            # The WHERE status='pending' clause makes the transition atomic even under races.
            cur = self._conn.execute(
                "UPDATE approvals SET status = ?, decided_at = ?, decided_by = ? "
                "WHERE request_id = ? AND status = 'pending'",
                (status, decided_at, decided_by, request_id),
            )
            if cur.rowcount != 1:
                raise ValueError(f"request {request_id} was decided concurrently")
        return current.model_copy(
            update={"status": status, "decided_at": decided_at, "decided_by": decided_by}
        )

    @staticmethod
    def _to_request(row: tuple) -> ApprovalRequest:
        keys = [
            "request_id",
            "tool_name",
            "args",
            "rationale",
            "status",
            "created_at",
            "decided_at",
            "decided_by",
            "requested_by",
        ]
        data = dict(zip(keys, row, strict=True))
        data["args"] = json.loads(data["args"])
        return ApprovalRequest(**data)


class ReviewCase(BaseModel):
    case_id: str
    reference: str
    reason: str
    opened_at: str
    approved_by: str
    source_request_id: str


class ReviewCaseLog:
    """Where approved ``flag_for_review`` actions land (a stand-in for case management)."""

    def __init__(self, db_path: str) -> None:
        self._conn = _connect(db_path)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_cases (
                    case_id TEXT PRIMARY KEY,
                    reference TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    approved_by TEXT NOT NULL,
                    source_request_id TEXT NOT NULL
                )
                """
            )

    def open_case(
        self, reference: str, reason: str, approved_by: str, source_request_id: str
    ) -> ReviewCase:
        case = ReviewCase(
            case_id=f"CASE-{uuid.uuid4().hex[:8].upper()}",
            reference=reference,
            reason=reason,
            opened_at=utc_now(),
            approved_by=approved_by,
            source_request_id=source_request_id,
        )
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO review_cases VALUES (?, ?, ?, ?, ?, ?)",
                (
                    case.case_id,
                    case.reference,
                    case.reason,
                    case.opened_at,
                    case.approved_by,
                    case.source_request_id,
                ),
            )
        return case

    def all(self) -> list[ReviewCase]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM review_cases ORDER BY opened_at").fetchall()
        keys = list(ReviewCase.model_fields)
        return [ReviewCase(**dict(zip(keys, r, strict=True))) for r in rows]
