"""The audit store."""

import pytest

from src.audit import AuditRecord, AuditStore, AuditWriteError


def _record(request_id: str, timestamp: str, fallback: bool) -> AuditRecord:
    return AuditRecord(
        request_id=request_id,
        timestamp=timestamp,
        user_input_redacted="My email is [REDACTED:email]. What is the TTR threshold?",
        guardrails_in={"injection": {"passed": True, "reason": None}},
        retrieved=[{"doc_id": "a.md", "chunk_index": 2, "score": 0.71}],
        prompt_version="v1",
        model="claude-opus-5-5",
        tool_calls=[{"name": "search_corpus", "args": {"query": "ttr"}, "status": "executed"}],
        output_redacted="AUD 10,000 [1]",
        guardrails_out={"grounding": {"passed": True, "reason": "grounding_score=0.90"}},
        confidence=0.71,
        fallback_used=fallback,
        approval_required=False,
        approval_status="not_required",
        latency_ms=812.5,
        token_usage={"prompt_tokens": 900, "completion_tokens": 40},
        outcome="fallback" if fallback else "answered",
        fallback_reason="llm_unavailable" if fallback else None,
        user_id="analyst1",
    )


def test_every_field_round_trips(tmp_path):
    store = AuditStore(str(tmp_path / "audit.db"))
    record = _record("req-1", "2026-09-30T01:00:00.000Z", fallback=False)
    store.write(record)
    assert store.get("req-1") == record


def test_get_unknown_returns_none(tmp_path):
    assert AuditStore(str(tmp_path / "audit.db")).get("nope") is None


def test_query_filters_fallback_and_since(tmp_path):
    store = AuditStore(str(tmp_path / "audit.db"))
    store.write(_record("req-1", "2026-09-30T01:00:00.000Z", fallback=False))
    store.write(_record("req-2", "2026-09-30T02:00:00.000Z", fallback=True))
    store.write(_record("req-3", "2026-09-30T03:00:00.000Z", fallback=True))

    assert [r.request_id for r in store.query(fallback_only=True)] == ["req-2", "req-3"]
    assert [r.request_id for r in store.query(since="2026-09-30T02:30:00Z")] == ["req-3"]
    assert len(store.query()) == 3


def test_records_are_immutable(tmp_path):
    store = AuditStore(str(tmp_path / "audit.db"))
    store.write(_record("req-1", "2026-09-30T01:00:00.000Z", fallback=False))
    with pytest.raises(AuditWriteError):
        store.write(_record("req-1", "2026-09-30T01:00:00.000Z", fallback=True))
    assert not hasattr(store, "update") and not hasattr(store, "delete")
