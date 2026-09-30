"""Human approval queue."""

import pytest

from src.approval import ApprovalQueue, ApprovalRequest, SelfApprovalError, utc_now


def _request(request_id: str = "req-1") -> ApprovalRequest:
    return ApprovalRequest(
        request_id=request_id,
        tool_name="flag_for_review",
        args={"reference": "TXN-48213", "reason": "structured deposits"},
        rationale="agent requested",
        status="pending",
        created_at=utc_now(),
        requested_by="analyst1",
    )


def test_submit_list_approve_then_second_decision_raises(tmp_path):
    queue = ApprovalQueue(str(tmp_path / "db.sqlite"))
    queue.submit(_request())
    assert [r.request_id for r in queue.pending()] == ["req-1"]

    decided = queue.decide("req-1", approved=True, decided_by="supervisor1")
    assert decided.status == "approved" and decided.decided_by == "supervisor1"
    assert decided.decided_at is not None
    assert queue.pending() == []

    with pytest.raises(ValueError):
        queue.decide("req-1", approved=False, decided_by="supervisor2")
    assert queue.get("req-1").status == "approved"  # unchanged


def test_requester_cannot_approve_own_request(tmp_path):
    queue = ApprovalQueue(str(tmp_path / "db.sqlite"))
    queue.submit(_request())
    with pytest.raises(SelfApprovalError):
        queue.decide("req-1", approved=True, decided_by="Analyst1")
    assert queue.get("req-1").status == "pending"


def test_unknown_request_raises_key_error(tmp_path):
    with pytest.raises(KeyError):
        ApprovalQueue(str(tmp_path / "db.sqlite")).decide("missing", True, "supervisor1")


def test_rejection_is_recorded(tmp_path):
    queue = ApprovalQueue(str(tmp_path / "db.sqlite"))
    queue.submit(_request())
    assert queue.decide("req-1", approved=False, decided_by="supervisor1").status == "rejected"
