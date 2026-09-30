"""HTTP API contract."""

import pytest
from fastapi.testclient import TestClient

from app import create_app
from src.llm import ExtractiveLLM


@pytest.fixture
def client(make_pipeline):
    with TestClient(create_app(make_pipeline(ExtractiveLLM()))) as test_client:
        yield test_client


def test_health_reports_components(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_reachable"] is True
    assert body["index_chunks"] > 0
    assert body["audit_store_ok"] is True


def test_ask_returns_contract_fields_and_audit_is_retrievable(client):
    response = client.post("/ask", json={"question": "How long must transaction records be kept?"})
    assert response.status_code == 200
    body = response.json()
    for field in (
        "request_id",
        "answer",
        "citations",
        "confidence",
        "fallback_used",
        "approval_required",
    ):
        assert field in body
    audit = client.get(f"/audit/{body['request_id']}")
    assert audit.status_code == 200
    assert audit.json()["record"]["request_id"] == body["request_id"]


def test_approval_flow_over_http(client):
    asked = client.post(
        "/ask",
        json={
            "question": "Please flag TXN-48213 for review: split cash deposits.",
            "user_id": "analyst1",
        },
    ).json()
    assert asked["approval_required"] is True
    request_id = asked["request_id"]

    assert [a["request_id"] for a in client.get("/approvals").json()] == [request_id]
    self_approval = client.post(
        f"/approvals/{request_id}", json={"approved": True, "decided_by": "analyst1"}
    )
    assert self_approval.status_code == 403

    decided = client.post(
        f"/approvals/{request_id}", json={"approved": True, "decided_by": "supervisor1"}
    )
    assert decided.status_code == 200
    assert decided.json()["execution_result"].startswith("Opened review case")

    again = client.post(
        f"/approvals/{request_id}", json={"approved": False, "decided_by": "supervisor2"}
    )
    assert again.status_code == 409
    assert (
        client.post("/approvals/missing", json={"approved": True, "decided_by": "x"}).status_code
        == 404
    )


def test_metrics_endpoint_is_prometheus_text(client):
    client.post("/ask", json={"question": "What is the TTR threshold for cash?"})
    text = client.get("/metrics").text
    assert "gas_request_count" in text
    assert "gas_latency_ms_bucket" in text
    assert "gas_pending_approvals" in text


def test_invalid_identity_rejected(client):
    response = client.post("/ask", json={"question": "TTR?", "user_id": "bad id; drop"})
    assert response.status_code == 422


def test_audit_query_filters_fallbacks(client):
    client.post("/ask", json={"question": "What's a good recipe for banana bread?"})
    rows = client.get("/audit", params={"fallback_only": True}).json()
    assert rows and all(r["fallback_used"] for r in rows)
