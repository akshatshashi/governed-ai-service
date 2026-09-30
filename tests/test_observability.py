"""Structured logging and metrics."""

import json
import logging

from src.llm import ExtractiveLLM
from src.observability import JsonFormatter, Metrics, request_id_var


def test_json_log_line_carries_request_id():
    token = request_id_var.set("req-123")
    try:
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "ask.completed", None, None)
        record.fields = {"outcome": "answered"}
        line = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert line["request_id"] == "req-123"
    assert line["event"] == "ask.completed" and line["outcome"] == "answered"


def test_metrics_render_prometheus_format():
    metrics = Metrics()
    metrics.inc("gas_request_count", outcome="answered")
    metrics.observe_latency(120)
    text = metrics.render(gauges={"gas_pending_approvals": 2})
    assert 'gas_request_count{outcome="answered"} 1' in text
    assert 'gas_latency_ms_bucket{le="250"} 1' in text
    assert "gas_pending_approvals 2" in text


def test_pipeline_logs_never_contain_raw_input(make_pipeline, caplog):
    secret = "jane.citizen@example.com"
    pipeline = make_pipeline(ExtractiveLLM())
    with caplog.at_level(logging.INFO, logger="governed_ai_service"):
        pipeline.ask(f"I am {secret}. What is the TTR threshold for cash?")
    formatted = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert "ask.completed" in formatted
    assert secret not in formatted and "TTR threshold" not in formatted
