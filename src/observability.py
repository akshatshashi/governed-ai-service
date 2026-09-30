"""Structured logging and in-process metrics.

Logging: one JSON object per line on stdout, always carrying ``request_id`` (taken from a
context variable set by the pipeline). Log calls pass structured fields, never free text built
from user input — raw user input is never logged, only counts, kinds and identifiers.

Metrics: counters and a latency histogram rendered in the Prometheus text exposition format at
``GET /metrics``, so any Prometheus-compatible scraper can alert on them (see RUNBOOK.md).
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
from datetime import UTC, datetime

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Send JSON logs to stdout and quieten chatty third-party loggers."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpx2", "chromadb", "sentence_transformers", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(
    logger: logging.Logger, event: str, level: int = logging.INFO, **fields: object
) -> None:
    """Log a named event with structured fields."""
    logger.log(level, event, extra={"fields": fields})


LATENCY_BUCKETS_MS: tuple[float, ...] = (
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    20000,
    40000,
    60000,
)

_HELP = {
    "gas_request_count": "Requests to /ask by outcome (answered, refused, abstained, fallback).",
    "gas_fallback_count": "Responses served by the deterministic fallback, by reason.",
    "gas_refusal_count": "Requests blocked by an input guardrail, by guardrail.",
    "gas_tool_call_count": "Tool calls requested by the model, by tool and control outcome.",
    "gas_token_usage": "LLM tokens consumed, by kind.",
    "gas_approval_decisions": "Human approval decisions, by decision.",
    "gas_latency_ms": "End-to-end /ask latency in milliseconds.",
}


class Metrics:
    """Thread-safe counters plus one latency histogram."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._bucket_counts = [0] * len(LATENCY_BUCKETS_MS)
        self._latency_sum = 0.0
        self._latency_count = 0

    def inc(self, name: str, value: float = 1, **labels: str) -> None:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe_latency(self, ms: float) -> None:
        with self._lock:
            self._latency_sum += ms
            self._latency_count += 1
            for i, bound in enumerate(LATENCY_BUCKETS_MS):
                if ms <= bound:
                    self._bucket_counts[i] += 1

    def value(self, name: str, **labels: str) -> float:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            return self._counters.get(key, 0)

    def render(self, gauges: dict[str, float] | None = None) -> str:
        """Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            by_name: dict[str, list[tuple[tuple[tuple[str, str], ...], float]]] = {}
            for (name, labels), value in sorted(self._counters.items()):
                by_name.setdefault(name, []).append((labels, value))
            for name in sorted(by_name):
                lines.append(f"# HELP {name} {_HELP.get(name, name)}")
                lines.append(f"# TYPE {name} counter")
                for labels, value in by_name[name]:
                    lines.append(f"{name}{_labels(labels)} {value:g}")
            lines.append(f"# HELP gas_latency_ms {_HELP['gas_latency_ms']}")
            lines.append("# TYPE gas_latency_ms histogram")
            for bound, count in zip(LATENCY_BUCKETS_MS, self._bucket_counts, strict=True):
                lines.append(f'gas_latency_ms_bucket{{le="{bound:g}"}} {count}')
            lines.append(f'gas_latency_ms_bucket{{le="+Inf"}} {self._latency_count}')
            lines.append(f"gas_latency_ms_sum {self._latency_sum:.2f}")
            lines.append(f"gas_latency_ms_count {self._latency_count}")
        for name, value in sorted((gauges or {}).items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value:g}")
        return "\n".join(lines) + "\n"


def _labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v.replace(chr(34), "")}"' for k, v in labels)
    return "{" + inner + "}"
