"""The governed request pipeline — the /ask flow, in strict order.

1. Generate a ``request_id`` (uuid4).
2. Redact PII from the input. Nothing downstream — retrieval, the model, logs, the audit
   store — ever sees the raw question.
3. Input guardrails: length, prompt injection, scope. A block returns a deterministic refusal.
4. Retrieve, then ``Agent.run()`` (model + controlled tool loop).
5. Output checks: citations must resolve, lexical grounding, retrieval confidence. A failure
   returns the deterministic fallback instead of the model's text.
6. Redact PII from the output.
7. If a high-impact tool was requested, submit it to the human approval queue.
8. Write an ``AuditRecord`` — on every path, including refusals, fallbacks and internal errors.
   If the audit write itself fails, the request fails closed (``AuditWriteError``).
9. Return the response.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from pydantic import BaseModel

from src.agent.loop import PROMPT_VERSION, Agent, AgentResult
from src.agent.tools import ToolContext, ToolRegistry, build_default_registry, load_thresholds
from src.approval import (
    ApprovalQueue,
    ApprovalRequest,
    ReviewCaseLog,
    utc_now,
)
from src.audit import AuditRecord, AuditStore
from src.config import Settings, get_settings
from src.fallback import FallbackReason, build_fallback
from src.guardrails import pii
from src.guardrails.grounding import check_grounding
from src.guardrails.injection import GuardResult, check_injection, check_scope
from src.llm import LLM, LLMRefusal, LLMUnavailable, build_llm
from src.observability import Metrics, log_event, request_id_var
from src.retriever import RetrievedChunk, Retriever

logger = logging.getLogger("governed_ai_service")


class Citation(BaseModel):
    number: int
    doc_id: str
    source_path: str
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[Citation]
    confidence: float
    fallback_used: bool
    approval_required: bool
    outcome: str  # "answered" | "refused" | "abstained" | "fallback"
    fallback_reason: str | None
    model: str
    prompt_version: str
    latency_ms: float


class ApprovalDecisionResult(BaseModel):
    approval: ApprovalRequest
    execution_result: str | None


class GovernedPipeline:
    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        retriever: Retriever,
        registry: ToolRegistry,
        audit: AuditStore,
        approvals: ApprovalQueue,
        cases: ReviewCaseLog,
        metrics: Metrics | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.registry = registry
        self.audit = audit
        self.approvals = approvals
        self.cases = cases
        self.metrics = metrics or Metrics()
        self.agent = Agent(llm, retriever, registry, top_k=settings.top_k)
        self._reachability: tuple[float, bool] | None = None
        self._reach_lock = threading.Lock()

    # -----------------------------------------------------------------------------------------
    # /ask
    # -----------------------------------------------------------------------------------------

    def ask(self, question: str, user_id: str = "anonymous") -> AskResponse:
        request_id = str(uuid.uuid4())
        token = request_id_var.set(request_id)
        try:
            return self._ask(request_id, question, user_id)
        finally:
            request_id_var.reset(token)

    def _ask(self, request_id: str, question: str, user_id: str) -> AskResponse:
        start = time.perf_counter()
        s = self.settings

        # Step 2: redact input before anything else touches it.
        redacted_q, pii_in = pii.redact(question)
        guardrails_in: dict[str, Any] = {
            "pii_redaction": {"passed": True, "found": pii.summarise(pii_in)}
        }
        guardrails_out: dict[str, Any] = {}
        chunks: list[RetrievedChunk] = []
        result: AgentResult | None = None
        outcome = "answered"
        reason: FallbackReason | None = None
        answer = ""
        model = self.llm.model
        confidence = 0.0
        citations: list[Citation] = []
        approval_status = "not_required"

        try:
            # Step 3: input guardrails.
            length = GuardResult(passed=len(question) <= s.max_question_chars)
            if not length.passed:
                length.reason = f"input_too_long>{s.max_question_chars}"
            injection = check_injection(redacted_q)
            scope = check_scope(redacted_q, s.allowed_topics)
            guardrails_in.update(
                length=length.model_dump(),
                injection=injection.model_dump(),
                scope=scope.model_dump(),
            )
            blocked_by = next((g.reason for g in (length, injection, scope) if not g.passed), None)

            if blocked_by:
                outcome, reason, model = "refused", FallbackReason.GUARDRAIL_BLOCKED, "not_called"
                answer = build_fallback(reason, [], blocked_by=blocked_by)
                self.metrics.inc("gas_refusal_count", guard=blocked_by.split(":")[0])
            else:
                # Step 4: retrieve, then run the agent.
                chunks = self.retriever.search(redacted_q, s.top_k)
                try:
                    result = self.agent.run(redacted_q, chunks, request_id)
                except LLMUnavailable as exc:
                    outcome, reason = "fallback", FallbackReason.LLM_UNAVAILABLE
                    guardrails_out["llm_error"] = exc.reason
                    answer = build_fallback(reason, chunks)
                except LLMRefusal as exc:
                    outcome, reason = "fallback", FallbackReason.MODEL_REFUSAL
                    guardrails_out["refusal_category"] = exc.category
                    answer = build_fallback(reason, chunks)
                else:
                    model, confidence = result.model, result.confidence
                    outcome, reason, answer = self._check_output(result, guardrails_out)
                    if outcome == "answered":
                        citations = [
                            Citation(
                                number=n, **result.retrieved[n - 1].model_dump(exclude={"text"})
                            )
                            for n in result.cited
                        ]
                        # Step 7: high-impact actions go to the human approval queue.
                        if result.approval_required:
                            self._submit_approvals(request_id, redacted_q, user_id, result)
                            approval_status = "pending"
        except Exception:
            logger.exception("ask.internal_error", extra={"fields": {"stage": "pipeline"}})
            outcome, reason = "fallback", FallbackReason.INTERNAL_ERROR
            answer = build_fallback(reason, chunks)
            citations = []
            approval_status = "not_required"

        # Step 6: redact output.
        answer, pii_out = pii.redact(answer)
        guardrails_out["pii_redaction"] = {"passed": True, "found": pii.summarise(pii_out)}

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        evidence = result.retrieved if result else chunks
        cited = set(result.cited) if result and outcome == "answered" else set()
        token_usage = result.token_usage if result else {}

        # Step 8: audit — every path.
        self.audit.write(
            AuditRecord(
                request_id=request_id,
                timestamp=utc_now(),
                user_input_redacted=redacted_q,
                guardrails_in=guardrails_in,
                retrieved=[
                    {
                        "n": i,
                        "doc_id": c.doc_id,
                        "chunk_index": c.chunk_index,
                        "score": c.score,
                        "cited": i in cited,
                    }
                    for i, c in enumerate(evidence, start=1)
                ],
                prompt_version=PROMPT_VERSION,
                model=model,
                tool_calls=_redact_tool_calls(result.tool_calls if result else []),
                output_redacted=answer,
                guardrails_out=guardrails_out,
                confidence=confidence,
                fallback_used=reason is not None,
                approval_required=approval_status == "pending",
                approval_status=approval_status,
                latency_ms=latency_ms,
                token_usage=token_usage,
                outcome=outcome,
                fallback_reason=reason.value if reason else None,
                user_id=user_id,
            )
        )

        self._record_metrics(outcome, reason, latency_ms, result)
        log_event(
            logger,
            "ask.completed",
            outcome=outcome,
            fallback_reason=reason.value if reason else None,
            model=model,
            latency_ms=latency_ms,
            question_chars=len(question),
            pii_in=pii.summarise(pii_in),
            pii_out=pii.summarise(pii_out),
            retrieved=len(evidence),
            cited=sorted(cited),
            tool_calls=[
                c["name"] + ":" + c["status"] for c in (result.tool_calls if result else [])
            ],
            approval_status=approval_status,
        )

        # Step 9: respond.
        return AskResponse(
            request_id=request_id,
            answer=answer,
            citations=citations,
            confidence=confidence,
            fallback_used=reason is not None,
            approval_required=approval_status == "pending",
            outcome=outcome,
            fallback_reason=reason.value if reason else None,
            model=model,
            prompt_version=PROMPT_VERSION,
            latency_ms=latency_ms,
        )

    def _check_output(
        self, result: AgentResult, guardrails_out: dict[str, Any]
    ) -> tuple[str, FallbackReason | None, str]:
        """Step 5: decide whether the model's answer may be returned."""
        s = self.settings
        if result.abstained:
            guardrails_out["grounding"] = {"passed": True, "reason": "abstention"}
            return "abstained", None, result.answer

        grounding = check_grounding(
            result.answer,
            [c.text for c in result.retrieved] + result.tool_evidence,
            s.grounding_threshold,
        )
        citations_ok = not result.invalid_citations
        confident = result.confidence >= s.confidence_threshold
        guardrails_out["grounding"] = grounding.model_dump()
        guardrails_out["citations"] = {"passed": citations_ok, "invalid": result.invalid_citations}
        guardrails_out["confidence"] = {
            "passed": confident,
            "value": result.confidence,
            "threshold": s.confidence_threshold,
        }
        if not grounding.passed or not citations_ok:
            reason = FallbackReason.GROUNDING_FAILED
            return "fallback", reason, build_fallback(reason, result.retrieved)
        if not confident:
            reason = FallbackReason.LOW_CONFIDENCE
            return "abstained", reason, build_fallback(reason, result.retrieved)
        return "answered", None, result.answer

    def _submit_approvals(
        self, request_id: str, redacted_q: str, user_id: str, result: AgentResult
    ) -> None:
        for action in result.pending_actions:
            self.approvals.submit(
                ApprovalRequest(
                    request_id=request_id,
                    tool_name=action["tool"],
                    args=_redact_args(action["args"]),
                    rationale=f"Requested by the agent while answering: {redacted_q[:300]}",
                    status="pending",
                    created_at=utc_now(),
                    requested_by=user_id,
                )
            )
            log_event(logger, "approval.submitted", tool=action["tool"])

    def _record_metrics(
        self,
        outcome: str,
        reason: FallbackReason | None,
        latency_ms: float,
        result: AgentResult | None,
    ) -> None:
        self.metrics.inc("gas_request_count", outcome=outcome)
        self.metrics.observe_latency(latency_ms)
        if reason is not None:
            self.metrics.inc("gas_fallback_count", reason=reason.value)
        if result:
            for call in result.tool_calls:
                self.metrics.inc("gas_tool_call_count", tool=call["name"], status=call["status"])
            self.metrics.inc(
                "gas_token_usage", result.token_usage.get("prompt_tokens", 0), kind="prompt"
            )
            self.metrics.inc(
                "gas_token_usage",
                result.token_usage.get("completion_tokens", 0),
                kind="completion",
            )

    # -----------------------------------------------------------------------------------------
    # Approvals
    # -----------------------------------------------------------------------------------------

    def decide(self, request_id: str, approved: bool, decided_by: str) -> ApprovalDecisionResult:
        """Record a human decision; only an approval causes the gated tool to run."""
        token = request_id_var.set(request_id)
        try:
            decision = self.approvals.decide(request_id, approved, decided_by)
            execution: str | None = None
            if decision.status == "approved":
                ctx = ToolContext(request_id=request_id, approved_by=decided_by)
                execution = self.registry.execute_approved(decision.tool_name, decision.args, ctx)
            self.metrics.inc("gas_approval_decisions", decision=decision.status)
            log_event(
                logger,
                "approval.decided",
                decision=decision.status,
                tool=decision.tool_name,
                executed=execution is not None,
            )
            return ApprovalDecisionResult(approval=decision, execution_result=execution)
        finally:
            request_id_var.reset(token)

    # -----------------------------------------------------------------------------------------
    # Read side
    # -----------------------------------------------------------------------------------------

    def audit_view(self, request_id: str) -> dict | None:
        """The audit record joined with any later approval decision and resulting case."""
        record = self.audit.get(request_id)
        if record is None:
            return None
        approval = self.approvals.get(request_id)
        cases = [c for c in self.cases.all() if c.source_request_id == request_id]
        return {
            "record": record.model_dump(),
            "approval": approval.model_dump() if approval else None,
            "cases": [c.model_dump() for c in cases],
        }

    def llm_reachable(self, ttl_seconds: float = 30.0) -> bool:
        """Cached reachability probe, so /health cannot hammer the model provider."""
        with self._reach_lock:
            now = time.monotonic()
            if self._reachability and now - self._reachability[0] < ttl_seconds:
                return self._reachability[1]
            ok = self.llm.is_reachable()
            self._reachability = (now, ok)
            return ok

    def health(self) -> dict:
        llm_ok = self.llm_reachable()
        chunks = self.retriever.count()
        audit_ok = self.audit.ping()
        return {
            "status": "ok" if (llm_ok and chunks > 0 and audit_ok) else "degraded",
            "llm_reachable": llm_ok,
            "index_chunks": chunks,
            "audit_store_ok": audit_ok,
            "llm_backend": self.settings.llm_backend,
            "model": self.llm.model,
            "prompt_version": PROMPT_VERSION,
            "pending_approvals": len(self.approvals.pending()),
        }


def _redact_args(args: dict) -> dict:
    return {k: pii.redact(v)[0] if isinstance(v, str) else v for k, v in args.items()}


def _redact_tool_calls(calls: list[dict]) -> list[dict]:
    out = []
    for call in calls:
        clean = dict(call)
        clean["args"] = _redact_args(call.get("args", {}))
        if "result_summary" in clean:
            clean["result_summary"] = pii.redact(clean["result_summary"])[0]
        out.append(clean)
    return out


def build_pipeline(settings: Settings | None = None, llm: LLM | None = None) -> GovernedPipeline:
    """Wire the production pipeline from settings."""
    settings = settings or get_settings()
    retriever = Retriever(settings.index_dir, settings.embedding_model)
    audit = AuditStore(settings.audit_db_path)
    approvals = ApprovalQueue(settings.audit_db_path)
    cases = ReviewCaseLog(settings.audit_db_path)
    registry = build_default_registry(
        retriever, load_thresholds(settings.reference_dir), cases.open_case
    )
    return GovernedPipeline(
        settings, llm or build_llm(settings), retriever, registry, audit, approvals, cases
    )
