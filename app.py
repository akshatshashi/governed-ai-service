"""FastAPI entry point. Deliberately thin: HTTP concerns only; all logic lives in ``src/``.

Run locally:  uvicorn app:app --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.approval import ApprovalRequest, SelfApprovalError
from src.audit import AuditWriteError
from src.config import get_settings
from src.observability import configure_logging
from src.pipeline import ApprovalDecisionResult, AskResponse, GovernedPipeline, build_pipeline

__version__ = "1.0.0"

_IDENTITY = r"^[A-Za-z0-9._@-]{1,64}$"


class AskRequest(BaseModel):
    # The hard cap here only protects the process; the configurable business limit is enforced
    # (and audited) inside the pipeline.
    question: str = Field(min_length=1, max_length=20000)
    user_id: str = Field(default="anonymous", pattern=_IDENTITY)


class DecisionRequest(BaseModel):
    approved: bool
    decided_by: str = Field(pattern=_IDENTITY)


def create_app(pipeline: GovernedPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        configure_logging(settings.log_level)
        app.state.pipeline = pipeline or build_pipeline(settings)
        yield

    app = FastAPI(
        title="Governed AI Service",
        version=__version__,
        description="AML/CTF compliance Q&A where the controls are the product.",
        lifespan=lifespan,
    )

    def _pipeline(request: Request) -> GovernedPipeline:
        return request.app.state.pipeline

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        try:
            return _pipeline(request).ask(body.question, body.user_id)
        except AuditWriteError as exc:
            # Fail closed: no answer leaves the service without an audit record.
            raise HTTPException(503, "audit store unavailable; request was not served") from exc

    @app.get("/approvals", response_model=list[ApprovalRequest])
    def list_approvals(request: Request) -> list[ApprovalRequest]:
        return _pipeline(request).approvals.pending()

    @app.post("/approvals/{request_id}", response_model=ApprovalDecisionResult)
    def decide(request_id: str, body: DecisionRequest, request: Request) -> ApprovalDecisionResult:
        try:
            return _pipeline(request).decide(request_id, body.approved, body.decided_by)
        except KeyError as exc:
            raise HTTPException(404, "no approval request with that id") from exc
        except SelfApprovalError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/health")
    def health(request: Request) -> dict:
        return _pipeline(request).health()

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request) -> str:
        p = _pipeline(request)
        return p.metrics.render(
            gauges={
                "gas_pending_approvals": len(p.approvals.pending()),
                "gas_index_chunks": p.retriever.count(),
            }
        )

    @app.get("/audit/{request_id}")
    def audit_record(request_id: str, request: Request) -> dict:
        view = _pipeline(request).audit_view(request_id)
        if view is None:
            raise HTTPException(404, "no audit record with that id")
        return view

    @app.get("/audit")
    def audit_query(
        request: Request,
        since: str | None = Query(default=None, description="ISO 8601 UTC timestamp"),
        fallback_only: bool = False,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict]:
        records = _pipeline(request).audit.query(since=since, fallback_only=fallback_only)
        return [r.model_dump() for r in records[-limit:]]

    return app


app = create_app()
