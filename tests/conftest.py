"""Shared fixtures: a real index over the committed synthetic corpus, and scripted fake models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.agent.tools import build_default_registry, load_thresholds
from src.approval import ApprovalQueue, ReviewCaseLog
from src.audit import AuditStore
from src.config import Settings
from src.ingest import index_chunks, load_documents, open_client
from src.llm import LLMResponse, ToolCall, ToolDefinition, ToolResult
from src.pipeline import GovernedPipeline
from src.retriever import Retriever

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus" / "synthetic"
REFERENCE_DIR = ROOT / "data" / "reference"


def make_response(
    text: str = "", tool_calls: list[ToolCall] | None = None, model: str = "scripted-model"
) -> LLMResponse:
    return LLMResponse(
        text=text,
        model=model,
        prompt_tokens=100,
        completion_tokens=20,
        latency_ms=1.0,
        tool_calls=tool_calls or [],
        stop_reason="tool_use" if tool_calls else "end_turn",
    )


class ScriptedLLM:
    """A fake model that plays back a fixed list of turns (or raises a scripted exception)."""

    model = "scripted-model"

    def __init__(self, turns: list[LLMResponse | Exception]) -> None:
        self.turns = list(turns)
        self.prompts: list[str] = []
        self.tool_results: list[list[ToolResult]] = []
        self.systems: list[str] = []

    def _next(self) -> LLMResponse:
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        return self._next()

    def start_session(self, system: str, tools: list[ToolDefinition]) -> ScriptedLLM:
        self.systems.append(system)
        return self

    def send(self, text: str) -> LLMResponse:
        self.prompts.append(text)
        return self._next()

    def send_tool_results(self, results: list[ToolResult]) -> LLMResponse:
        self.tool_results.append(results)
        return self._next()

    def is_reachable(self) -> bool:
        return True


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A persistent Chroma index over the committed synthetic corpus (built once per session)."""
    path = tmp_path_factory.mktemp("index")
    settings = Settings()
    count = index_chunks(
        open_client(str(path)),
        load_documents(str(CORPUS_DIR)),
        settings.embedding_model,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    assert count > 0
    return str(path)


@pytest.fixture(scope="session")
def retriever(index_dir: str) -> Retriever:
    return Retriever(index_dir, Settings().embedding_model)


@pytest.fixture
def make_pipeline(tmp_path: Path, retriever: Retriever) -> Callable[..., GovernedPipeline]:
    """Build a fully wired pipeline around any LLM, with a fresh audit database per test."""

    def _make(llm, **overrides) -> GovernedPipeline:
        db = str(tmp_path / "audit.db")
        settings = Settings(
            audit_db_path=db,
            reference_dir=str(REFERENCE_DIR),
            llm_backend="extractive",
            **overrides,
        )
        cases = ReviewCaseLog(db)
        registry = build_default_registry(
            retriever, load_thresholds(str(REFERENCE_DIR)), cases.open_case
        )
        return GovernedPipeline(
            settings, llm, retriever, registry, AuditStore(db), ApprovalQueue(db), cases
        )

    return _make
