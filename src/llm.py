"""Language-model interface.

Everything else in the service depends only on the ``LLM`` / ``LLMSession`` protocols defined
here, so the model provider is swappable. Two implementations ship:

``ClaudeLLM``
    Anthropic Claude through the official ``anthropic`` SDK. This is the default backend.

``ExtractiveLLM``
    A deterministic, offline, *non-generative* baseline. It answers by selecting sentences from
    the context it is given and uses simple rules to decide on tool calls. It exists so that CI
    can run the full evaluation gate with no API key and no network, reproducibly, and it doubles
    as a documented degraded mode.
"""

from __future__ import annotations

import re
import time
from typing import Any, Protocol

import anthropic
from pydantic import BaseModel, Field

from src.text import content_terms, split_sentences

ABSTAIN_TEXT = "I don't have enough information in the provided sources to answer that."

# Server-side refusal fallback: if Claude's safety classifiers decline a request, the API re-runs
# it on Anthropic's recommended fallback model inside the same call. The model that actually
# served the answer is reported in the response and recorded in the audit trail.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(Exception):
    """The model backend could not be reached or could not serve the request."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class LLMRefusal(Exception):
    """The model (and any server-side fallback model) declined to answer."""

    def __init__(self, category: str | None) -> None:
        super().__init__(f"model refusal (category={category})")
        self.category = category


class ToolDefinition(BaseModel):
    """A tool as advertised to the model. Execution is governed separately by the registry."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    """A tool invocation *requested* by the model. Nothing runs until the registry allows it."""

    id: str
    name: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    """The outcome of a tool call, returned to the model."""

    call_id: str
    content: str
    is_error: bool = False


class LLMResponse(BaseModel):
    """One model turn."""

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str | None = None


class LLMSession(Protocol):
    """A multi-turn exchange used by the agent's tool loop."""

    def send(self, text: str) -> LLMResponse: ...

    def send_tool_results(self, results: list[ToolResult]) -> LLMResponse: ...


class LLM(Protocol):
    """The contract every model backend implements."""

    model: str

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse: ...

    def start_session(self, system: str, tools: list[ToolDefinition]) -> LLMSession: ...

    def is_reachable(self) -> bool: ...


# ---------------------------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------------------------


class ClaudeLLM:
    """Anthropic Claude backend.

    Retries: the SDK retries connection errors, 408/409/429 and 5xx twice with backoff; this class
    adds none of its own. Any failure after that surfaces as ``LLMUnavailable`` with a short
    machine-readable reason, which the pipeline turns into a deterministic fallback response.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
        timeout_seconds: float = 60.0,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout_seconds, max_retries=2
        )

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        message, latency_ms = self._create([{"role": "user", "content": prompt}], system, [])
        return self._to_response(message, latency_ms)

    def start_session(self, system: str, tools: list[ToolDefinition]) -> LLMSession:
        return _ClaudeSession(self, system, tools)

    def is_reachable(self) -> bool:
        """Cheap liveness probe: the Models API needs valid credentials and costs no tokens."""
        try:
            self._client.models.retrieve(self.model)
        except (anthropic.AnthropicError, TypeError):
            return False
        return True

    def _create(
        self, messages: list[dict[str, Any]], system: str | None, tools: list[ToolDefinition]
    ) -> tuple[Any, float]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            "output_config": {"effort": self._effort},
            "betas": [_FALLBACK_BETA],
            "fallbacks": "default",
            # Auto-cache the conversation prefix so the second turn of a tool loop re-reads the
            # retrieved context from cache instead of paying for it again.
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [t.model_dump() for t in tools]

        start = time.perf_counter()
        try:
            message = self._client.beta.messages.create(**params)
        except TypeError as exc:
            # The SDK raises TypeError when no API key / auth token can be resolved.
            if "authentication" in str(exc).lower():
                raise LLMUnavailable("no_credentials") from exc
            raise
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailable("authentication_failed") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMUnavailable("permission_denied") from exc
        except anthropic.NotFoundError as exc:
            raise LLMUnavailable("model_not_found") from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailable("rate_limited") from exc
        except anthropic.BadRequestError as exc:
            raise LLMUnavailable("bad_request", exc.message) from exc
        except anthropic.APIStatusError as exc:
            raise LLMUnavailable(f"api_error_{exc.status_code}") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMUnavailable("timeout") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable("connection_error") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        # Branch on stop_reason before reading content: a refused turn may have no content.
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMRefusal(getattr(details, "category", None) if details else None)
        return message, latency_ms

    @staticmethod
    def _to_response(message: Any, latency_ms: float) -> LLMResponse:
        text = "".join(block.text for block in message.content if block.type == "text")
        calls = [
            ToolCall(id=block.id, name=block.name, args=dict(block.input))
            for block in message.content
            if block.type == "tool_use"
        ]
        usage = message.usage
        prompt_tokens = (
            (usage.input_tokens or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        )
        return LLMResponse(
            text=text.strip(),
            model=message.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            tool_calls=calls,
            stop_reason=message.stop_reason,
        )


class _ClaudeSession:
    """Conversation state for one agent run.

    History is append-only: each assistant turn is stored exactly as returned (including thinking
    blocks) and replayed unchanged, which the API requires for multi-turn tool use.
    """

    def __init__(self, llm: ClaudeLLM, system: str, tools: list[ToolDefinition]) -> None:
        self._llm = llm
        self._system = system
        self._tools = tools
        self._messages: list[dict[str, Any]] = []

    def send(self, text: str) -> LLMResponse:
        self._messages.append({"role": "user", "content": text})
        return self._step()

    def send_tool_results(self, results: list[ToolResult]) -> LLMResponse:
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ]
        self._messages.append({"role": "user", "content": blocks})
        return self._step()

    def _step(self) -> LLMResponse:
        message, latency_ms = self._llm._create(self._messages, self._system, self._tools)
        self._messages.append({"role": "assistant", "content": message.content})
        return self._llm._to_response(message, latency_ms)


# ---------------------------------------------------------------------------------------------
# Extractive baseline (deterministic, offline)
# ---------------------------------------------------------------------------------------------

_PASSAGE_RE = re.compile(
    r"^\[(\d+)\] \(source: [^)]*\)\n(.*?)(?=^\[\d+\] \(source:|^</context>|\Z)", re.S | re.M
)
_QUESTION_RE = re.compile(r"^Question:\s*(.+)$", re.M)
_REFERENCE_RE = re.compile(r"\b[A-Z]{2,6}-\d{3,}\b")
_FLAG_RE = re.compile(r"\b(flag|escalate)\b", re.I)
_THRESHOLD_RE = re.compile(r"\b(threshold|limit)s?\b", re.I)
_REDACTION_RE = re.compile(r"\[REDACTED:\w+\]")

# Words too common in this corpus to count as evidence that a sentence answers a question.
_GENERIC_TERMS = content_terms(
    "austrac australia australian kestrel fx internal policy under need long quickly "
    "happen required require requirement"
)

# Country names the rule-based tool selector recognises (Australia is answered from the corpus).
_COUNTRY_ALIASES = {
    "new zealand": "NZ",
    "united states": "US",
    "usa": "US",
    "america": "US",
    "canada": "CA",
    "united kingdom": "GB",
    "uk": "GB",
    "britain": "GB",
    "singapore": "SG",
}


class ExtractiveLLM:
    """Deterministic sentence-selection baseline. Makes no network calls."""

    model = "extractive-baseline-v1"

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        start = time.perf_counter()
        question, passages = _parse_prompt(prompt)
        return _extractive_response(extractive_answer(question, passages), start)

    def start_session(self, system: str, tools: list[ToolDefinition]) -> LLMSession:
        return _ExtractiveSession({t.name for t in tools})

    def is_reachable(self) -> bool:
        return True


class _ExtractiveSession:
    def __init__(self, tool_names: set[str]) -> None:
        self._tools = tool_names
        self._question = ""
        self._passages: list[tuple[int, str]] = []

    def send(self, text: str) -> LLMResponse:
        start = time.perf_counter()
        self._question, self._passages = _parse_prompt(text)
        call = self._choose_tool(self._question)
        if call is not None:
            return LLMResponse(
                text="",
                model=ExtractiveLLM.model,
                prompt_tokens=None,
                completion_tokens=None,
                latency_ms=(time.perf_counter() - start) * 1000,
                tool_calls=[call],
                stop_reason="tool_use",
            )
        return _extractive_response(extractive_answer(self._question, self._passages), start)

    def send_tool_results(self, results: list[ToolResult]) -> LLMResponse:
        start = time.perf_counter()
        parts: list[str] = []
        for result in results:
            if result.is_error:
                parts.append(f"The requested action could not be completed: {result.content}")
            else:
                parts.append(result.content)
        return _extractive_response(" ".join(parts) or ABSTAIN_TEXT, start)

    def _choose_tool(self, question: str) -> ToolCall | None:
        reference = _REFERENCE_RE.search(question)
        if "flag_for_review" in self._tools and _FLAG_RE.search(question) and reference:
            reason = _REFERENCE_RE.sub("", _FLAG_RE.sub("", question))
            reason = re.sub(r"\s+", " ", reason).strip(" .,:;-")[:300] or "flagged by user"
            return ToolCall(
                id="call_1",
                name="flag_for_review",
                args={"reference": reference.group(0), "reason": reason},
            )
        if "lookup_threshold" in self._tools and _THRESHOLD_RE.search(question):
            lowered = question.lower()
            for alias, code in _COUNTRY_ALIASES.items():
                if re.search(rf"\b{re.escape(alias)}\b", lowered):
                    return ToolCall(id="call_1", name="lookup_threshold", args={"country": code})
        return None


def _parse_prompt(prompt: str) -> tuple[str, list[tuple[int, str]]]:
    match = _QUESTION_RE.search(prompt)
    question = match.group(1).strip() if match else prompt.strip()
    passages = [(int(n), body.strip()) for n, body in _PASSAGE_RE.findall(prompt)]
    return question, passages


def extractive_answer(
    question: str, passages: list[tuple[int, str]], max_sentences: int = 2
) -> str:
    """Pick the context sentences that share the most content words with the question.

    A sentence qualifies only if it shares at least two specific (non-generic) terms with the
    question and covers at least 40% of them; otherwise the baseline abstains.
    """
    q_terms = content_terms(_REDACTION_RE.sub(" ", question)) - _GENERIC_TERMS
    if not q_terms:
        return ABSTAIN_TEXT
    scored: list[tuple[float, int, int, str]] = []
    for number, body in passages:
        for sentence in split_sentences(body):
            if sentence.startswith(("#", "Provenance:")):
                continue  # headings and provenance notes are not answers
            overlap = q_terms & content_terms(sentence)
            coverage = len(overlap) / len(q_terms)
            if len(overlap) >= 2 and coverage >= 0.4:
                scored.append((coverage, len(overlap), -number, sentence))
    if not scored:
        return ABSTAIN_TEXT
    scored.sort(reverse=True)
    chosen: list[str] = []
    for _, _, neg_number, sentence in scored:
        cited = f"{sentence} [{-neg_number}]"
        if sentence not in " ".join(chosen):
            chosen.append(cited)
        if len(chosen) == max_sentences:
            break
    return " ".join(chosen)


def _extractive_response(text: str, start: float) -> LLMResponse:
    return LLMResponse(
        text=text,
        model=ExtractiveLLM.model,
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=(time.perf_counter() - start) * 1000,
        stop_reason="end_turn",
    )


def build_llm(settings: Any) -> LLM:
    """Construct the backend named in settings."""
    if settings.llm_backend == "extractive":
        return ExtractiveLLM()
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    key = key or None  # an empty ANTHROPIC_API_KEY= line means "not configured"
    return ClaudeLLM(
        model=settings.claude_model,
        effort=settings.claude_effort,
        max_tokens=settings.claude_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
        api_key=key,
    )
