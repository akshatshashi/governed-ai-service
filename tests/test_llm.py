"""The model interface: protocol conformance, Claude response/error mapping, extractive baseline."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from src.llm import (
    ABSTAIN_TEXT,
    LLM,
    ClaudeLLM,
    ExtractiveLLM,
    LLMRefusal,
    LLMResponse,
    LLMUnavailable,
    ToolDefinition,
    ToolResult,
    extractive_answer,
)
from tests.conftest import ScriptedLLM, make_response


def test_fake_llm_implementing_protocol_returns_llm_response():
    fake: LLM = ScriptedLLM([make_response("hello")])
    response = fake.generate("hi", system="be brief")
    assert isinstance(response, LLMResponse)
    assert response.text == "hello"


# --- Claude backend, with the SDK client replaced by a stub (no network, no key) -------------


class _StubMessages:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls: list[dict] = []

    def create(self, **params):
        self.calls.append(params)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _stub_client(outcome):
    messages = _StubMessages(outcome)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def _message(content, stop_reason="end_turn", model="claude-opus-5-5", stop_details=None):
    usage = SimpleNamespace(
        input_tokens=120, output_tokens=30, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage=usage,
        stop_details=stop_details,
    )


def test_claude_maps_text_tool_calls_and_usage():
    content = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="Checking the threshold."),
        SimpleNamespace(
            type="tool_use", id="tu_1", name="lookup_threshold", input={"country": "NZ"}
        ),
    ]
    client, messages = _stub_client(_message(content, stop_reason="tool_use"))
    llm = ClaudeLLM(client=client)
    tool = ToolDefinition(name="lookup_threshold", description="d", input_schema={"type": "object"})
    session = llm.start_session("system prompt", [tool])
    response = session.send("question")

    assert response.text == "Checking the threshold."
    assert response.tool_calls[0].name == "lookup_threshold"
    assert response.tool_calls[0].args == {"country": "NZ"}
    assert response.prompt_tokens == 120 and response.completion_tokens == 30
    params = messages.calls[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["fallbacks"] == "default"
    assert params["output_config"] == {"effort": "medium"}
    assert params["tools"][0]["name"] == "lookup_threshold"

    # History is append-only: the assistant turn is replayed exactly, then the tool result.
    messages.outcome = _message([SimpleNamespace(type="text", text="NZD 10,000 [1]")])
    session.send_tool_results([ToolResult(call_id="tu_1", content="NZ: NZD 10,000")])
    history = messages.calls[1]["messages"]
    assert history[1] == {"role": "assistant", "content": content}
    assert history[2]["content"][0]["tool_use_id"] == "tu_1"


def test_claude_refusal_raises_before_reading_content():
    details = SimpleNamespace(category="cyber", explanation=None)
    client, _ = _stub_client(_message([], stop_reason="refusal", stop_details=details))
    with pytest.raises(LLMRefusal) as err:
        ClaudeLLM(client=client).generate("q")
    assert err.value.category == "cyber"


_REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (anthropic.APIConnectionError(request=_REQ), "connection_error"),
        (anthropic.APITimeoutError(request=_REQ), "timeout"),
        (
            anthropic.RateLimitError(
                "slow down", response=httpx2.Response(429, request=_REQ), body=None
            ),
            "rate_limited",
        ),
        (
            anthropic.InternalServerError(
                "boom", response=httpx2.Response(500, request=_REQ), body=None
            ),
            "api_error_500",
        ),
        (TypeError("Could not resolve authentication method."), "no_credentials"),
    ],
)
def test_claude_errors_become_llm_unavailable(error, reason):
    client, _ = _stub_client(error)
    with pytest.raises(LLMUnavailable) as err:
        ClaudeLLM(client=client).generate("q")
    assert err.value.reason == reason


def test_unrelated_type_error_is_not_swallowed():
    client, _ = _stub_client(TypeError("unexpected keyword argument"))
    with pytest.raises(TypeError):
        ClaudeLLM(client=client).generate("q")


# --- Extractive baseline ----------------------------------------------------------------------


def test_extractive_answer_selects_and_cites_matching_sentence():
    passages = [
        (1, "Unrelated text about opening hours. Branches close at 5pm."),
        (
            2,
            "A TTR must be submitted for cash of AUD 10,000 or more. It is due in 10 business days.",
        ),
    ]
    answer = extractive_answer("What is the TTR cash threshold amount?", passages)
    assert "AUD 10,000" in answer and "[2]" in answer


def test_extractive_answer_abstains_without_overlap():
    passages = [(1, "Branches close at 5pm on weekdays.")]
    assert (
        extractive_answer("Which currency pairs have the tightest spreads?", passages)
        == ABSTAIN_TEXT
    )


def test_extractive_session_requests_approval_tool_for_flag_requests():
    tools = [
        ToolDefinition(name="flag_for_review", description="", input_schema={}),
        ToolDefinition(name="lookup_threshold", description="", input_schema={}),
    ]
    session = ExtractiveLLM().start_session("sys", tools)
    response = session.send(
        "<context>\n</context>\n\nQuestion: Please flag TXN-12345 for review now"
    )
    assert response.tool_calls[0].name == "flag_for_review"
    assert response.tool_calls[0].args["reference"] == "TXN-12345"
