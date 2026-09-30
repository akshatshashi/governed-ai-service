"""The Claude backend end to end through the real SDK, against a mock HTTP transport.

No network and no API key: this checks what goes over the wire (request shape, tool schemas,
append-only history with thinking blocks replayed) and that the full pipeline works on Claude's
response format, including a tool-use round trip.
"""

import json

import anthropic
import httpx2

from src.llm import ClaudeLLM

TOOL_TURN = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "lookup_threshold",
            "input": {"country": "New Zealand"},
        },
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 900, "output_tokens": 40},
}

ANSWER_TURN = {
    "id": "msg_2",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "sig-2"},
        {
            "type": "text",
            "text": "New Zealand requires prescribed transaction reports for large cash "
            "transactions of NZD 10,000 or more.",
        },
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1000, "output_tokens": 30},
}


def _mock_claude(requests: list[dict]) -> ClaudeLLM:
    turns = [TOOL_TURN, ANSWER_TURN]

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.startswith("/v1/models/"):
            return httpx2.Response(200, json={"id": "claude-opus-5-5", "type": "model"})
        requests.append(
            {"headers": dict(request.headers), "body": json.loads(request.content.decode())}
        )
        return httpx2.Response(200, json=turns.pop(0))

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return ClaudeLLM(client=client)


def test_pipeline_on_claude_wire_format(make_pipeline):
    requests: list[dict] = []
    pipeline = make_pipeline(_mock_claude(requests))

    response = pipeline.ask("What is the cash transaction reporting threshold in New Zealand?")

    assert response.outcome == "answered", response
    assert "NZD 10,000" in response.answer
    assert response.model == "claude-opus-5-5"

    first, second = requests[0], requests[1]
    assert "server-side-fallback-2026-07-01" in first["headers"]["anthropic-beta"]
    body = first["body"]
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "medium"}
    assert {t["name"] for t in body["tools"]} == {
        "search_corpus",
        "lookup_threshold",
        "flag_for_review",
    }
    assert "<context>" in body["messages"][0]["content"]

    # Turn 2 replays turn 1's assistant content exactly (thinking signature included),
    # then the tool result for the matching tool_use id.
    history = second["body"]["messages"]
    assert history[1]["role"] == "assistant"
    assert history[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
    tool_result = history[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert "NZD 10,000" in tool_result["content"]

    record = pipeline.audit.get(response.request_id)
    assert record.tool_calls[0]["name"] == "lookup_threshold"
    assert record.token_usage["prompt_tokens"] == 1900
    assert record.token_usage["llm_calls"] == 2


def test_reachability_probe_uses_models_api():
    assert _mock_claude([]).is_reachable()
