"""Agent loop behaviour with a scripted model."""

from src.agent.loop import PROMPT_VERSION, Agent, build_prompt
from src.agent.tools import build_default_registry, load_thresholds
from src.llm import ToolCall
from src.retriever import RetrievedChunk
from tests.conftest import REFERENCE_DIR, ScriptedLLM, make_response


def _agent(retriever, llm, **kwargs) -> Agent:
    registry = build_default_registry(retriever, load_thresholds(str(REFERENCE_DIR)))
    return Agent(llm, retriever, registry, **kwargs)


def test_prompt_numbers_passages_and_puts_question_on_one_line():
    chunk = RetrievedChunk(text="Body", doc_id="a.md", source_path="a.md", chunk_index=0, score=0.5)
    prompt = build_prompt("line one\nline two", [chunk])
    assert "[1] (source: a.md, chunk 0)\nBody" in prompt
    assert prompt.endswith("Question: line one line two")


def test_answer_with_citations_and_confidence_from_cited_chunks(retriever):
    llm = ScriptedLLM([make_response("A TTR is required for cash of AUD 10,000 or more [1].")])
    result = _agent(retriever, llm).run("What is the TTR threshold?")
    assert result.cited == [1]
    assert result.confidence == result.retrieved[0].score
    assert result.prompt_version == PROMPT_VERSION
    assert not result.approval_required


def test_tool_loop_executes_allowed_tool_and_returns_result(retriever):
    llm = ScriptedLLM(
        [
            make_response(
                tool_calls=[ToolCall(id="t1", name="lookup_threshold", args={"country": "NZ"})]
            ),
            make_response("New Zealand requires PTRs for cash of NZD 10,000 or more."),
        ]
    )
    result = _agent(retriever, llm).run("NZ cash threshold?")
    assert result.tool_calls[0]["status"] == "executed"
    assert "NZD 10,000" in llm.tool_results[0][0].content
    assert result.tool_evidence and result.confidence > 0


def test_unknown_tool_is_rejected_and_reported_to_model(retriever):
    llm = ScriptedLLM(
        [
            make_response(tool_calls=[ToolCall(id="t1", name="wire_money", args={"amount": 1})]),
            make_response(
                "I don't have enough information in the provided sources to answer that."
            ),
        ]
    )
    result = _agent(retriever, llm).run("Send money")
    assert result.tool_calls[0]["status"] == "rejected"
    assert llm.tool_results[0][0].is_error
    assert result.abstained


def test_high_impact_request_is_collected_not_executed(retriever):
    call = ToolCall(
        id="t1", name="flag_for_review", args={"reference": "TXN-48213", "reason": "structuring"}
    )
    llm = ScriptedLLM(
        [make_response(tool_calls=[call]), make_response("Queued; pending approval.")]
    )
    result = _agent(retriever, llm).run("Flag TXN-48213")
    assert result.approval_required
    assert result.pending_actions == [{"tool": "flag_for_review", "args": call.args}]
    assert result.tool_calls[0]["status"] == "pending_approval"


def test_step_budget_stops_runaway_tool_loops(retriever):
    call = ToolCall(id="t", name="search_corpus", args={"query": "cash reporting"})
    llm = ScriptedLLM([make_response(tool_calls=[call])] * 4)
    result = _agent(retriever, llm, max_steps=2).run("cash")
    statuses = [c["status"] for c in result.tool_calls]
    assert statuses.count("executed") == 2
    assert statuses[-1] == "not_run_step_limit"


def test_invalid_citations_are_detected(retriever):
    llm = ScriptedLLM([make_response("It is AUD 10,000 [42].")])
    result = _agent(retriever, llm).run("TTR threshold?")
    assert result.invalid_citations == [42]
