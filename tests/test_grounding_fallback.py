"""Grounding check and deterministic fallback."""

from src.fallback import FallbackReason, build_fallback
from src.guardrails.grounding import check_grounding, grounding_score
from src.retriever import RetrievedChunk

CHUNKS = [
    "A threshold transaction report must be submitted for physical currency of AUD 10,000 or more.",
    "The report is due within 10 business days after the transaction.",
]


def test_grounded_answer_passes():
    answer = "A threshold transaction report is required for cash of AUD 10,000 or more [1]."
    assert check_grounding(answer, CHUNKS).passed


def test_clearly_ungrounded_answer_fails():
    answer = "Penguins migrate across Antarctica every winter to find better fishing grounds."
    result = check_grounding(answer, CHUNKS)
    assert not result.passed
    assert grounding_score(answer, CHUNKS) < 0.25


def test_numbers_count_as_evidence():
    assert grounding_score("10,000", CHUNKS) == 1.0
    assert grounding_score("50,000", CHUNKS) == 0.0


def _chunk(doc_id: str, idx: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(text=text, doc_id=doc_id, source_path=doc_id, chunk_index=idx, score=0.8)


def test_fallback_includes_source_attribution_for_each_passage():
    chunks = [_chunk("a.md", 0, "Passage A text."), _chunk("b.md", 3, "Passage B text.")]
    text = build_fallback(FallbackReason.LLM_UNAVAILABLE, chunks)
    assert "source: a.md, chunk 0" in text and "Passage A text." in text
    assert "source: b.md, chunk 3" in text and "Passage B text." in text


def test_fallback_is_deterministic():
    chunks = [_chunk("a.md", 0, "Passage A text.")]
    assert build_fallback(FallbackReason.LOW_CONFIDENCE, chunks) == build_fallback(
        FallbackReason.LOW_CONFIDENCE, chunks
    )


def test_blocked_fallback_shows_no_passages_and_does_not_echo_input():
    text = build_fallback(
        FallbackReason.GUARDRAIL_BLOCKED, [_chunk("a.md", 0, "secret")], blocked_by="out_of_scope"
    )
    assert "secret" not in text
    assert "compliance questions only" in text
