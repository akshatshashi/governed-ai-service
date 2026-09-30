"""Deterministic degraded-mode responses.

When the model is unavailable, refuses, produces an ungrounded answer, or retrieval confidence is
too low, the service does not guess. It returns a fixed explanation plus the most relevant
retrieved passages *verbatim* with their sources. No text in a fallback response is generated —
so the service stays useful (the passages are often enough to answer the question) and stays safe.
"""

from __future__ import annotations

from enum import StrEnum

from src.retriever import RetrievedChunk


class FallbackReason(StrEnum):
    LLM_UNAVAILABLE = "llm_unavailable"
    LOW_CONFIDENCE = "low_confidence"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    GROUNDING_FAILED = "grounding_failed"
    MODEL_REFUSAL = "model_refusal"
    INTERNAL_ERROR = "internal_error"


_EXPLANATIONS: dict[FallbackReason, str] = {
    FallbackReason.LLM_UNAVAILABLE: (
        "The language model is currently unavailable, so no answer was generated. "
        "The most relevant source passages are shown below."
    ),
    FallbackReason.LOW_CONFIDENCE: (
        "I don't have enough information in the provided sources to answer that confidently. "
        "The closest matching passages are shown below for your review."
    ),
    FallbackReason.GUARDRAIL_BLOCKED: (
        "This request was blocked by an input control and was not processed. "
        "The service answers AML/CTF compliance questions only."
    ),
    FallbackReason.GROUNDING_FAILED: (
        "A draft answer was produced but could not be verified against the source documents, "
        "so it was withheld. The most relevant source passages are shown below."
    ),
    FallbackReason.MODEL_REFUSAL: (
        "The model declined to answer this request. "
        "The most relevant source passages are shown below."
    ),
    FallbackReason.INTERNAL_ERROR: (
        "An internal error occurred, so no answer was generated. "
        "The most relevant source passages are shown below."
    ),
}


_BLOCK_DETAIL = {
    "out_of_scope": "The service answers AML/CTF compliance questions only.",
    "prompt_injection": (
        "It appears to contain instructions aimed at the assistant rather than a compliance "
        "question."
    ),
    "input_too_long": "The question is longer than the service accepts.",
}


def build_fallback(
    reason: FallbackReason,
    chunks: list[RetrievedChunk],
    max_passages: int = 3,
    blocked_by: str | None = None,
) -> str:
    """Return a fixed explanation plus up to ``max_passages`` passages with source attribution.

    ``blocked_by`` (the guardrail's reason code) selects a fixed, non-echoing explanation for a
    blocked request.
    """
    explanation = _EXPLANATIONS[reason]
    if reason is FallbackReason.GUARDRAIL_BLOCKED and blocked_by:
        detail = _BLOCK_DETAIL.get(blocked_by.split(":")[0].split(">")[0])
        if detail:
            explanation = (
                "This request was blocked by an input control and was not processed. " + detail
            )
    lines = [explanation]
    if reason is not FallbackReason.GUARDRAIL_BLOCKED:
        passages = chunks[:max_passages]
        if passages:
            lines.append("")
            for i, chunk in enumerate(passages, start=1):
                lines.append(
                    f"[{i}] (source: {chunk.doc_id}, chunk {chunk.chunk_index}, "
                    f"similarity {chunk.score:.2f})"
                )
                lines.append(chunk.text)
                lines.append("")
        else:
            lines.append("No relevant passages were found.")
    lines.append("This is decision support only; a qualified person must review before acting.")
    return "\n".join(lines).strip()
