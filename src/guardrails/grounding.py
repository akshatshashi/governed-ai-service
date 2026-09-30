"""Output guardrail: is the answer grounded in the evidence it was given?

This is a **heuristic proxy for faithfulness, not proof of it.** It measures lexical overlap: the
share of the answer's content words (stemmed, stopwords removed, numbers kept) that also occur
somewhere in the evidence (retrieved chunks plus tool results).

Known failure modes, stated plainly:
- *False fail on paraphrase*: a faithful answer that rephrases the source with different words
  scores low.
- *False pass on copied-but-irrelevant text*: an answer that stitches together words from the
  chunks can score high while asserting something the chunks do not support, or answering a
  different question.
- *Blind to negation and numbers in context*: "must" vs "must not", or a number attached to the
  wrong obligation, overlap equally well.

It is cheap, deterministic and explainable, which makes it a reasonable *gate*. A stronger
production design would add an NLI/entailment model or an LLM-as-judge check per claim.
"""

from __future__ import annotations

from src.guardrails.injection import GuardResult
from src.text import content_terms

DEFAULT_THRESHOLD = 0.25


def grounding_score(answer: str, chunks: list[str]) -> float:
    """Proportion of the answer's content words found in the evidence, in [0, 1]."""
    answer_terms = content_terms(answer)
    if not answer_terms:
        return 0.0
    evidence_terms: set[str] = set()
    for chunk in chunks:
        evidence_terms |= content_terms(chunk)
    return len(answer_terms & evidence_terms) / len(answer_terms)


def check_grounding(
    answer: str, chunks: list[str], threshold: float = DEFAULT_THRESHOLD
) -> GuardResult:
    """Fail when fewer than ``threshold`` of the answer's content words appear in the evidence."""
    score = grounding_score(answer, chunks)
    if score < threshold:
        return GuardResult(passed=False, reason=f"grounding_score={score:.2f}<{threshold}")
    return GuardResult(passed=True, reason=f"grounding_score={score:.2f}")
