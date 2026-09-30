"""Regex-based PII detection and redaction for Australian financial-services text.

Runs on **both** the user's input (before anything reaches retrieval, the model or the audit
store) and the model's output (before it reaches the caller).

Privacy rule: a ``PIIMatch`` carries only the *kind* and *position* of a match — never the value.
Nothing in this module logs or returns the raw matched text.

Design bias: when in doubt, redact. Over-redacting a nine-digit reference number costs a little
answer quality; under-redacting a Tax File Number is a privacy incident. For that reason TFN and
ABN detection is pattern-based and does not require a valid checksum. Card numbers *do* require a
Luhn check, because 13-19 digit runs are common in non-sensitive identifiers.
"""

from __future__ import annotations

import re

from pydantic import BaseModel


class PIIMatch(BaseModel):
    """One detected PII span. Deliberately has no field for the matched value."""

    kind: str  # "email" | "phone" | "tfn" | "abn" | "card" | "name_like"
    start: int
    end: int


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Australian numbers: mobiles (04xx xxx xxx / +61 4xx xxx xxx), landlines with area code
# (0[2378] xxxx xxxx / +61 [2378] xxxx xxxx) and 13/1300/1800 service numbers.
_PHONE = re.compile(
    r"(?<![\w+])(?:"
    r"(?:\+61\s?|0)4\d{2}[\s-]?\d{3}[\s-]?\d{3}"
    r"|(?:\+61\s?|0)[2378][\s-]?\d{4}[\s-]?\d{4}"
    r"|\(0[2378]\)\s?\d{4}[\s-]?\d{4}"
    r"|1[38]00[\s-]?\d{3}[\s-]?\d{3}"
    r")(?!\d)"
)

# Tax File Number: 9 digits, optionally grouped 3-3-3 with spaces or hyphens.
_TFN = re.compile(r"(?<![\d-])\d{3}[\s-]?\d{3}[\s-]?\d{3}(?![\d-])")

# Australian Business Number: 11 digits, commonly grouped 2-3-3-3.
_ABN = re.compile(r"(?<![\d-])\d{2}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}(?![\d-])")

# Candidate card numbers: 13-19 digits with optional single space/hyphen separators.
_CARD_CANDIDATE = re.compile(r"(?<![\d-])\d(?:[\s-]?\d){12,18}(?![\d-])")

# Name-like phrases: an honorific or an explicit self-identification followed by capitalised words.
_NAME_LIKE = re.compile(
    r"\b(?:(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
    r"|(?i:my name is|name:|customer name is|customer named)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
)


def luhn_valid(digits: str) -> bool:
    """Luhn (mod 10) checksum used by payment card numbers."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def detect(text: str) -> list[PIIMatch]:
    """Return non-overlapping PII spans, sorted by position.

    When two detectors overlap (e.g. a phone number whose tail also looks like a TFN) the longer
    span wins, so the whole sensitive value is redacted as one token.
    """
    candidates: list[PIIMatch] = []
    for m in _EMAIL.finditer(text):
        candidates.append(PIIMatch(kind="email", start=m.start(), end=m.end()))
    for m in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            candidates.append(PIIMatch(kind="card", start=m.start(), end=m.end()))
    for m in _PHONE.finditer(text):
        candidates.append(PIIMatch(kind="phone", start=m.start(), end=m.end()))
    for m in _ABN.finditer(text):
        candidates.append(PIIMatch(kind="abn", start=m.start(), end=m.end()))
    for m in _TFN.finditer(text):
        candidates.append(PIIMatch(kind="tfn", start=m.start(), end=m.end()))
    for m in _NAME_LIKE.finditer(text):
        candidates.append(PIIMatch(kind="name_like", start=m.start(), end=m.end()))

    # Longest span first, then keep only spans that do not overlap an already-kept span.
    candidates.sort(key=lambda c: (-(c.end - c.start), c.start))
    kept: list[PIIMatch] = []
    for cand in candidates:
        if all(cand.end <= k.start or cand.start >= k.end for k in kept):
            kept.append(cand)
    return sorted(kept, key=lambda c: c.start)


def redact(text: str) -> tuple[str, list[PIIMatch]]:
    """Replace each PII span with ``[REDACTED:<kind>]``.

    Returns the cleaned text and the matches (positions refer to the *original* text).
    """
    matches = detect(text)
    out: list[str] = []
    cursor = 0
    for m in matches:
        out.append(text[cursor : m.start])
        out.append(f"[REDACTED:{m.kind}]")
        cursor = m.end
    out.append(text[cursor:])
    return "".join(out), matches


def summarise(matches: list[PIIMatch]) -> dict[str, int]:
    """Count matches by kind — the only form in which PII findings are logged or audited."""
    counts: dict[str, int] = {}
    for m in matches:
        counts[m.kind] = counts.get(m.kind, 0) + 1
    return counts
