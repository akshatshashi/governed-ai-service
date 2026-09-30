"""Input guardrails: prompt-injection detection and scope checking.

Both checks are deliberately simple, deterministic and explainable, and both are **heuristics,
not guarantees**. They are the first layer of a defence-in-depth design: the system prompt also
tells the model to treat retrieved text as data, tools are allow-listed and validated, high-impact
actions need human approval, and outputs are grounding-checked.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata

from pydantic import BaseModel

from src.text import stem


class GuardResult(BaseModel):
    """Outcome of one guardrail. ``reason`` names a pattern class, never the user's text."""

    passed: bool
    reason: str | None = None


# Patterns matched against a *compacted* form of the input: lower-cased, leetspeak folded, and
# every non-alphanumeric character removed. This defeats spacing tricks ("i g n o r e"),
# punctuation tricks ("ig.nore") and simple character substitution ("1gn0re").
#
# Because compaction erases word boundaries, every compact pattern must end in a high-signal
# object ("...instructions", "...prompt") so benign questions such as "can I skip the previous
# step?" or "show me the rules for IFTIs" do not trip it.
_COMPACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "instruction_override": re.compile(
        r"(ignore|disregard|forget|override)(all|any|the|your|my|of|these|those)*"
        r"(previous|prior|above|earlier|preceding|former|original|system)?"
        r"(instructions?|prompts?|directions|guidelines)"
        r"|(disregard|ignore|forget)(all|everything)?(the)?above"
        r"|ignoreallprevious|newinstructions(are|follow)"
    ),
    "role_hijack": re.compile(
        r"youarenow(a|an|the|in|my)?(unrestricted|unfiltered|jailbroken|dan|evil|developer|"
        r"freed|uncensored)"
        r"|developermode|jailbreak|doanythingnow|actasan?(unrestricted|unfiltered|jailbroken)"
        r"|pretend(to|that)(be|you)(are)?(an?)?(unrestricted|unfiltered|evil)"
        r"|fromnowonyouare"
    ),
    "prompt_exfiltration": re.compile(
        r"systemprompt"
        r"|(reveal|print|show|repeat|output|display|tellme|leak|dump)(me)?your"
        r"(full|entire|original|hidden|initial|secret)?(system)?(prompt|instructions|configuration)"
        r"|whatareyour(instructions|rules)|initialinstructions|hiddeninstructions"
        r"|repeatthe(text|words)above"
    ),
}

# Patterns matched against whitespace-normalised lower-case text (phrases that are only
# suspicious as whole words, e.g. "you are now" but not "if you are now required to report").
_WORD_PATTERNS: dict[str, re.Pattern[str]] = {
    "role_hijack": re.compile(
        r"\byou are now\b(?!\s+(?:required|obliged|obligated|subject|registered|enrolled|"
        r"able|a reporting entity|responsible|expected|in breach))"
    ),
    "delimiter_injection": re.compile(
        r"</?\s*(system|assistant|instructions?)\s*>|<\|im_(start|end)\|>|\[/?inst\]"
        r"|^\s*#{2,}\s*(system|instruction)"
    ),
}

_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_BASE64_TOKEN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", text).lower().strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _normalise(text).translate(_LEET))


def _decoded_payloads(text: str) -> list[str]:
    """Decode base64-looking tokens so instructions cannot be smuggled in encoded form."""
    decoded: list[str] = []
    for token in _BASE64_TOKEN.findall(text):
        try:
            raw = base64.b64decode(token + "=" * (-len(token) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        candidate = raw.decode("utf-8", errors="ignore")
        if sum(ch.isprintable() for ch in candidate) >= 0.9 * max(1, len(candidate)):
            decoded.append(candidate)
    return decoded


def _scan(text: str) -> str | None:
    compact = _compact(text)
    for name, pattern in _COMPACT_PATTERNS.items():
        if pattern.search(compact):
            return name
    words = _normalise(text)
    for name, pattern in _WORD_PATTERNS.items():
        if pattern.search(words):
            return name
    return None


def check_injection(user_input: str) -> GuardResult:
    """Flag common instruction-override and prompt-extraction attempts.

    Pattern classes: ``instruction_override``, ``role_hijack``, ``prompt_exfiltration``,
    ``delimiter_injection``, and ``encoded_payload`` (any of the above hidden in base64).
    The reason names the class only — the user's input is never echoed back.
    """
    hit = _scan(user_input)
    if hit:
        return GuardResult(passed=False, reason=f"prompt_injection:{hit}")
    for payload in _decoded_payloads(user_input):
        if _scan(payload):
            return GuardResult(passed=False, reason="prompt_injection:encoded_payload")
    return GuardResult(passed=True, reason=None)


def check_scope(user_input: str, allowed_topics: list[str]) -> GuardResult:
    """Keyword heuristic: pass if the question mentions at least one allowed topic.

    Topics and input are lower-cased, tokenised and crudely stemmed, and a multi-word topic
    matches when its words appear consecutively. This is a coarse relevance gate to stop the
    service being used as a general chatbot — **a heuristic, not a guarantee**. A question can
    mention "cash" and still be off-topic; the retrieval-confidence and grounding checks
    downstream catch most of those.
    """
    words = [stem(w) for w in re.findall(r"[a-z0-9]+", _normalise(user_input))]
    joined = f" {' '.join(words)} "
    for topic in allowed_topics:
        topic_words = [stem(w) for w in re.findall(r"[a-z0-9]+", topic.lower())]
        if topic_words and f" {' '.join(topic_words)} " in joined:
            return GuardResult(passed=True, reason=None)
    return GuardResult(passed=False, reason="out_of_scope")
