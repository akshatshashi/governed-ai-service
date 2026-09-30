"""Small, dependency-free text utilities shared by the guardrails and the extractive baseline.

Keeping tokenisation in one place means the grounding check, the scope check and the offline
extractive model all agree on what a "content word" is.
"""

import re

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves also may might must
    shall within without upon per via etc e g i e s eg ie one two any every either neither
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_CITATION_RE = re.compile(r"\[\d+\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n|\n(?=\s*(?:#{1,6}\s|[-*]\s|\d+\.\s))")


def normalise_number(token: str) -> str:
    """Strip thousands separators so "10,000" and "10000" compare equal."""
    return token.replace(",", "") if any(ch.isdigit() for ch in token) else token


def stem(word: str) -> str:
    """A deliberately crude suffix stripper — enough to match "reports" with "report"."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            # "submitted" -> "submitt" -> "submit"; keep "ll"/"ss" ("fall", "pass").
            if suffix in ("ing", "ed") and word[-1] == word[-2] and word[-1] not in "aeiouls":
                word = word[:-1]
            return word
    return word


def tokens(text: str) -> list[str]:
    """Lower-case word tokens with citation markers removed and numbers normalised."""
    text = _CITATION_RE.sub(" ", text.lower())
    return [normalise_number(t) for t in _WORD_RE.findall(text)]


def content_terms(text: str) -> set[str]:
    """Stemmed tokens that carry meaning: stopwords and 1-2 letter tokens removed.

    Numbers are always kept, because in this domain the numbers *are* the facts
    (thresholds, deadlines, retention periods).
    """
    out: set[str] = set()
    for tok in tokens(text):
        if tok.replace(".", "").isdigit():
            out.add(tok)
        elif len(tok) > 2 and tok not in STOPWORDS:
            out.add(stem(tok))
    return out


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Blank lines, Markdown headings and list items are hard boundaries; within a block, split on
    terminal punctuation followed by a capital or digit. Single line breaks inside a block (as in
    text extracted from PDFs) are treated as spaces.
    """
    sentences: list[str] = []
    for block in _BLOCK_SPLIT_RE.split(text):
        flat = re.sub(r"\s+", " ", block).strip()
        if flat:
            sentences.extend(p.strip() for p in _SENTENCE_SPLIT_RE.split(flat) if p.strip())
    return sentences
