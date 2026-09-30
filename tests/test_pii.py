"""PII detection and redaction (Australian context)."""

from src.guardrails.pii import detect, luhn_valid, redact, summarise

CARD = "4111 1111 1111 1111"  # a standard Luhn-valid test card number
SAMPLE = (
    "Contact jane.citizen@example.com or 0412 345 678. "
    "TFN 123 456 782, ABN 51 824 753 556, card " + CARD + ". "
    "My name is Jane Citizen."
)


def test_detects_one_of_each_type():
    kinds = {m.kind for m in detect(SAMPLE)}
    assert kinds == {"email", "phone", "tfn", "abn", "card", "name_like"}


def test_redact_removes_every_card_digit_group():
    cleaned, matches = redact(SAMPLE)
    assert "[REDACTED:card]" in cleaned
    assert "4111" not in cleaned and "1111" not in cleaned
    assert "jane.citizen@example.com" not in cleaned
    assert "123 456 782" not in cleaned
    assert len(matches) == 6


def test_matches_carry_positions_not_values():
    for match in detect(SAMPLE):
        assert set(match.model_dump()) == {"kind", "start", "end"}


def test_luhn_rejects_non_card_digit_runs():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    cleaned, _ = redact("Reference 1234567890123 is not a card.")
    assert "[REDACTED:card]" not in cleaned


def test_international_mobile_and_landline_formats():
    for number in ("+61 412 345 678", "(02) 9876 5432", "1300 123 456"):
        cleaned, matches = redact(f"Call {number} today")
        assert [m.kind for m in matches] == ["phone"], number
        assert number not in cleaned


def test_amounts_and_dates_are_not_redacted():
    text = "Cash of AUD 10,000 on 2026-09-30 must be reported within 10 business days."
    assert redact(text) == (text, [])


def test_summarise_counts_by_kind_only():
    assert summarise(detect(SAMPLE))["tfn"] == 1
