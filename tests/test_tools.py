"""Tool registry controls: allow-list, validation, limits, impact gate."""

import pytest

from src.agent.tools import (
    APPROVAL_MARKER,
    ToolContext,
    build_default_registry,
    load_thresholds,
)
from tests.conftest import REFERENCE_DIR


class _Case:
    case_id = "CASE-TEST"


@pytest.fixture
def opened():
    return []


@pytest.fixture
def registry(retriever, opened):
    def open_case(**kwargs):
        opened.append(kwargs)
        return _Case()

    return build_default_registry(retriever, load_thresholds(str(REFERENCE_DIR)), open_case)


def test_unknown_tool_rejected(registry):
    ok, result = registry.call("delete_customer", {"id": "1"}, {})
    assert not ok and "allow-list" in result


def test_bad_args_rejected_without_echoing_values(registry):
    ok, result = registry.call("lookup_threshold", {"country": 42, "extra": "tfn 123456782"}, {})
    assert not ok and result.startswith("rejected: invalid arguments")
    assert "123456782" not in result


def test_reference_format_is_validated(registry):
    ok, result = registry.call(
        "flag_for_review", {"reference": "drop table;", "reason": "looks odd"}, {}
    )
    assert not ok and "reference" in result


def test_call_limit_enforced(registry):
    counts: dict[str, int] = {}
    assert registry.call("lookup_threshold", {"country": "NZ"}, counts)[0]
    assert registry.call("lookup_threshold", {"country": "US"}, counts)[0]
    ok, result = registry.call("lookup_threshold", {"country": "CA"}, counts)
    assert not ok and "call limit" in result


def test_high_impact_tool_does_not_execute(registry, opened):
    ok, result = registry.call(
        "flag_for_review", {"reference": "TXN-48213", "reason": "structured cash deposits"}, {}
    )
    assert ok and result.startswith(APPROVAL_MARKER)
    assert opened == []  # nothing happened


def test_high_impact_tool_runs_only_with_named_approver(registry, opened):
    args = {"reference": "TXN-48213", "reason": "structured cash deposits"}
    with pytest.raises(PermissionError):
        registry.execute_approved("flag_for_review", args, ToolContext(request_id="r1"))
    result = registry.execute_approved(
        "flag_for_review", args, ToolContext(request_id="r1", approved_by="supervisor1")
    )
    assert "CASE-TEST" in result
    assert opened[0]["approved_by"] == "supervisor1"


def test_lookup_threshold_accepts_names_and_codes(registry):
    assert "NZD 10,000" in registry.call("lookup_threshold", {"country": "New Zealand"}, {})[1]
    assert "FinCEN" in registry.call("lookup_threshold", {"country": "us"}, {})[1]
    assert (
        "No threshold reference data"
        in registry.call("lookup_threshold", {"country": "Mars"}, {})[1]
    )


def test_search_corpus_extends_numbered_evidence(registry):
    ctx = ToolContext()
    ok, result = registry.call("search_corpus", {"query": "structuring offence"}, {}, ctx)
    assert ok and result.startswith("[1] (source:")
    assert len(ctx.evidence) >= 1


def test_definitions_advertise_schemas(registry):
    names = {d.name for d in registry.definitions()}
    assert names == {"search_corpus", "lookup_threshold", "flag_for_review"}
