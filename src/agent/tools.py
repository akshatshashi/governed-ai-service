"""A restricted tool system. The control layer is the point of this file.

The model can *ask* for any tool it likes; this registry decides what actually happens.
``ToolRegistry.call`` applies the controls in a fixed order:

1. **Allow-list** — unknown tool names are rejected.
2. **Argument validation** — arguments must satisfy the tool's pydantic schema (types, lengths,
   formats; unexpected fields rejected).
3. **Per-request call limits** — each tool has a ``max_calls_per_request`` budget.
4. **Impact gate** — ``high_impact`` tools never execute here; they return an approval marker and
   the pipeline submits them to the human approval queue.
5. Only then is a low-impact tool executed.

Validation errors are summarised by field and error type only, so rejected arguments (which may
contain personal information) are never echoed into logs or the audit trail.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.llm import ToolDefinition
from src.retriever import RetrievedChunk, Retriever

APPROVAL_MARKER = "PENDING_APPROVAL"


class ToolSpec(BaseModel):
    name: str
    description: str
    args_schema: type[BaseModel]
    high_impact: bool  # True => requires human approval before it can run
    max_calls_per_request: int


@dataclass
class ToolContext:
    """Per-request state tools may read or extend."""

    request_id: str = ""
    evidence: list[RetrievedChunk] = field(default_factory=list)
    top_k: int = 3
    approved_by: str | None = None  # set only when executing an approved high-impact action


ToolFn = Callable[[BaseModel, ToolContext], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolFn]] = {}

    def register(self, spec: ToolSpec, fn: ToolFn) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._tools[spec.name] = (spec, fn)

    def get(self, name: str) -> ToolSpec | None:
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def definitions(self) -> list[ToolDefinition]:
        """Tool descriptions advertised to the model (advisory; enforcement happens in call())."""
        return [
            ToolDefinition(
                name=spec.name,
                description=spec.description,
                input_schema=spec.args_schema.model_json_schema(),
            )
            for spec, _ in self._tools.values()
        ]

    def call(
        self,
        name: str,
        raw_args: dict,
        call_counts: dict[str, int],
        ctx: ToolContext | None = None,
    ) -> tuple[bool, str]:
        """Apply the controls in order and, if all pass, run the tool.

        Returns ``(success, result_string)``. For a high-impact tool, success is True and the
        result starts with ``APPROVAL_MARKER`` — the tool has *not* run.
        """
        entry = self._tools.get(name)
        if entry is None:
            return False, "rejected: unknown tool (not on the allow-list)"
        spec, fn = entry

        try:
            args = spec.args_schema.model_validate(raw_args)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'args'}: {err['type']}"
                for err in exc.errors()
            )
            return False, f"rejected: invalid arguments ({problems})"

        used = call_counts.get(name, 0)
        if used >= spec.max_calls_per_request:
            return False, (
                f"rejected: call limit reached ({spec.max_calls_per_request} per request)"
            )
        call_counts[name] = used + 1

        if spec.high_impact:
            return True, (
                f"{APPROVAL_MARKER}: {name} has been queued for human approval and has NOT been "
                "executed. A second person must approve it before any action is taken."
            )

        try:
            return True, fn(args, ctx or ToolContext())
        except Exception as exc:  # a failing tool must not crash the request
            return False, f"error: tool failed ({type(exc).__name__})"

    def execute_approved(self, name: str, raw_args: dict, ctx: ToolContext) -> str:
        """Run a high-impact tool after a human has approved it.

        Only the approval flow calls this, and only with ``ctx.approved_by`` set. Arguments are
        re-validated: what runs is exactly what the approver saw.
        """
        entry = self._tools.get(name)
        if entry is None:
            raise KeyError(name)
        spec, fn = entry
        if not spec.high_impact:
            raise ValueError(f"{name} is not an approval-gated tool")
        if not ctx.approved_by:
            raise PermissionError("execute_approved requires a named approver")
        return fn(spec.args_schema.model_validate(raw_args), ctx)


# ---------------------------------------------------------------------------------------------
# The three tools this service exposes
# ---------------------------------------------------------------------------------------------


class SearchCorpusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=300, description="What to search the corpus for.")


class LookupThresholdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    country: str = Field(
        min_length=2,
        max_length=60,
        description="Country name or ISO 3166 alpha-2 code, e.g. 'NZ' or 'New Zealand'.",
    )


class FlagForReviewArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference: str = Field(
        pattern=r"^[A-Z]{2,6}-\d{3,12}$",
        description="The transaction or customer reference exactly as given, e.g. 'TXN-48213'.",
    )
    reason: str = Field(min_length=5, max_length=500, description="Why it should be reviewed.")


def format_passage(number: int, chunk: RetrievedChunk) -> str:
    """The one format passages are shown to the model in (the extractive baseline parses it)."""
    return f"[{number}] (source: {chunk.doc_id}, chunk {chunk.chunk_index})\n{chunk.text}"


def load_thresholds(reference_dir: str) -> dict[str, dict]:
    path = Path(reference_dir) / "thresholds.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {code.upper(): entry for code, entry in data["countries"].items()}


def build_default_registry(
    retriever: Retriever,
    thresholds: dict[str, dict],
    open_case: Callable[..., object] | None = None,
) -> ToolRegistry:
    """Register search_corpus, lookup_threshold and flag_for_review with their controls."""
    registry = ToolRegistry()

    def search_corpus(args: SearchCorpusArgs, ctx: ToolContext) -> str:
        lines: list[str] = []
        for chunk in retriever.search(args.query, ctx.top_k):
            key = (chunk.doc_id, chunk.chunk_index)
            known = [(c.doc_id, c.chunk_index) for c in ctx.evidence]
            if key in known:
                number = known.index(key) + 1
            else:
                ctx.evidence.append(chunk)
                number = len(ctx.evidence)
            lines.append(format_passage(number, chunk))
        return "\n\n".join(lines) if lines else "No passages found."

    aliases = {entry["name"].lower(): code for code, entry in thresholds.items()}
    aliases.update(
        {alias.lower(): code for code, e in thresholds.items() for alias in e.get("aliases", [])}
    )

    def lookup_threshold(args: LookupThresholdArgs, ctx: ToolContext) -> str:
        wanted = args.country.strip()
        code = wanted.upper() if wanted.upper() in thresholds else aliases.get(wanted.lower())
        if code is None:
            return f"No threshold reference data is held for '{wanted[:60]}'."
        entry = thresholds[code]
        return (
            f"{entry['name']} ({code}): {entry['summary']} "
            f"Regulator: {entry['regulator']}. (Reference data: data/reference/thresholds.yaml)"
        )

    def flag_for_review(args: FlagForReviewArgs, ctx: ToolContext) -> str:
        if open_case is None:
            raise RuntimeError("no case log configured")
        case = open_case(
            reference=args.reference,
            reason=args.reason,
            approved_by=ctx.approved_by,
            source_request_id=ctx.request_id,
        )
        return f"Opened review case {case.case_id} for {args.reference}."

    registry.register(
        ToolSpec(
            name="search_corpus",
            description=(
                "Search the compliance document corpus for more passages when the provided "
                "context is not enough. Returns numbered passages you can cite."
            ),
            args_schema=SearchCorpusArgs,
            high_impact=False,
            max_calls_per_request=3,
        ),
        search_corpus,
    )
    registry.register(
        ToolSpec(
            name="lookup_threshold",
            description=(
                "Look up the cash/transaction reporting threshold for a country from the "
                "service's reference table. Use for countries other than Australia."
            ),
            args_schema=LookupThresholdArgs,
            high_impact=False,
            max_calls_per_request=2,
        ),
        lookup_threshold,
    )
    registry.register(
        ToolSpec(
            name="flag_for_review",
            description=(
                "Flag a specific transaction or customer reference for compliance review. "
                "Use ONLY when the user explicitly asks to flag or escalate a reference. This is "
                "a high-impact action: it is queued for human approval and does not execute "
                "immediately."
            ),
            args_schema=FlagForReviewArgs,
            high_impact=True,
            max_calls_per_request=1,
        ),
        flag_for_review,
    )
    return registry
