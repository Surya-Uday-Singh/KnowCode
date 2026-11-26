"""Turn pipeline results into what the user receives.

Two granularities, two formats:

- function level: one summary per part (function, class, method or script
  section), each with its review status;
- file level: the same checked summaries stitched into one document: a
  table of the parts, each of the reader's fields gathered across the
  whole file, how the parts connect, the file's setup, open questions and
  anything that couldn't be summarized;
- either as Markdown (for people) or JSON (for other programs).

Everything here is ordinary code. No LLM is called: the file-level document
is assembled from summaries that have already been through the validator
and the critic. `Report.file_summary` is a reserved slot for a short
whole-file summary written by the model, should that ever be added; it is
always empty for now and simply not shown.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Literal

from pydantic import BaseModel

from knowcode.pipeline import ChunkOutcome
from knowcode.preprocessor import Chunk, PreprocessResult
from knowcode.prompts import ROLES, Role

Granularity = Literal["function", "file"]
OutputFormat = Literal["md", "json"]

FIELD_TITLES: dict[str, str] = {
    "core_logic": "Core logic",
    "state_changes": "State changes",
    "dependencies": "Dependencies",
    "expected_behavior": "Expected behavior",
    "boundary_values": "Boundary values",
    "failure_modes": "Failure modes",
    "business_rules": "Business rules",
    "user_facing_behavior": "What users experience",
}

# Plain-language descriptions of the validator's rule problems. Worded so
# they're fine to show a product manager.
RULE_PROBLEMS: dict[str, str] = {
    "unknown_name": "mentions names that don't appear in the code",
    "pm_jargon": "contains programming terms",
    "wrong_name": "is labelled with the wrong part name",
    "wrong_kind": "is labelled with the wrong kind of part",
}

# Why a part has no summary, in words any reader can follow.
NOT_SUMMARIZED: dict[str, str] = {
    "format_failure": "the model's replies could not be used",
    "api_failure": "the AI service could not be reached",
    "empty_response": "the AI service kept returning empty replies",
    "content_filtered": "the AI service's content filter blocked the reply",
    "too_large": "this part is too long to summarize in one piece",
    "component_failure": "an internal error occurred",
}


class Report(BaseModel):
    """Everything needed to render the results for one input."""

    source: str  # the file path, or "snippet" for pasted code
    role: Role
    granularity: Granularity
    outcomes: list[ChunkOutcome]  # in file order
    imports: list[str]
    constants: list[str]
    connections: dict[str, list[str]]  # part name -> parts it uses
    file_summary: str | None = None  # reserved; see the module docstring

    @property
    def reader(self) -> str:
        return ROLES[self.role].display_name

    @property
    def shows_code(self) -> bool:
        """Whether this reader sees code names, imports and the call map."""
        return ROLES[self.role].uses_code_names


def build_report(
    source: str,
    role: Role,
    granularity: Granularity,
    result: PreprocessResult,
    outcomes: list[ChunkOutcome],
) -> Report:
    return Report(
        source=source,
        role=role,
        granularity=granularity,
        outcomes=outcomes,
        imports=result.imports,
        constants=result.constants,
        connections=part_connections(result.chunks),
    )


def render(report: Report, output_format: OutputFormat) -> str:
    if output_format == "json":
        return json.dumps(to_json(report), indent=2, ensure_ascii=False) + "\n"
    if report.granularity == "file":
        return _file_markdown(report)
    return _function_markdown(report)


# ---------------------------------------------------------------------------
# How the parts connect
# ---------------------------------------------------------------------------


def part_connections(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Which parts use which, worked out by reading the code (no LLM).

    A part "uses" another when it calls a name the other part defines: a
    function's name, a class's name, or one of a class's methods. Matching
    is by name only, so `cart.add(...)` counts as using a part that defines
    `add` even if `cart` is some other object; treat the result as a guide.
    """
    defines = {chunk.name: _defined_names(chunk) for chunk in chunks}
    connections: dict[str, list[str]] = {}
    for chunk in chunks:
        called = _called_names(chunk.source)
        used = [
            other.name
            for other in chunks
            if other.name != chunk.name and called & defines[other.name]
        ]
        if used:
            connections[chunk.name] = used
    return connections


def _defined_names(chunk: Chunk) -> set[str]:
    if chunk.kind == "script":
        return set()
    # A function, method or class chunk is exactly one definition.
    node = ast.parse(chunk.source).body[0]
    assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    names = {node.name}
    if isinstance(node, ast.ClassDef):
        names |= {
            item.name
            for item in node.body
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
        }
    # Special methods like __init__ are never called by name.
    return {name for name in names if not (name.startswith("__") and name.endswith("__"))}


def _called_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


# ---------------------------------------------------------------------------
# Status wording
# ---------------------------------------------------------------------------


def status_text(outcome: ChunkOutcome) -> str:
    """One full line describing how the part's summary fared."""
    score = f"score {outcome.score}/5"
    if outcome.status == "passed":
        return f"✅ Passed review ({score})"
    if outcome.status == "passed_with_rule_errors":
        return f"⚠️ Passed review ({score}), but the summary {_rule_problems(outcome)}"
    if outcome.status == "below_threshold":
        return f"⚠️ Below the quality bar: best {score} after one revision"
    if outcome.status == "unreviewed":
        return "⚠️ Not reviewed: the reviewer could not score this summary"
    return f"⛔ Not summarized: {NOT_SUMMARIZED[outcome.status]}"


def status_short(outcome: ChunkOutcome) -> str:
    """A few words for a table cell or a progress line."""
    if outcome.status == "passed":
        return f"✅ passed ({outcome.score}/5)"
    if outcome.status == "passed_with_rule_errors":
        return f"⚠️ passed ({outcome.score}/5), rule problem"
    if outcome.status == "below_threshold":
        return f"⚠️ below threshold ({outcome.score}/5)"
    if outcome.status == "unreviewed":
        return "⚠️ not reviewed"
    return "⛔ not summarized"


def _rule_problems(outcome: ChunkOutcome) -> str:
    problems = [RULE_PROBLEMS.get(issue.code, issue.code) for issue in outcome.rule_errors]
    return " and ".join(dict.fromkeys(problems)) or "broke a style rule"


def _not_summarized_reason(outcome: ChunkOutcome, report: Report) -> str:
    reason = NOT_SUMMARIZED[outcome.status]
    # Developers and testers also get the technical detail, if there is one.
    if report.shows_code and outcome.error:
        reason += f" ({_last_line(outcome.error)})"
    return reason


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _function_markdown(report: Report) -> str:
    lines = [
        f"# KnowCode summary: {report.source}",
        "",
        f"Reader: {report.reader} · one summary per part · {_count_parts(report)}",
    ]
    for outcome in report.outcomes:
        lines += ["", f"## {_heading(outcome, report)}", ""]
        if outcome.summary is None:
            lines.append(f"⛔ Not summarized: {_not_summarized_reason(outcome, report)}")
            continue
        lines.append(status_text(outcome))
        summary = outcome.summary
        lines += ["", f"**What it does:** {summary.what_it_does}"]
        for field, items in summary.highlights.items():
            lines += ["", f"**{FIELD_TITLES[field]}**", *_bullets(items)]
        if summary.unclear_from_code:
            lines += ["", "**Unclear from the code**", *_bullets(summary.unclear_from_code)]
    return "\n".join(lines) + "\n"


def _file_markdown(report: Report) -> str:
    summarized = [o for o in report.outcomes if o.summary is not None]
    skipped = [o for o in report.outcomes if o.summary is None]

    lines = [
        f"# KnowCode file overview: {report.source}",
        "",
        f"Reader: {report.reader} · whole file · {_count_parts(report)}",
    ]
    if report.file_summary:
        lines += ["", "## Overview", "", report.file_summary]

    lines += ["", "## Parts", ""]
    if report.shows_code:
        lines += ["| Part | Kind | Lines | What it does | Status |", "|---|---|---|---|---|"]
    else:
        lines += ["| Part | Lines | What it does | Status |", "|---|---|---|---|"]
    for outcome in report.outcomes:
        does = _first_sentence(outcome.summary.what_it_does) if outcome.summary else "—"
        cells = [outcome.chunk_name]
        if report.shows_code:
            cells.append(outcome.kind)
        cells += [_lines(outcome), does, status_short(outcome)]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")

    if summarized:
        lines += ["", "## What each part does"]
        for outcome in summarized:
            lines += ["", f"**{outcome.chunk_name}:** {outcome.summary.what_it_does}"]

    for field in ROLES[report.role].fields:
        lines += ["", f"## {FIELD_TITLES[field]} across the file", ""]
        items = [
            f"- **{o.chunk_name}:** {item}"
            for o in summarized
            for item in o.summary.highlights[field]
        ]
        lines += items or ["_Nothing noted._"]

    if report.shows_code:
        lines += ["", "## How the parts connect", ""]
        if report.connections:
            for name, used in report.connections.items():
                lines.append(f"- `{name}` uses " + ", ".join(f"`{u}`" for u in used))
            lines += ["", "_Worked out from the calls in the code, matched by name._"]
        else:
            lines.append("_No part calls another part._")
        lines += ["", "## File setup", ""]
        lines.append("**Imports:** " + (", ".join(f"`{i}`" for i in report.imports) or "none"))
        lines.append("")
        # A constant can span several lines; show each on one line so it fits in inline code.
        shown = [" ".join(c.split()) for c in report.constants]
        lines.append("**Constants:** " + (", ".join(f"`{c}`" for c in shown) or "none"))

    unclear = [
        f"- **{o.chunk_name}:** {item}" for o in summarized for item in o.summary.unclear_from_code
    ]
    if unclear:
        lines += ["", "## Unclear from the code", "", *unclear]

    if skipped:
        lines += ["", "## Not summarized", ""]
        for outcome in skipped:
            reason = _not_summarized_reason(outcome, report)
            lines.append(f"- **{outcome.chunk_name}** ({_lines(outcome)}): {reason}")

    return "\n".join(lines) + "\n"


def _heading(outcome: ChunkOutcome, report: Report) -> str:
    # A product manager doesn't read code: no backticks and no "function"/"class".
    if not report.shows_code:
        return f"{outcome.chunk_name} · {_lines(outcome)}"
    return f"`{outcome.chunk_name}` · {outcome.kind} · {_lines(outcome)}"


def _count_parts(report: Report) -> str:
    total = len(report.outcomes)
    done = sum(1 for o in report.outcomes if o.summary is not None)
    noun = "part" if total == 1 else "parts"
    return f"{total} {noun}" if done == total else f"{total} {noun} ({done} summarized)"


def _lines(outcome: ChunkOutcome) -> str:
    if outcome.start_line == outcome.end_line:
        return f"line {outcome.start_line}"
    return f"lines {outcome.start_line}-{outcome.end_line}"


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def _first_sentence(text: str) -> str:
    # The sentence ends at ". " (so decimals like 1.24 don't end it).
    end = text.find(". ")
    return text if end == -1 else text[: end + 1]


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _last_line(text: str) -> str:
    return [line for line in text.strip().splitlines() if line.strip()][-1].strip()


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def to_json(report: Report) -> dict[str, Any]:
    """The results as plain data for other programs (and the experiment harness)."""
    data: dict[str, Any] = {
        "source": report.source,
        "role": report.role.value,
        "reader": report.reader,
        "granularity": report.granularity,
        "parts": [_part_json(outcome) for outcome in report.outcomes],
    }
    if report.granularity == "function":
        return data

    summarized = [o for o in report.outcomes if o.summary is not None]
    data["file_summary"] = report.file_summary
    data["by_field"] = {
        field: [
            {"part": o.chunk_name, "item": item}
            for o in summarized
            for item in o.summary.highlights[field]
        ]
        for field in ROLES[report.role].fields
    }
    data["unclear_from_code"] = [
        {"part": o.chunk_name, "item": item}
        for o in summarized
        for item in o.summary.unclear_from_code
    ]
    if report.shows_code:
        data["connections"] = report.connections
        data["setup"] = {"imports": report.imports, "constants": report.constants}
    data["not_summarized"] = [
        {
            "part": o.chunk_name,
            "start_line": o.start_line,
            "end_line": o.end_line,
            "status": o.status,
            "reason": NOT_SUMMARIZED[o.status],
        }
        for o in report.outcomes
        if o.summary is None
    ]
    return data


def _part_json(outcome: ChunkOutcome) -> dict[str, Any]:
    return {
        "name": outcome.chunk_name,
        "kind": outcome.kind,
        "start_line": outcome.start_line,
        "end_line": outcome.end_line,
        "status": outcome.status,
        "failure_category": outcome.failure_category,
        "score": outcome.score,
        "rule_problems": [
            {"code": issue.code, "message": issue.message} for issue in outcome.rule_errors
        ],
        "error": outcome.error,
        "summary": outcome.summary.model_dump() if outcome.summary else None,
    }
