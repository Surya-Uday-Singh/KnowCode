"""Generate generic baseline summaries: the same code, context, model and
settings as KnowCode, but with no target role and no critic.

For each part of the code:

    generator (general reader) → validator → [one retry if the validator
    found a problem] → keep the best draft

The validator's retry is kept so a formatting accident (a cut-off reply,
broken JSON, an invented name) doesn't leave a part without a summary, and
so the validator plays the same role here as in KnowCode. There is no critic
review: the baseline stands for what a plain LLM summary looks like.

Every run is logged like a KnowCode run, tagged "generic_baseline":

    python -m experiment.baseline FILE --granularity {function,file}
                                  [--format {md,json}] [--output OUT] [--chunk NAME]
                                  [--log-dir DIR | --no-log]

Exit codes match the knowcode command: 0 finished, 1 bad input, 2 stopped
(no API key, rejected key, empty account), 130 interrupted with Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel

from experiment.generic import GENERIC_CONDITION, GENERIC_DESCRIPTION, GENERIC_READER
from knowcode.config import MissingApiKeyError, Settings, get_settings
from knowcode.llm import (
    ContentFilteredError,
    DeepSeekClient,
    EmptyResponseError,
    LLMClient,
    LLMError,
    LLMReply,
    estimate_cost_cny,
)
from knowcode.output import NOT_SUMMARIZED, RULE_PROBLEMS, part_connections
from knowcode.pipeline import Draft
from knowcode.preprocessor import (
    Chunk,
    FileContext,
    InputError,
    PreprocessResult,
    preprocess,
    read_source,
)
from knowcode.prompts import build_generator_messages, build_retry_messages
from knowcode.runlog import RunLog
from knowcode.validator import Issue, Summary, validate_summary

BaselineStatus = Literal[
    "generated",  # a usable summary that passed the validator's checks
    "generated_with_rule_errors",  # usable, but rule errors remained after the retry
    "format_failure",  # no usable summary (cut off, not JSON, wrong structure)
    "api_failure",  # the generator call failed after its technical retries
    "empty_response",  # the generator kept replying with nothing
    "content_filtered",  # the provider's filter blocked the reply
    "too_large",  # the preprocessor refused the part
    "component_failure",  # a bug in our own code
]

FAILURE_CATEGORY: dict[str, str] = {
    "format_failure": "output",
    "empty_response": "output",
    "content_filtered": "output",
    "api_failure": "api",
    "too_large": "input",
    "component_failure": "component",
}

KEY_DETAILS_TITLE = "Key details"

EXIT_OK, EXIT_BAD_INPUT, EXIT_STOPPED, EXIT_INTERRUPTED = 0, 1, 2, 130


class BaselineOutcome(BaseModel):
    """The result for one part: its status, the summary kept (if any), and
    every draft with its prompt, reply and validator result."""

    chunk_name: str
    kind: str
    start_line: int
    end_line: int
    status: BaselineStatus
    best_draft: int | None = None
    summary: Summary | None = None
    rule_errors: list[Issue] = []
    drafts: list[Draft] = []
    error: str | None = None  # for too_large and component_failure

    @property
    def failure_category(self) -> str | None:
        return FAILURE_CATEGORY.get(self.status)

    def replies(self) -> list[LLMReply]:
        """Every LLM reply received for this part (generator only: no critic)."""
        return [draft.reply for draft in self.drafts if draft.reply is not None]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def summarize_chunk_generic(
    chunk: Chunk,
    context: FileContext,
    client: LLMClient,
    settings: Settings | None = None,
) -> BaselineOutcome:
    """Write a generic summary of one part, retrying once on validator problems.

    Raises LLMError only when it should stop the whole run (bad key, no balance).
    """
    settings = settings or get_settings()
    retries_left = settings.max_quality_retries
    drafts: list[Draft] = []
    messages = build_generator_messages(chunk, GENERIC_READER, context)
    reason: Literal["validator"] | None = None

    while True:
        draft = Draft(number=len(drafts) + 1, retry_reason=reason, messages=messages)
        drafts.append(draft)

        try:
            draft.reply = client.complete(messages, "generator")
        except LLMError as error:
            if error.stops_run:
                raise
            draft.error = f"{type(error).__name__}: {error}"
            draft.error_status = _status_for(error)
            break  # technical retries are already used up; keep any earlier draft

        draft.check = validate_summary(
            draft.reply.content, chunk, GENERIC_READER, context, draft.reply.truncated
        )
        # Done if the draft is usable and broke no rules, or if the retry is used up.
        if draft.check.ok or retries_left == 0:
            break
        retries_left -= 1
        reason = "validator"
        messages = build_retry_messages(
            chunk,
            GENERIC_READER,
            context,
            draft.reply.content,
            draft.check.problems_text(),
            found_by="validator",
        )

    return _outcome(chunk, drafts)


def summarize_file_generic(
    result: PreprocessResult,
    client: LLMClient,
    settings: Settings | None = None,
    progress: Callable[[int, int, BaselineOutcome], None] | None = None,
) -> list[BaselineOutcome]:
    """Every part of a preprocessed file, in file order, the same way KnowCode
    runs a file: refused parts become "too_large", a bug in our own code
    while handling one part is recorded as "component_failure", and only
    errors that should stop the run are raised."""
    settings = settings or get_settings()
    outcomes: list[BaselineOutcome] = []
    for done, chunk in enumerate(result.chunks, 1):
        try:
            outcome = summarize_chunk_generic(chunk, result.context, client, settings)
        except LLMError as error:
            if error.stops_run:
                raise
            outcome = _component_failure(chunk, error)
        except Exception as error:  # noqa: BLE001 - recorded as a bug in our code, not raised
            outcome = _component_failure(chunk, error)
        outcomes.append(outcome)
        if progress is not None:
            progress(done, len(result.chunks), outcome)

    for failure in result.failures:
        outcomes.append(
            BaselineOutcome(
                chunk_name=failure.name,
                kind=failure.kind,
                start_line=failure.start_line,
                end_line=failure.end_line,
                status="too_large",
                error=failure.message,
            )
        )
    outcomes.sort(key=lambda o: o.start_line)
    return outcomes


def _outcome(chunk: Chunk, drafts: list[Draft]) -> BaselineOutcome:
    base = dict(
        chunk_name=chunk.name,
        kind=chunk.kind,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        drafts=drafts,
    )
    usable = [draft for draft in drafts if draft.usable]
    if not usable:
        # Nothing to keep. Report what happened to the last attempt.
        return BaselineOutcome(**base, status=drafts[-1].error_status or "format_failure")

    # Fewer rule errors first; on a tie, the later draft (it had feedback).
    best = max(usable, key=lambda draft: (-len(draft.rule_errors), draft.number))
    assert best.check is not None
    return BaselineOutcome(
        **base,
        status="generated_with_rule_errors" if best.rule_errors else "generated",
        best_draft=best.number,
        summary=best.check.summary,
        rule_errors=best.rule_errors,
    )


def _status_for(error: LLMError) -> BaselineStatus:
    if isinstance(error, EmptyResponseError):
        return "empty_response"
    if isinstance(error, ContentFilteredError):
        return "content_filtered"
    return "api_failure"


def _component_failure(chunk: Chunk, error: BaseException) -> BaselineOutcome:
    return BaselineOutcome(
        chunk_name=chunk.name,
        kind=chunk.kind,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        status="component_failure",
        error="".join(traceback.format_exception(error)).strip(),
    )


# ---------------------------------------------------------------------------
# Rendering (same layout as KnowCode's output, with generic wording)
# ---------------------------------------------------------------------------


def status_text(outcome: BaselineOutcome) -> str:
    if outcome.status == "generated":
        return "✅ Generated (passed the automatic checks)"
    if outcome.status == "generated_with_rule_errors":
        problems = " and ".join(
            dict.fromkeys(RULE_PROBLEMS.get(e.code, e.code) for e in outcome.rule_errors)
        )
        return f"⚠️ Generated, but the summary {problems or 'broke a rule'}"
    return f"⛔ Not summarized: {NOT_SUMMARIZED[outcome.status]}"


def status_short(outcome: BaselineOutcome) -> str:
    if outcome.status == "generated":
        return "✅ generated"
    if outcome.status == "generated_with_rule_errors":
        return "⚠️ generated, rule problem"
    return "⛔ not summarized"


def render(
    source: str,
    granularity: str,
    result: PreprocessResult,
    outcomes: list[BaselineOutcome],
    output_format: str,
) -> str:
    if output_format == "json":
        data = to_json(source, granularity, result, outcomes)
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if granularity == "file":
        return _file_markdown(source, result, outcomes)
    return _function_markdown(source, outcomes)


def _function_markdown(source: str, outcomes: list[BaselineOutcome]) -> str:
    lines = [
        f"# KnowCode generic baseline: {source}",
        "",
        f"Generic summary (no target role) · one summary per part · {_count(outcomes)}",
    ]
    for outcome in outcomes:
        lines += ["", f"## `{outcome.chunk_name}` · {outcome.kind} · {_lines(outcome)}", ""]
        lines.append(_status_line(outcome))
        if outcome.summary is None:
            continue
        summary = outcome.summary
        lines += ["", f"**What it does:** {summary.what_it_does}"]
        lines += ["", f"**{KEY_DETAILS_TITLE}**", *_bullets(summary.highlights["key_details"])]
        if summary.unclear_from_code:
            lines += ["", "**Unclear from the code**", *_bullets(summary.unclear_from_code)]
    return "\n".join(lines) + "\n"


def _file_markdown(source: str, result: PreprocessResult, outcomes: list[BaselineOutcome]) -> str:
    summarized = [o for o in outcomes if o.summary is not None]
    skipped = [o for o in outcomes if o.summary is None]
    lines = [
        f"# KnowCode generic baseline file overview: {source}",
        "",
        f"Generic summary (no target role) · whole file · {_count(outcomes)}",
        "",
        "## Parts",
        "",
        "| Part | Kind | Lines | What it does | Status |",
        "|---|---|---|---|---|",
    ]
    for outcome in outcomes:
        does = _first_sentence(outcome.summary.what_it_does) if outcome.summary else "—"
        cells = [outcome.chunk_name, outcome.kind, _lines(outcome), does, status_short(outcome)]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")

    if summarized:
        lines += ["", "## What each part does"]
        for outcome in summarized:
            lines += ["", f"**{outcome.chunk_name}:** {outcome.summary.what_it_does}"]

    lines += ["", f"## {KEY_DETAILS_TITLE} across the file", ""]
    items = [
        f"- **{o.chunk_name}:** {item}"
        for o in summarized
        for item in o.summary.highlights["key_details"]
    ]
    lines += items or ["_Nothing noted._"]

    connections = part_connections(result.chunks)
    lines += ["", "## How the parts connect", ""]
    if connections:
        for name, used in connections.items():
            lines.append(f"- `{name}` uses " + ", ".join(f"`{u}`" for u in used))
        lines += ["", "_Worked out from the calls in the code, matched by name._"]
    else:
        lines.append("_No part calls another part._")
    lines += ["", "## File setup", ""]
    lines.append("**Imports:** " + (", ".join(f"`{i}`" for i in result.imports) or "none"))
    lines.append("")
    # A constant can span several lines; show each on one line so it fits in inline code.
    shown = [" ".join(c.split()) for c in result.constants]
    lines.append("**Constants:** " + (", ".join(f"`{c}`" for c in shown) or "none"))

    unclear = [
        f"- **{o.chunk_name}:** {item}" for o in summarized for item in o.summary.unclear_from_code
    ]
    if unclear:
        lines += ["", "## Unclear from the code", "", *unclear]
    if skipped:
        lines += ["", "## Not summarized", ""]
        for outcome in skipped:
            lines.append(f"- **{outcome.chunk_name}** ({_lines(outcome)}): {_reason(outcome)}")
    return "\n".join(lines) + "\n"


def to_json(
    source: str, granularity: str, result: PreprocessResult, outcomes: list[BaselineOutcome]
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source": source,
        "condition": GENERIC_CONDITION,
        "role": None,  # generic baseline summaries have no target role
        "reader": GENERIC_READER.display_name,
        "granularity": granularity,
        "parts": [
            {
                "name": o.chunk_name,
                "kind": o.kind,
                "start_line": o.start_line,
                "end_line": o.end_line,
                "status": o.status,
                "failure_category": o.failure_category,
                "rule_problems": [{"code": e.code, "message": e.message} for e in o.rule_errors],
                "error": o.error,
                "summary": o.summary.model_dump() if o.summary else None,
            }
            for o in outcomes
        ],
    }
    if granularity == "function":
        return data
    summarized = [o for o in outcomes if o.summary is not None]
    data["by_field"] = {
        "key_details": [
            {"part": o.chunk_name, "item": item}
            for o in summarized
            for item in o.summary.highlights["key_details"]
        ]
    }
    data["unclear_from_code"] = [
        {"part": o.chunk_name, "item": item}
        for o in summarized
        for item in o.summary.unclear_from_code
    ]
    data["connections"] = part_connections(result.chunks)
    data["setup"] = {"imports": result.imports, "constants": result.constants}
    data["not_summarized"] = [
        {
            "part": o.chunk_name,
            "start_line": o.start_line,
            "end_line": o.end_line,
            "status": o.status,
            "reason": NOT_SUMMARIZED[o.status],
        }
        for o in outcomes
        if o.summary is None
    ]
    return data


def _status_line(outcome: BaselineOutcome) -> str:
    if outcome.summary is None:
        return f"⛔ Not summarized: {_reason(outcome)}"
    return status_text(outcome)


def _reason(outcome: BaselineOutcome) -> str:
    reason = NOT_SUMMARIZED[outcome.status]
    if outcome.error:  # the technical detail, e.g. the size of a part that was too large
        last_line = [line for line in outcome.error.strip().splitlines() if line.strip()][-1]
        reason += f" ({last_line.strip()})"
    return reason


def _count(outcomes: list[BaselineOutcome]) -> str:
    total = len(outcomes)
    done = sum(1 for o in outcomes if o.summary is not None)
    noun = "part" if total == 1 else "parts"
    return f"{total} {noun}" if done == total else f"{total} {noun} ({done} summarized)"


def _lines(outcome: BaselineOutcome) -> str:
    if outcome.start_line == outcome.end_line:
        return f"line {outcome.start_line}"
    return f"lines {outcome.start_line}-{outcome.end_line}"


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def _first_sentence(text: str) -> str:
    end = text.find(". ")  # ". " so decimals like 1.24 don't end the sentence
    return text if end == -1 else text[: end + 1]


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    """Usage mistakes exit with 1 (argparse's own 2 means "stopped" here)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise _UsageError(f"{self.format_usage()}{self.prog}: error: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m experiment.baseline",
        description="Generic baseline summaries (no target role, no critic) for the experiment. "
        "Uses the same code, context, model and settings as `knowcode summarize`.",
    )
    parser.add_argument("path", help='a .py file, or "-" to read a snippet from standard input')
    parser.add_argument(
        "--granularity",
        required=True,
        choices=["function", "file"],
        help="function: one summary per part; file: those summaries combined into one document",
    )
    parser.add_argument(
        "--format", choices=["md", "json"], default="md", help="output format (default: md)"
    )
    parser.add_argument("--output", type=Path, help="write the result to this file")
    parser.add_argument("--chunk", help="only summarize the part with this name")
    logging = parser.add_mutually_exclusive_group()
    logging.add_argument(
        "--log-dir", type=Path, default=Path("runs"), help="folder for run logs (default: runs)"
    )
    logging.add_argument("--no-log", action="store_true", help="don't save a run log")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    client_factory: Callable[[], LLMClient] | None = None,
    settings: Settings | None = None,
) -> int:
    """Run the command and return its exit code. The keyword arguments let
    tests supply their own streams, a fake LLM client and settings."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    try:
        args = _parser().parse_args(argv)
    except _UsageError as error:
        print(error, file=stderr)
        return EXIT_BAD_INPUT

    settings = settings or get_settings()
    try:
        if args.path == "-":
            source, source_text = "snippet", stdin.read()
        else:
            source, source_text = args.path, read_source(args.path)
        result = preprocess(source_text, settings.max_chunk_tokens)
    except (InputError, OSError, UnicodeDecodeError) as error:
        print(f"baseline: cannot read {args.path}: {error}", file=stderr)
        return EXIT_BAD_INPUT

    if args.chunk:
        result = result.model_copy(
            update={
                "chunks": [c for c in result.chunks if c.name == args.chunk],
                "failures": [f for f in result.failures if f.name == args.chunk],
            }
        )
        if not result.chunks and not result.failures:
            print(f"baseline: no part named {args.chunk!r} in {source}", file=stderr)
            return EXIT_BAD_INPUT

    try:
        client = client_factory() if client_factory else DeepSeekClient(settings)
    except MissingApiKeyError as error:
        print(f"baseline: {error}", file=stderr)
        return EXIT_STOPPED

    log = None
    if not args.no_log:
        try:
            log = RunLog(
                args.log_dir,
                inputs={
                    "source": source,
                    "role": None,  # no target role: that's what makes it the baseline
                    "reader": GENERIC_READER.display_name,
                    "granularity": args.granularity,
                    "format": args.format,
                    "chunk": args.chunk,
                    "output": str(args.output) if args.output else None,
                },
                settings=settings,
                source_text=source_text,
                condition=GENERIC_CONDITION,
                condition_description=GENERIC_DESCRIPTION,
                label="generic",  # e.g. runs/20261125-101500-a1b2-generic/
            )
        except OSError as error:
            print(f"baseline: cannot create a run log in {args.log_dir}: {error}", file=stderr)
            return EXIT_BAD_INPUT

    finished: list[BaselineOutcome] = []

    def on_progress(done: int, total: int, outcome: BaselineOutcome) -> None:
        finished.append(outcome)
        if log:
            log.add_part(outcome)
        replies = outcome.replies()
        print(
            f"[{done}/{total}] {outcome.chunk_name} ... {status_short(outcome)}"
            f" · {len(replies)} calls · {sum(r.elapsed_seconds for r in replies):.0f} s",
            file=stderr,
            flush=True,
        )

    try:
        outcomes = summarize_file_generic(result, client, settings, progress=on_progress)
    except LLMError as error:  # only errors that stop the whole run get this far
        print(f"baseline: run stopped: {error}", file=stderr)
        if log:
            log.finish("stopped", finished, error=f"{type(error).__name__}: {error}")
            print(f"Run log (partial): {log.path}", file=stderr)
        return EXIT_STOPPED
    except KeyboardInterrupt:
        print("baseline: interrupted", file=stderr)
        if log:
            log.finish("interrupted", finished)
            print(f"Run log (partial): {log.path}", file=stderr)
        return EXIT_INTERRUPTED

    text = render(source, args.granularity, result, outcomes, args.format)
    if log:
        log.finish("finished", outcomes, text, args.format)
    if args.output:
        try:
            args.output.write_text(text, encoding="utf-8")
        except OSError as error:
            print(f"baseline: cannot write {args.output}: {error}", file=stderr)
            return EXIT_BAD_INPUT
    else:
        stdout.write(text)

    print(_totals(outcomes), file=stderr)
    if args.output:
        print(f"Written to {args.output}", file=stderr)
    if log:
        print(f"Run log: {log.path}", file=stderr)
    return EXIT_OK


def _totals(outcomes: list[BaselineOutcome]) -> str:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    replies = [reply for outcome in outcomes for reply in outcome.replies()]
    cost = sum(estimate_cost_cny(reply) or 0 for reply in replies)
    statuses = ", ".join(f"{n} {status.replace('_', ' ')}" for status, n in counts.items())
    noun = "part" if len(outcomes) == 1 else "parts"
    return (
        f"Done: {len(outcomes)} {noun} ({statuses}) · {len(replies)} LLM calls"
        f" · about ¥{cost:.3f} (deepseek-flash off-peak estimate)"
    )


if __name__ == "__main__":
    sys.exit(main())
