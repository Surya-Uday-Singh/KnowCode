"""The `knowcode` command.

    knowcode summarize FILE --role {developer,qa,pm} --granularity {function,file}
                            [--format {md,json}] [--output OUT] [--chunk NAME]

FILE may be "-" to read a pasted snippet from standard input. The result goes
to standard output (or to --output); progress lines go to standard error, so
a saved result never has progress mixed into it.

Exit codes: 0 when the run finished (even if some parts couldn't be
summarized), 1 for bad input or usage, 2 when the run was stopped (no API
key, a rejected key, an empty account).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from knowcode.config import MissingApiKeyError, Settings
from knowcode.llm import DeepSeekClient, LLMClient, LLMError, estimate_cost_cny
from knowcode.output import build_report, render, status_short
from knowcode.pipeline import ChunkOutcome, summarize_file
from knowcode.preprocessor import InputError, PreprocessResult, preprocess, preprocess_file
from knowcode.prompts import Role

EXIT_OK, EXIT_BAD_INPUT, EXIT_STOPPED = 0, 1, 2


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    """argparse exits with code 2 on a usage mistake; we use 2 for "run stopped",
    so usage mistakes are turned into exit code 1 instead."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise _UsageError(f"{self.format_usage()}{self.prog}: error: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="knowcode",
        description="Summarize Python code for a specific reader.",
    )
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    summarize = commands.add_parser(
        "summarize",
        help="summarize a Python file or snippet",
        description="Summarize a Python file or pasted snippet for one reader.",
    )
    summarize.add_argument("path", help='a .py file, or "-" to read a snippet from standard input')
    summarize.add_argument(
        "--role",
        required=True,
        choices=[role.value for role in Role],
        help="who the summary is for: developer, qa (QA engineer) or pm (product manager)",
    )
    summarize.add_argument(
        "--granularity",
        required=True,
        choices=["function", "file"],
        help="function: one summary per function, class, method or script section; "
        "file: those summaries combined into one document for the whole file",
    )
    summarize.add_argument(
        "--format", choices=["md", "json"], default="md", help="output format (default: md)"
    )
    summarize.add_argument("--output", type=Path, help="write the result to this file")
    summarize.add_argument("--chunk", help="only summarize the part with this name")
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

    try:
        source, result = _read_input(args.path, stdin)
    except (InputError, OSError, UnicodeDecodeError) as error:
        print(f"knowcode: cannot read {args.path}: {error}", file=stderr)
        return EXIT_BAD_INPUT

    if args.chunk:
        result = _only_chunk(result, args.chunk)
        if not result.chunks and not result.failures:
            print(f"knowcode: no part named {args.chunk!r} in {source}", file=stderr)
            return EXIT_BAD_INPUT

    role = Role(args.role)
    try:
        client = client_factory() if client_factory else DeepSeekClient(settings)
        outcomes = summarize_file(
            result, role, client, settings, progress=lambda d, t, o: _progress(d, t, o, stderr)
        )
    except MissingApiKeyError as error:
        print(f"knowcode: {error}", file=stderr)
        return EXIT_STOPPED
    except LLMError as error:  # only errors that stop the whole run get this far
        print(f"knowcode: run stopped: {error}", file=stderr)
        return EXIT_STOPPED

    report = build_report(source, role, args.granularity, result, outcomes)
    text = render(report, args.format)
    if args.output:
        try:
            args.output.write_text(text, encoding="utf-8")
        except OSError as error:
            print(f"knowcode: cannot write {args.output}: {error}", file=stderr)
            return EXIT_BAD_INPUT
    else:
        stdout.write(text)

    print(_totals(outcomes), file=stderr)
    if args.output:
        print(f"Written to {args.output}", file=stderr)
    return EXIT_OK


def _read_input(path: str, stdin: TextIO) -> tuple[str, PreprocessResult]:
    if path == "-":
        return "snippet", preprocess(stdin.read())
    return path, preprocess_file(path)


def _only_chunk(result: PreprocessResult, name: str) -> PreprocessResult:
    return result.model_copy(
        update={
            "chunks": [c for c in result.chunks if c.name == name],
            "failures": [f for f in result.failures if f.name == name],
        }
    )


def _progress(done: int, total: int, outcome: ChunkOutcome, stderr: TextIO) -> None:
    replies = outcome.replies()
    seconds = sum(reply.elapsed_seconds for reply in replies)
    print(
        f"[{done}/{total}] {outcome.chunk_name} ... {status_short(outcome)}"
        f" · {len(replies)} calls · {seconds:.0f} s",
        file=stderr,
        flush=True,
    )


def _totals(outcomes: list[ChunkOutcome]) -> str:
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
