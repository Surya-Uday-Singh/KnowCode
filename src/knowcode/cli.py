"""The `knowcode` command.

    knowcode summarize FILE --role {developer,qa,pm} --granularity {function,file}
                            [--format {md,json}] [--output OUT] [--chunk NAME]
                            [--log-dir DIR | --no-log]

FILE may be "-" to read a pasted snippet from standard input. The result goes
to standard output (or to --output); progress lines go to standard error, so
a saved result never has progress mixed into it. Every run is logged to its
own folder under runs/ (see runlog.py) unless --no-log is given.

Exit codes: 0 when the run finished (even if some parts couldn't be
summarized), 1 for bad input or usage, 2 when the run was stopped (no API
key, a rejected key, an empty account), 130 when interrupted with Ctrl-C.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from knowcode.config import MissingApiKeyError, Settings, get_settings
from knowcode.llm import DeepSeekClient, LLMClient, LLMError, estimate_cost_cny
from knowcode.output import build_report, render, status_short
from knowcode.pipeline import ChunkOutcome, summarize_file
from knowcode.preprocessor import InputError, PreprocessResult, preprocess, read_source
from knowcode.prompts import Role
from knowcode.runlog import RunLog

EXIT_OK, EXIT_BAD_INPUT, EXIT_STOPPED = 0, 1, 2
EXIT_INTERRUPTED = 130  # the usual exit code after Ctrl-C


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
    logging = summarize.add_mutually_exclusive_group()
    logging.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="folder for run logs (default: runs); each run gets its own subfolder",
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
        source, source_text = _read_input(args.path, stdin)
        result = preprocess(source_text, settings.max_chunk_tokens)
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
    except MissingApiKeyError as error:
        print(f"knowcode: {error}", file=stderr)
        return EXIT_STOPPED

    log = None
    if not args.no_log:
        try:
            log = RunLog(
                args.log_dir,
                inputs=_inputs(args, source),
                settings=settings,
                source_text=source_text,
            )
        except OSError as error:
            print(f"knowcode: cannot create a run log in {args.log_dir}: {error}", file=stderr)
            return EXIT_BAD_INPUT

    finished: list[ChunkOutcome] = []

    def on_progress(done: int, total: int, outcome: ChunkOutcome) -> None:
        finished.append(outcome)
        if log:
            log.add_part(outcome)
        _progress(done, total, outcome, stderr)

    try:
        outcomes = summarize_file(result, role, client, settings, progress=on_progress)
    except LLMError as error:  # only errors that stop the whole run get this far
        print(f"knowcode: run stopped: {error}", file=stderr)
        if log:
            log.finish("stopped", finished, error=f"{type(error).__name__}: {error}")
            print(f"Run log (partial): {log.path}", file=stderr)
        return EXIT_STOPPED
    except KeyboardInterrupt:
        print("knowcode: interrupted", file=stderr)
        if log:
            log.finish("interrupted", finished)
            print(f"Run log (partial): {log.path}", file=stderr)
        return EXIT_INTERRUPTED

    report = build_report(source, role, args.granularity, result, outcomes)
    text = render(report, args.format)
    if log:
        log.finish("finished", outcomes, text, args.format)
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
    if log:
        print(f"Run log: {log.path}", file=stderr)
    return EXIT_OK


def _read_input(path: str, stdin: TextIO) -> tuple[str, str]:
    """The name to show for the input, and its source text."""
    if path == "-":
        return "snippet", stdin.read()
    return path, read_source(path)


def _inputs(args: argparse.Namespace, source: str) -> dict[str, object]:
    return {
        "source": source,
        "role": args.role,
        "granularity": args.granularity,
        "format": args.format,
        "chunk": args.chunk,
        "output": str(args.output) if args.output else None,
    }


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
