"""Parse Python source and split it into chunks for summarization.

This is the first step of the pipeline and uses no LLM:

1. Syntax check: `ast.parse` either builds a syntax tree or raises
   SyntaxError. Invalid code is rejected here, before any API call is made.
2. Chunking: every top-level function and class becomes its own Chunk, so
   the model only ever sees one focused piece of code at a time.
3. Size check: a class that is too large is split into its methods. A single
   function that is still too large is reported as a failure; we never
   silently cut code off, because a summary of half a function is misleading.

The file's import statements are collected separately so they can be passed
to the model as context alongside each chunk.

Run it on a file to see what it produces:
    python -m knowcode.preprocessor path/to/file.py
"""

from __future__ import annotations

import ast
import math
import sys
import textwrap
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Common rule of thumb: one token is roughly 4 characters of English or code.
# Accurate enough for a size limit and avoids depending on a tokenizer library.
CHARS_PER_TOKEN = 4

# A method split out of a class is still kind "function"; its `parent` field
# records which class it came from.
ChunkKind = Literal["function", "class"]

# The syntax tree node types that we turn into chunks.
_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


# ---------------------------------------------------------------------------
# Data models passed on to the rest of the pipeline
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """One unit of code to summarize: a function, a class, or a method."""

    model_config = ConfigDict(frozen=True)  # read-only once created

    name: str  # e.g. "round_price", or "Cart.add" for a method split out of a class
    kind: ChunkKind
    source: str  # exact source text including decorators, dedented to column 0
    start_line: int  # 1-based, inclusive, in the original file
    end_line: int
    parent: str | None = None  # enclosing class name when this is a method
    estimated_tokens: int


class ChunkFailure(BaseModel):
    """A function or class that was found but couldn't be turned into a chunk."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: ChunkKind
    start_line: int
    end_line: int
    reason: Literal["too_large"]
    message: str
    estimated_tokens: int


class PreprocessResult(BaseModel):
    """Everything later steps need from one source file."""

    imports: list[str]  # import statements exactly as written in the file
    chunks: list[Chunk]  # in the order they appear in the file
    failures: list[ChunkFailure]  # definitions we refused to send (e.g. too large)


# ---------------------------------------------------------------------------
# Errors for input we can't process at all
# ---------------------------------------------------------------------------


class InputError(Exception):
    """The input code can't be processed. Raised before any LLM call."""

    # Used when logging failures, to separate bad input from API or model problems.
    category = "input"


class CodeSyntaxError(InputError):
    """The source has a syntax error, so it can't be parsed into chunks."""

    def __init__(self, message: str, line: int | None, column: int | None):
        self.line = line
        self.column = column
        where = f" (line {line}, column {column})" if line is not None else ""
        super().__init__(f"Syntax error{where}: {message}")


class NothingToSummarizeError(InputError):
    """The source is valid Python but defines no functions or classes."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens `text` uses (characters / 4, rounded up)."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def preprocess(source: str, max_chunk_tokens: int | None = None) -> PreprocessResult:
    """Validate Python source and split it into chunks.

    `max_chunk_tokens` defaults to the value in settings; tests pass it explicitly.
    Raises CodeSyntaxError or NothingToSummarizeError (both are InputError).
    """
    if max_chunk_tokens is None:
        # Imported here so that using this module doesn't require loading settings.
        from knowcode.config import get_settings

        max_chunk_tokens = get_settings().max_chunk_tokens

    tree = _parse(source)
    # keepends=True keeps the "\n" on each line, so joining lines back
    # together reproduces the original text exactly.
    lines = source.splitlines(keepends=True)

    imports = _collect_imports(tree, source)
    chunks: list[Chunk] = []
    failures: list[ChunkFailure] = []

    # Only look at top-level statements. Anything nested inside a function or
    # class is handled as part of that definition.
    for node in tree.body:
        if isinstance(node, _DEF_NODES):
            _chunk_node(
                node, lines, max_chunk_tokens, parent=None, chunks=chunks, failures=failures
            )

    # Valid code with no definitions (e.g. a plain script) gives the model
    # nothing to work on, so treat it as unusable input.
    if not chunks and not failures:
        raise NothingToSummarizeError(
            "No functions or classes were found, so there is nothing to summarize."
        )

    return PreprocessResult(imports=imports, chunks=chunks, failures=failures)


def preprocess_file(path: str | Path, max_chunk_tokens: int | None = None) -> PreprocessResult:
    """Read a .py file and run `preprocess` on its contents."""
    try:
        source = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"{path} is not valid UTF-8 text: {exc}") from exc
    return preprocess(source, max_chunk_tokens)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse(source: str) -> ast.Module:
    """Parse source into a syntax tree, converting parse errors into InputErrors."""
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise CodeSyntaxError(exc.msg, exc.lineno, exc.offset) from exc
    except ValueError as exc:
        # ast.parse raises ValueError (not SyntaxError) for null bytes in the source.
        raise CodeSyntaxError(str(exc), None, None) from exc


def _collect_imports(tree: ast.Module, source: str) -> list[str]:
    """Return the module-level import statements as written in the source.

    Also finds imports wrapped in module-level blocks, such as the common
    `try: import x / except ImportError:` pattern. Imports inside a function or
    class are skipped because they're already part of that chunk's source.
    """
    found: list[str] = []

    def visit(statements: list[ast.stmt]) -> None:
        for stmt in statements:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                # get_source_segment returns the original text; unparse is a
                # fallback that rebuilds equivalent code from the tree.
                found.append(ast.get_source_segment(source, stmt) or ast.unparse(stmt))
            elif isinstance(stmt, _DEF_NODES):
                continue
            else:
                # Compound statements (if / try / with / for ...) keep their
                # inner statements in these fields; walk into each one.
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(stmt, field, []))
                for handler in getattr(stmt, "handlers", []):  # try's except blocks
                    visit(handler.body)

    visit(tree.body)
    return found


def _node_source(node: ast.AST, lines: list[str]) -> tuple[str, int, int]:
    """Return (source text, start line, end line) for a function or class node.

    `node.lineno` points at the `def`/`class` line, but decorators sit above
    it and change behavior (e.g. @property, @staticmethod), so start from the
    first decorator when there is one.

    The text is dedented so a method pulled out of a class body starts at
    column 0 and is valid Python on its own.
    """
    decorators = getattr(node, "decorator_list", [])
    start = min([d.lineno for d in decorators] + [node.lineno])
    end = node.end_lineno
    text = textwrap.dedent("".join(lines[start - 1 : end]))  # line numbers are 1-based
    return text, start, end


def _chunk_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    lines: list[str],
    max_tokens: int,
    parent: str | None,
    chunks: list[Chunk],
    failures: list[ChunkFailure],
) -> None:
    """Turn one definition into a chunk, split it into smaller chunks, or record
    a failure. Results are appended to `chunks` / `failures`.
    """
    source, start, end = _node_source(node, lines)
    tokens = estimate_tokens(source)
    kind: ChunkKind = "class" if isinstance(node, ast.ClassDef) else "function"
    name = f"{parent}.{node.name}" if parent else node.name

    # Normal case: it fits, so the whole definition is one chunk. Nested
    # functions and methods stay inside it.
    if tokens <= max_tokens:
        chunks.append(
            Chunk(
                name=name,
                kind=kind,
                source=source,
                start_line=start,
                end_line=end,
                parent=parent,
                estimated_tokens=tokens,
            )
        )
        return

    # A class that's too large is split into its methods and nested classes.
    # This calls itself recursively, so an oversized nested class gets split
    # the same way. Class-level statements that aren't definitions (e.g.
    # attributes like `MAX = 10`) aren't included in any method chunk.
    if kind == "class":
        members = [child for child in node.body if isinstance(child, _DEF_NODES)]
        if members:
            for child in members:
                _chunk_node(child, lines, max_tokens, name, chunks, failures)
            return

    # Too large and can't be split any further (a single function, or a class
    # with no methods): report it instead of truncating the code.
    failures.append(
        ChunkFailure(
            name=name,
            kind=kind,
            start_line=start,
            end_line=end,
            reason="too_large",
            message=(
                f"{kind} `{name}` is about {tokens} tokens, over the limit of "
                f"{max_tokens}, and cannot be split further."
            ),
            estimated_tokens=tokens,
        )
    )


# ---------------------------------------------------------------------------
# Command-line demo: python -m knowcode.preprocessor some_file.py
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m knowcode.preprocessor <file.py>")
        return 2
    try:
        result = preprocess_file(argv[1])
    except InputError as exc:
        print(f"Input failure: {exc}")
        return 1

    print(f"Imports ({len(result.imports)}):")
    for imp in result.imports:
        print(f"  {imp}")
    print(f"\nChunks ({len(result.chunks)}):")
    for c in result.chunks:
        parent = f", parent={c.parent}" if c.parent else ""
        print(
            f"  {c.kind:8} {c.name:30} lines {c.start_line}-{c.end_line}"
            f"  ~{c.estimated_tokens} tokens{parent}"
        )
    if result.failures:
        print(f"\nFailures ({len(result.failures)}):")
        for f in result.failures:
            print(f"  {f.reason}: {f.message}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
