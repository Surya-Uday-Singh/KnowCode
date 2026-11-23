"""Parse Python source and split it into chunks for summarization.

This is the first step of the pipeline and uses no LLM:

1. Syntax check: `ast.parse` either builds a syntax tree or raises
   SyntaxError. Invalid code is rejected here, before any API call is made.
2. Chunking: the file is split into focused pieces so the model only ever
   sees one piece of code at a time:
     - every top-level function and class is its own chunk;
     - code outside any function or class (a "script" section, such as a
       loop that processes a file or an `if __name__ == "__main__":` block)
       is grouped into chunks too, one per stretch of statements between
       definitions.
3. Size check: anything over the size limit is split at safe boundaries (a
   class into its methods, a script section between statements). A piece
   that still doesn't fit is reported as a failure; we never silently cut
   code off, because a summary of half a function is misleading.

Lines that only set things up (imports and constant values such as
`MAX_ROWS = 1000`) aren't worth a summary of their own. Instead they are
collected as file context and given to the model alongside every chunk.

Run it on a file to see what it produces:
    python -m knowcode.preprocessor path/to/file.py
"""

from __future__ import annotations

import ast
import copy
import math
import sys
import textwrap
import tokenize
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Common rule of thumb: one token is roughly 4 characters of English or code.
# Accurate enough for a size limit and avoids depending on a tokenizer library.
CHARS_PER_TOKEN = 4

# Constants longer than this aren't copied into the shared file context,
# because that context is repeated in every prompt for the file.
MAX_CONSTANT_CONTEXT_CHARS = 200

# Limits for the outlines of the file's other definitions, also repeated in
# every prompt: one outline line, and all outlines sent with one chunk.
MAX_OUTLINE_CHARS = 300
MAX_DEFINITIONS_CONTEXT_CHARS = 1500
MAX_DOC_SUMMARY_CHARS = 150

# A method split out of a class is still kind "function"; its `parent` field
# records which class it came from. "script" is code outside any definition.
ChunkKind = Literal["function", "class", "script"]

# The syntax tree node types for function and class definitions.
_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


# ---------------------------------------------------------------------------
# Data models passed on to the rest of the pipeline
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """One unit of code to summarize: a function, a class, a method, or a
    section of script code."""

    model_config = ConfigDict(frozen=True)  # read-only once created

    # e.g. "round_price", "Cart.add" for a method split out of a class,
    # or "script (lines 7-11)" for script code, which has no name of its own.
    name: str
    kind: ChunkKind
    source: str  # exact source text including decorators, dedented to column 0
    start_line: int  # 1-based, inclusive, in the original file
    end_line: int
    parent: str | None = None  # enclosing class name when this is a method
    estimated_tokens: int


class ChunkFailure(BaseModel):
    """A piece of code that was found but couldn't be turned into a chunk."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: ChunkKind
    start_line: int
    end_line: int
    reason: Literal["too_large"]
    message: str
    estimated_tokens: int


class Definition(BaseModel):
    """A one-line outline of a top-level function or class, built from its
    code: signature (or a class's fields and methods) and the first line of
    its docstring. No summary is generated; this is read straight from the code."""

    model_config = ConfigDict(frozen=True)

    name: str
    # e.g. "def round_price(value: float) -> float — Round a price up to the nearest cent."
    outline: str


class FileContext(BaseModel):
    """What the model is told about the rest of the file alongside one chunk.
    Each chunk is summarized on its own, so without this it can't know that,
    say, a helper it calls is defined a few lines above."""

    model_config = ConfigDict(frozen=True)

    imports: list[str] = []
    constants: list[str] = []
    definitions: list[Definition] = []

    def outlines_for(self, chunk: Chunk) -> list[str]:
        """Outlines of the file's other definitions, for `chunk`.

        The chunk's own definition is left out (its full code is already in
        the prompt), but a method split out of a class keeps its class's
        outline, which lists its sibling methods. The list is capped so the
        context stays small; anything cut off is counted in a last line.
        """
        own = chunk.name if chunk.parent is None else None
        others = [d.outline for d in self.definitions if d.name != own]
        shown: list[str] = []
        used = 0
        for index, outline in enumerate(others):
            if shown and used + len(outline) > MAX_DEFINITIONS_CONTEXT_CHARS:
                shown.append(f"…(+{len(others) - index} more)")
                break
            shown.append(outline)
            used += len(outline) + 1
        return shown


class PreprocessResult(BaseModel):
    """Everything later steps need from one source file."""

    # File context, sent to the model with every chunk from this file.
    imports: list[str]  # import statements exactly as written
    constants: list[str]  # short module-level constant assignments, e.g. "MAX_ROWS = 1000"
    definitions: list[Definition] = []  # outline of every top-level function and class

    chunks: list[Chunk]  # in the order they appear in the file
    failures: list[ChunkFailure]  # pieces we refused to send (e.g. too large)

    @property
    def context(self) -> FileContext:
        return FileContext(
            imports=self.imports, constants=self.constants, definitions=self.definitions
        )


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
    """The source contains no code at all (it's empty or only comments)."""


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

    source = _normalize(source)
    tree = _parse(source)
    if not tree.body:
        raise NothingToSummarizeError("The input contains no code, only blank lines or comments.")

    # keepends=True keeps the "\n" on each line, so joining lines back
    # together reproduces the original text exactly.
    lines = source.splitlines(keepends=True)

    imports = _collect_imports(tree, source)
    definitions = [_definition(node) for node in tree.body if isinstance(node, _DEF_NODES)]
    constants: list[str] = []
    chunks: list[Chunk] = []
    failures: list[ChunkFailure] = []

    # Walk the top-level statements in order. Definitions become chunks right
    # away; everything else is gathered into `section` until the next
    # definition (or the end of the file) closes that stretch of script code.
    section: list[ast.stmt] = []

    def close_section() -> None:
        if not section:
            return
        if all(_is_setup(stmt) for stmt in section):
            # Only imports/constants: no summary, but keep constants as context.
            constants.extend(_constant_lines(section, source))
        else:
            # Imports and docstrings at the top of a script section are already
            # sent as file context, so leave them out of the chunk to avoid
            # showing the model the same lines twice.
            work = list(section)
            while _is_import_or_docstring(work[0]):
                work.pop(0)
            _chunk_script(work, lines, max_chunk_tokens, chunks, failures)
        section.clear()

    for node in tree.body:
        if isinstance(node, _DEF_NODES):
            close_section()
            _chunk_node(
                node, lines, max_chunk_tokens, parent=None, chunks=chunks, failures=failures
            )
        else:
            section.append(node)
    close_section()

    # A file made only of setup lines (e.g. a settings file full of constants)
    # would otherwise produce nothing. Summarize the whole file as script code
    # instead of rejecting it. Its constants are now in the chunk itself, so
    # they don't also need to be sent as context.
    if not chunks and not failures:
        constants.clear()
        _chunk_script(tree.body, lines, max_chunk_tokens, chunks, failures)

    return PreprocessResult(
        imports=imports,
        constants=constants,
        definitions=definitions,
        chunks=chunks,
        failures=failures,
    )


def preprocess_file(path: str | Path, max_chunk_tokens: int | None = None) -> PreprocessResult:
    """Read a .py file and run `preprocess` on its contents."""
    return preprocess(read_source(path), max_chunk_tokens)


def read_source(path: str | Path) -> str:
    """Read a .py file as text.

    tokenize.open decodes the file the same way the Python interpreter does:
    it honours an encoding declaration such as `# -*- coding: latin-1 -*-`,
    defaults to UTF-8, and strips a UTF-8 byte order mark if there is one.
    """
    try:
        with tokenize.open(path) as f:
            return f.read()
    except (SyntaxError, UnicodeDecodeError) as exc:
        # SyntaxError here means an invalid encoding declaration, not bad code.
        raise InputError(f"Could not read {path} as Python source text: {exc}") from exc


# ---------------------------------------------------------------------------
# Outlines of the file's definitions
# ---------------------------------------------------------------------------


def _definition(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> Definition:
    outline = _outline(node)
    if len(outline) > MAX_OUTLINE_CHARS:
        outline = outline[: MAX_OUTLINE_CHARS - 1] + "…"
    return Definition(name=node.name, outline=outline)


def _outline(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """One line describing a definition, e.g.
    "def round_price(value: float) -> float — Round a price up to the nearest cent."
    "@dataclass class CartItem — fields: name: str, quantity: int = 1; methods: total() -> float"
    """
    details = []
    doc = _doc_summary(node)
    if doc:
        details.append(doc)

    if isinstance(node, ast.ClassDef):
        head = f"{_decorators(node)}class {node.name}"
        bases = [ast.unparse(b) for b in [*node.bases, *node.keywords]]
        if bases:
            head += f"({', '.join(bases)})"
        fields = [
            ast.unparse(item)
            for item in node.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        ]
        if fields:
            details.append("fields: " + ", ".join(fields))
        methods = [
            _method_outline(item)
            for item in node.body
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and _is_shown_method(item)
        ]
        if methods:
            details.append("methods: " + ", ".join(methods))
    else:
        head = _decorators(node) + _signature(node, keyword=True)

    return head + (" — " + " · ".join(details) if details else "")


def _doc_summary(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """The first paragraph of the docstring on one line, shortened if long.
    Docstrings often wrap their first sentence over several lines."""
    doc = ast.get_docstring(node)
    if not doc or not doc.strip():
        return ""
    paragraph = doc.strip().split("\n\n")[0]
    text = " ".join(paragraph.split())
    return text if len(text) <= MAX_DOC_SUMMARY_CHARS else text[: MAX_DOC_SUMMARY_CHARS - 1] + "…"


def _signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef, keyword: bool, drop_first: bool = False
) -> str:
    args = node.args
    if drop_first:
        # Leave out `self` / `cls`; the reader knows a method gets its object.
        args = copy.copy(args)
        if args.posonlyargs:
            args.posonlyargs = args.posonlyargs[1:]
        elif args.args:
            args.args = args.args[1:]
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    is_async = isinstance(node, ast.AsyncFunctionDef)
    prefix = ("async def " if is_async else "def ") if keyword else ("async " if is_async else "")
    return f"{prefix}{node.name}({ast.unparse(args)}){returns}"


def _method_outline(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    decorators = {ast.unparse(d) for d in node.decorator_list}
    if decorators & {"property", "cached_property", "functools.cached_property"}:
        return f"{node.name} (property)"
    return _signature(node, keyword=False, drop_first="staticmethod" not in decorators)


def _is_shown_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Special methods aren't called by name, so they're left out of a class's
    outline, except an __init__ that takes arguments (how the class is created)."""
    if not (node.name.startswith("__") and node.name.endswith("__")):
        return True
    has_arguments = len(node.args.posonlyargs) + len(node.args.args) > 1 or bool(
        node.args.vararg or node.args.kwonlyargs or node.args.kwarg
    )
    return node.name == "__init__" and has_arguments


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    return "".join(f"@{ast.unparse(d)} " for d in node.decorator_list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _normalize(source: str) -> str:
    """Clean up pasted source so valid code isn't rejected for cosmetic reasons.

    - Drops a leading byte order mark (an invisible character some Windows
      editors put at the start of a file), which ast.parse rejects.
    - Removes indentation shared by every line. A method copied out of a class
      arrives indented, and Python refuses code whose first line is indented.
      Normal files already start at column 0, so this changes nothing for them,
      and line numbers are never affected.
    """
    return textwrap.dedent(source.removeprefix("﻿"))


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


# ---------------------------------------------------------------------------
# Setup lines and file context
# ---------------------------------------------------------------------------


def _is_literal(node: ast.expr) -> bool:
    """True if `node` is a plain value written directly in the code, such as
    1000, "abc", None, [1, 2] or {"a": 1}, as opposed to something computed
    by calling code, such as open("data.csv") or load_config()."""
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


def _is_import_or_docstring(stmt: ast.stmt) -> bool:
    """True for an import statement or a bare string (docstring) on its own line."""
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return True
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _is_setup(stmt: ast.stmt) -> bool:
    """True if a top-level statement only sets things up rather than doing work.

    Setup means: imports, the module docstring, `pass`, and assignments of
    literal values (MAX_ROWS = 1000). An `if` or `try` block counts as setup
    when everything inside it is setup, which covers patterns like
    `try: import json / except ImportError: json = None`.

    Anything else (function calls, loops, `with` blocks, assignments whose
    value is computed) is real work and belongs in a script chunk.
    """
    if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
        return True
    if isinstance(stmt, ast.Expr):
        # A bare string on its own line is a docstring or a comment-like note.
        return isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
    if isinstance(stmt, ast.Assign):
        return _is_literal(stmt.value)
    if isinstance(stmt, ast.AnnAssign):
        # `x: int` (no value) is just a type declaration.
        return stmt.value is None or _is_literal(stmt.value)
    if isinstance(stmt, (ast.If, ast.Try, ast.TryStar)):
        inner = [*stmt.body, *stmt.orelse, *getattr(stmt, "finalbody", [])]
        for handler in getattr(stmt, "handlers", []):
            inner.extend(handler.body)
        return all(_is_setup(s) for s in inner)
    return False


def _constant_lines(statements: list[ast.stmt], source: str) -> list[str]:
    """Return the short, one-line constant assignments among `statements`.

    These are shared with every chunk as context, so a function that uses
    MAX_ROWS is summarized knowing that MAX_ROWS is 1000. Multi-line values
    (e.g. a large lookup table) are left out to keep every prompt small.
    """
    found: list[str] = []
    for stmt in statements:
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        if stmt.lineno != stmt.end_lineno:
            continue
        text = ast.get_source_segment(source, stmt) or ast.unparse(stmt)
        if len(text) <= MAX_CONSTANT_CONTEXT_CHARS:
            found.append(text)
    return found


# ---------------------------------------------------------------------------
# Building chunks
# ---------------------------------------------------------------------------


def _source_lines(lines: list[str], start: int, end: int) -> str:
    """Return lines start..end (1-based, inclusive), dedented to column 0."""
    return textwrap.dedent("".join(lines[start - 1 : end]))


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
    return _source_lines(lines, start, end), start, end


def _too_large(
    name: str, kind: ChunkKind, start: int, end: int, tokens: int, max_tokens: int
) -> ChunkFailure:
    return ChunkFailure(
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
    failures.append(_too_large(name, kind, start, end, tokens, max_tokens))


def _chunk_script(
    statements: list[ast.stmt],
    lines: list[str],
    max_tokens: int,
    chunks: list[Chunk],
    failures: list[ChunkFailure],
) -> None:
    """Turn a stretch of top-level statements into one or more "script" chunks.

    Statements are packed into a chunk until adding the next one would go over
    the size limit; then a new chunk starts. Splits only ever happen between
    complete statements, so a loop or `with` block is never cut in half. A
    single statement that's over the limit on its own is reported as a failure.
    """
    group: list[ast.stmt] = []

    def emit() -> None:
        start, end = group[0].lineno, group[-1].end_lineno
        source = _source_lines(lines, start, end)
        tokens = estimate_tokens(source)
        name = f"script (lines {start}-{end})"
        if tokens <= max_tokens:
            chunks.append(
                Chunk(
                    name=name,
                    kind="script",
                    source=source,
                    start_line=start,
                    end_line=end,
                    estimated_tokens=tokens,
                )
            )
        else:
            failures.append(_too_large(name, "script", start, end, tokens, max_tokens))

    for stmt in statements:
        if group:
            candidate = _source_lines(lines, group[0].lineno, stmt.end_lineno)
            # Statements sharing a line (`a = 1; b = 2`) can't be separated,
            # because chunks are cut on line boundaries.
            starts_new_line = stmt.lineno > group[-1].end_lineno
            if estimate_tokens(candidate) > max_tokens and starts_new_line:
                emit()
                group = []
        group.append(stmt)
    if group:
        emit()


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
    print(f"\nConstants ({len(result.constants)}):")
    for const in result.constants:
        print(f"  {const}")
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
