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
import re
import sys
import textwrap
import tokenize
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Common rule of thumb: one token is roughly 4 characters of English or code.
# Accurate enough for a size limit and avoids depending on a tokenizer library.
CHARS_PER_TOKEN = 4

# The file's constants are repeated in every prompt for the file, so they're
# kept small: one constant is shown in full up to this many characters (it may
# span several lines); a longer one is listed by name only. All constants sent
# with one chunk are capped at the second limit.
MAX_CONSTANT_CONTEXT_CHARS = 400
MAX_CONSTANTS_CONTEXT_CHARS = 2000

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
        return _capped(others, MAX_DEFINITIONS_CONTEXT_CHARS)

    def shown_constants(self) -> list[str]:
        """The file's constants as sent to the model, capped like the outlines."""
        return _capped(self.constants, MAX_CONSTANTS_CONTEXT_CHARS)


def _capped(lines: list[str], limit: int) -> list[str]:
    """`lines` up to about `limit` characters, then one line counting the rest."""
    shown: list[str] = []
    used = 0
    for index, line in enumerate(lines):
        if shown and used + len(line) > limit:
            shown.append(f"…(+{len(lines) - index} more)")
            break
        shown.append(line)
        used += len(line) + 1
    return shown


class PreprocessResult(BaseModel):
    """Everything later steps need from one source file."""

    # File context, sent to the model with every chunk from this file.
    imports: list[str]  # import statements exactly as written
    # module-level constants and settings as written, e.g. "MAX_ROWS = 1000" or
    # '_WORDS = frozenset("a b".split())'; too-long ones by name only
    constants: list[str]
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
    # Names of the constants found so far, so a later constant built from an
    # earlier one (e.g. re.compile(f"[{CHARS}]")) is recognised as a constant too.
    known_constants: set[str] = set()
    chunks: list[Chunk] = []
    failures: list[ChunkFailure] = []

    # Walk the top-level statements in order. Definitions become chunks right
    # away; everything else is gathered into `section` until the next
    # definition (or the end of the file) closes that stretch of script code.
    section: list[ast.stmt] = []

    def close_section() -> None:
        if not section:
            return
        all_setup, section_constants = _section_constants(section, source, known_constants)
        # Constants are shared as context whether or not the section also does
        # real work, so every chunk is summarized knowing their values.
        constants.extend(section_constants)
        if not all_setup:
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


# Python's convention for constants: UPPER_CASE names, optionally with a
# leading underscore (MAX_ROWS, _DESIGNATORS).
_CONSTANT_NAME = re.compile(r"_*[A-Z][A-Z0-9_]*")

# Calls that only build a value out of plain values, without running any of
# the project's own code: collections, conversions, string methods and
# regular-expression compilation. Any other call (load_config(), open(...))
# makes an assignment real work.
_SAFE_BUILDERS = frozenset(
    "frozenset set tuple list dict sorted str int float bool bytes range".split()
)
_SAFE_STRING_METHODS = frozenset(
    "split rsplit splitlines strip lstrip rstrip lower upper casefold replace join format".split()
)
_SAFE_RE_FUNCTIONS = frozenset({"compile", "escape"})


def _is_setup(stmt: ast.stmt, known: set[str]) -> bool:
    """True if a top-level statement only sets things up rather than doing work.

    Setup means: imports, the module docstring, `pass`, and assignments of
    constant values (see `_is_constant_value`). An `if` or `try` block counts
    as setup when everything inside it is setup, which covers patterns like
    `try: import json / except ImportError: json = None`.

    Anything else (function calls, loops, `with` blocks, assignments whose
    value comes from running code) is real work and belongs in a script chunk.
    """
    if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
        return True
    if isinstance(stmt, ast.Expr):
        # A bare string on its own line is a docstring or a comment-like note.
        return isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
    if isinstance(stmt, ast.Assign):
        return _is_constant_value(stmt.value, known)
    if isinstance(stmt, ast.AnnAssign):
        # `x: int` (no value) is just a type declaration.
        return stmt.value is None or _is_constant_value(stmt.value, known)
    if isinstance(stmt, (ast.If, ast.Try, ast.TryStar)):
        inner = [*stmt.body, *stmt.orelse, *getattr(stmt, "finalbody", [])]
        for handler in getattr(stmt, "handlers", []):
            inner.extend(handler.body)
        return all(_is_setup(s, known) for s in inner)
    return False


def _is_constant_value(node: ast.expr, known: set[str]) -> bool:
    r"""True if `node` is a value built only from plain values, decided by
    reading the code (nothing is run). Examples:

        400, "abc", [1, 2]                         plain values
        frozenset("inc ltd llc".split())           a collection built from them
        re.compile(r"\d+", re.IGNORECASE)         a compiled pattern
        re.compile(f"[{CHARS}]")                   built from an earlier constant

    Anything that calls the project's own code or reads from outside
    (load_config(), open("data.csv"), os.environ[...]) is not a constant value.
    """
    if _is_literal(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in known
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_constant_value(item, known) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(key is not None and _is_constant_value(key, known) for key in node.keys) and all(
            _is_constant_value(value, known) for value in node.values
        )
    if isinstance(node, ast.BinOp):  # e.g. A + B, re.I | re.M
        return _is_constant_value(node.left, known) and _is_constant_value(node.right, known)
    if isinstance(node, ast.UnaryOp):
        return _is_constant_value(node.operand, known)
    if isinstance(node, ast.JoinedStr):  # an f-string
        return all(_is_constant_value(part, known) for part in node.values)
    if isinstance(node, ast.FormattedValue):  # one {...} inside an f-string
        return _is_constant_value(node.value, known) and (
            node.format_spec is None or _is_constant_value(node.format_spec, known)
        )
    if isinstance(node, ast.Attribute):  # a regular-expression flag, e.g. re.IGNORECASE
        return isinstance(node.value, ast.Name) and node.value.id == "re" and node.attr.isupper()
    if isinstance(node, ast.Call):
        return (
            _is_safe_call(node.func, known)
            and all(_is_constant_value(arg, known) for arg in node.args)
            and all(_is_constant_value(kw.value, known) for kw in node.keywords)
        )
    return False


def _is_safe_call(func: ast.expr, known: set[str]) -> bool:
    if isinstance(func, ast.Name):
        return func.id in _SAFE_BUILDERS
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "re":
            return func.attr in _SAFE_RE_FUNCTIONS
        # A string method on a constant value, e.g. "inc ltd".split()
        return func.attr in _SAFE_STRING_METHODS and _is_constant_value(func.value, known)
    return False


def _section_constants(
    section: list[ast.stmt], source: str, known: set[str]
) -> tuple[bool, list[str]]:
    """Look at one module-level section: is it only setup, and which of its
    assignments are constants to share as context?

    In a section that is only setup, every constant assignment is shared. In
    a section that also does real work, only UPPER_CASE names are: a plain
    `rows = []` there is a working variable that the code below fills in,
    while `MAX_ROWS = 1000` is a constant. `known` is updated with the names
    of the constants shared, in order.
    """
    # First pass: does the section do any real work? Assume every
    # constant-looking assignment is a constant while deciding.
    trial = set(known)
    all_setup = all(_note_constant(stmt, trial) or _is_setup(stmt, trial) for stmt in section)

    # Second pass, now that we know which rule applies. In a working section
    # a lower-case name never counts, so nothing built from it does either
    # (e.g. COPY = list(rows) after rows.append(...)).
    shared: list[str] = []
    for stmt in section:
        value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        if value is None or not _is_constant_value(value, known):
            continue
        names = _assigned_names(stmt)
        is_upper_case = bool(names) and all(_CONSTANT_NAME.fullmatch(n) for n in names)
        if all_setup or is_upper_case:
            known.update(names)
            shared.append(_constant_text(stmt, source))
    return all_setup, shared


def _note_constant(stmt: ast.stmt, known: set[str]) -> bool:
    """If `stmt` assigns a constant value, add its names to `known` and return True."""
    value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
    if value is None or not _is_constant_value(value, known):
        return False
    known.update(_assigned_names(stmt))
    return True


def _assigned_names(stmt: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    names: list[str] = []
    for target in targets:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                names.append(node.id)
    return names


def _constant_text(stmt: ast.Assign | ast.AnnAssign, source: str) -> str:
    """The constant as written in the source. Too long to repeat in every
    prompt, it's listed by name only, so the model still knows it exists."""
    text = ast.get_source_segment(source, stmt) or ast.unparse(stmt)
    if len(text) <= MAX_CONSTANT_CONTEXT_CHARS:
        return text
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    name = " = ".join(ast.unparse(target) for target in targets)
    lines = (stmt.end_lineno or stmt.lineno) - stmt.lineno + 1
    return f"{name} = …  # {lines} line{'s' if lines != 1 else ''}, too long to show here"


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
