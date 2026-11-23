"""Check LLM replies before the pipeline uses them.

Two kinds of reply are checked:

- a generator summary: is it usable (complete, valid JSON, the right
  fields for this reader), and does it follow the rules (right name and
  kind, no invented code names, no programming jargon for a product
  manager)?
- a critic verdict: is there a usable score from 1 to 5?

Every problem is an Issue with a short code and a message written to the
model, so the messages can go straight into the retry prompt. Errors are
problems worth a retry; warnings are only recorded.

This module only reports. Deciding whether to retry, and enforcing the
one-retry limit, is the pipeline's job.

What these checks can't do: tell whether a summary is *correct*. A
summary can use only real names, follow every rule, and still describe the
code wrongly. That is what the critic, and in the end human reviewers, are for.
"""

from __future__ import annotations

import ast
import builtins
import json
import keyword
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from knowcode.preprocessor import Chunk
from knowcode.prompts import ROLES, Role

# Length guidelines given to the generator. Breaking them is only a warning.
MIN_SENTENCES, MAX_SENTENCES = 2, 10
MAX_ITEM_WORDS = 30
MAX_LIST_ITEMS = 6

# Words a product manager summary must not use (singular and plural).
PM_FORBIDDEN_TERMS = re.compile(
    r"\b(functions?|variables?|loops?|exceptions?|parameters?|null|apis?)\b", re.IGNORECASE
)

_REQUIRED_SUMMARY_FIELDS = ("name", "kind", "what_it_does", "highlights")
_SUMMARY_FIELDS = (*_REQUIRED_SUMMARY_FIELDS, "unclear_from_code")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class Issue(BaseModel):
    """One problem found in a reply."""

    model_config = ConfigDict(frozen=True)

    code: str  # short and stable, for counting in the run log, e.g. "unknown_name"
    message: str  # addressed to the model, usable as-is in a retry prompt


class Summary(BaseModel):
    """A generator reply that passed the usability checks."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: str
    what_it_does: str
    highlights: dict[str, list[str]]  # exactly the role's fields, in the role's order
    unclear_from_code: list[str]


class Verdict(BaseModel):
    """A critic reply that passed the checks."""

    model_config = ConfigDict(frozen=True)

    score: int  # 1-5
    passed: bool  # always consistent with the score and the pass threshold
    critique: str


class SummaryCheck(BaseModel):
    summary: Summary | None  # None when the reply can't be used at all
    errors: list[Issue]
    warnings: list[Issue]

    @property
    def usable(self) -> bool:
        return self.summary is not None

    @property
    def ok(self) -> bool:
        """Usable and broke no rules."""
        return self.usable and not self.errors

    def problems_text(self) -> str:
        """The errors as a numbered list, for the retry prompt."""
        return _numbered(self.errors)


class VerdictCheck(BaseModel):
    verdict: Verdict | None  # None when there's no usable score
    errors: list[Issue]
    warnings: list[Issue]

    @property
    def usable(self) -> bool:
        return self.verdict is not None


# ---------------------------------------------------------------------------
# Generator summaries
# ---------------------------------------------------------------------------


def validate_summary(
    text: str,
    chunk: Chunk,
    role: Role,
    imports: list[str],
    constants: list[str],
    truncated: bool = False,
) -> SummaryCheck:
    """Check a generator reply for `chunk`, written for `role`.

    `truncated` comes from the LLM client: a reply cut off at the token
    limit is never usable, whatever its text looks like.
    """
    errors: list[Issue] = []
    warnings: list[Issue] = []

    if truncated:
        errors.append(
            Issue(
                code="truncated",
                message="Your previous reply was cut off at the length limit, so it was "
                "incomplete. Write a shorter summary.",
            )
        )
        return SummaryCheck(summary=None, errors=errors, warnings=warnings)

    data = _parse_object(text, errors)
    if data is None:
        return SummaryCheck(summary=None, errors=errors, warnings=warnings)

    summary = _summary_from(data, ROLES[role].fields, errors, warnings)
    if summary is None:
        return SummaryCheck(summary=None, errors=errors, warnings=warnings)

    _check_name_and_kind(summary, chunk, errors)
    reader_text = _reader_text(summary)
    if ROLES[role].uses_code_names:
        _check_names_exist(reader_text, chunk, imports, constants, errors)
    else:
        _check_no_jargon(reader_text, errors)
    _check_lengths(summary, warnings)

    return SummaryCheck(summary=summary, errors=errors, warnings=warnings)


def _summary_from(
    data: dict[str, Any],
    role_fields: tuple[str, ...],
    errors: list[Issue],
    warnings: list[Issue],
) -> Summary | None:
    """Check the structure of a parsed reply; return None if it's unusable."""
    problems: list[str] = []

    missing = [f for f in _REQUIRED_SUMMARY_FIELDS if f not in data]
    if missing:
        problems.append(f"missing field(s): {', '.join(missing)}")
    for field in ("name", "kind"):
        if field in data and not isinstance(data[field], str):
            problems.append(f'"{field}" must be a string')
    if "what_it_does" in data and not _is_text(data["what_it_does"]):
        problems.append('"what_it_does" must be a non-empty string')

    highlights = data.get("highlights")
    if "highlights" in data:
        if not isinstance(highlights, dict):
            problems.append('"highlights" must be an object')
        else:
            missing_fields = [f for f in role_fields if f not in highlights]
            extra_fields = [f for f in highlights if f not in role_fields]
            if missing_fields:
                problems.append(f'"highlights" is missing: {", ".join(missing_fields)}')
            if extra_fields:
                problems.append(
                    f'"highlights" has fields that are not for this reader: '
                    f"{', '.join(extra_fields)} (use only {', '.join(role_fields)})"
                )
            for field in role_fields:
                if field in highlights and not _is_text_list(highlights[field]):
                    problems.append(f'"highlights.{field}" must be a list of non-empty strings')

    unclear = data.get("unclear_from_code", [])
    if "unclear_from_code" not in data:
        warnings.append(Issue(code="missing_unclear", message='"unclear_from_code" was missing.'))
    elif not _is_text_list(unclear):
        problems.append('"unclear_from_code" must be a list of non-empty strings')

    extra = [f for f in data if f not in _SUMMARY_FIELDS]
    if extra:
        warnings.append(
            Issue(code="extra_fields", message=f"Unexpected field(s) ignored: {', '.join(extra)}.")
        )

    if problems:
        errors.append(
            Issue(
                code="invalid_schema",
                message="Your previous reply did not have the required JSON structure: "
                + "; ".join(problems)
                + ".",
            )
        )
        return None

    return Summary(
        name=data["name"],
        kind=data["kind"],
        what_it_does=data["what_it_does"],
        highlights={f: list(highlights[f]) for f in role_fields},
        unclear_from_code=list(unclear),
    )


def _check_name_and_kind(summary: Summary, chunk: Chunk, errors: list[Issue]) -> None:
    if summary.name != chunk.name:
        errors.append(
            Issue(
                code="wrong_name",
                message=f'"name" was "{summary.name}" but the code summarized is "{chunk.name}".',
            )
        )
    if summary.kind != chunk.kind:
        errors.append(
            Issue(
                code="wrong_kind",
                message=f'"kind" was "{summary.kind}" but it should be "{chunk.kind}".',
            )
        )


def _check_names_exist(
    texts: list[str],
    chunk: Chunk,
    imports: list[str],
    constants: list[str],
    errors: list[Issue],
) -> None:
    known = known_names(chunk, imports, constants)
    unknown = sorted({name for text in texts for name in code_names(text)} - known)
    if unknown:
        errors.append(
            Issue(
                code="unknown_name",
                message="The summary mentions names that do not appear in the code: "
                + ", ".join(f"`{n}`" for n in unknown)
                + ". Mention only names that appear in the code.",
            )
        )


def _check_no_jargon(texts: list[str], errors: list[Issue]) -> None:
    found: list[str] = []
    if any("`" in text for text in texts):
        found.append("backticks")
    terms = sorted(
        {m.group(0).lower() for text in texts for m in PM_FORBIDDEN_TERMS.finditer(text)}
    )
    if terms:
        found.append("programming terms (" + ", ".join(terms) + ")")
    names = sorted({name for text in texts for name in _bare_code_tokens(text)})
    if names:
        found.append("code names (" + ", ".join(names) + ")")
    if found:
        errors.append(
            Issue(
                code="pm_jargon",
                message="The summary is for a product manager but contains "
                + "; ".join(found)
                + ". Use plain business language only.",
            )
        )


def _check_lengths(summary: Summary, warnings: list[Issue]) -> None:
    sentences = _count_sentences(summary.what_it_does)
    if not MIN_SENTENCES <= sentences <= MAX_SENTENCES:
        warnings.append(
            Issue(
                code="length",
                message=f'"what_it_does" has {sentences} sentence(s); the guideline is '
                f"{MIN_SENTENCES}-{MAX_SENTENCES}.",
            )
        )
    lists = {**summary.highlights, "unclear_from_code": summary.unclear_from_code}
    for field, items in lists.items():
        if len(items) > MAX_LIST_ITEMS:
            warnings.append(
                Issue(
                    code="length",
                    message=f'"{field}" has {len(items)} items; the guideline is at most '
                    f"{MAX_LIST_ITEMS}.",
                )
            )
        for item in items:
            words = len(item.split())
            if words > MAX_ITEM_WORDS:
                warnings.append(
                    Issue(
                        code="length",
                        message=f'An item in "{field}" has {words} words; the guideline is at '
                        f"most {MAX_ITEM_WORDS}.",
                    )
                )


def _reader_text(summary: Summary) -> list[str]:
    """Every piece of text the reader sees. `name` and `kind` are metadata the
    schema requires, so they're left out of the name and jargon checks."""
    return [
        summary.what_it_does,
        *(item for items in summary.highlights.values() for item in items),
        *summary.unclear_from_code,
    ]


# ---------------------------------------------------------------------------
# Finding code names in text, and the names a chunk knows about
# ---------------------------------------------------------------------------

# An identifier not glued to a preceding letter, digit or underscore, so the
# "e5" in "1e5" isn't picked up.
_IDENTIFIER = re.compile(r"(?<![0-9A-Za-z_])[A-Za-z_][A-Za-z0-9_]*")
_BACKTICK_SPAN = re.compile(r"`([^`]+)`")
_STRING_LITERAL = re.compile(r"""(["'])(?:\\.|(?!\1).)*\1""")

# Patterns that mark a word as code even without backticks.
_DOTTED = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
# Word ends: not followed by more of a word, or by ".name" (which makes it a
# dotted name instead). A full stop ending a sentence is fine.
_END = r"(?!\w|\.\w)"
_SNAKE = re.compile(r"(?<![\w.])_*[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+_*" + _END)
# name( ... ) but not English plurals like "item(s)" or "box(es)".
_CALL = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\((?!s\)|es\))")
_CAMEL = re.compile(r"(?<![\w.])(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+|[a-z]{2,}[A-Z]\w*)" + _END)


def code_names(text: str) -> set[str]:
    """Identifiers mentioned as code in `text`.

    Plain English words can't be told apart from variable names ("price"),
    so only clear signs of code count:
    - anything inside backticks, e.g. `len(member.loans)` gives len, member, loans;
    - outside backticks, words that look like code: snake_case, dotted.names,
      calls like round_price(1.1), and CamelCase like ValueError.
    """
    names: set[str] = set()
    for span in _BACKTICK_SPAN.findall(text):
        span = _STRING_LITERAL.sub(" ", span)  # quoted text isn't a name
        names.update(_IDENTIFIER.findall(span))
    names.update(_bare_code_tokens(_BACKTICK_SPAN.sub(" ", text)))
    return names


def _bare_code_tokens(text: str) -> set[str]:
    """Code-looking identifiers in text that isn't inside backticks."""
    tokens: set[str] = set()
    for match in _DOTTED.finditer(text):
        parts = match.group(0).split(".")
        # Abbreviations like "e.g" and "i.e" are made of one-letter parts.
        if any(len(part) > 1 for part in parts):
            tokens.update(parts)
    tokens.update(m.group(0) for m in _SNAKE.finditer(text))
    tokens.update(m.group(1) for m in _CALL.finditer(text))
    tokens.update(m.group(0) for m in _CAMEL.finditer(text))
    return tokens


def known_names(chunk: Chunk, imports: list[str], constants: list[str]) -> set[str]:
    """Every name a summary of `chunk` may mention: names in the chunk's own
    code, the file's imports and constants, and Python's built-in names
    (a summary may rightly say a call raises TypeError even if the code
    never spells it out)."""
    names = set(dir(builtins)) | set(keyword.kwlist) | set(keyword.softkwlist)
    names |= set(_SUMMARY_FIELDS)
    names |= _names_in_source(chunk.source)
    names |= set(_IDENTIFIER.findall(chunk.name))  # "Cart.add" gives Cart and add
    if chunk.parent:
        names.add(chunk.parent)
    for line in [*imports, *constants]:
        names |= _names_in_source(line)
    return names


def _names_in_source(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Our own chunks always parse; this guards against odd context lines.
        return set(_IDENTIFIER.findall(source))

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.update(node.name.split("."))
            if node.asname:
                names.add(node.asname)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.update(node.module.split("."))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            names.update(node.names)
        elif isinstance(node, ast.MatchAs | ast.MatchStar) and node.name:
            names.add(node.name)
    return names


# ---------------------------------------------------------------------------
# Critic verdicts
# ---------------------------------------------------------------------------


def validate_verdict(
    text: str, truncated: bool = False, pass_threshold: int | None = None
) -> VerdictCheck:
    """Check a critic reply.

    The score is what matters: a verdict without a whole-number score from
    1 to 5 is unusable. "pass" is always recomputed from the score, so the
    verdict can't contradict itself; a disagreement is recorded as a warning.
    `pass_threshold` defaults to the value in settings.
    """
    if pass_threshold is None:
        from knowcode.config import get_settings

        pass_threshold = get_settings().pass_threshold

    errors: list[Issue] = []
    warnings: list[Issue] = []

    if truncated:
        errors.append(
            Issue(code="truncated", message="The review was cut off at the length limit.")
        )
        return VerdictCheck(verdict=None, errors=errors, warnings=warnings)

    data = _parse_object(text, errors)
    if data is None:
        return VerdictCheck(verdict=None, errors=errors, warnings=warnings)

    score = _whole_number(data.get("score"))
    if score is None or not 1 <= score <= 5:
        errors.append(
            Issue(
                code="invalid_score",
                message=f'"score" must be a whole number from 1 to 5, got {data.get("score")!r}.',
            )
        )
        return VerdictCheck(verdict=None, errors=errors, warnings=warnings)

    passed = score >= pass_threshold
    stated = data.get("pass")
    if not isinstance(stated, bool):
        warnings.append(
            Issue(
                code="missing_pass",
                message=f'"pass" was missing or not true/false; set from the score ({score}).',
            )
        )
    elif stated != passed:
        warnings.append(
            Issue(
                code="pass_mismatch",
                message=f'"pass" was {str(stated).lower()} but the score is {score}; '
                f"the score was used.",
            )
        )

    critique = data.get("critique", "")
    if not isinstance(critique, str):
        warnings.append(
            Issue(code="invalid_critique", message='"critique" was not a string; ignored.')
        )
        critique = ""
    if not passed and not critique.strip():
        warnings.append(
            Issue(code="empty_critique", message="The draft failed but the critique is empty.")
        )

    verdict = Verdict(score=score, passed=passed, critique=critique.strip())
    return VerdictCheck(verdict=verdict, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^```[A-Za-z]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)


def _parse_object(text: str, errors: list[Issue]) -> dict[str, Any] | None:
    """Parse a reply as a JSON object, allowing Markdown code fences around it."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        data = json.loads(stripped)
    except ValueError as error:
        errors.append(
            Issue(
                code="invalid_json",
                message=f"Your previous reply was not valid JSON ({error}). Reply with a single "
                "JSON object and nothing else.",
            )
        )
        return None
    if not isinstance(data, dict):
        errors.append(
            Issue(
                code="invalid_json",
                message="Your previous reply was JSON but not an object. Reply with a single "
                "JSON object.",
            )
        )
        return None
    return data


def _whole_number(value: Any) -> int | None:
    # bool is a subclass of int in Python; true/false is not a score.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_text_list(value: Any) -> bool:
    return isinstance(value, list) and all(_is_text(item) for item in value)


def _count_sentences(text: str) -> int:
    # A sentence ends with . ! or ? followed by whitespace or the end of the
    # text, so decimals like 1.24 don't count as a break.
    return len([s for s in re.split(r"[.!?]+(?:\s+|$)", text.strip()) if s.strip()])


def _numbered(issues: list[Issue]) -> str:
    return "\n".join(f"{i}. {issue.message}" for i, issue in enumerate(issues, 1))
