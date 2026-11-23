"""Build the messages sent to the LLM for each chunk.

There are three kinds of call, and this module builds the message list for each:

- generator: write a summary of one chunk for one reader (role);
- critic:    score a draft summary against that role's rubric;
- retry:     the generator call again, plus the previous draft and the
             problems found in it, so the model can fix them.

Every call is two messages: a system message with the instructions and a
user message with the code. All three roles share the same templates; what
differs per role (who the reader is, which fields they need, the style rules,
the rubric and a worked example) comes from the ROLES table below.

Nothing here talks to the API. Each builder returns plain
[{"role": ..., "content": ...}] dicts that can be passed straight to the
OpenAI-compatible client.

Print the prompts for a file to see what the model receives:
    python -m knowcode.prompts path/to/file.py --role qa
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict

from knowcode.preprocessor import Chunk


class Role(StrEnum):
    DEVELOPER = "developer"
    QA = "qa"
    PM = "pm"


class Message(TypedDict):
    role: Literal["system", "user"]
    content: str


# Who found the problems that a retry is asked to fix: the LLM critic, our
# own validator (unparseable JSON, programming terms in a PM summary, ...),
# or both at once.
ProblemSource = Literal["critic", "validator", "both"]


# ---------------------------------------------------------------------------
# Role table
# ---------------------------------------------------------------------------

# What each highlight field should contain. Shown to the generator next to the
# field names, so it knows what to put under each one.
FIELD_DEFINITIONS: dict[str, str] = {
    "core_logic": "the main steps or algorithm, in order.",
    "state_changes": (
        "anything modified (object attributes, globals, arguments changed in place, "
        "files, databases) and what is returned."
    ),
    "dependencies": (
        "imports, other functions and classes called, outside services, and "
        "assumptions about the inputs."
    ),
    "expected_behavior": "what the code returns for typical inputs, with examples.",
    "boundary_values": (
        "limits and thresholds in the code (empty, zero, negative, maximum, and the "
        "exact edges of conditions)."
    ),
    "failure_modes": "errors raised, invalid inputs, and what happens on each error path.",
    "business_rules": "the rules or policies the code enforces.",
    "user_facing_behavior": (
        "what a user would see or experience, including errors in user terms."
    ),
}


@dataclass(frozen=True)
class FailingExample:
    """A worked example of the critic rejecting a weak draft."""

    draft_problem: str  # what was wrong with the draft, in words
    verdict: dict


@dataclass(frozen=True)
class RoleSpec:
    display_name: str  # how the role is named inside prompts
    reader: str
    fields: tuple[str, ...]  # keys of the "highlights" object, in order
    style: tuple[str, ...]
    rubric: tuple[str, ...]
    example_summary: dict  # a good generator answer for EXAMPLE_SOURCE
    failing_example: FailingExample
    # False for readers who don't read code: their prompt gets versions of the
    # general rules that don't ask for names from the code.
    uses_code_names: bool = True


ROLES: dict[Role, RoleSpec] = {
    Role.DEVELOPER: RoleSpec(
        display_name="Developer",
        reader=(
            "A software engineer who will maintain or extend this code. "
            "Knows Python but has not read this code."
        ),
        fields=("core_logic", "state_changes", "dependencies"),
        style=(
            "Technical terms are fine.",
            "Put identifiers in backticks.",
            "Give steps in the order they happen.",
        ),
        rubric=(
            "core_logic gives the real steps in the order they happen, naming the actual "
            "conditions, loops and calls in the code rather than describing them vaguely.",
            "state_changes lists everything the code modifies (object attributes, globals, "
            "arguments changed in place, files, databases) and what it returns. If nothing is "
            "modified, it says so plainly instead of leaving the reader to guess.",
            "dependencies names the imports, other functions or classes called, and outside "
            "services that the code actually uses, plus any assumptions it makes about its "
            "inputs.",
            "Identifiers are written in backticks and match the code exactly.",
            "The summary tells a maintainer something they could not get from reading the "
            "function signature alone.",
        ),
        example_summary={
            "name": "borrow_book",
            "kind": "function",
            "what_it_does": (
                "Lends a book to a library member if the member is allowed to borrow and the "
                "book is free. It refuses suspended members and members already at the loan "
                "limit, and reports whether the loan was made."
            ),
            "highlights": {
                "core_logic": [
                    "Raises `PermissionError` first if `member.is_suspended` is true.",
                    "Then raises `ValueError` if `len(member.loans)` is already `MAX_LOANS` (3) "
                    "or more.",
                    "Returns `False` with no changes if `book.available` is false; otherwise "
                    "records the loan and returns `True`.",
                ],
                "state_changes": [
                    "Sets `book.available` to `False` on a successful loan.",
                    "Appends `book` to `member.loans` in place.",
                    "Returns `True` when the loan is made and `False` when the book is "
                    "unavailable.",
                ],
                "dependencies": [
                    "Uses the module constant `MAX_LOANS`.",
                    "Assumes `member` has `is_suspended` and a `loans` list, and `book` has a "
                    "boolean `available`; neither is type-checked.",
                ],
            },
            "unclear_from_code": [
                "Whether the caller saves the changed `member` and `book` objects anywhere."
            ],
        },
        failing_example=FailingExample(
            draft_problem='state_changes said "Does not modify anything."',
            verdict={
                "score": 2,
                "pass": False,
                "critique": (
                    "1. state_changes says nothing is modified, but the code sets "
                    "`book.available` to False and appends `book` to `member.loans`; list both "
                    "and the True/False return value."
                ),
            },
        ),
    ),
    Role.QA: RoleSpec(
        display_name="QA Engineer",
        reader="A tester designing test cases. May not read Python fluently.",
        fields=("expected_behavior", "boundary_values", "failure_modes"),
        style=(
            "Give concrete example inputs with the expected result.",
            "Use actual numbers from the code.",
            "Avoid internal details.",
        ),
        rubric=(
            "expected_behavior gives at least one concrete example input together with the "
            "result the code would produce for it.",
            "boundary_values uses the actual numbers and conditions from the code, including the "
            'exact edge of each comparison (for example "more than 50" versus "50 or more"). '
            'Generic advice such as "test edge cases" does not count.',
            "failure_modes names each error raised or invalid input the code handles, and says "
            "what happens in each case. If the code does not check something it obviously "
            "should, that may be noted as unclear rather than invented.",
            "A tester who does not read Python fluently could design test cases from this "
            "summary alone.",
            "Internal implementation details that do not affect behavior are left out.",
        ),
        example_summary={
            "name": "borrow_book",
            "kind": "function",
            "what_it_does": (
                "Lends a book to a library member if the member is allowed to borrow and the "
                "book is free. Suspended members and members at the loan limit are refused with "
                "an error, while an unavailable book is reported by returning False."
            ),
            "highlights": {
                "expected_behavior": [
                    "An active member with 1 loan borrowing an available book gets True, and the "
                    "book becomes unavailable.",
                    "An active member with 1 loan borrowing an unavailable book gets False, and "
                    "nothing changes.",
                ],
                "boundary_values": [
                    "A member with 2 loans can still borrow; a member with exactly 3 loans is "
                    "refused.",
                    "A suspended member with 3 loans gets the suspension error, because "
                    "suspension is checked first.",
                ],
                "failure_modes": [
                    'A suspended member causes a PermissionError with the message "member is '
                    'suspended".',
                    'A member with 3 or more loans causes a ValueError with the message "loan '
                    'limit reached".',
                    "An unavailable book does not raise an error; it returns False, so a test "
                    "must check the result.",
                ],
            },
            "unclear_from_code": [],
        },
        failing_example=FailingExample(
            draft_problem=(
                'boundary_values only said "Test the loan limit" and suspension was never '
                "mentioned."
            ),
            verdict={
                "score": 3,
                "pass": False,
                "critique": (
                    "1. boundary_values is generic: state that 2 loans is allowed and 3 is "
                    "refused, since the check is len(member.loans) >= MAX_LOANS. "
                    "2. failure_modes omits the PermissionError raised when member.is_suspended "
                    "is true."
                ),
            },
        ),
    ),
    Role.PM: RoleSpec(
        display_name="Product Manager",
        reader="A product manager. Does not read code.",
        fields=("business_rules", "user_facing_behavior"),
        uses_code_names=False,
        style=(
            "Plain business language only.",
            "No code syntax, backticks, identifiers or programming terms (function, variable, "
            "loop, exception, parameter, null, API).",
            "State thresholds as business rules.",
        ),
        rubric=(
            "business_rules states each rule, limit or policy the code enforces, written as a "
            'business rule (for example "an order cannot contain more than 50 items").',
            "user_facing_behavior describes what a user would see or experience, including what "
            "happens when something goes wrong, in everyday terms.",
            "The summary contains no code syntax, backticks, identifiers, or programming terms "
            "(function, variable, loop, exception, parameter, null, API).",
            "Thresholds and numbers from the code appear as plain rules, not as conditions or "
            "comparisons.",
            "A product manager could explain this behavior to a customer or stakeholder without "
            "asking an engineer.",
        ),
        example_summary={
            "name": "borrow_book",
            "kind": "function",
            "what_it_does": (
                "This handles a library member borrowing a book. The loan only goes ahead if the "
                "member is in good standing, has not reached their borrowing limit, and the book "
                "is on the shelf."
            ),
            "highlights": {
                "business_rules": [
                    "Suspended members cannot borrow any books.",
                    "A member can have at most 3 books on loan at once; a fourth is refused.",
                    "A book that is already lent out cannot be borrowed until it is returned.",
                ],
                "user_facing_behavior": [
                    "A suspended member is told they are suspended and the loan is refused.",
                    "A member who already has 3 books is told they have reached their loan limit.",
                    "If the book is already out, the loan does not go ahead.",
                ],
            },
            "unclear_from_code": [
                "What the member is shown when the book is already out is decided elsewhere."
            ],
        },
        failing_example=FailingExample(
            draft_problem=(
                'The draft said "raises a PermissionError" and "returns False", and left out '
                "the 3-book limit."
            ),
            verdict={
                "score": 2,
                "pass": False,
                "critique": (
                    "1. user_facing_behavior uses programming terms ('raises a PermissionError', "
                    "'returns False'); describe what the member experiences instead. "
                    "2. business_rules omits the limit of 3 books on loan at once."
                ),
            },
        ),
    ),
}


# The code every worked example is about. Deliberately unrelated to any real
# input, and the prompts say so, so the model doesn't borrow its names or facts.
EXAMPLE_CONSTANTS = ["MAX_LOANS = 3"]
EXAMPLE_SOURCE = '''\
def borrow_book(member, book):
    """Lend a book to a library member."""
    if member.is_suspended:
        raise PermissionError("member is suspended")
    if len(member.loans) >= MAX_LOANS:
        raise ValueError("loan limit reached")
    if not book.available:
        return False
    book.available = False
    member.loans.append(book)
    return True
'''

PASSING_VERDICT = {"score": 5, "pass": True, "critique": ""}

# Two of the general rules talk about names from the code. Readers who don't
# read code are told not to use identifiers at all, so they get versions of
# these two rules without names; every other rule is shared by all roles.
NO_INVENTED_NAMES = (
    "Never invent function, variable or class names. Only mention names that appear in the code."
)
NO_INVENTED_FACTS = (
    "Never invent rules, limits or behavior. Only describe what the code actually does."
)
BE_SPECIFIC_WITH_NAMES = (
    "Be specific. Use real names, values, thresholds and conditions from the code. Avoid "
    'generic statements such as "handles errors" or "processes data".'
)
BE_SPECIFIC_IN_BUSINESS_TERMS = (
    "Be specific. Use the real values, thresholds and conditions from the code, described in "
    'business terms. Avoid generic statements such as "handles errors" or "processes data".'
)
ONLY_FROM_CODE = (
    "Use only information found in the code. Do not guess about code that is not shown."
)
OTHER_RULES = (
    "If something important cannot be determined from the code, put it in "
    '"unclear_from_code" instead of guessing.',
    "Do not write generic checklists or advice that would apply to any code.",
    "Do not copy the source code into your answer.",
    'Keep "what_it_does" to 2-10 sentences. Keep every list item to one sentence of at most '
    "30 words. Use 0-6 items per list.",
)

SCORING_SCALE = (
    "5 = Accurate, complete, and well suited to the reader.",
    "4 = Accurate and useful, with only minor gaps.",
    "3 = Notable gap, or too vague or generic to be useful.",
    "2 = Significant errors or missing key content.",
    "1 = Wrong, unusable, or written for the wrong reader.",
)

# Header lines for the problems in a retry prompt, depending on who found them.
_PROBLEM_HEADERS: dict[ProblemSource, str] = {
    "critic": "A reviewer found these problems:",
    "validator": "An automatic check found these problems:",
    "both": "A reviewer and an automatic check found these problems:",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_generator_messages(
    chunk: Chunk, role: Role, imports: list[str], constants: list[str]
) -> list[Message]:
    """Messages asking the model to summarize `chunk` for `role`."""
    return [
        {"role": "system", "content": _generator_system(chunk, ROLES[role])},
        {"role": "user", "content": _generator_user(chunk, imports, constants)},
    ]


def build_critic_messages(
    chunk: Chunk,
    role: Role,
    draft: str,
    imports: list[str],
    constants: list[str],
    pass_threshold: int | None = None,
) -> list[Message]:
    """Messages asking the model to score `draft` (the generator's raw reply).

    `pass_threshold` defaults to the value in settings; tests pass it explicitly.
    """
    if pass_threshold is None:
        from knowcode.config import get_settings

        pass_threshold = get_settings().pass_threshold

    spec = ROLES[role]
    user = "\n\n".join(
        [
            _file_context(imports, constants),
            f"Code:\n{chunk.source.rstrip()}",
            f"Draft summary:\n{draft.strip()}",
        ]
    )
    return [
        {"role": "system", "content": _critic_system(chunk, spec, pass_threshold)},
        {"role": "user", "content": user},
    ]


def build_retry_messages(
    chunk: Chunk,
    role: Role,
    imports: list[str],
    constants: list[str],
    previous_draft: str,
    problems: str,
    found_by: ProblemSource = "critic",
) -> list[Message]:
    """The generator messages again, with the previous draft and its problems added.

    `previous_draft` is the raw reply, which may not even be valid JSON when
    the validator rejected it. `problems` is the critic's critique, the
    validator's error messages, or both (`found_by="both"`).
    """
    messages = build_generator_messages(chunk, role, imports, constants)
    retry_note = "\n\n".join(
        [
            f"Your previous draft:\n{previous_draft.strip()}",
            f"{_PROBLEM_HEADERS[found_by]}\n{problems.strip()}",
            "Rewrite the summary and fix these problems. Keep the same JSON format and follow "
            "all the rules above. Do not add new claims that are not supported by the code.",
        ]
    )
    messages[1] = {"role": "user", "content": f"{messages[1]['content']}\n\n{retry_note}"}
    return messages


# ---------------------------------------------------------------------------
# Template pieces
# ---------------------------------------------------------------------------


def _generator_system(chunk: Chunk, spec: RoleSpec) -> str:
    sections = [
        "You are an expert code documentation assistant. You write summaries of Python code "
        f"for one specific reader: a {spec.display_name}.",
        f"## Your reader\n{spec.reader}",
        "## Your task\n"
        f"Summarize the {_describe(chunk)} shown in the user message. Work in two steps:\n"
        "1. EXPLAIN: In plain language, say what the code does and why it exists.\n"
        "2. HIGHLIGHT: Give the details this reader needs, using the fields below.",
        "## What this reader needs\n"
        + "\n".join(f"- {name}: {FIELD_DEFINITIONS[name]}" for name in spec.fields),
        f"## Style for this reader\n{_bullets(spec.style)}",
        f"## Rules for all roles\n{_bullets(_rules_for(spec))}",
        "## Output format\n"
        "Respond with a single JSON object and nothing else. No Markdown, no code fences, no "
        "text before or after the JSON. Use exactly this structure:\n"
        + _output_schema(chunk, spec),
        "## Example\n"
        "This example is about a different function. It only shows the expected shape and "
        "level of detail; do not reuse its names or facts.\n\n"
        f"{_file_context([], EXAMPLE_CONSTANTS)}\n\n"
        f"Python function to summarize:\n{EXAMPLE_SOURCE.rstrip()}\n\n"
        f"Answer for a {spec.display_name}:\n{_json(spec.example_summary)}",
    ]
    return "\n\n".join(sections)


def _generator_user(chunk: Chunk, imports: list[str], constants: list[str]) -> str:
    return (
        f"{_file_context(imports, constants)}\n\n"
        f"Python {_kind_word(chunk)} to summarize:\n\n{chunk.source.rstrip()}"
    )


def _critic_system(chunk: Chunk, spec: RoleSpec, pass_threshold: int) -> str:
    failing = spec.failing_example
    sections = [
        f"You are a strict reviewer of code documentation written for a {spec.display_name}. "
        f"You are given the {_describe(chunk)} and a draft summary of it. Score the draft from "
        "1 to 5 against the rubric below and, if it does not pass, explain what to fix.",
        f"## The reader\n{spec.reader}",
        "## How to review\n"
        "1. ACCURACY: Compare every claim in the draft with the code. Look for invented names, "
        "wrong behavior, and missing key behavior.\n"
        "2. ROLE RUBRIC: Check the draft against each criterion below.\n"
        "3. RULES: Check that the draft follows the style rules for this reader.",
        f"## Rubric for a {spec.display_name}\n{_bullets(spec.rubric)}",
        f"## Style rules for this reader\n{_bullets(spec.style)}",
        "## Scoring scale\n"
        + "\n".join(SCORING_SCALE)
        + f"\nThe draft passes only if the score is {pass_threshold} or higher. "
        f'"pass" must be true exactly when the score is {pass_threshold} or higher.',
        "## Writing the critique\n"
        "If the draft does not pass, write short, concrete feedback (at most 3 points). For "
        "each point, say what is wrong or missing and refer to the specific part of the code. "
        "If the draft passes, leave the critique empty.",
        "## Output format\n"
        "Respond with a single JSON object and nothing else:\n"
        "{\n"
        '  "score": <integer 1-5>,\n'
        '  "pass": <true or false>,\n'
        '  "critique": "<short feedback, or empty string if pass>"\n'
        "}",
        "## Examples\n"
        "These examples are about a different function. They only show how to score and how "
        "to write a critique; do not reuse their names or facts.\n\n"
        f"Example code:\n{_file_context([], EXAMPLE_CONSTANTS)}\n\n{EXAMPLE_SOURCE.rstrip()}\n\n"
        f"Example 1. Draft:\n{_json(spec.example_summary)}\n"
        f"Answer: {_json(PASSING_VERDICT, indent=None)}\n\n"
        f"Example 2. Draft problem: {failing.draft_problem}\n"
        f"Answer: {_json(failing.verdict, indent=None)}",
    ]
    return "\n\n".join(sections)


def _output_schema(chunk: Chunk, spec: RoleSpec) -> str:
    """The JSON structure the generator must return, with this chunk's name and
    kind already filled in so they come back exactly as we expect them."""
    highlights = ",\n".join(f'    "{name}": ["<item>", "<item>"]' for name in spec.fields)
    return (
        "{\n"
        f'  "name": {json.dumps(chunk.name)},\n'
        f'  "kind": "{chunk.kind}",\n'
        '  "what_it_does": "<2-10 sentences in plain language>",\n'
        '  "highlights": {\n'
        f"{highlights}\n"
        "  },\n"
        '  "unclear_from_code": ["<anything that cannot be determined>"]\n'
        "}"
    )


def _file_context(imports: list[str], constants: list[str]) -> str:
    return (
        f"File imports (for reference only):\n{_lines_or_none(imports)}\n\n"
        f"File constants (for reference only):\n{_lines_or_none(constants)}"
    )


def _describe(chunk: Chunk) -> str:
    """Phrase naming the chunk in the task sentence, e.g. "Python function named `f`"."""
    if chunk.kind == "script":
        return (
            f"Python script code (module-level statements, lines {chunk.start_line}-"
            f"{chunk.end_line})"
        )
    if chunk.parent is not None:
        return f"Python method named `{chunk.name}` (a method of the class `{chunk.parent}`)"
    return f"Python {chunk.kind} named `{chunk.name}`"


def _kind_word(chunk: Chunk) -> str:
    if chunk.kind == "script":
        return "script code"
    if chunk.parent is not None:
        return "method"
    return chunk.kind


def _rules_for(spec: RoleSpec) -> tuple[str, ...]:
    """The general rules, with the two name rules worded for this reader."""
    if spec.uses_code_names:
        no_inventing, be_specific = NO_INVENTED_NAMES, BE_SPECIFIC_WITH_NAMES
    else:
        no_inventing, be_specific = NO_INVENTED_FACTS, BE_SPECIFIC_IN_BUSINESS_TERMS
    return (ONLY_FROM_CODE, no_inventing, be_specific, *OTHER_RULES)


def _bullets(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _lines_or_none(lines: list[str]) -> str:
    return "\n".join(lines) if lines else "(none)"


def _json(value: dict, indent: int | None = 2) -> str:
    return json.dumps(value, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Command line: print the prompts for a file
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    import argparse

    from knowcode.preprocessor import InputError, preprocess_file

    parser = argparse.ArgumentParser(description="Print the generator prompts for a Python file.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--role", choices=[r.value for r in Role], default=Role.DEVELOPER.value)
    parser.add_argument("--critic", action="store_true", help="print critic prompts instead")
    args = parser.parse_args(argv)

    try:
        result = preprocess_file(args.path)
    except InputError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 1

    role = Role(args.role)
    for chunk in result.chunks:
        if args.critic:
            messages = build_critic_messages(
                chunk, role, "<draft summary goes here>", result.imports, result.constants
            )
        else:
            messages = build_generator_messages(chunk, role, result.imports, result.constants)
        print(f"{'=' * 30} {chunk.name} ({chunk.kind}) {'=' * 30}")
        for message in messages:
            print(f"--- {message['role'].upper()} ---\n{message['content']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
