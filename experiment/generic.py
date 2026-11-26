"""The "general reader": the instructions used for generic baseline summaries.

It plugs into KnowCode's prompt template exactly like the Developer, QA
Engineer and Product Manager roles do, so a generic summary is written with
the same template, rules, output format, file context and worked-example
function as a role-specific one. Only the reader-specific parts differ:

- no particular reader is described;
- one neutral highlight field, key_details, instead of a role's fields;
- neutral style rules (nothing role-flavoured such as backticks or "no jargon");
- the worked example answer is a neutral summary of the same example function.

There is no critic rubric: generic baseline summaries are not critic-reviewed.
"""

from knowcode.prompts import RoleSpec

GENERIC_CONDITION = "generic_baseline"
GENERIC_DESCRIPTION = (
    "Generic summary with no target role: the experiment's comparison baseline "
    "(Condition A in the design doc)."
)

KEY_DETAILS_DEFINITION = (
    "the most important facts about how the code behaves: its main steps, inputs and "
    "outputs, conditions and limits, and what happens when something goes wrong."
)

# A good generic answer for the worked-example function (borrow_book) that
# every prompt shows. It covers the same facts as the role examples, written
# for nobody in particular.
GENERIC_EXAMPLE_SUMMARY = {
    "name": "borrow_book",
    "kind": "function",
    "what_it_does": (
        "Lends a book to a library member if the member may borrow and the book is available. "
        "It refuses suspended members and members who already have 3 loans, and reports "
        "whether the loan was made."
    ),
    "highlights": {
        "key_details": [
            'Raises PermissionError ("member is suspended") if the member is suspended; this is '
            "checked first.",
            'Raises ValueError ("loan limit reached") if the member already has MAX_LOANS (3) or '
            "more loans.",
            "Returns False without changing anything if the book is not available.",
            "Otherwise marks the book unavailable, adds it to member.loans and returns True.",
        ]
    },
    "unclear_from_code": [],
}

GENERIC_READER = RoleSpec(
    # Lower-case because the template reads "...for one specific reader: a {display_name}."
    display_name="general reader",
    reader=(
        "Someone who wants to understand what this code does. No particular job role is assumed."
    ),
    fields=("key_details",),
    field_definitions={"key_details": KEY_DETAILS_DEFINITION},
    style=(
        "Write in clear, plain language.",
        "Be precise about what the code does.",
    ),
    example_summary=GENERIC_EXAMPLE_SUMMARY,
    # A general reader may see the code's names; the validator then checks
    # that every name mentioned really appears in the code.
    uses_code_names=True,
)
