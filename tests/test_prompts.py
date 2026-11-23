"""Tests for prompt building: generator, critic and retry messages for every role."""

import ast
import json
import re
from pathlib import Path

import pytest

from knowcode.preprocessor import FileContext, preprocess_file
from knowcode.prompts import (
    BE_SPECIFIC_IN_BUSINESS_TERMS,
    BE_SPECIFIC_WITH_NAMES,
    EXAMPLE_SOURCE,
    FIELD_DEFINITIONS,
    NO_INVENTED_FACTS,
    NO_INVENTED_NAMES,
    ONLY_FROM_CODE,
    OTHER_RULES,
    REFERENCE_RULE_IN_BUSINESS_TERMS,
    REFERENCE_RULE_WITH_NAMES,
    ROLES,
    Role,
    build_critic_messages,
    build_generator_messages,
    build_retry_messages,
)

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cart.py"
NO_LIMIT = 100_000
# Small enough that the Cart class is split into methods (Cart.add, ...), so
# method chunks can be tested; every other chunk in the sample still fits.
SPLIT_CLASSES = 100
ALL_ROLES = list(Role)

# Words a Product Manager summary must not contain.
PM_FORBIDDEN_TERMS = ["function", "variable", "loop", "exception", "parameter", "null", "api"]


@pytest.fixture(scope="module")
def sample():
    return preprocess_file(SAMPLE, SPLIT_CLASSES)


def chunk_named(sample, name):
    return next(c for c in sample.chunks if c.name == name)


def generator(sample, name, role):
    return build_generator_messages(chunk_named(sample, name), role, sample.context)


def critic(sample, name, role, draft='{"name": "x"}'):
    return build_critic_messages(
        chunk_named(sample, name), role, draft, sample.context, pass_threshold=4
    )


# --- Shape shared by every call ---------------------------------------------


@pytest.mark.parametrize("role", ALL_ROLES)
def test_every_call_is_a_system_then_a_user_message(sample, role):
    for messages in (generator(sample, "round_price", role), critic(sample, "round_price", role)):
        assert [m["role"] for m in messages] == ["system", "user"]
        assert all(m["content"].strip() for m in messages)


@pytest.mark.parametrize("role", ALL_ROLES)
def test_prompts_mention_json_for_json_mode(sample, role):
    # The API's JSON mode refuses requests whose prompt never mentions json.
    for messages in (generator(sample, "round_price", role), critic(sample, "round_price", role)):
        assert "json" in messages[0]["content"].lower()


@pytest.mark.parametrize("role", ALL_ROLES)
def test_no_template_placeholder_is_left_unfilled(sample, role):
    # Placeholders would look like {ROLE} or {CHUNK_NAME}; none should survive.
    for messages in (generator(sample, "Cart.add", role), critic(sample, "Cart.add", role)):
        for message in messages:
            assert not re.search(r"\{[A-Z_]+\}", message["content"])


# --- Generator ---------------------------------------------------------------


@pytest.mark.parametrize("role", ALL_ROLES)
def test_generator_lists_exactly_the_roles_fields_with_definitions(sample, role):
    system = generator(sample, "round_price", role)[0]["content"]
    for field, definition in FIELD_DEFINITIONS.items():
        if field in ROLES[role].fields:
            assert f"- {field}: {definition}" in system
            assert f'"{field}": ["<item>", "<item>"]' in system
        else:
            assert field not in system.split("## Example")[0]


def test_product_manager_gets_two_fields_and_the_others_three():
    assert len(ROLES[Role.PM].fields) == 2
    assert len(ROLES[Role.DEVELOPER].fields) == 3
    assert len(ROLES[Role.QA].fields) == 3


@pytest.mark.parametrize("role", ALL_ROLES)
def test_generator_names_the_reader_and_includes_the_style_rules(sample, role):
    system = generator(sample, "round_price", role)[0]["content"]
    spec = ROLES[role]
    assert f"for one specific reader: a {spec.display_name}." in system
    assert spec.reader in system
    for rule in spec.style:
        assert rule in system


def test_pm_rules_do_not_ask_for_names_from_the_code(sample):
    system = generator(sample, "round_price", Role.PM)[0]["content"]
    rules = system.split("## Rules for all roles", 1)[1].split("## Output format", 1)[0]
    assert NO_INVENTED_FACTS in rules
    assert BE_SPECIFIC_IN_BUSINESS_TERMS in rules
    assert "real names" not in rules
    assert "Only mention names that appear in the code" not in rules


@pytest.mark.parametrize("role", [Role.DEVELOPER, Role.QA])
def test_developer_and_qa_keep_the_name_rules(sample, role):
    system = generator(sample, "round_price", role)[0]["content"]
    rules = system.split("## Rules for all roles", 1)[1].split("## Output format", 1)[0]
    assert NO_INVENTED_NAMES in rules
    assert BE_SPECIFIC_WITH_NAMES in rules
    assert NO_INVENTED_FACTS not in rules
    assert BE_SPECIFIC_IN_BUSINESS_TERMS not in rules


@pytest.mark.parametrize("role", ALL_ROLES)
def test_every_role_gets_eight_rules_in_the_same_order(sample, role):
    system = generator(sample, "round_price", role)[0]["content"]
    rules = system.split("## Rules for all roles\n", 1)[1].split("\n\n## Output format", 1)[0]
    lines = rules.splitlines()
    assert len(lines) == 8
    assert lines[0] == f"- {ONLY_FROM_CODE}"
    reference = (
        REFERENCE_RULE_WITH_NAMES
        if ROLES[role].uses_code_names
        else REFERENCE_RULE_IN_BUSINESS_TERMS
    )
    assert lines[1] == f"- {reference}"
    assert lines[4:] == [f"- {rule}" for rule in OTHER_RULES]


def test_generator_user_message_has_the_code_and_file_context(sample):
    user = generator(sample, "round_price", Role.QA)[1]["content"]
    assert chunk_named(sample, "round_price").source.rstrip() in user
    for line in sample.imports + sample.constants:
        assert line in user
    assert "MAX_ITEMS = 50" in user


def test_empty_file_context_is_shown_as_none(sample):
    chunk = chunk_named(sample, "round_price")
    user = build_generator_messages(chunk, Role.DEVELOPER, FileContext())[1]["content"]
    assert "File imports (for reference only):\n(none)" in user
    assert "File constants (for reference only):\n(none)" in user


def test_output_schema_has_the_chunks_real_name_and_kind(sample):
    system = generator(sample, "Cart.add", Role.DEVELOPER)[0]["content"]
    assert '"name": "Cart.add"' in system
    # Methods are kind "function"; the class they belong to is said in words.
    assert '"kind": "function"' in system
    assert "method named `Cart.add` (a method of the class `Cart`)" in system


def test_class_chunk_is_described_as_a_class(sample):
    system = generator(sample, "CartItem", Role.DEVELOPER)[0]["content"]
    assert "Python class named `CartItem`" in system
    assert '"kind": "class"' in system


def test_script_chunk_is_described_as_script_code(tmp_path):
    path = tmp_path / "job.py"
    path.write_text("import sys\n\nfor arg in sys.argv[1:]:\n    print(arg.upper())\n")
    result = preprocess_file(path, NO_LIMIT)
    [chunk] = result.chunks
    messages = build_generator_messages(chunk, Role.QA, result.context)
    assert "Python script code (module-level statements, lines" in messages[0]["content"]
    assert '"kind": "script"' in messages[0]["content"]
    assert "Python script code to summarize:" in messages[1]["content"]


def test_chunk_name_with_quotes_still_gives_valid_json_in_the_schema(sample):
    chunk = chunk_named(sample, "round_price").model_copy(update={"name": 'weird"name'})
    system = build_generator_messages(chunk, Role.DEVELOPER, FileContext())[0]["content"]
    assert '"name": "weird\\"name"' in system


@pytest.mark.parametrize("role", ALL_ROLES)
def test_generator_example_is_marked_as_a_different_function(sample, role):
    system = generator(sample, "round_price", role)[0]["content"]
    example = system.split("## Example", 1)[1]
    assert "different function" in example
    assert EXAMPLE_SOURCE.rstrip() in example
    assert json.dumps(ROLES[role].example_summary, indent=2) in example


# --- Worked examples follow our own rules -----------------------------------


@pytest.mark.parametrize("role", ALL_ROLES)
def test_example_summary_has_the_schema_the_role_asks_for(role):
    summary = ROLES[role].example_summary
    assert set(summary) == {"name", "kind", "what_it_does", "highlights", "unclear_from_code"}
    assert tuple(summary["highlights"]) == ROLES[role].fields


@pytest.mark.parametrize("role", ALL_ROLES)
def test_example_summary_respects_the_length_rules(role):
    summary = ROLES[role].example_summary
    lists = [*summary["highlights"].values(), summary["unclear_from_code"]]
    for items in lists:
        assert len(items) <= 6
        for item in items:
            assert len(item.split()) <= 30, item


def test_pm_example_contains_no_programming_terms_or_code():
    summary = ROLES[Role.PM].example_summary
    # name and kind are metadata; the reader-facing text is everything else.
    text = " ".join(
        [summary["what_it_does"], *summary["unclear_from_code"]]
        + [item for items in summary["highlights"].values() for item in items]
    ).lower()
    assert "`" not in text
    for term in PM_FORBIDDEN_TERMS:
        assert not re.search(rf"\b{term}\b", text), term
    for identifier in ("borrow_book", "max_loans", "is_suspended", "member.loans"):
        assert identifier not in text


def test_example_source_is_valid_python():
    ast.parse(EXAMPLE_SOURCE)


@pytest.mark.parametrize("role", ALL_ROLES)
def test_failing_example_verdict_is_consistent(role):
    verdict = ROLES[role].failing_example.verdict
    assert verdict["score"] < 4
    assert verdict["pass"] is False
    assert verdict["critique"]


# --- Critic ------------------------------------------------------------------


@pytest.mark.parametrize("role", ALL_ROLES)
def test_critic_has_rubric_reader_and_style_for_the_role(sample, role):
    system = critic(sample, "round_price", role)[0]["content"]
    spec = ROLES[role]
    assert f"## Rubric for a {spec.display_name}" in system
    for criterion in spec.rubric:
        assert criterion in system
    assert spec.reader in system
    for rule in spec.style:
        assert rule in system


def test_critic_rubric_is_only_the_requested_roles(sample):
    system = critic(sample, "round_price", Role.QA)[0]["content"]
    for other in (Role.DEVELOPER, Role.PM):
        for criterion in ROLES[other].rubric:
            assert criterion not in system


def test_critic_user_message_has_code_context_and_draft(sample):
    draft = '{"name": "round_price", "kind": "function"}'
    user = critic(sample, "round_price", Role.DEVELOPER, draft)[1]["content"]
    assert user.index("File imports") < user.index("Code:") < user.index("Draft summary:")
    assert chunk_named(sample, "round_price").source.rstrip() in user
    assert draft in user
    assert "MAX_ITEMS = 50" in user


def test_critic_pass_rule_uses_the_threshold(sample):
    chunk = chunk_named(sample, "round_price")
    system = build_critic_messages(chunk, Role.QA, "{}", FileContext(), pass_threshold=3)[0][
        "content"
    ]
    assert '"pass" must be true exactly when the score is 3 or higher.' in system


def test_critic_threshold_defaults_to_settings(sample, monkeypatch):
    from knowcode.config import get_settings

    monkeypatch.setenv("KNOWCODE_PASS_THRESHOLD", "5")
    get_settings.cache_clear()
    try:
        chunk = chunk_named(sample, "round_price")
        system = build_critic_messages(chunk, Role.QA, "{}", FileContext())[0]["content"]
        assert "the score is 5 or higher" in system
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("role", ALL_ROLES)
def test_critic_has_a_passing_and_a_failing_example(sample, role):
    system = critic(sample, "round_price", role)[0]["content"]
    examples = system.split("## Examples", 1)[1]
    assert '{"score": 5, "pass": true, "critique": ""}' in examples
    assert json.dumps(ROLES[role].failing_example.verdict) in examples
    assert ROLES[role].failing_example.draft_problem in examples


# --- Retry -------------------------------------------------------------------


def test_retry_is_the_generator_prompt_plus_draft_and_critique(sample):
    chunk = chunk_named(sample, "Cart.add")
    first = build_generator_messages(chunk, Role.QA, sample.context)
    retry = build_retry_messages(
        chunk,
        Role.QA,
        sample.context,
        previous_draft='{"name": "Cart.add"}',
        problems="1. boundary_values is generic.",
    )
    assert retry[0] == first[0]
    assert retry[1]["content"].startswith(first[1]["content"])
    added = retry[1]["content"][len(first[1]["content"]) :]
    assert added.index("Your previous draft:") < added.index("A reviewer found these problems:")
    assert '{"name": "Cart.add"}' in added
    assert "1. boundary_values is generic." in added
    assert "Rewrite the summary and fix these problems." in added


def test_retry_with_problems_from_both_sources_says_so(sample):
    chunk = chunk_named(sample, "round_price")
    retry = build_retry_messages(
        chunk, Role.QA, FileContext(), "{}", "From the reviewer:\n1. a", found_by="both"
    )
    user = retry[1]["content"]
    assert "A reviewer and an automatic check found these problems:\nFrom the reviewer:" in user


def test_retry_after_a_validator_error_says_so(sample):
    chunk = chunk_named(sample, "round_price")
    retry = build_retry_messages(
        chunk,
        Role.PM,
        FileContext(),
        previous_draft='{"name": "round_price", "what_it_does": "Rounds',
        problems="The reply is not valid JSON: it ends in the middle of a string.",
    )
    user = retry[1]["content"]
    assert "An automatic check found these problems:" not in user  # default is the critic
    retry = build_retry_messages(
        chunk,
        Role.PM,
        FileContext(),
        previous_draft='{"name": "round_price", "what_it_does": "Rounds',
        problems="The reply is not valid JSON: it ends in the middle of a string.",
        found_by="validator",
    )
    user = retry[1]["content"]
    assert "An automatic check found these problems:" in user
    assert "A reviewer found these problems:" not in user
    # An unparseable draft is passed through as-is so the model sees what went wrong.
    assert '"what_it_does": "Rounds' in user


def test_building_prompts_does_not_change_the_chunk(sample):
    chunk = chunk_named(sample, "Cart.add")
    before = chunk.model_dump()
    build_retry_messages(chunk, Role.DEVELOPER, FileContext(), "{}", "problem")
    build_critic_messages(chunk, Role.DEVELOPER, "{}", FileContext(), pass_threshold=4)
    assert chunk.model_dump() == before


# --- Other definitions in the file -------------------------------------------------


DEFINITIONS_HEADER = "Other definitions in this file (for reference only; their code is not shown):"


def test_generator_sees_the_other_definitions_in_the_file_but_not_its_own(sample):
    user = generator(sample, "CartItem", Role.QA)[1]["content"]
    block = user.split(DEFINITIONS_HEADER + "\n", 1)[1].split("\n\n", 1)[0]
    assert block.splitlines()[0] == (
        "def round_price(value: float) -> float — Round a price up to the nearest cent."
    )
    assert "class Cart — methods: add(item: CartItem) -> None" in block
    assert "class CartItem" not in block
    # the reference sections come before the code
    assert user.index(DEFINITIONS_HEADER) < user.index("Python class to summarize:")


def test_no_other_definitions_shows_none(sample):
    chunk = chunk_named(sample, "round_price")
    user = build_generator_messages(chunk, Role.QA, FileContext())[1]["content"]
    assert f"{DEFINITIONS_HEADER}\n(none)" in user


def test_critic_sees_the_definitions_and_is_told_missing_claims_are_errors(sample):
    system, user = (m["content"] for m in critic(sample, "CartItem", Role.QA))
    assert DEFINITIONS_HEADER in user and "def round_price(value: float)" in user
    assert "a claim that one of them is missing or undefined is an error" in system


@pytest.mark.parametrize("role", list(Role))
def test_worked_examples_show_the_definitions_block_as_none(sample, role):
    for system in (
        generator(sample, "round_price", role)[0],
        critic(sample, "round_price", role)[0],
    ):
        assert f"{DEFINITIONS_HEADER}\n(none)" in system["content"]


def test_rule_one_covers_the_reference_sections(sample):
    system = generator(sample, "round_price", Role.DEVELOPER)[0]["content"]
    assert "Use only information found in the code and the reference sections" in system
    assert "You may say this code uses them" in system
    pm_system = generator(sample, "round_price", Role.PM)[0]["content"]
    assert "You may say this code uses them" not in pm_system
