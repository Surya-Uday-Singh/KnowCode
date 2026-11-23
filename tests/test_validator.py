"""Tests for the validator: summary structure, name check, jargon check,
length warnings and critic verdicts."""

import json
from pathlib import Path

import pytest

from knowcode.preprocessor import preprocess, preprocess_file
from knowcode.prompts import EXAMPLE_CONSTANTS, EXAMPLE_SOURCE, ROLES, Role
from knowcode.validator import (
    code_names,
    known_names,
    validate_summary,
    validate_verdict,
)

FIXTURES = Path(__file__).parent / "fixtures"
NO_LIMIT = 100_000
SPLIT_CLASSES = 100  # small enough to split Cart into methods


@pytest.fixture(scope="module")
def cart():
    return preprocess_file(FIXTURES / "sample_cart.py", SPLIT_CLASSES)


def chunk_named(result, name):
    return next(c for c in result.chunks if c.name == name)


def summary_for(role, chunk_name="round_price", kind="function", **changes):
    """A well-formed summary for `role`, with any field replaced via `changes`."""
    fields = ROLES[role].fields
    data = {
        "name": chunk_name,
        "kind": kind,
        "what_it_does": "Rounds a price up to the nearest cent. It never rounds down.",
        "highlights": {field: ["Prices are rounded up to the next cent."] for field in fields},
        "unclear_from_code": [],
    }
    data.update(changes)
    return data


def check(cart, data, role=Role.DEVELOPER, chunk="round_price", **kwargs):
    text = data if isinstance(data, str) else json.dumps(data)
    return validate_summary(
        text, chunk_named(cart, chunk), role, cart.imports, cart.constants, **kwargs
    )


def codes(issues):
    return [issue.code for issue in issues]


# --- Good replies pass -------------------------------------------------------


@pytest.mark.parametrize("role", list(Role))
def test_our_worked_examples_pass_with_no_errors_or_warnings(role):
    chunk = preprocess(EXAMPLE_SOURCE, NO_LIMIT).chunks[0]
    text = json.dumps(ROLES[role].example_summary)
    result = validate_summary(text, chunk, role, [], EXAMPLE_CONSTANTS)
    assert result.ok
    assert result.errors == [] and result.warnings == []
    assert result.summary.highlights == ROLES[role].example_summary["highlights"]


def test_real_model_draft_from_a_live_call_passes(cart):
    # A QA draft DeepSeek actually wrote for round_price. It has factual
    # mistakes (only the critic can catch those) but follows every rule.
    text = (FIXTURES / "live_qa_draft_round_price.json").read_text()
    result = check(cart, text, Role.QA)
    assert result.ok, result.errors
    assert result.warnings == []


def test_summary_highlights_keep_the_roles_field_order(cart):
    data = summary_for(Role.QA)
    data["highlights"] = dict(reversed(list(data["highlights"].items())))
    result = check(cart, data, Role.QA)
    assert list(result.summary.highlights) == list(ROLES[Role.QA].fields)


# --- Unusable replies -----------------------------------------------------------


def test_truncated_reply_is_unusable_even_if_it_parses(cart):
    result = check(cart, summary_for(Role.DEVELOPER), truncated=True)
    assert not result.usable
    assert codes(result.errors) == ["truncated"]
    assert "shorter" in result.problems_text()


@pytest.mark.parametrize(
    "text",
    ['{"name": "round_price", "what_it', "Here is the summary!", "", "[1, 2]", '"just a string"'],
    ids=["cut_off", "prose", "empty", "array", "string"],
)
def test_reply_that_is_not_a_json_object_is_unusable(cart, text):
    result = check(cart, text)
    assert not result.usable
    assert codes(result.errors) == ["invalid_json"]


def test_json_inside_markdown_fences_is_accepted(cart):
    text = "```json\n" + json.dumps(summary_for(Role.DEVELOPER)) + "\n```"
    assert check(cart, text).ok


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"what_it_does": ""}, "what_it_does"),
        ({"what_it_does": None}, "what_it_does"),
        ({"name": 5}, '"name" must be a string'),
        ({"highlights": ["a"]}, '"highlights" must be an object'),
        ({"unclear_from_code": "nothing"}, "unclear_from_code"),
    ],
)
def test_wrong_types_make_the_reply_unusable(cart, change, expected):
    result = check(cart, summary_for(Role.DEVELOPER, **change))
    assert not result.usable
    assert codes(result.errors) == ["invalid_schema"]
    assert expected in result.errors[0].message


def test_missing_required_fields_are_named(cart):
    data = summary_for(Role.DEVELOPER)
    del data["what_it_does"], data["kind"]
    result = check(cart, data)
    assert not result.usable
    assert "missing field(s): kind, what_it_does" in result.errors[0].message


def test_highlights_for_the_wrong_reader_are_unusable(cart):
    # A PM summary that came back with developer fields.
    data = summary_for(Role.PM, highlights={"core_logic": ["x"], "business_rules": ["y"]})
    result = check(cart, data, Role.PM)
    assert not result.usable
    message = result.errors[0].message
    assert "missing: user_facing_behavior" in message
    assert "not for this reader: core_logic" in message


def test_highlight_list_items_must_be_non_empty_strings(cart):
    data = summary_for(Role.QA)
    data["highlights"]["boundary_values"] = ["fine", ""]
    data["highlights"]["failure_modes"] = "none"
    result = check(cart, data, Role.QA)
    assert not result.usable
    assert "highlights.boundary_values" in result.errors[0].message
    assert "highlights.failure_modes" in result.errors[0].message


def test_empty_highlight_lists_are_allowed(cart):
    # Code with no error paths should be allowed to say nothing rather than invent one.
    data = summary_for(Role.QA)
    data["highlights"]["failure_modes"] = []
    assert check(cart, data, Role.QA).ok


def test_missing_unclear_from_code_is_only_a_warning(cart):
    data = summary_for(Role.DEVELOPER)
    del data["unclear_from_code"]
    result = check(cart, data)
    assert result.ok
    assert result.summary.unclear_from_code == []
    assert codes(result.warnings) == ["missing_unclear"]


def test_extra_top_level_fields_are_ignored_with_a_warning(cart):
    result = check(cart, summary_for(Role.DEVELOPER, confidence="high"))
    assert result.ok
    assert codes(result.warnings) == ["extra_fields"]


# --- Name and kind -----------------------------------------------------------------


def test_wrong_name_and_kind_are_errors_but_the_summary_is_usable(cart):
    result = check(cart, summary_for(Role.DEVELOPER, name="roundPrice", kind="class"))
    assert result.usable and not result.ok
    assert codes(result.errors) == ["wrong_name", "wrong_kind"]


def test_method_chunk_uses_its_qualified_name(cart):
    data = summary_for(Role.DEVELOPER, chunk_name="Cart.add")
    assert check(cart, data, chunk="Cart.add").ok


# --- Name check (Developer and QA) -------------------------------------------------


@pytest.mark.parametrize(
    "text, names",
    [
        (
            "Raises `ValueError` when `len(self.items) >= MAX_ITEMS`.",
            {"ValueError", "len", "self", "items", "MAX_ITEMS"},
        ),
        ('Raises `ValueError("cart is full")`.', {"ValueError"}),  # quoted text isn't a name
        ("round_price(1.1) returns 1.11.", {"round_price"}),
        ("See member.loans and MAX_LOANS.", {"member", "loans", "MAX_LOANS"}),
        ("Raises PermissionError or uses isSuspended.", {"PermissionError", "isSuspended"}),
        ("Rounds the price, e.g. 1.234 to 1.24, for each item(s); 1e5 too.", set()),
        ("Plain words like price, total and items are not checked.", set()),
    ],
)
def test_code_names_finds_only_what_looks_like_code(text, names):
    assert code_names(text) == names


def test_known_names_cover_code_context_and_builtins(cart):
    known = known_names(chunk_named(cart, "Cart.add"), cart.imports, cart.constants)
    # from the chunk's code
    assert {"add", "self", "item", "items", "append", "quantity", "len"} <= known
    # the method's class, the file's imports and constants
    assert {"Cart", "math", "dataclass", "dataclasses", "json", "MAX_ITEMS"} <= known
    # built-ins a summary may rightly mention without the code spelling them out
    assert {"TypeError", "OverflowError", "None", "True"} <= known
    assert "discount_rate" not in known


# Code shaped like the real chunks where the first version of the name check
# raised false alarms: names that only appear in a docstring, a comment, a
# string value or a dictionary key.
REAL_WORLD_CODE = '''\
def send(payload, settings):
    """Send the payload. Raises CodeSyntaxError (an InputError) on bad input."""
    # DeepSeek needs JSON mode switched on for every request.
    request = {"response_format": {"type": "json_object"}, "extra_body": payload}
    if payload is None:
        return Issue(code="pass_mismatch", message="content_filter")
    return request
'''


def test_names_from_docstrings_comments_and_strings_are_known():
    chunk = preprocess(REAL_WORLD_CODE, NO_LIMIT).chunks[0]
    known = known_names(chunk, [], [])
    assert {"CodeSyntaxError", "InputError", "DeepSeek"} <= known  # docstring, comment
    assert {"pass_mismatch", "content_filter"} <= known  # string values
    assert {"response_format", "json_object", "extra_body"} <= known  # dictionary keys


@pytest.mark.parametrize(
    "sentence",
    [
        # Sentences from real DeepSeek drafts that were wrongly flagged.
        "Invalid Python syntax causes CodeSyntaxError, which is an InputError.",
        "Sets response_format to json_object, and extra_body to the payload.",
        "A missing payload returns an Issue with code pass_mismatch.",
        "Sends the request to the DeepSeek API.",
    ],
)
def test_real_draft_sentences_using_names_from_the_code_pass(sentence):
    chunk = preprocess(REAL_WORLD_CODE, NO_LIMIT).chunks[0]
    data = summary_for(Role.QA, chunk_name="send")
    data["highlights"]["failure_modes"] = [sentence]
    result = validate_summary(json.dumps(data), chunk, Role.QA, [], [])
    assert result.ok, result.errors


def test_invented_setting_name_from_a_real_draft_is_still_caught():
    # A real draft said the client "uses generator_model"; the setting is
    # actually deepseek_model, so this name is invented.
    result = preprocess_file(Path("src/knowcode/llm.py"))
    chunk = chunk_named(result, "DeepSeekClient")
    data = summary_for(Role.QA, chunk_name="DeepSeekClient", kind="class")
    data["highlights"]["expected_behavior"] = [
        "A generator call uses generator_model, generator_thinking and generator_max_tokens.",
        "Every request sets response_format to json_object and extra_body thinking on or off.",
    ]
    check_result = validate_summary(
        json.dumps(data), chunk, Role.QA, result.imports, result.constants
    )
    assert codes(check_result.errors) == ["unknown_name"]
    assert "`generator_model`" in check_result.errors[0].message
    assert "response_format" not in check_result.errors[0].message


@pytest.mark.parametrize("role", [Role.DEVELOPER, Role.QA])
def test_invented_names_are_errors_for_developer_and_qa(cart, role):
    item = "Stores the result in `self.discount_rate` and calls math.floor()."
    data = summary_for(role, chunk_name="Cart.add")
    data["highlights"][ROLES[role].fields[0]] = [item]
    result = check(cart, data, role, chunk="Cart.add")
    assert result.usable and not result.ok
    assert codes(result.errors) == ["unknown_name"]
    message = result.errors[0].message
    assert "`discount_rate`" in message and "`floor`" in message
    assert "`self`" not in message and "`math`" not in message


def test_real_names_builtins_and_constants_pass_the_name_check(cart):
    data = summary_for(Role.DEVELOPER, chunk_name="Cart.add")
    data["highlights"]["core_logic"] = [
        "Raises `ValueError` once `len(self.items)` reaches `MAX_ITEMS` (50).",
        "Then rejects an `item` whose `quantity` is 0 or less.",
        "A non-numeric quantity would raise TypeError from the comparison.",
    ]
    assert check(cart, data, chunk="Cart.add").ok


def test_name_check_ignores_the_name_field_itself(cart):
    # name is checked against the chunk separately; it's not reader text.
    data = summary_for(Role.DEVELOPER)
    assert "round_price" not in data["what_it_does"]
    assert check(cart, data).ok


# --- Jargon check (Product Manager) -------------------------------------------------


@pytest.mark.parametrize(
    "item, found",
    [
        ("The `cart` is refused.", "backticks"),
        ("The function refuses a 51st item.", "programming terms (function)"),
        ("An exception is raised when the cart is full.", "programming terms (exception)"),
        ("Several Parameters and APIs are involved.", "programming terms (apis, parameters)"),
        ("Returns null for empty carts.", "programming terms (null)"),
        ("A cart can hold at most MAX_ITEMS products.", "code names (MAX_ITEMS)"),
        ("Shoppers see a ValueError message.", "code names (ValueError)"),
        ("The total uses self.items.", "code names (items, self)"),
        ("Calls round_price(total) at checkout.", "code names (round_price)"),
    ],
)
def test_pm_jargon_is_an_error(cart, item, found):
    data = summary_for(Role.PM)
    data["highlights"]["business_rules"] = [item]
    result = check(cart, data, Role.PM)
    assert result.usable and not result.ok
    assert codes(result.errors) == ["pm_jargon"]
    assert found in result.errors[0].message


def test_plain_business_language_passes_the_jargon_check(cart):
    data = summary_for(Role.PM, chunk_name="Cart.add")
    data["highlights"] = {
        "business_rules": [
            "A cart can hold at most 50 items; adding a 51st is refused.",
            "Each item must have a quantity of at least 1.",
        ],
        "user_facing_behavior": ["A shopper with a full cart is told the cart is full."],
    }
    assert check(cart, data, Role.PM, chunk="Cart.add").ok


def test_pm_name_and_kind_fields_are_not_jargon(cart):
    # "round_price" in the name field is required by the schema.
    assert check(cart, summary_for(Role.PM), Role.PM).ok


def test_pm_summaries_are_not_name_checked_and_others_not_jargon_checked(cart):
    data = summary_for(Role.DEVELOPER)
    data["highlights"]["core_logic"] = ["The function uses `math.ceil`."]
    assert check(cart, data, Role.DEVELOPER).ok  # "function" is fine for a developer


# --- Length guidelines (warnings only) ----------------------------------------------


def test_length_guidelines_are_warnings_not_errors(cart):
    long_item = " ".join(["word"] * 31) + "."
    data = summary_for(Role.DEVELOPER, what_it_does="Rounds a price up.")
    data["highlights"]["core_logic"] = ["Step."] * 7
    data["highlights"]["dependencies"] = [long_item]
    result = check(cart, data)
    assert result.ok
    messages = [w.message for w in result.warnings]
    assert codes(result.warnings) == ["length"] * 3
    assert any("1 sentence(s)" in m for m in messages)
    assert any("7 items" in m for m in messages)
    assert any("31 words" in m for m in messages)


def test_decimals_do_not_split_sentences(cart):
    text = "Rounds 1.234 up to 1.24 for a price. Values like 0.07 become 0.08."
    data = summary_for(Role.QA, what_it_does=text)
    assert check(cart, data, Role.QA).warnings == []


# --- Retry message ---------------------------------------------------------------------


def test_problems_text_numbers_every_error(cart):
    data = summary_for(Role.PM, name="wrong")
    data["highlights"]["business_rules"] = ["The function refuses it."]
    result = check(cart, data, Role.PM)
    text = result.problems_text()
    assert text.startswith("1. ")
    assert "\n2. " in text
    assert "wrong" in text and "function" in text


# --- Critic verdicts ---------------------------------------------------------------------


def verdict(**data):
    return validate_verdict(json.dumps(data), pass_threshold=4)


def test_good_verdicts_are_accepted():
    passing = verdict(score=5, **{"pass": True}, critique="")
    failing = verdict(score=2, **{"pass": False}, critique="1. Missing the limit.")
    assert passing.usable and passing.verdict.passed and passing.warnings == []
    assert failing.verdict.score == 2 and not failing.verdict.passed
    assert failing.verdict.critique == "1. Missing the limit."


@pytest.mark.parametrize("score", [0, 6, None, "4", True, 3.5, [4]])
def test_verdict_without_a_score_from_1_to_5_is_unusable(score):
    result = verdict(score=score, **{"pass": True}, critique="")
    assert not result.usable
    assert codes(result.errors) == ["invalid_score"]


def test_whole_number_float_score_is_accepted():
    assert verdict(score=4.0, **{"pass": True}, critique="").verdict.score == 4


def test_missing_score_is_unusable():
    result = validate_verdict('{"pass": true, "critique": ""}', pass_threshold=4)
    assert codes(result.errors) == ["invalid_score"]


def test_pass_that_contradicts_the_score_is_corrected_from_the_score():
    result = verdict(score=3, **{"pass": True}, critique="1. Vague.")
    assert result.verdict.passed is False
    assert codes(result.warnings) == ["pass_mismatch"]


def test_missing_pass_is_derived_from_the_score():
    result = validate_verdict('{"score": 4, "critique": ""}', pass_threshold=4)
    assert result.verdict.passed is True
    assert codes(result.warnings) == ["missing_pass"]


def test_failing_verdict_with_an_empty_critique_gets_a_warning():
    result = verdict(score=2, **{"pass": False}, critique="  ")
    assert result.usable
    assert codes(result.warnings) == ["empty_critique"]


def test_threshold_decides_pass():
    data = json.dumps({"score": 3, "pass": True, "critique": ""})
    assert validate_verdict(data, pass_threshold=3).verdict.passed is True
    assert validate_verdict(data, pass_threshold=4).verdict.passed is False


def test_truncated_or_broken_verdict_is_unusable():
    assert codes(validate_verdict('{"score": 4', pass_threshold=4).errors) == ["invalid_json"]
    result = validate_verdict(
        '{"score": 4, "pass": true, "critique": ""}', truncated=True, pass_threshold=4
    )
    assert codes(result.errors) == ["truncated"]


def test_real_critic_verdict_from_a_live_call_is_accepted():
    text = (
        '{"score": 3, "pass": false, "critique": "1. The boundary claim that exact cents are '
        'always unchanged is wrong: round_price(1.1) returns 1.11."}'
    )
    result = validate_verdict(text, pass_threshold=4)
    assert result.verdict.score == 3 and not result.verdict.passed
    assert result.warnings == []
