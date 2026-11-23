"""Tests for rendering results: connections between parts, status wording,
function- and file-level Markdown, and JSON. Outcomes come from the real
pipeline driven by a scripted FakeLLMClient."""

import json
import os
from pathlib import Path

import pytest

from knowcode.config import Settings
from knowcode.llm import FakeLLMClient
from knowcode.output import build_report, part_connections, render, status_text, to_json
from knowcode.pipeline import ChunkOutcome, summarize_file
from knowcode.preprocessor import preprocess, preprocess_file
from knowcode.prompts import ROLES, Role
from knowcode.validator import PM_FORBIDDEN_TERMS, Issue

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cart.py"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("KNOWCODE_", "DEEPSEEK_")):
            monkeypatch.delenv(name)


def summary(
    role,
    name,
    kind,
    does="Rounds a price up to the nearest cent. It never rounds down.",
    item="Prices are rounded up to the next cent.",
    unclear=(),
):
    return json.dumps(
        {
            "name": name,
            "kind": kind,
            "what_it_does": does,
            "highlights": {field: [f"{item}"] for field in ROLES[role].fields},
            "unclear_from_code": list(unclear),
        }
    )


def verdict(score):
    return json.dumps(
        {"score": score, "pass": score >= 4, "critique": "" if score >= 4 else "1. Vague."}
    )


def run_sample(role, granularity, script, limit=100_000):
    """Run the pipeline on sample_cart.py with a scripted client and build a report."""
    result = preprocess_file(SAMPLE, limit)
    client = FakeLLMClient(script)
    outcomes = summarize_file(result, role, client, Settings(_env_file=None))
    assert client.remaining == 0
    return build_report("sample_cart.py", role, granularity, result, outcomes)


def all_pass(role, item="Prices are rounded up to the next cent."):
    """Script where every part of sample_cart.py (4 parts) passes first time."""
    parts = [
        ("round_price", "function"),
        ("CartItem", "class"),
        ("Cart", "class"),
        ("export_cart", "function"),
    ]
    script = []
    for name, kind in parts:
        script += [summary(role, name, kind, item=item), verdict(5), verdict(5)]
    return script


def mixed(role):
    """round_price passes, CartItem below threshold, Cart unusable twice, export_cart passes."""
    return [
        summary(role, "round_price", "function", unclear=["Whether negative prices are allowed."]),
        verdict(5),
        verdict(4),
        summary(role, "CartItem", "class"),
        verdict(3),
        verdict(3),
        summary(role, "CartItem", "class"),
        verdict(3),
        verdict(2),
        "not json",
        "still not json",
        summary(role, "export_cart", "function"),
        verdict(5),
        verdict(5),
    ]


def outcome(status, **fields):
    base = dict(
        chunk_name="huge", kind="function", start_line=4, end_line=204, role=Role.QA, status=status
    )
    base.update(fields)
    return ChunkOutcome(**base)


# --- How the parts connect -------------------------------------------------------------


def test_connections_in_the_sample_file():
    result = preprocess_file(SAMPLE, 100_000)
    # CartItem.total calls round_price; Cart calls i.total() (defined by
    # CartItem) and round_price at checkout.
    assert part_connections(result.chunks) == {
        "CartItem": ["round_price"],
        "Cart": ["round_price", "CartItem"],
    }


def test_connections_between_methods_split_out_of_a_class():
    result = preprocess_file(SAMPLE, 100)
    connections = part_connections(result.chunks)
    assert connections["Cart.subtotal"] == ["CartItem"]
    assert connections["Cart.checkout_total"] == ["round_price"]  # its nested helper isn't a part
    assert "Cart.add" not in connections


def test_script_sections_connect_to_the_functions_they_call():
    source = (
        "def load(p):\n    return p\n\n"
        "def clean(x):\n    return x\n\n"
        "rows = clean(load('a.csv'))\nprint(rows)\n"
    )
    result = preprocess(source, 100_000)
    script = result.chunks[-1].name
    assert part_connections(result.chunks) == {script: ["load", "clean"]}


def test_special_methods_do_not_create_connections():
    source = (
        "class A:\n    def __init__(self):\n        pass\n\n"
        "def make():\n    return A().__init__()\n"
    )
    result = preprocess(source, 100_000)
    # make() calls A() (the class) and __init__ (never matched by name)
    assert part_connections(result.chunks) == {"make": ["A"]}


# --- Status wording -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, fields, text",
    [
        ("passed", {"score": 5}, "✅ Passed review (score 5/5)"),
        (
            "below_threshold",
            {"score": 3},
            "⚠️ Below the quality bar: best score 3/5 after one revision",
        ),
        ("unreviewed", {}, "⚠️ Not reviewed: the reviewer could not score this summary"),
        ("too_large", {}, "⛔ Not summarized: this part is too long to summarize in one piece"),
        ("api_failure", {}, "⛔ Not summarized: the AI service could not be reached"),
        ("component_failure", {}, "⛔ Not summarized: an internal error occurred"),
    ],
)
def test_status_text(status, fields, text):
    assert status_text(outcome(status, **fields)) == text


def test_rule_problems_are_described_in_plain_words():
    o = outcome(
        "passed_with_rule_errors",
        score=4,
        rule_errors=[Issue(code="pm_jargon", message="…"), Issue(code="unknown_name", message="…")],
    )
    assert status_text(o) == (
        "⚠️ Passed review (score 4/5), but the summary contains programming terms "
        "and mentions names that don't appear in the code"
    )


# --- Function-level Markdown ----------------------------------------------------------------


def test_function_level_has_one_section_per_part_with_status_and_fields():
    text = render(run_sample(Role.QA, "function", mixed(Role.QA)), "md")
    assert text.startswith("# KnowCode summary: sample_cart.py\n")
    assert "Reader: QA Engineer · one summary per part · 4 parts (3 summarized)" in text
    assert "## `round_price` · function · lines 15-17\n\n✅ Passed review (score 4/5)" in text
    assert "## `CartItem` · class · lines 20-27\n\n⚠️ Below the quality bar" in text
    assert (
        "## `Cart` · class · lines 30-49\n\n"
        "⛔ Not summarized: the model's replies could not be used" in text
    )
    assert "**What it does:** Rounds a price up" in text
    for field in ("Expected behavior", "Boundary values", "Failure modes"):
        assert f"**{field}**\n- Prices are rounded up" in text
    assert "**Unclear from the code**\n- Whether negative prices are allowed." in text


def test_pm_headings_have_no_backticks_or_kind_words():
    text = render(run_sample(Role.PM, "function", all_pass(Role.PM)), "md")
    assert "## round_price · lines 15-17" in text
    assert "`" not in text
    assert "**Business rules**" in text and "**What users experience**" in text


def test_technical_detail_for_failures_is_shown_to_developers_but_not_pms():
    source = "def huge():\n" + "".join(f"    x{i} = {i}\n" for i in range(200))
    result = preprocess(source, 200)
    for role, detail_shown in [(Role.DEVELOPER, True), (Role.PM, False)]:
        report = build_report(
            "big.py", role, "function", result, summarize_file(result, role, FakeLLMClient([]))
        )
        text = render(report, "md")
        assert "too long to summarize in one piece" in text
        assert ("over the limit of 200" in text) is detail_shown


# --- File-level Markdown ------------------------------------------------------------------------


def test_file_level_has_the_parts_table_fields_across_the_file_and_connections():
    text = render(run_sample(Role.QA, "file", mixed(Role.QA)), "md")
    assert text.startswith("# KnowCode file overview: sample_cart.py\n")
    assert "| Part | Kind | Lines | What it does | Status |" in text
    assert (
        "| round_price | function | lines 15-17 "
        "| Rounds a price up to the nearest cent. | ✅ passed (4/5) |" in text
    )
    assert "| Cart | class | lines 30-49 | — | ⛔ not summarized |" in text
    assert "## What each part does" in text
    for field in ("Expected behavior", "Boundary values", "Failure modes"):
        assert f"## {field} across the file\n\n- **round_price:** Prices are rounded up" in text
    assert "## How the parts connect\n\n- `CartItem` uses `round_price`" in text
    assert "**Constants:** `MAX_ITEMS = 50`, `DISCOUNT_THRESHOLD = 100.0`" in text
    assert (
        "## Unclear from the code\n\n- **round_price:** Whether negative prices are allowed."
        in text
    )
    assert (
        "## Not summarized\n\n- **Cart** (lines 30-49): the model's replies could not be used"
        in text
    )


def test_pm_file_level_leaves_out_code_sections_and_programming_terms():
    script = mixed(Role.PM)
    text = render(run_sample(Role.PM, "file", script), "md")
    assert "| Part | Lines | What it does | Status |" in text
    assert "How the parts connect" not in text
    assert "File setup" not in text and "import math" not in text
    assert "`" not in text
    assert not PM_FORBIDDEN_TERMS.search(text), PM_FORBIDDEN_TERMS.search(text)
    assert "## Business rules across the file" in text


def test_empty_field_across_the_file_says_nothing_noted():
    report = run_sample(Role.QA, "file", mixed(Role.QA))
    for o in report.outcomes:
        if o.summary:
            o.summary.highlights["failure_modes"].clear()
    assert "## Failure modes across the file\n\n_Nothing noted._" in render(report, "md")


def test_table_cells_escape_pipes_and_keep_decimals_in_the_first_sentence():
    does = "Rounds 1.234 up to 1.24 | never down. Second sentence."
    script = all_pass(Role.QA)
    script[0] = summary(Role.QA, "round_price", "function", does=does)
    text = render(run_sample(Role.QA, "file", script), "md")
    assert "| Rounds 1.234 up to 1.24 \\| never down. |" in text


def test_file_summary_slot_is_empty_by_default_and_shown_when_filled():
    report = run_sample(Role.QA, "file", all_pass(Role.QA))
    assert "## Overview" not in render(report, "md")
    report.file_summary = "This file models a shopping cart."
    assert "## Overview\n\nThis file models a shopping cart." in render(report, "md")


# --- JSON ---------------------------------------------------------------------------------------


def test_function_level_json_lists_every_part_with_status_and_summary():
    data = json.loads(render(run_sample(Role.QA, "function", mixed(Role.QA)), "json"))
    assert data["role"] == "qa" and data["reader"] == "QA Engineer"
    assert data["granularity"] == "function"
    assert [p["name"] for p in data["parts"]] == ["round_price", "CartItem", "Cart", "export_cart"]
    first, cart = data["parts"][0], data["parts"][2]
    assert first["status"] == "passed" and first["score"] == 4
    assert first["summary"]["highlights"]["boundary_values"] == [
        "Prices are rounded up to the next cent."
    ]
    assert cart["summary"] is None and cart["failure_category"] == "output"
    assert "by_field" not in data


def test_file_level_json_adds_grouped_fields_connections_and_setup():
    data = to_json(run_sample(Role.QA, "file", mixed(Role.QA)))
    assert data["file_summary"] is None
    assert set(data["by_field"]) == {"expected_behavior", "boundary_values", "failure_modes"}
    assert data["by_field"]["boundary_values"][0] == {
        "part": "round_price",
        "item": "Prices are rounded up to the next cent.",
    }
    assert data["connections"]["Cart"] == ["round_price", "CartItem"]
    assert data["setup"]["constants"] == ["MAX_ITEMS = 50", "DISCOUNT_THRESHOLD = 100.0"]
    assert data["unclear_from_code"] == [
        {"part": "round_price", "item": "Whether negative prices are allowed."}
    ]
    assert data["not_summarized"][0]["part"] == "Cart"


def test_pm_file_level_json_has_no_code_sections():
    data = to_json(run_sample(Role.PM, "file", all_pass(Role.PM)))
    assert "connections" not in data and "setup" not in data
    assert set(data["by_field"]) == {"business_rules", "user_facing_behavior"}
