"""Tests for the generic baseline in the experiment folder: its prompt, its
generator → validator loop (no critic), its output and its run log. Also
checks that the KnowCode tool itself is unchanged: `knowcode summarize`
still only accepts the three roles. A scripted FakeLLMClient stands in
for DeepSeek."""

import io
import json
import os
import re
from pathlib import Path

import pytest

from experiment.baseline import main, summarize_chunk_generic, summarize_file_generic
from experiment.generic import GENERIC_EXAMPLE_SUMMARY, GENERIC_READER
from knowcode.cli import main as knowcode_main
from knowcode.config import Settings
from knowcode.llm import APIFailure, FakeLLMClient, LLMSetupError
from knowcode.preprocessor import FileContext, preprocess, preprocess_file
from knowcode.prompts import (
    EXAMPLE_CONSTANTS,
    EXAMPLE_SOURCE,
    ROLES,
    Role,
    build_critic_messages,
    build_generator_messages,
)
from knowcode.validator import validate_summary

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cart.py"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("KNOWCODE_", "DEEPSEEK_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)  # run logs land in a temporary folder


@pytest.fixture(scope="module")
def cart():
    return preprocess_file(SAMPLE, 100_000)


def settings(**overrides):
    return Settings(_env_file=None, **overrides)


def generic_summary(name="round_price", kind="function", details=None):
    return json.dumps(
        {
            "name": name,
            "kind": kind,
            "what_it_does": "Rounds a price up to the nearest cent. It never rounds down.",
            "highlights": {"key_details": details or ["Uses math.ceil on the value times 100."]},
            "unclear_from_code": [],
        }
    )


def run_chunk(cart, script, chunk="round_price", **config):
    client = FakeLLMClient(script)
    target = next(c for c in cart.chunks if c.name == chunk)
    outcome = summarize_chunk_generic(target, cart.context, client, settings(**config))
    assert client.remaining == 0, "the script had unused replies"
    return outcome, client


# --- The generic prompt ----------------------------------------------------------


def test_generic_prompt_uses_the_same_template_as_the_roles(cart):
    chunk = cart.chunks[0]
    generic = build_generator_messages(chunk, GENERIC_READER, cart.context)
    qa = build_generator_messages(chunk, Role.QA, cart.context)

    def headings(messages):
        return re.findall(r"^## .+$", messages[0]["content"], re.MULTILINE)

    assert headings(generic) == headings(qa)  # same sections, in the same order
    assert generic[1] == qa[1]  # identical user message: same code and file context
    system = generic[0]["content"]
    assert "for one specific reader: a general reader." in system
    assert "No particular job role is assumed." in system
    assert "- key_details: the most important facts about how the code behaves" in system
    assert '"key_details": ["<item>", "<item>"]' in system


def test_generic_prompt_has_no_role_content(cart):
    system = build_generator_messages(cart.chunks[0], GENERIC_READER, cart.context)[0]["content"]
    for spec in ROLES.values():
        assert spec.reader not in system
        for field in spec.fields:
            assert field not in system.split("## Example")[0]
        for rule in spec.style:
            assert rule not in system


def test_generic_rules_are_the_same_as_for_developer_and_qa(cart):
    def rules(role):
        system = build_generator_messages(cart.chunks[0], role, cart.context)[0]["content"]
        return system.split("## Rules for all roles\n", 1)[1].split("\n\n## ", 1)[0]

    assert rules(GENERIC_READER) == rules(Role.QA) == rules(Role.DEVELOPER)


def test_generic_worked_example_passes_the_validator():
    chunk = preprocess(EXAMPLE_SOURCE, 100_000).chunks[0]
    check = validate_summary(
        json.dumps(GENERIC_EXAMPLE_SUMMARY),
        chunk,
        GENERIC_READER,
        FileContext(constants=EXAMPLE_CONSTANTS),
    )
    assert check.ok and check.warnings == []


def test_generic_reader_cannot_be_critiqued(cart):
    with pytest.raises(ValueError, match="no critic rubric"):
        build_critic_messages(cart.chunks[0], GENERIC_READER, "{}", cart.context, 4)


def test_validator_checks_names_in_generic_summaries(cart):
    chunk = next(c for c in cart.chunks if c.name == "round_price")
    text = generic_summary(details=["Stores the result in self.discount_rate."])
    check = validate_summary(text, chunk, GENERIC_READER, cart.context)
    assert [e.code for e in check.errors] == ["unknown_name"]


# --- The loop: generator → validator, no critic -------------------------------------


def test_good_draft_is_done_after_one_generator_call_and_no_critic(cart):
    outcome, client = run_chunk(cart, [generic_summary()])
    assert outcome.status == "generated"
    assert outcome.failure_category is None
    assert [call for _, call in client.calls] == ["generator"]
    assert outcome.summary.highlights == {"key_details": ["Uses math.ceil on the value times 100."]}
    assert outcome.drafts[0].reads == []


def test_unusable_draft_gets_one_retry_with_the_validator_message(cart):
    outcome, client = run_chunk(cart, ["Here is my summary!", generic_summary()])
    assert outcome.status == "generated" and outcome.best_draft == 2
    assert outcome.drafts[1].retry_reason == "validator"
    retry_prompt = client.calls[1][0][1]["content"]
    assert (
        "An automatic check found these problems:\n1. Your previous reply was not valid JSON"
        in (retry_prompt)
    )


def test_rule_errors_get_one_retry_and_are_labelled_if_they_remain(cart):
    invented = generic_summary(details=["Stores the result in self.discount_rate."])
    outcome, client = run_chunk(cart, [invented, invented])
    assert outcome.status == "generated_with_rule_errors"
    assert [e.code for e in outcome.rule_errors] == ["unknown_name"]
    assert outcome.summary is not None  # kept, not discarded
    assert len(client.calls) == 2


def test_two_unusable_drafts_are_a_format_failure(cart):
    outcome, _ = run_chunk(cart, ["nope", '{"name": "round_price"}'])
    assert outcome.status == "format_failure"
    assert outcome.failure_category == "output"
    assert outcome.summary is None


def test_no_quality_retries_means_one_attempt(cart):
    outcome, client = run_chunk(cart, ["nope"], max_quality_retries=0)
    assert outcome.status == "format_failure"
    assert len(client.calls) == 1


def test_generator_failure_on_the_retry_keeps_the_first_draft(cart):
    invented = generic_summary(details=["Stores the result in self.discount_rate."])
    outcome, _ = run_chunk(cart, [invented, APIFailure("down", attempts=4)])
    assert outcome.status == "generated_with_rule_errors" and outcome.best_draft == 1
    assert outcome.drafts[1].error_status == "api_failure"


def test_setup_errors_stop_the_run(cart):
    with pytest.raises(LLMSetupError):
        run_chunk(cart, [LLMSetupError("HTTP 401: bad key.", attempts=1)])


def test_file_run_includes_too_large_parts_and_records_bugs():
    source = "def small():\n    return 1\n\ndef huge():\n" + "".join(
        f"    x{i} = {i}\n" for i in range(200)
    )
    result = preprocess(source, 200)
    outcomes = summarize_file_generic(result, FakeLLMClient([RuntimeError("bug")]), settings())
    assert [(o.chunk_name, o.status) for o in outcomes] == [
        ("small", "component_failure"),
        ("huge", "too_large"),
    ]
    assert "RuntimeError: bug" in outcomes[0].error


# --- The command and its run log ------------------------------------------------------------


def run_command(args, script, stdin=""):
    out, err = io.StringIO(), io.StringIO()
    client = FakeLLMClient(script)
    code = main(
        args,
        stdin=io.StringIO(stdin),
        stdout=out,
        stderr=err,
        client_factory=lambda: client,
        settings=settings(),
    )
    return code, out.getvalue(), err.getvalue(), client


def sample_script():
    parts = [
        ("round_price", "function"),
        ("CartItem", "class"),
        ("Cart", "class"),
        ("export_cart", "function"),
    ]
    return [generic_summary(name, kind) for name, kind in parts]


def test_command_needs_granularity_and_has_no_role_option():
    code, _, err, _ = run_command([str(SAMPLE)], [])
    assert code == 1 and "--granularity" in err
    code, _, err, _ = run_command([str(SAMPLE), "--granularity", "file", "--role", "qa"], [])
    assert code == 1 and "unrecognized arguments: --role qa" in err


def test_function_level_markdown(tmp_path):
    code, out, err, _ = run_command(
        [str(SAMPLE), "--granularity", "function", "--chunk", "round_price"], [generic_summary()]
    )
    assert code == 0
    assert out.startswith("# KnowCode generic baseline:")
    assert "Generic summary (no target role) · one summary per part · 1 part" in out
    assert (
        "## `round_price` · function · lines 15-17\n\n✅ Generated (passed the automatic checks)"
        in out
    )
    assert "**Key details**\n- Uses math.ceil on the value times 100." in out
    assert "[1/1] round_price ... ✅ generated · 1 calls" in err


def test_file_level_markdown_and_json():
    code, out, _, _ = run_command([str(SAMPLE), "--granularity", "file"], sample_script())
    assert code == 0
    assert "## Key details across the file\n\n- **round_price:** Uses math.ceil" in out
    assert "- `Cart` uses `round_price`, `CartItem`" in out
    code, out, _, _ = run_command(
        [str(SAMPLE), "--granularity", "file", "--format", "json"], sample_script()
    )
    data = json.loads(out)
    assert data["condition"] == "generic_baseline"
    assert data["role"] is None and data["reader"] == "general reader"
    assert data["by_field"]["key_details"][0] == {
        "part": "round_price",
        "item": "Uses math.ceil on the value times 100.",
    }


def test_run_log_is_tagged_generic_baseline():
    run_command(
        [str(SAMPLE), "--granularity", "function", "--chunk", "round_price"], [generic_summary()]
    )
    [folder] = list(Path("runs").iterdir())
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}-generic", folder.name)
    run_json = json.loads((folder / "run.json").read_text())
    assert run_json["condition"] == "generic_baseline"
    assert "Generic summary with no target role" in run_json["condition_description"]
    assert run_json["inputs"]["role"] is None
    assert run_json["inputs"]["reader"] == "general reader"
    assert run_json["totals"]["llm_calls"] == 1
    [part] = [json.loads(line) for line in (folder / "parts.jsonl").read_text().splitlines()]
    assert part["task_id"] == f"{folder.name}/round_price"
    assert part["status"] == "generated"
    assert part["drafts"][0]["messages"][0]["content"].startswith("You are an expert")
    assert part["drafts"][0]["reads"] == []  # no critic


def test_stopped_baseline_run_keeps_finished_parts():
    script = [generic_summary(), LLMSetupError("HTTP 402: no balance.", attempts=1)]
    code, _, err, _ = run_command([str(SAMPLE), "--granularity", "file"], script)
    assert code == 2 and "run stopped" in err
    [folder] = list(Path("runs").iterdir())
    assert json.loads((folder / "run.json").read_text())["status"] == "stopped"
    assert len((folder / "parts.jsonl").read_text().splitlines()) == 1


# --- The KnowCode tool is unchanged ------------------------------------------------------------


def test_knowcode_summarize_still_only_accepts_the_three_roles():
    err = io.StringIO()
    code = knowcode_main(
        ["summarize", str(SAMPLE), "--role", "generic", "--granularity", "file"],
        stdout=io.StringIO(),
        stderr=err,
        settings=settings(),
    )
    assert code == 1
    assert "argument --role: invalid choice: 'generic'" in err.getvalue()


def test_knowcode_runs_are_tagged_role_specific():
    verdict = json.dumps({"score": 5, "pass": True, "critique": ""})
    role_summary = json.dumps(
        {
            "name": "round_price",
            "kind": "function",
            "what_it_does": "Rounds a price up to the nearest cent. It never rounds down.",
            "highlights": {f: ["Prices are rounded up."] for f in ROLES[Role.QA].fields},
            "unclear_from_code": [],
        }
    )
    client = FakeLLMClient([role_summary, verdict, verdict])
    knowcode_main(
        [
            "summarize",
            str(SAMPLE),
            "--role",
            "qa",
            "--granularity",
            "function",
            "--chunk",
            "round_price",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        client_factory=lambda: client,
        settings=settings(),
    )
    [folder] = list(Path("runs").iterdir())
    assert folder.name.endswith("-qa")
    assert json.loads((folder / "run.json").read_text())["condition"] == "role_specific"
