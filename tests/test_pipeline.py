"""Tests for the pipeline loop. Every LLM reply comes from a scripted
FakeLLMClient, so each scenario is exact and nothing calls the API.

Script order for one chunk: generator reply, then the critic reads for that
draft (2 by default), then the retry's generator reply and its critic reads.
"""

import json
import os
from pathlib import Path

import pytest

from knowcode.config import Settings
from knowcode.llm import (
    APIFailure,
    ContentFilteredError,
    EmptyResponseError,
    FakeLLMClient,
    LLMReply,
    LLMSetupError,
)
from knowcode.pipeline import summarize_chunk, summarize_file
from knowcode.preprocessor import preprocess, preprocess_file
from knowcode.prompts import ROLES, Role

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cart.py"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("KNOWCODE_", "DEEPSEEK_")):
            monkeypatch.delenv(name)


@pytest.fixture(scope="module")
def cart():
    return preprocess_file(SAMPLE, 100_000)


def settings(**overrides):
    return Settings(_env_file=None, **overrides)


def summary(role=Role.QA, name="round_price", kind="function", note="", **highlights):
    """A summary that passes the validator for round_price."""
    fields = {
        field: ["Prices are always rounded up to the next cent."] for field in ROLES[role].fields
    }
    fields.update(highlights)
    return json.dumps(
        {
            "name": name,
            "kind": kind,
            "what_it_does": f"Rounds a price up to the nearest cent. It never rounds down.{note}",
            "highlights": fields,
            "unclear_from_code": [],
        }
    )


INVENTED_NAME = summary(failure_modes=["Stores the result in `self.discount_rate`."])
PM_JARGON = summary(Role.PM, business_rules=["The function always rounds up."])


def verdict(score, critique=""):
    return json.dumps({"score": score, "pass": score >= 4, "critique": critique})


def run(cart, script, role=Role.QA, chunk="round_price", **config):
    client = FakeLLMClient(script)
    target = next(c for c in cart.chunks if c.name == chunk)
    outcome = summarize_chunk(target, role, cart.context, client, settings(**config))
    assert client.remaining == 0, "the script had unused replies"
    return outcome, client


def kinds(client):
    return [call for _, call in client.calls]


def retry_prompt(client):
    """The user message of the second generator call."""
    generator_calls = [messages for messages, call in client.calls if call == "generator"]
    return generator_calls[1][1]["content"]


# --- Passing ---------------------------------------------------------------------


def test_draft_passing_both_critic_reads_is_done_after_three_calls(cart):
    outcome, client = run(cart, [summary(), verdict(5), verdict(4)])
    assert outcome.status == "passed"
    assert outcome.failure_category is None
    assert outcome.score == 4  # the lower of the two reads
    assert outcome.best_draft == 1 and not outcome.retry_used
    assert outcome.summary.name == "round_price"
    assert kinds(client) == ["generator", "critic", "critic"]
    assert [read.score for read in outcome.drafts[0].reads] == [5, 4]


def test_both_critic_reads_see_the_same_prompt(cart):
    _, client = run(cart, [summary(), verdict(5), verdict(5)])
    critic_prompts = [messages for messages, call in client.calls if call == "critic"]
    assert critic_prompts[0] == critic_prompts[1]
    assert summary() in critic_prompts[0][1]["content"]


# --- Critic-driven retry ----------------------------------------------------------


def test_one_failing_read_triggers_the_retry_with_its_critique(cart):
    outcome, client = run(
        cart,
        [
            summary(),
            verdict(5),
            verdict(3, "1. round_price(1.1) returns 1.11, not 1.1."),
            summary(note=" Values like 1.1 become 1.11."),
            verdict(4),
            verdict(5),
        ],
    )
    assert outcome.status == "passed"
    assert outcome.best_draft == 2 and outcome.score == 4
    assert outcome.drafts[0].score == 3
    assert outcome.drafts[1].retry_reason == "critic"
    prompt = retry_prompt(client)
    assert "Your previous draft:\n" + summary() in prompt
    assert "A reviewer found these problems:\n1. round_price(1.1) returns 1.11" in prompt


def test_failing_after_the_retry_keeps_the_better_draft(cart):
    outcome, client = run(
        cart,
        [
            summary(),
            verdict(3, "1. Vague."),
            verdict(3),
            summary(note=" Later."),
            verdict(2, "1. Worse."),
            verdict(4),
        ],
    )
    assert outcome.status == "below_threshold"
    assert outcome.failure_category == "quality"
    assert outcome.best_draft == 1 and outcome.score == 3
    assert len(client.calls) == 6  # the doc's 2 drafts, each read twice


def test_tie_in_score_keeps_the_later_draft(cart):
    outcome, _ = run(
        cart,
        [
            summary(),
            verdict(3, "1. a"),
            verdict(3),
            summary(note=" B."),
            verdict(3, "1. b"),
            verdict(3),
        ],
    )
    assert outcome.status == "below_threshold"
    assert outcome.best_draft == 2


def test_failing_critique_without_details_still_gives_the_retry_feedback(cart):
    _, client = run(
        cart, [summary(), verdict(2, ""), verdict(5), summary(), verdict(5), verdict(5)]
    )
    assert "scored this draft 2/5 but gave no details" in retry_prompt(client)


def test_no_quality_retries_means_one_draft_only(cart):
    outcome, client = run(
        cart, [summary(), verdict(2, "1. Bad."), verdict(2)], max_quality_retries=0
    )
    assert outcome.status == "below_threshold"
    assert len(outcome.drafts) == 1
    assert kinds(client) == ["generator", "critic", "critic"]


# --- Validator-driven retry -----------------------------------------------------------


def test_unusable_draft_skips_the_critic_and_retries_with_the_validator_message(cart):
    outcome, client = run(cart, ["Here is my summary!", summary(), verdict(5), verdict(5)])
    assert outcome.status == "passed"
    assert kinds(client) == ["generator", "generator", "critic", "critic"]
    assert outcome.drafts[0].reads == []
    assert outcome.drafts[1].retry_reason == "validator"
    assert (
        "An automatic check found these problems:\n1. Your previous reply was not valid JSON"
        in retry_prompt(client)
    )


def test_truncated_draft_is_retried_with_a_request_to_be_shorter(cart):
    cut_off = LLMReply(
        call="generator",
        model="fake",
        content='{"name": "round_price", "what_it',
        finish_reason="length",
        elapsed_seconds=0.0,
        attempts=1,
    )
    outcome, client = run(cart, [cut_off, summary(), verdict(4), verdict(4)])
    assert outcome.status == "passed"
    assert "cut off at the length limit" in retry_prompt(client)


def test_two_unusable_drafts_are_a_format_failure(cart):
    outcome, client = run(cart, ["nope", '{"name": "round_price"}'])
    assert outcome.status == "format_failure"
    assert outcome.failure_category == "output"
    assert outcome.summary is None and outcome.best_draft is None
    assert kinds(client) == ["generator", "generator"]


def test_rule_errors_and_critic_failure_go_into_one_combined_retry(cart):
    outcome, client = run(
        cart,
        [
            INVENTED_NAME,
            verdict(3, "1. Missing the 1.1 case."),
            verdict(4),
            summary(),
            verdict(5),
            verdict(5),
        ],
    )
    assert outcome.status == "passed"
    assert outcome.drafts[1].retry_reason == "both"
    prompt = retry_prompt(client)
    assert "A reviewer and an automatic check found these problems:" in prompt
    assert "From the reviewer:\n1. Missing the 1.1 case." in prompt
    assert "From the automatic checks:\n1. The summary mentions names" in prompt
    assert "`discount_rate`" in prompt


def test_rule_errors_alone_trigger_a_validator_retry_even_if_the_critic_passed(cart):
    outcome, client = run(
        cart, [INVENTED_NAME, verdict(5), verdict(4), summary(), verdict(5), verdict(4)]
    )
    assert outcome.status == "passed"
    assert outcome.drafts[1].retry_reason == "validator"
    assert "A reviewer" not in retry_prompt(client)


def test_rule_errors_left_after_the_retry_are_kept_and_labelled(cart):
    outcome, _ = run(
        cart, [PM_JARGON, verdict(5), verdict(4), PM_JARGON, verdict(4), verdict(4)], Role.PM
    )
    assert outcome.status == "passed_with_rule_errors"
    assert outcome.failure_category is None
    assert outcome.summary is not None  # kept, not discarded
    assert [e.code for e in outcome.rule_errors] == ["pm_jargon"]
    assert outcome.best_draft == 2  # same score and errors: the later draft


# --- Critic problems -------------------------------------------------------------------------


def test_unusable_critic_reply_is_asked_for_once_more(cart):
    outcome, client = run(cart, [summary(), "not a verdict", verdict(4), verdict(5)])
    assert outcome.status == "passed"
    first_read = outcome.drafts[0].reads[0]
    assert len(first_read.replies) == 2 and first_read.score == 4
    assert kinds(client) == ["generator", "critic", "critic", "critic"]


def test_critic_failing_on_every_read_leaves_the_draft_unreviewed(cart):
    outcome, client = run(cart, [summary(), "x", '{"score": 9}', "{}", '{"score": "high"}'])
    assert outcome.status == "unreviewed"
    assert outcome.score is None and outcome.summary is not None
    assert len(outcome.drafts) == 1  # no feedback to retry with
    assert len(client.calls) == 5


def test_critic_api_failure_loses_that_read_without_asking_again(cart):
    outcome, client = run(cart, [summary(), APIFailure("down", attempts=4), verdict(4)])
    assert outcome.status == "passed"
    assert outcome.drafts[0].reads[0].error.startswith("APIFailure")
    assert outcome.score == 4  # from the read that worked
    assert len(client.calls) == 3


def test_critic_api_failure_on_both_reads_is_unreviewed(cart):
    outcome, _ = run(
        cart, [summary(), APIFailure("down", attempts=4), APIFailure("down", attempts=4)]
    )
    assert outcome.status == "unreviewed"


def test_scored_draft_beats_an_unreviewed_one(cart):
    outcome, _ = run(
        cart,
        [summary(), verdict(3, "1. Vague."), verdict(3), summary(note=" B."), "x", "y", "x", "y"],
    )
    assert outcome.drafts[1].score is None
    assert outcome.best_draft == 1
    assert outcome.status == "below_threshold"


def test_unreviewed_draft_with_rule_errors_is_still_retried(cart):
    outcome, client = run(
        cart, [INVENTED_NAME, "x", "y", "x", "y", summary(), verdict(4), verdict(4)]
    )
    assert outcome.status == "passed"
    assert outcome.drafts[1].retry_reason == "validator"


def test_calls_per_chunk_are_bounded_even_when_everything_goes_wrong(cart):
    # Rule errors keep the retry going while every critic reply is unusable:
    # 2 drafts x (1 generator + 2 reads x 2 asks) = 10 calls, never more.
    outcome, client = run(cart, [INVENTED_NAME, *["x"] * 4, INVENTED_NAME, *["x"] * 4])
    assert len(client.calls) == 10
    assert len(outcome.drafts) == 2
    assert outcome.status == "unreviewed"


# --- Number of critic reads ------------------------------------------------------------------


def test_one_critic_read_matches_the_design_doc(cart):
    outcome, client = run(cart, [summary(), verdict(4)], critic_runs=1)
    assert outcome.status == "passed"
    assert kinds(client) == ["generator", "critic"]


@pytest.mark.parametrize(
    "scores, combined, status",
    [([3, 4, 5], 4, "passed"), ([3, 3, 4], 3, "below_threshold"), ([5, 2, 5], 5, "passed")],
)
def test_three_critic_reads_use_the_middle_score(cart, scores, combined, status):
    script = [summary(), *[verdict(s, "1. Problem." if s < 4 else "") for s in scores]]
    if status != "passed":
        script += [summary(), *[verdict(s, "1. Problem." if s < 4 else "") for s in scores]]
    outcome, _ = run(cart, script, critic_runs=3)
    assert outcome.drafts[0].score == combined
    assert outcome.status == status


def test_critique_comes_from_the_lowest_scoring_read(cart):
    outcome, client = run(
        cart,
        [
            summary(),
            verdict(3, "1. Minor."),
            verdict(2, "1. Two real errors."),
            summary(),
            verdict(4),
            verdict(4),
        ],
    )
    assert outcome.drafts[0].critique == "1. Two real errors."
    assert "1. Two real errors." in retry_prompt(client)
    assert "1. Minor." not in retry_prompt(client)


# --- Generator failures --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error, status, category",
    [
        (APIFailure("down", attempts=4), "api_failure", "api"),
        (EmptyResponseError("empty", attempts=4), "empty_response", "output"),
        (ContentFilteredError("blocked", attempts=1), "content_filtered", "output"),
    ],
)
def test_generator_failure_on_the_first_draft_is_the_chunk_status(cart, error, status, category):
    outcome, client = run(cart, [error])
    assert outcome.status == status
    assert outcome.failure_category == category
    assert kinds(client) == ["generator"]


def test_generator_failure_on_the_retry_keeps_the_first_draft(cart):
    outcome, _ = run(
        cart, [summary(), verdict(3, "1. Vague."), verdict(4), APIFailure("down", attempts=4)]
    )
    assert outcome.status == "below_threshold"
    assert outcome.best_draft == 1
    assert outcome.drafts[1].error_status == "api_failure"


@pytest.mark.parametrize("position", [0, 1])
def test_setup_errors_stop_the_run(cart, position):
    script = [summary(), verdict(5)]
    script[position] = LLMSetupError("HTTP 401: bad key.", attempts=1)
    client = FakeLLMClient(script)
    chunk = cart.chunks[0]
    with pytest.raises(LLMSetupError):
        summarize_chunk(chunk, Role.QA, cart.context, client, settings())


# --- Whole files ---------------------------------------------------------------


def passing_script(role, names_and_kinds):
    script = []
    for name, kind in names_and_kinds:
        script += [summary(role, name=name, kind=kind), verdict(5), verdict(5)]
    return script


def test_file_outcomes_come_in_file_order_with_too_large_chunks_included():
    source = (
        "def small():\n    return 1\n\n"
        + "def huge():\n"
        + "".join(f"    x{i} = {i}\n" for i in range(200))
        + "\ndef tail():\n    return 2\n"
    )
    result = preprocess(source, max_chunk_tokens=200)
    client = FakeLLMClient(
        passing_script(Role.DEVELOPER, [("small", "function"), ("tail", "function")])
    )
    outcomes = summarize_file(result, Role.DEVELOPER, client, settings())
    assert [(o.chunk_name, o.status) for o in outcomes] == [
        ("small", "passed"),
        ("huge", "too_large"),
        ("tail", "passed"),
    ]
    assert outcomes[1].failure_category == "input"
    assert "over the limit" in outcomes[1].error


def test_bug_in_one_chunk_is_recorded_and_the_next_chunk_still_runs():
    result = preprocess("def a():\n    return 1\n\ndef b():\n    return 2\n", 100_000)
    client = FakeLLMClient(
        [RuntimeError("bug in our code"), *passing_script(Role.QA, [("b", "function")])]
    )
    outcomes = summarize_file(result, Role.QA, client, settings())
    assert outcomes[0].status == "component_failure"
    assert outcomes[0].failure_category == "component"
    assert "RuntimeError: bug in our code" in outcomes[0].error
    assert outcomes[1].status == "passed"


def test_setup_error_stops_the_whole_file():
    result = preprocess("def a():\n    return 1\n\ndef b():\n    return 2\n", 100_000)
    client = FakeLLMClient([LLMSetupError("HTTP 402: no balance.", attempts=1)])
    with pytest.raises(LLMSetupError):
        summarize_file(result, Role.QA, client, settings())


def test_outcome_lists_every_reply_received(cart):
    outcome, _ = run(cart, [summary(), "x", verdict(4), verdict(5)])
    replies = outcome.replies()
    assert [r.call for r in replies] == ["generator", "critic", "critic", "critic"]
