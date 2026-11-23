"""Tests for the LLM client.

The real DeepSeekClient runs against a fake server: httpx2's MockTransport
answers requests inside the test, so the openai library's real request and
error handling is exercised without any network call. Waits between retries
are recorded instead of slept.
"""

import json
import os

import httpx2
import pytest

from knowcode.config import MissingApiKeyError, Settings
from knowcode.llm import (
    APIFailure,
    ContentFilteredError,
    DeepSeekClient,
    EmptyResponseError,
    FakeLLMClient,
    LLMReply,
    LLMSetupError,
)

MESSAGES = [
    {"role": "system", "content": "Respond with a JSON object."},
    {"role": "user", "content": "def f(): pass"},
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Keep a developer's shell settings out of the tests."""
    for name in list(os.environ):
        if name.startswith(("KNOWCODE_", "DEEPSEEK_")):
            monkeypatch.delenv(name)


def completion(
    content='{"ok": true}',
    finish_reason="stop",
    reasoning=None,
    model="deepseek-flash",
):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 200,
            "prompt_cache_hit_tokens": 100,
            "prompt_cache_miss_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 50},
        },
    }


class FakeServer:
    """Answers each request with the next scripted outcome and records what was sent.

    An outcome is a completion dict (HTTP 200), an int (that HTTP status with
    an error body), or an httpx2 exception class (raised as a network error).
    """

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, int):
            return httpx2.Response(outcome, json={"error": {"message": "scripted error"}})
        if isinstance(outcome, type):
            raise outcome("scripted network error", request=request)
        return httpx2.Response(200, json=outcome)


class Recorder:
    """Stands in for time.sleep: records the waits instead of sleeping."""

    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)


def make_client(server, jitter=0.0, **settings):
    sleeps = Recorder()
    ticks = iter(range(0, 1000, 2))  # a clock that moves 2 seconds per reading
    client = DeepSeekClient(
        Settings(_env_file=None, deepseek_api_key="test-key", **settings),
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
        sleep=sleeps,
        clock=lambda: float(next(ticks)),
        jitter=lambda: jitter,
    )
    return client, sleeps


# --- Request parameters ------------------------------------------------------


def test_generator_request_uses_generator_settings_with_thinking_off():
    server = FakeServer(completion())
    client, _ = make_client(server)
    client.complete(MESSAGES, "generator")
    [sent] = server.requests
    assert sent["model"] == "deepseek-flash"
    assert sent["messages"] == MESSAGES
    assert sent["temperature"] == 0.3
    assert sent["max_tokens"] == 1000
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["stream"] is False
    assert sent["thinking"] == {"type": "disabled"}


def test_critic_request_uses_thinking_and_leaves_out_temperature():
    server = FakeServer(completion())
    client, _ = make_client(server)
    client.complete(MESSAGES, "critic")
    [sent] = server.requests
    assert sent["thinking"] == {"type": "enabled"}
    assert "temperature" not in sent  # ignored by the API in thinking mode
    assert sent["max_tokens"] == 12000


def test_critic_with_thinking_off_sends_its_temperature():
    server = FakeServer(completion())
    client, _ = make_client(server, critic_thinking=False)
    client.complete(MESSAGES, "critic")
    assert server.requests[0]["temperature"] == 0.0
    assert server.requests[0]["thinking"] == {"type": "disabled"}


def test_critic_can_use_its_own_model():
    server = FakeServer(completion(), completion())
    client, _ = make_client(server, deepseek_critic_model="deepseek-v4-pro")
    client.complete(MESSAGES, "generator")
    client.complete(MESSAGES, "critic")
    assert [r["model"] for r in server.requests] == ["deepseek-flash", "deepseek-v4-pro"]


def test_missing_api_key_fails_before_any_request():
    with pytest.raises(MissingApiKeyError):
        DeepSeekClient(Settings(_env_file=None))


# --- The reply ---------------------------------------------------------------


def test_reply_carries_text_reasoning_tokens_and_timing():
    server = FakeServer(completion('{"score": 5}', reasoning="The draft is accurate."))
    client, _ = make_client(server)
    reply = client.complete(MESSAGES, "critic")
    assert reply.call == "critic"
    assert reply.model == "deepseek-flash"
    assert reply.content == '{"score": 5}'
    assert reply.reasoning == "The draft is accurate."
    assert reply.finish_reason == "stop"
    assert reply.truncated is False
    assert (reply.prompt_tokens, reply.cached_prompt_tokens) == (120, 100)
    assert (reply.completion_tokens, reply.reasoning_tokens) == (80, 50)
    assert reply.attempts == 1
    assert reply.elapsed_seconds == 2.0


def test_truncated_reply_is_returned_flagged_and_not_retried():
    server = FakeServer(completion('{"name": "f", "what_it', finish_reason="length"))
    client, sleeps = make_client(server)
    reply = client.complete(MESSAGES, "generator")
    assert reply.truncated is True
    assert reply.content == '{"name": "f", "what_it'
    assert len(server.requests) == 1
    assert sleeps.waits == []
    # truncated is included when the reply is saved, e.g. to the run log
    assert reply.model_dump()["truncated"] is True


def test_reply_cut_off_while_reasoning_is_truncated_not_empty():
    server = FakeServer(completion("", finish_reason="length", reasoning="Let me check…"))
    client, _ = make_client(server)
    reply = client.complete(MESSAGES, "critic")
    assert reply.truncated is True
    assert reply.content == ""


# --- Technical retries ---------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        httpx2.ReadTimeout,  # timeout
        httpx2.ConnectError,  # network error
        429,  # rate limit
        500,  # server error
        503,  # server overloaded
        408,  # request timeout
        completion(finish_reason="insufficient_system_resource"),
        completion(finish_reason="aborted"),
        completion(content=""),  # empty reply
        completion(content="   \n "),  # spaces only, which JSON mode can return
    ],
    ids=[
        "timeout",
        "connection",
        "429",
        "500",
        "503",
        "408",
        "insufficient_resource",
        "aborted",
        "empty",
        "blank",
    ],
)
def test_technical_failure_is_retried_with_the_same_request(failure):
    server = FakeServer(failure, completion())
    client, sleeps = make_client(server)
    reply = client.complete(MESSAGES, "generator")
    assert reply.content == '{"ok": true}'
    assert reply.attempts == 2
    assert server.requests[0] == server.requests[1]
    assert sleeps.waits == [1.0]


def test_waits_double_between_attempts():
    server = FakeServer(500, 500, 500, completion())
    client, sleeps = make_client(server)
    reply = client.complete(MESSAGES, "generator")
    assert reply.attempts == 4
    assert sleeps.waits == [1.0, 2.0, 4.0]


def test_jitter_adds_up_to_a_quarter_to_each_wait():
    server = FakeServer(500, 500, completion())
    client, sleeps = make_client(server, jitter=1.0)
    client.complete(MESSAGES, "generator")
    assert sleeps.waits == [1.25, 2.5]


def test_three_retries_means_four_attempts_then_api_failure():
    server = FakeServer(*[httpx2.ReadTimeout] * 4)
    client, sleeps = make_client(server)
    with pytest.raises(APIFailure) as info:
        client.complete(MESSAGES, "generator")
    assert len(server.requests) == 4
    assert len(sleeps.waits) == 3
    assert info.value.attempts == 4
    assert info.value.category == "api"
    assert info.value.stops_run is False
    assert "timed out" in str(info.value)


def test_always_empty_is_an_output_failure():
    server = FakeServer(*[completion(content="")] * 4)
    client, _ = make_client(server)
    with pytest.raises(EmptyResponseError) as info:
        client.complete(MESSAGES, "generator")
    assert info.value.category == "output"
    assert info.value.attempts == 4


def test_retry_count_comes_from_settings():
    server = FakeServer(500)
    client, sleeps = make_client(server, max_technical_retries=0)
    with pytest.raises(APIFailure):
        client.complete(MESSAGES, "generator")
    # One request only: the openai library's own hidden retries are off too.
    assert len(server.requests) == 1
    assert sleeps.waits == []


# --- Failures that are not retried ---------------------------------------------


@pytest.mark.parametrize(
    "status, hint",
    [
        (401, "API key"),
        (402, "no balance"),
        (400, "invalid"),
        (404, "DEEPSEEK_MODEL"),
        (422, "invalid"),
    ],
)
def test_setup_problems_stop_the_run_without_retrying(status, hint):
    server = FakeServer(status)
    client, sleeps = make_client(server)
    with pytest.raises(LLMSetupError) as info:
        client.complete(MESSAGES, "generator")
    assert info.value.stops_run is True
    assert hint in str(info.value)
    assert len(server.requests) == 1
    assert sleeps.waits == []


def test_error_message_never_contains_the_api_key():
    server = FakeServer(401)
    client, _ = make_client(server)
    with pytest.raises(LLMSetupError) as info:
        client.complete(MESSAGES, "generator")
    assert "test-key" not in str(info.value)


def test_content_filter_is_an_output_failure_without_retry():
    server = FakeServer(completion(content="", finish_reason="content_filter"))
    client, _ = make_client(server)
    with pytest.raises(ContentFilteredError) as info:
        client.complete(MESSAGES, "generator")
    assert info.value.category == "output"
    assert len(server.requests) == 1


# --- Fake client for pipeline tests ---------------------------------------------


def test_fake_client_plays_its_script_in_order_and_records_calls():
    truncated = LLMReply(
        call="generator",
        model="fake",
        content='{"name": ',
        finish_reason="length",
        elapsed_seconds=0.0,
        attempts=1,
    )
    fake = FakeLLMClient(['{"a": 1}', truncated, APIFailure("down", attempts=4)])

    first = fake.complete(MESSAGES, "generator")
    assert first.content == '{"a": 1}'
    assert first.call == "generator"
    assert first.truncated is False
    assert fake.complete(MESSAGES, "critic") is truncated
    with pytest.raises(APIFailure):
        fake.complete(MESSAGES, "generator")

    assert [call for _, call in fake.calls] == ["generator", "critic", "generator"]
    assert fake.calls[0][0] == MESSAGES
    assert fake.remaining == 0


def test_fake_client_complains_when_its_script_runs_out():
    fake = FakeLLMClient([])
    with pytest.raises(AssertionError, match="script is empty"):
        fake.complete(MESSAGES, "critic")
