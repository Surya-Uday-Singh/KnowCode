"""Send prompts to DeepSeek and return the raw reply.

DeepSeek's API follows the OpenAI format, so the standard `openai` client is
used with DeepSeek's base URL. This module only gets a reply back; it doesn't
read the JSON inside it (the validator does) or decide whether a summary is
good enough (the pipeline does).

Two kinds of call are made, each with its own settings from config:
- "generator": writes a summary (thinking off, temperature 0.3 by default);
- "critic":    scores a summary (thinking on by default).

Technical retries
-----------------
When a request fails for reasons that have nothing to do with the prompt
(timeout, network error, rate limit, server error, empty reply), the exact
same request is sent again after a growing wait: 1s, 2s, 4s by default, plus
up to 25% random extra so parallel requests don't all retry in lockstep.
`max_technical_retries=3` means up to 4 attempts in total. The openai
library's own automatic retries are turned off so this is the only retry
logic and the attempt count we record is accurate.

Some failures are not retried because resending can't help:
- a reply cut off by the token limit comes back with `truncated=True`;
  the caller decides what to do (the same request would be cut off again);
- a reply blocked by the content filter raises ContentFilteredError;
- a bad API key, an empty account or a malformed request raises
  LLMSetupError, which should stop the whole run, since every other request
  would fail the same way.

Try one real generator call and one critic call (needs DEEPSEEK_API_KEY):
    python -m knowcode.llm
"""

from __future__ import annotations

import random
import sys
import time
from collections.abc import Callable, Iterable
from typing import Any, Literal, Protocol

import openai
from pydantic import BaseModel, ConfigDict, computed_field

from knowcode.config import Settings, get_settings
from knowcode.prompts import Message

CallKind = Literal["generator", "critic"]

# finish_reason values meaning the server gave up on the request rather than
# the model finishing its answer. Sending the same request again can work.
_SERVER_GAVE_UP = {"insufficient_system_resource", "aborted"}

# Errors where the request never completed and resending it can work:
# timeouts and network errors, rate limits (429) and server errors (5xx).
_RETRYABLE_ERRORS = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)

# Other HTTP statuses worth retrying: request timeout and conflict.
_RETRYABLE_STATUSES = {408, 409}

# Random extra added to each wait, as a fraction of it.
_MAX_JITTER = 0.25


# ---------------------------------------------------------------------------
# Data passed back to the caller
# ---------------------------------------------------------------------------


class LLMReply(BaseModel):
    """One completed LLM call: the raw text plus what we record about it."""

    model_config = ConfigDict(frozen=True)

    call: CallKind
    model: str  # the model that actually answered, as reported by the API
    content: str  # the raw reply text, normally a JSON object
    reasoning: str | None = None  # the model's reasoning, when thinking mode is on
    finish_reason: str  # "stop" for a complete answer, "length" if cut off
    prompt_tokens: int | None = None
    cached_prompt_tokens: int | None = None  # prompt tokens served from cache (billed less)
    completion_tokens: int | None = None  # includes reasoning tokens
    reasoning_tokens: int | None = None
    elapsed_seconds: float  # first attempt to final reply, including waits between retries
    attempts: int  # 1 when the first attempt worked

    @computed_field
    @property
    def truncated(self) -> bool:
        """True when the reply hit the token limit, so its JSON is incomplete."""
        return self.finish_reason == "length"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """An LLM call that produced no usable reply."""

    # Failure category used when logging: "api" if the call never completed,
    # "output" if the model replied but the reply can't be used.
    category = "api"
    # True when every other request would fail the same way, so the whole run
    # should stop instead of moving on to the next chunk.
    stops_run = False

    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


class APIFailure(LLMError):
    """The request still failed after all technical retries."""


class EmptyResponseError(LLMError):
    """The model kept replying with nothing (or only whitespace)."""

    category = "output"


class ContentFilteredError(LLMError):
    """The provider's content filter blocked the reply."""

    category = "output"


class LLMSetupError(LLMError):
    """A problem with our setup or request (bad API key, no balance, invalid
    parameters) rather than a passing failure."""

    stops_run = True


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Anything that can complete a call. The pipeline depends only on this,
    so tests can pass FakeLLMClient instead of the real one."""

    def complete(self, messages: list[Message], call: CallKind) -> LLMReply: ...


class DeepSeekClient:
    """Calls the DeepSeek API, with technical retries."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: Any = None,  # tests pass one with a fake network layer
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,  # returns a number in [0, 1)
    ):
        self.settings = settings or get_settings()
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self._client = openai.OpenAI(
            api_key=self.settings.require_api_key(),
            base_url=self.settings.deepseek_base_url,
            timeout=self.settings.request_timeout_seconds,
            max_retries=0,  # retries are handled below, not hidden inside the library
            http_client=http_client,
        )

    def complete(self, messages: list[Message], call: CallKind) -> LLMReply:
        """Send `messages` and return the reply, retrying technical failures.

        Raises APIFailure or EmptyResponseError when retries run out,
        ContentFilteredError, or LLMSetupError (which should stop the run).
        """
        request = self.request_for(messages, call)
        max_attempts = self.settings.max_technical_retries + 1
        started = self._clock()
        problem = ""
        last_was_empty = False

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self._sleep(self._retry_delay(attempt - 1))

            try:
                response = self._client.chat.completions.create(**request)
            except _RETRYABLE_ERRORS as error:
                problem, last_was_empty = _describe(error), False
                continue
            except openai.APIStatusError as error:
                if error.status_code in _RETRYABLE_STATUSES:
                    problem, last_was_empty = _describe(error), False
                    continue
                raise LLMSetupError(_setup_message(error), attempt) from error

            choice = response.choices[0] if response.choices else None
            finish_reason = (choice.finish_reason if choice else None) or ""
            content = (choice.message.content if choice else None) or ""

            if finish_reason in _SERVER_GAVE_UP:
                problem, last_was_empty = f"the server ended the request ({finish_reason})", False
                continue
            if finish_reason == "content_filter":
                raise ContentFilteredError("The reply was blocked by the content filter.", attempt)
            # A cut-off reply can be blank (all its tokens went on reasoning).
            # It's still returned, flagged as truncated, rather than retried.
            if finish_reason != "length" and not content.strip():
                problem, last_was_empty = "the reply was empty", True
                continue

            return self._reply(call, response, content, attempt, self._clock() - started)

        message = f"No usable reply after {max_attempts} attempt(s); last problem: {problem}."
        if last_was_empty:
            raise EmptyResponseError(message, max_attempts)
        raise APIFailure(message, max_attempts)

    def request_for(self, messages: list[Message], call: CallKind) -> dict[str, Any]:
        """The keyword arguments sent to the API for this kind of call."""
        s = self.settings
        if call == "generator":
            model, thinking = s.deepseek_model, s.generator_thinking
            temperature, max_tokens = s.generator_temperature, s.generator_max_tokens
        else:
            model, thinking = s.critic_model, s.critic_thinking
            temperature, max_tokens = s.critic_temperature, s.critic_max_tokens

        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
            # DeepSeek-specific, so it goes in the request body as an extra field.
            "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
        }
        # The API ignores temperature in thinking mode; leave it out so the
        # request shows what actually applies.
        if not thinking:
            request["temperature"] = temperature
        return request

    def _retry_delay(self, retry_number: int) -> float:
        base = self.settings.retry_base_delay_seconds * 2 ** (retry_number - 1)
        return base * (1 + _MAX_JITTER * self._jitter())

    @staticmethod
    def _reply(
        call: CallKind, response: Any, content: str, attempts: int, elapsed: float
    ) -> LLMReply:
        choice = response.choices[0]
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        return LLMReply(
            call=call,
            model=response.model,
            content=content,
            # DeepSeek returns the reasoning next to the answer, as an extra field.
            reasoning=getattr(choice.message, "reasoning_content", None),
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else None,
            cached_prompt_tokens=getattr(usage, "prompt_cache_hit_tokens", None),
            completion_tokens=usage.completion_tokens if usage else None,
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
            elapsed_seconds=elapsed,
            attempts=attempts,
        )


class FakeLLMClient:
    """Stands in for DeepSeekClient in tests: plays back a script instead of
    calling the API, and records every call it receives.

    Each script item is used for one call, in order:
    - a string becomes a complete reply with that text;
    - an LLMReply is returned as it is (e.g. a truncated one);
    - an exception is raised (e.g. APIFailure).
    """

    def __init__(self, script: Iterable[str | LLMReply | Exception]):
        self._script = list(script)
        self.calls: list[tuple[list[Message], CallKind]] = []

    @property
    def remaining(self) -> int:
        return len(self._script)

    def complete(self, messages: list[Message], call: CallKind) -> LLMReply:
        self.calls.append((messages, call))
        if not self._script:
            raise AssertionError(f"FakeLLMClient got an unexpected {call} call: script is empty")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return LLMReply(
                call=call,
                model="fake",
                content=item,
                finish_reason="stop",
                elapsed_seconds=0.0,
                attempts=1,
            )
        return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _describe(error: openai.APIError) -> str:
    if isinstance(error, openai.APITimeoutError):
        return "the request timed out"
    if isinstance(error, openai.APIConnectionError):
        return "could not connect to the API"
    if isinstance(error, openai.APIStatusError):
        return f"HTTP {error.status_code} from the API"
    return type(error).__name__


def _setup_message(error: openai.APIStatusError) -> str:
    hints = {
        401: "the API key was rejected; check DEEPSEEK_API_KEY in .env",
        402: "the DeepSeek account has no balance left; top it up on the DeepSeek platform",
        403: "this API key isn't allowed to make this request",
        404: "the model or URL wasn't found; check DEEPSEEK_MODEL and DEEPSEEK_BASE_URL",
    }
    hint = hints.get(error.status_code, "the API rejected the request as invalid")
    return f"HTTP {error.status_code}: {hint}."


# ---------------------------------------------------------------------------
# Command line: one real generator call and one critic call
# ---------------------------------------------------------------------------

# Approximate deepseek-flash list prices in yuan per million tokens, off-peak,
# as of November 2026. Peak hours cost double. Only used for the estimate
# printed below; check DeepSeek's pricing page for current numbers.
_FLASH_PRICES_CNY = {"cache_hit": 0.02, "cache_miss": 1.0, "output": 4.0}


def estimate_cost_cny(reply: LLMReply) -> float | None:
    if reply.prompt_tokens is None or reply.completion_tokens is None:
        return None
    cached = reply.cached_prompt_tokens or 0
    return (
        cached * _FLASH_PRICES_CNY["cache_hit"]
        + (reply.prompt_tokens - cached) * _FLASH_PRICES_CNY["cache_miss"]
        + reply.completion_tokens * _FLASH_PRICES_CNY["output"]
    ) / 1_000_000


def _print_reply(reply: LLMReply) -> None:
    import json

    print(f"--- {reply.call} reply ({reply.model}) ---")
    if reply.reasoning:
        shown = reply.reasoning if len(reply.reasoning) <= 1500 else reply.reasoning[:1500] + " …"
        print(f"[reasoning]\n{shown}\n")
    print(reply.content)
    try:
        json.loads(reply.content)
        valid = "yes"
    except ValueError:
        valid = "NO"
    cost = estimate_cost_cny(reply)
    print(
        f"\nvalid JSON: {valid} | finish: {reply.finish_reason} | attempts: {reply.attempts}"
        f" | time: {reply.elapsed_seconds:.1f}s"
        f"\ntokens: prompt {reply.prompt_tokens} (cached {reply.cached_prompt_tokens}),"
        f" output {reply.completion_tokens} (reasoning {reply.reasoning_tokens})"
        + (f"\nestimated cost (flash, off-peak): ¥{cost:.5f}" if cost is not None else "")
        + "\n"
    )


def _main(argv: list[str]) -> int:
    import argparse
    from pathlib import Path

    from knowcode.config import MissingApiKeyError
    from knowcode.preprocessor import preprocess_file
    from knowcode.prompts import Role, build_critic_messages, build_generator_messages

    parser = argparse.ArgumentParser(
        description="Make one real generator call and one critic call."
    )
    parser.add_argument("path", type=Path, nargs="?", default=Path("tests/fixtures/sample_cart.py"))
    parser.add_argument("--role", choices=[r.value for r in Role], default=Role.QA.value)
    parser.add_argument("--chunk", help="chunk name (default: the first chunk)")
    args = parser.parse_args(argv)

    result = preprocess_file(args.path)
    chunks = [c for c in result.chunks if args.chunk in (None, c.name)]
    if not chunks:
        print(f"No chunk named {args.chunk!r} in {args.path}.", file=sys.stderr)
        return 1
    chunk, role = chunks[0], Role(args.role)

    try:
        client = DeepSeekClient()
        print(f"Summarizing {chunk.name} for role {role.value}…\n")
        draft = client.complete(
            build_generator_messages(chunk, role, result.imports, result.constants), "generator"
        )
        _print_reply(draft)
        verdict = client.complete(
            build_critic_messages(chunk, role, draft.content, result.imports, result.constants),
            "critic",
        )
        _print_reply(verdict)
    except MissingApiKeyError as error:
        print(error, file=sys.stderr)
        return 1
    except LLMError as error:
        print(f"{type(error).__name__} ({error.category}): {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
