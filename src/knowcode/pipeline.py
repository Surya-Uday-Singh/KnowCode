"""Run the generate → validate → critique → retry loop for each chunk.

For one chunk:

1. The generator writes a draft.
2. The validator checks it. An unusable draft (cut off, not JSON, wrong
   structure) skips the critic: there's nothing to review.
3. A usable draft is read by the critic `critic_runs` times (2 by default).
   With two reads the lower score counts, with three the middle one. A read
   whose reply can't be used is asked again once; if it still fails, or the
   call itself fails, that read is lost. With no usable read the draft is
   "unreviewed".
4. The draft is done if the critic passed it and the validator found no
   rule errors. Otherwise, if the quality retry is unused, the generator is
   asked again with everything that was wrong: the critic's critique, the
   validator's errors, or both in one prompt, so the single retry can fix
   them all.
5. When the retry is used up, the best draft is kept and the chunk gets a
   status (passed, below threshold, unreviewed, ...).

The number of quality retries is capped (1 by default), so the loop always
ends. Technical retries (timeouts, empty replies) happen inside the LLM
client and don't count against it.

Try it on real code (needs DEEPSEEK_API_KEY):
    python -m knowcode.pipeline tests/fixtures/sample_cart.py --role qa --chunk round_price
"""

from __future__ import annotations

import sys
import traceback
from typing import Literal

from pydantic import BaseModel

from knowcode.config import Settings, get_settings
from knowcode.llm import (
    ContentFilteredError,
    EmptyResponseError,
    LLMClient,
    LLMError,
    LLMReply,
)
from knowcode.preprocessor import Chunk, PreprocessResult
from knowcode.prompts import (
    Message,
    ProblemSource,
    Role,
    build_critic_messages,
    build_generator_messages,
    build_retry_messages,
)
from knowcode.validator import (
    Issue,
    Summary,
    SummaryCheck,
    VerdictCheck,
    validate_summary,
    validate_verdict,
)

ChunkStatus = Literal[
    "passed",  # critic passed it and the validator found no rule errors
    "passed_with_rule_errors",  # critic passed it, but rule errors remained after the retry
    "below_threshold",  # the best draft still scored below the pass threshold
    "unreviewed",  # a usable draft, but the critic never produced a usable score
    "format_failure",  # no usable draft (cut off, not JSON, wrong structure)
    "api_failure",  # the generator call failed after its technical retries
    "empty_response",  # the generator kept replying with nothing
    "content_filtered",  # the provider's filter blocked the generator's reply
    "too_large",  # the preprocessor refused the chunk
    "component_failure",  # a bug in our own code
]

# Failure categories used in the results. Statuses not listed aren't failures.
FAILURE_CATEGORY: dict[str, str] = {
    "below_threshold": "quality",
    "format_failure": "output",
    "empty_response": "output",
    "content_filtered": "output",
    "api_failure": "api",
    "too_large": "input",
    "component_failure": "component",
}

# Sent in a retry when the critic failed a draft without saying why.
_NO_CRITIQUE = (
    "The reviewer scored this draft {score}/5 but gave no details. Check every claim against "
    "the code and make sure the summary covers what this reader needs."
)


# ---------------------------------------------------------------------------
# What happened to each chunk
# ---------------------------------------------------------------------------


class CriticRead(BaseModel):
    """One independent critic read of a draft. `replies` has two entries when
    the first reply couldn't be used and the critic was asked again."""

    replies: list[LLMReply] = []
    check: VerdictCheck | None = None  # check of the last reply
    error: str | None = None  # an LLM call that failed outright

    @property
    def score(self) -> int | None:
        if self.check is None or self.check.verdict is None:
            return None
        return self.check.verdict.score

    @property
    def critique(self) -> str:
        if self.check is None or self.check.verdict is None:
            return ""
        return self.check.verdict.critique


class Draft(BaseModel):
    """One generator attempt and everything that happened to it."""

    number: int  # 1 for the first draft, 2 after the retry
    retry_reason: ProblemSource | None = None  # why this draft was asked for
    messages: list[Message]  # the full prompt sent to the generator
    reply: LLMReply | None = None
    error: str | None = None  # the generator call failed outright
    error_status: ChunkStatus | None = None
    check: SummaryCheck | None = None
    critic_messages: list[Message] | None = None
    reads: list[CriticRead] = []
    score: int | None = None  # combined critic score; None if unreviewed or unusable
    critique: str = ""  # from the lowest-scoring read

    @property
    def usable(self) -> bool:
        return self.check is not None and self.check.usable

    @property
    def rule_errors(self) -> list[Issue]:
        return list(self.check.errors) if self.usable else []

    def critic_passed(self, threshold: int) -> bool:
        return self.score is not None and self.score >= threshold

    def fully_passed(self, threshold: int) -> bool:
        return self.critic_passed(threshold) and not self.rule_errors


class ChunkOutcome(BaseModel):
    """The result for one chunk: its status, the summary kept (if any), and
    every draft and LLM reply along the way."""

    chunk_name: str
    kind: str
    start_line: int
    end_line: int
    role: Role
    status: ChunkStatus
    best_draft: int | None = None  # number of the draft kept
    summary: Summary | None = None
    score: int | None = None
    rule_errors: list[Issue] = []  # still present in the kept draft
    drafts: list[Draft] = []
    error: str | None = None  # for too_large and component_failure

    @property
    def failure_category(self) -> str | None:
        return FAILURE_CATEGORY.get(self.status)

    @property
    def retry_used(self) -> bool:
        return len(self.drafts) > 1

    def replies(self) -> list[LLMReply]:
        """Every LLM reply received for this chunk, in order."""
        found: list[LLMReply] = []
        for draft in self.drafts:
            if draft.reply is not None:
                found.append(draft.reply)
            for read in draft.reads:
                found.extend(read.replies)
        return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_chunk(
    chunk: Chunk,
    role: Role,
    imports: list[str],
    constants: list[str],
    client: LLMClient,
    settings: Settings | None = None,
) -> ChunkOutcome:
    """Run the full loop for one chunk.

    Raises LLMError only when it should stop the whole run (bad API key, no
    balance); every other failure becomes the chunk's status.
    """
    settings = settings or get_settings()
    threshold = settings.pass_threshold
    retries_left = settings.max_quality_retries
    drafts: list[Draft] = []
    messages = build_generator_messages(chunk, role, imports, constants)
    reason: ProblemSource | None = None

    while True:
        draft = Draft(number=len(drafts) + 1, retry_reason=reason, messages=messages)
        drafts.append(draft)

        try:
            draft.reply = client.complete(messages, "generator")
        except LLMError as error:
            if error.stops_run:
                raise
            draft.error = f"{type(error).__name__}: {error}"
            draft.error_status = _status_for(error)
            break  # technical retries are already used up; keep any earlier draft

        draft.check = validate_summary(
            draft.reply.content, chunk, role, imports, constants, draft.reply.truncated
        )
        if draft.usable:
            _review(draft, chunk, role, imports, constants, client, settings)

        feedback = _retry_feedback(draft, threshold)
        if feedback is None or retries_left == 0:
            break
        problems, reason = feedback
        retries_left -= 1
        messages = build_retry_messages(
            chunk, role, imports, constants, draft.reply.content, problems, reason
        )

    return _outcome(chunk, role, drafts, threshold)


def summarize_file(
    result: PreprocessResult,
    role: Role,
    client: LLMClient,
    settings: Settings | None = None,
) -> list[ChunkOutcome]:
    """Run every chunk of a preprocessed file, in file order. Chunks the
    preprocessor refused appear as "too_large"; a bug in our own code while
    handling one chunk is recorded as "component_failure" and the run goes on
    with the next chunk."""
    settings = settings or get_settings()
    outcomes: list[ChunkOutcome] = []
    for chunk in result.chunks:
        try:
            outcome = summarize_chunk(
                chunk, role, result.imports, result.constants, client, settings
            )
        except LLMError as error:
            if error.stops_run:
                raise
            outcome = _component_failure(chunk, role, error)
        except Exception as error:  # noqa: BLE001 - recorded as a bug in our code, not raised
            outcome = _component_failure(chunk, role, error)
        outcomes.append(outcome)

    for failure in result.failures:
        outcomes.append(
            ChunkOutcome(
                chunk_name=failure.name,
                kind=failure.kind,
                start_line=failure.start_line,
                end_line=failure.end_line,
                role=role,
                status="too_large",
                error=failure.message,
            )
        )
    outcomes.sort(key=lambda o: o.start_line)
    return outcomes


# ---------------------------------------------------------------------------
# Steps of the loop
# ---------------------------------------------------------------------------


def _review(
    draft: Draft,
    chunk: Chunk,
    role: Role,
    imports: list[str],
    constants: list[str],
    client: LLMClient,
    settings: Settings,
) -> None:
    """Have the critic read the draft `critic_runs` times and combine the scores."""
    assert draft.reply is not None
    draft.critic_messages = build_critic_messages(
        chunk, role, draft.reply.content, imports, constants, settings.pass_threshold
    )
    draft.reads = [
        _read(draft.critic_messages, client, settings.pass_threshold)
        for _ in range(settings.critic_runs)
    ]

    scored = [read for read in draft.reads if read.score is not None]
    if not scored:
        return  # unreviewed
    scores = sorted(read.score for read in scored)
    # One read: its score. Two: the lower one, so the draft passes only if
    # both reads pass it. Three: the middle one (majority).
    draft.score = scores[0] if len(scores) <= 2 else scores[len(scores) // 2]
    lowest = min(scored, key=lambda read: read.score)
    draft.critique = lowest.critique


def _read(messages: list[Message], client: LLMClient, threshold: int) -> CriticRead:
    """One critic read; a reply that can't be used is asked for once more."""
    read = CriticRead()
    for _ in range(2):
        try:
            reply = client.complete(messages, "critic")
        except LLMError as error:
            if error.stops_run:
                raise
            # The client already retried technical failures; don't ask again.
            read.error = f"{type(error).__name__}: {error}"
            return read
        read.replies.append(reply)
        read.check = validate_verdict(reply.content, reply.truncated, threshold)
        if read.check.usable:
            break
    return read


def _retry_feedback(draft: Draft, threshold: int) -> tuple[str, ProblemSource] | None:
    """What the retry prompt should ask the generator to fix, or None if the
    draft needs no retry (it passed, or the critic couldn't score it and the
    validator found nothing)."""
    assert draft.check is not None
    if not draft.usable:
        return draft.check.problems_text(), "validator"

    critic_failed = draft.score is not None and draft.score < threshold
    critic_text = (
        (draft.critique or _NO_CRITIQUE.format(score=draft.score)) if critic_failed else ""
    )
    validator_text = draft.check.problems_text()

    if critic_text and validator_text:
        combined = (
            f"From the reviewer:\n{critic_text}\n\nFrom the automatic checks:\n{validator_text}"
        )
        return combined, "both"
    if critic_text:
        return critic_text, "critic"
    if validator_text:
        return validator_text, "validator"
    return None


def _outcome(chunk: Chunk, role: Role, drafts: list[Draft], threshold: int) -> ChunkOutcome:
    base = dict(
        chunk_name=chunk.name,
        kind=chunk.kind,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        role=role,
        drafts=drafts,
    )
    usable = [draft for draft in drafts if draft.usable]
    if not usable:
        # Nothing to keep. Report what happened to the last attempt.
        last = drafts[-1]
        return ChunkOutcome(**base, status=last.error_status or "format_failure")

    best = max(usable, key=lambda draft: _rank(draft, threshold))
    if best.fully_passed(threshold):
        status: ChunkStatus = "passed"
    elif best.critic_passed(threshold):
        status = "passed_with_rule_errors"
    elif best.score is None:
        status = "unreviewed"
    else:
        status = "below_threshold"

    assert best.check is not None
    return ChunkOutcome(
        **base,
        status=status,
        best_draft=best.number,
        summary=best.check.summary,
        score=best.score,
        rule_errors=best.rule_errors,
    )


def _rank(draft: Draft, threshold: int) -> tuple[bool, int, int, int]:
    """Sort key for choosing the best draft: a fully passing draft first, then
    the higher critic score (a scored draft beats an unreviewed one), then
    fewer rule errors, then the later draft (it was written with feedback)."""
    score = draft.score if draft.score is not None else -1
    return (draft.fully_passed(threshold), score, -len(draft.rule_errors), draft.number)


def _status_for(error: LLMError) -> ChunkStatus:
    if isinstance(error, EmptyResponseError):
        return "empty_response"
    if isinstance(error, ContentFilteredError):
        return "content_filtered"
    return "api_failure"  # APIFailure, or any other call that never completed


def _component_failure(chunk: Chunk, role: Role, error: BaseException) -> ChunkOutcome:
    details = "".join(traceback.format_exception(error)).strip()
    return ChunkOutcome(
        chunk_name=chunk.name,
        kind=chunk.kind,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        role=role,
        status="component_failure",
        error=details,
    )


# ---------------------------------------------------------------------------
# Command line: run the loop on a real file
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    import argparse
    import json
    from pathlib import Path

    from knowcode.config import MissingApiKeyError
    from knowcode.llm import DeepSeekClient, estimate_cost_cny
    from knowcode.preprocessor import InputError, preprocess_file

    parser = argparse.ArgumentParser(description="Summarize a Python file for one role.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--role", choices=[r.value for r in Role], default=Role.QA.value)
    parser.add_argument("--chunk", help="only this chunk (default: every chunk)")
    args = parser.parse_args(argv)

    try:
        result = preprocess_file(args.path)
    except InputError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 1
    if args.chunk:
        result = result.model_copy(
            update={
                "chunks": [c for c in result.chunks if c.name == args.chunk],
                "failures": [f for f in result.failures if f.name == args.chunk],
            }
        )
        if not result.chunks and not result.failures:
            print(f"No chunk named {args.chunk!r} in {args.path}.", file=sys.stderr)
            return 1

    try:
        outcomes = summarize_file(result, Role(args.role), DeepSeekClient())
    except MissingApiKeyError as error:
        print(error, file=sys.stderr)
        return 1
    except LLMError as error:
        print(f"Run stopped. {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    total = 0.0
    for outcome in outcomes:
        replies = outcome.replies()
        cost = sum(estimate_cost_cny(r) or 0 for r in replies)
        seconds = sum(r.elapsed_seconds for r in replies)
        total += cost
        print(f"\n=== {outcome.chunk_name} ({outcome.kind}): {outcome.status} ===")
        for draft in outcome.drafts:
            reads = [read.score for read in draft.reads]
            why = f" (retry: {draft.retry_reason})" if draft.retry_reason else ""
            line = f"draft {draft.number}{why}: "
            if draft.error:
                line += draft.error
            elif not draft.usable:
                line += "unusable: " + ", ".join(e.code for e in draft.check.errors)
            else:
                rule = ", ".join(e.code for e in draft.rule_errors) or "none"
                line += f"critic reads {reads} → {draft.score}; rule errors: {rule}"
            print(line)
            if draft.critique:
                print(f"  critique: {draft.critique}")
        if outcome.error:
            print(outcome.error)
        if outcome.summary:
            print(f"kept draft {outcome.best_draft}:")
            print(json.dumps(outcome.summary.model_dump(), indent=2, ensure_ascii=False))
        print(f"{len(replies)} LLM calls, {seconds:.0f} s, about ¥{cost:.4f}")
    print(f"\nTotal: about ¥{total:.4f} (deepseek-flash off-peak prices)")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
