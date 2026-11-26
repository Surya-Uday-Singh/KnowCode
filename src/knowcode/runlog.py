"""Save everything about a run to disk, so every result can be traced,
checked and analysed later.

Each run gets its own folder named after its start time and what was run,
e.g. runs/20261123-153012-a1b2-qa/:

    run.json      what was run and how it went: inputs, every setting (never
                  the API key), the KnowCode version and git commit, a
                  fingerprint of the source, status, and totals for calls,
                  tokens, time and estimated cost
    parts.jsonl   one JSON line per part, written as soon as the part
                  finishes: every prompt, every generator and critic reply
                  (including the critic's reasoning), every validator
                  result, the part's status and the summary kept
    source.py     the exact code that was summarized
    output.md     the result as the user received it (output.json for --format json)

Human ratings (relevance, coverage, correctness) are not part of the run
log; they are collected separately and refer back to each part's task_id.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import knowcode
from knowcode.config import Settings
from knowcode.llm import LLMReply, estimate_cost_cny

RunStatus = Literal["running", "finished", "stopped", "interrupted"]

# What kind of summaries a run produced. Spelled out so a log explains itself
# without the design document.
ROLE_SPECIFIC = "role_specific"
ROLE_SPECIFIC_DESCRIPTION = "Summary written for the chosen role (Condition B in the design doc)."


class LoggedPart(Protocol):
    """What the log needs from one part's result. The pipeline's ChunkOutcome
    has all of this; other runners (e.g. experiments) can log their own
    result type as long as it does too."""

    chunk_name: str
    start_line: int

    @property
    def failure_category(self) -> str | None: ...

    def replies(self) -> list[LLMReply]: ...

    def model_dump(self, *, mode: str) -> dict[str, Any]: ...


class RunLog:
    """The log folder for one run. Create it when the run starts, call
    `add_part` as each part finishes and `finish` at the end."""

    def __init__(
        self,
        root: Path,
        *,
        inputs: dict[str, Any],
        settings: Settings,
        source_text: str,
        condition: str = ROLE_SPECIFIC,
        condition_description: str = ROLE_SPECIFIC_DESCRIPTION,
        label: str | None = None,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ):
        """`label` is added to the folder name (e.g. the role, "qa") so runs
        can be told apart at a glance in the runs folder."""
        self._now = now
        self.started_at = now()
        self.run_id, self.path = _new_run_folder(root, self.started_at, label)
        self._task_ids: dict[tuple[str, int], str] = {}
        self._written: set[tuple[str, int]] = set()

        (self.path / "source.py").write_text(source_text, encoding="utf-8")
        self._run: dict[str, Any] = {
            "run_id": self.run_id,
            "condition": condition,
            "condition_description": condition_description,
            "status": "running",
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": None,
            "knowcode_version": knowcode.__version__,
            "git_commit": _git_commit(),
            "inputs": inputs,
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "settings": _settings_record(settings),
        }
        self._write_run()

    def task_id(self, outcome: LoggedPart) -> str:
        """A stable id for one part in this run, e.g. "20261123-153012-a1b2/Cart.add"."""
        key = (outcome.chunk_name, outcome.start_line)
        if key not in self._task_ids:
            task_id = f"{self.run_id}/{outcome.chunk_name}"
            # Two definitions can share a name (a function redefined later in the file).
            if task_id in self._task_ids.values():
                task_id += f"@{outcome.start_line}"
            self._task_ids[key] = task_id
        return self._task_ids[key]

    def add_part(self, outcome: LoggedPart) -> None:
        """Append one part's full record to parts.jsonl."""
        key = (outcome.chunk_name, outcome.start_line)
        if key in self._written:
            return
        self._written.add(key)
        record = {
            "task_id": self.task_id(outcome),
            "failure_category": outcome.failure_category,
            **_part_totals([outcome]),
            **outcome.model_dump(mode="json"),
        }
        with (self.path / "parts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(
        self,
        status: RunStatus,
        outcomes: list[LoggedPart],
        output_text: str | None = None,
        output_format: str = "md",
        error: str | None = None,
    ) -> None:
        """Record how the run ended. Parts not yet written (e.g. ones the
        preprocessor refused) are added first."""
        for outcome in outcomes:
            self.add_part(outcome)
        if output_text is not None:
            (self.path / f"output.{output_format}").write_text(output_text, encoding="utf-8")

        ended_at = self._now()
        counts: dict[str, int] = {}
        for outcome in outcomes:
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        self._run.update(
            {
                "status": status,
                "ended_at": ended_at.isoformat(timespec="seconds"),
                "wall_seconds": round((ended_at - self.started_at).total_seconds(), 1),
                "error": error,
                "parts": len(outcomes),
                "status_counts": counts,
                "totals": _part_totals(outcomes),
                "task_ids": [self.task_id(o) for o in outcomes],
            }
        )
        self._write_run()

    def _write_run(self) -> None:
        text = json.dumps(self._run, indent=2, ensure_ascii=False) + "\n"
        (self.path / "run.json").write_text(text, encoding="utf-8")


def _new_run_folder(root: Path, started_at: datetime, label: str | None) -> tuple[str, Path]:
    """Make a folder named after the start time plus a short random suffix,
    so two runs started in the same second never share a folder, plus the
    label if there is one."""
    root.mkdir(parents=True, exist_ok=True)
    # Keep the label safe to use in a folder name.
    suffix = f"-{re.sub(r'[^a-z0-9_]+', '-', label.lower()).strip('-')}" if label else ""
    while True:
        run_id = f"{started_at:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}{suffix}"
        path = root / run_id
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return run_id, path


def _settings_record(settings: Settings) -> dict[str, Any]:
    record = settings.model_dump(mode="json", exclude={"deepseek_api_key"})
    record["critic_model"] = settings.critic_model
    return record


def _part_totals(outcomes: list[LoggedPart]) -> dict[str, Any]:
    replies = [reply for outcome in outcomes for reply in outcome.replies()]

    def total(field: str) -> int:
        return sum(getattr(reply, field) or 0 for reply in replies)

    return {
        "llm_calls": len(replies),
        "prompt_tokens": total("prompt_tokens"),
        "cached_prompt_tokens": total("cached_prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "llm_seconds": round(sum(reply.elapsed_seconds for reply in replies), 1),
        "estimated_cost_cny": round(sum(estimate_cost_cny(reply) or 0 for reply in replies), 5),
    }


def _git_commit() -> dict[str, Any] | None:
    """The commit of the KnowCode code that ran, and whether it had
    uncommitted changes. None if KnowCode isn't running from a git checkout."""
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=here, capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()

    try:
        return {
            "hash": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return None
