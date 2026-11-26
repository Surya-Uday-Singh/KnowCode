"""Tests for run logging: the folder layout, every data field the experiment
needs, partial runs, and that the API key never reaches the logs. Runs go
through the real `knowcode` command with a scripted FakeLLMClient."""

import hashlib
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest

from knowcode.cli import main
from knowcode.config import Settings
from knowcode.llm import FakeLLMClient, LLMReply, LLMSetupError
from knowcode.pipeline import ChunkOutcome
from knowcode.prompts import ROLES, Role
from knowcode.runlog import RunLog

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cart.py"
SECRET = "sk-test-secret-key-123"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("KNOWCODE_", "DEEPSEEK_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


def passing(role, name, kind="function"):
    summary = json.dumps(
        {
            "name": name,
            "kind": kind,
            "what_it_does": "Rounds a price up to the nearest cent. It never rounds down.",
            "highlights": {field: ["Prices are rounded up."] for field in ROLES[role].fields},
            "unclear_from_code": [],
        }
    )
    verdict = LLMReply(
        call="critic",
        model="fake",
        content=json.dumps({"score": 5, "pass": True, "critique": ""}),
        reasoning="Checked every claim against the code.",
        finish_reason="stop",
        prompt_tokens=1000,
        completion_tokens=300,
        reasoning_tokens=250,
        elapsed_seconds=4.0,
        attempts=1,
    )
    return [summary, verdict, verdict]


SAMPLE_PARTS = [
    ("round_price", "function"),
    ("CartItem", "class"),
    ("Cart", "class"),
    ("export_cart", "function"),
]


def run(args, script, settings=None, client=None):
    out, err = io.StringIO(), io.StringIO()
    client = client or FakeLLMClient(script)
    code = main(
        args,
        stdout=out,
        stderr=err,
        client_factory=lambda: client,
        settings=settings or Settings(_env_file=None, deepseek_api_key=SECRET),
    )
    return code, out.getvalue(), err.getvalue()


def the_run_folder(root="runs"):
    [folder] = list(Path(root).iterdir())
    return folder


def full_sample_run(*extra):
    script = [line for name, kind in SAMPLE_PARTS for line in passing(Role.QA, name, kind)]
    return run(["summarize", str(SAMPLE), "--role", "qa", "--granularity", "file", *extra], script)


# --- A finished run ------------------------------------------------------------------


def test_finished_run_writes_the_four_files():
    code, out, err = full_sample_run()
    assert code == 0
    folder = the_run_folder()
    assert sorted(p.name for p in folder.iterdir()) == [
        "output.md",
        "parts.jsonl",
        "run.json",
        "source.py",
    ]
    assert f"Run log: {folder}" in err
    assert (folder / "output.md").read_text() == out  # exactly what the user got
    assert (folder / "source.py").read_text() == SAMPLE.read_text()


def test_run_json_records_inputs_settings_version_and_totals():
    full_sample_run()
    run_json = json.loads((the_run_folder() / "run.json").read_text())
    # start time, random suffix, then the role so runs are easy to tell apart
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}-qa", run_json["run_id"])
    assert run_json["condition"] == "role_specific"
    assert "Summary written for the chosen role" in run_json["condition_description"]
    assert run_json["status"] == "finished"
    assert run_json["started_at"] and run_json["ended_at"]
    assert run_json["inputs"] == {
        "source": str(SAMPLE),
        "role": "qa",
        "granularity": "file",
        "format": "md",
        "chunk": None,
        "output": None,
    }
    assert run_json["source_sha256"] == hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    settings = run_json["settings"]
    assert settings["deepseek_model"] == "deepseek-flash"
    assert settings["critic_model"] == "deepseek-flash"
    assert settings["critic_runs"] == 2 and settings["critic_thinking"] is True
    assert "deepseek_api_key" not in settings
    assert run_json["knowcode_version"]
    commit = run_json["git_commit"]
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit["hash"])
    assert run_json["parts"] == 4
    assert run_json["status_counts"] == {"passed": 4}
    totals = run_json["totals"]
    assert totals["llm_calls"] == 12
    assert totals["completion_tokens"] == 8 * 300 and totals["reasoning_tokens"] == 8 * 250
    assert totals["llm_seconds"] == 32.0
    assert run_json["task_ids"][0] == f"{run_json['run_id']}/round_price"


def test_each_part_line_has_every_field_the_experiment_needs():
    full_sample_run()
    folder = the_run_folder()
    parts = [json.loads(line) for line in (folder / "parts.jsonl").read_text().splitlines()]
    assert [p["chunk_name"] for p in parts] == [name for name, _ in SAMPLE_PARTS]
    part = parts[0]
    draft = part["drafts"][0]
    read = draft["reads"][0]
    assert part["task_id"] == f"{folder.name}/round_price"  # Task ID
    assert part["role"] == "qa"  # Target role
    assert draft["messages"][0]["role"] == "system"  # Complete prompt (generator)
    assert "round_price" in draft["messages"][1]["content"]
    assert draft["critic_messages"][1]["content"].startswith("File imports")  # critic prompt
    assert json.loads(draft["reply"]["content"])["name"] == "round_price"  # LLM response
    assert draft["reply"]["model"] == "fake"  # Model
    assert json.loads(read["replies"][0]["content"])["score"] == 5  # critic response
    assert read["replies"][0]["reasoning"] == "Checked every claim against the code."
    assert draft["check"]["errors"] == []  # Validator response
    assert part["status"] == "passed" and part["score"] == 5  # Execution result
    assert part["failure_category"] is None  # Error type
    assert part["llm_seconds"] == 8.0  # Response time
    assert part["llm_calls"] == 3
    assert part["summary"]["highlights"]["boundary_values"] == ["Prices are rounded up."]


def test_the_api_key_never_appears_in_any_log_file():
    full_sample_run()
    for path in the_run_folder().iterdir():
        assert SECRET not in path.read_text(), path.name


def test_json_output_is_saved_as_output_json():
    script = passing(Role.PM, "round_price")
    run(
        [
            "summarize",
            str(SAMPLE),
            "--role",
            "pm",
            "--granularity",
            "function",
            "--format",
            "json",
            "--chunk",
            "round_price",
        ],
        script,
    )
    folder = the_run_folder()
    assert json.loads((folder / "output.json").read_text())["role"] == "pm"
    assert not (folder / "output.md").exists()


def test_parts_the_preprocessor_refused_are_logged_too(tmp_path):
    source = tmp_path / "big.py"
    source.write_text(
        "def small():\n    return 1\n\ndef huge():\n"
        + "".join(f"    x{i} = {i}\n" for i in range(200))
    )
    settings = Settings(_env_file=None, deepseek_api_key=SECRET, max_chunk_tokens=200)
    code, _, _ = run(
        ["summarize", str(source), "--role", "developer", "--granularity", "function"],
        passing(Role.DEVELOPER, "small"),
        settings,
    )
    assert code == 0
    folder = the_run_folder()
    parts = [json.loads(line) for line in (folder / "parts.jsonl").read_text().splitlines()]
    assert [(p["chunk_name"], p["status"]) for p in parts] == [
        ("small", "passed"),
        ("huge", "too_large"),
    ]
    assert parts[1]["failure_category"] == "input"
    assert json.loads((folder / "run.json").read_text())["status_counts"] == {
        "passed": 1,
        "too_large": 1,
    }


# --- Partial runs ----------------------------------------------------------------------------


def test_stopped_run_keeps_the_parts_that_finished():
    script = [*passing(Role.QA, "round_price"), LLMSetupError("HTTP 402: no balance.", attempts=1)]
    code, out, err = run(
        ["summarize", str(SAMPLE), "--role", "qa", "--granularity", "file"], script
    )
    assert code == 2 and out == ""
    folder = the_run_folder()
    assert f"Run log (partial): {folder}" in err
    run_json = json.loads((folder / "run.json").read_text())
    assert run_json["status"] == "stopped"
    assert run_json["error"] == "LLMSetupError: HTTP 402: no balance."
    parts = (folder / "parts.jsonl").read_text().splitlines()
    assert [json.loads(p)["chunk_name"] for p in parts] == ["round_price"]
    assert not list(folder.glob("output.*"))


class InterruptingClient:
    """Answers the first chunk, then behaves as if the user pressed Ctrl-C."""

    def __init__(self):
        self.script = passing(Role.QA, "round_price")
        self.fake = FakeLLMClient(self.script)

    def complete(self, messages, call):
        if self.fake.remaining == 0:
            raise KeyboardInterrupt
        return self.fake.complete(messages, call)


def test_interrupted_run_is_marked_and_keeps_finished_parts():
    code, _, err = run(
        ["summarize", str(SAMPLE), "--role", "qa", "--granularity", "file"],
        [],
        client=InterruptingClient(),
    )
    assert code == 130
    assert "interrupted" in err
    folder = the_run_folder()
    assert json.loads((folder / "run.json").read_text())["status"] == "interrupted"
    assert len((folder / "parts.jsonl").read_text().splitlines()) == 1


def test_parts_are_written_as_each_one_finishes():
    client = FakeLLMClient(
        [line for name, kind in SAMPLE_PARTS for line in passing(Role.QA, name, kind)]
    )
    seen = []

    class Watching:
        def complete(self, messages, call):
            # Before each generator call, count the part lines already on disk.
            if call == "generator":
                logs = list(Path("runs").glob("*/parts.jsonl"))
                seen.append(len(logs[0].read_text().splitlines()) if logs else 0)
            return client.complete(messages, call)

    run(["summarize", str(SAMPLE), "--role", "qa", "--granularity", "file"], [], client=Watching())
    assert seen == [0, 1, 2, 3]


# --- Options and edge cases ------------------------------------------------------------------


def test_no_log_writes_nothing():
    code, _, err = full_sample_run("--no-log")
    assert code == 0
    assert not Path("runs").exists()
    assert "Run log" not in err


def test_log_dir_chooses_the_folder():
    full_sample_run("--log-dir", "my-logs")
    assert the_run_folder("my-logs").is_dir()
    assert not Path("runs").exists()


def test_log_dir_that_cannot_be_created_is_bad_input():
    Path("taken").write_text("a file, not a folder")
    code, _, err = full_sample_run("--log-dir", "taken")
    assert code == 1 and "cannot create a run log" in err


def test_no_log_and_log_dir_cannot_be_combined():
    code, _, err = full_sample_run("--no-log", "--log-dir", "x")
    assert code == 1 and "not allowed with" in err


def test_two_runs_in_the_same_second_get_different_folders():
    moment = datetime(2026, 11, 23, 15, 30, 12).astimezone()
    settings = Settings(_env_file=None)
    logs = [
        RunLog(
            Path("runs"), inputs={}, settings=settings, source_text="x = 1\n", now=lambda: moment
        )
        for _ in range(5)
    ]
    assert len({log.path for log in logs}) == 5
    assert all(log.run_id.startswith("20261123-153012-") for log in logs)


def test_parts_with_the_same_name_get_distinct_task_ids():
    log = RunLog(Path("runs"), inputs={}, settings=Settings(_env_file=None), source_text="")
    first = ChunkOutcome(
        chunk_name="f", kind="function", start_line=1, end_line=2, role=Role.QA, status="too_large"
    )
    second = first.model_copy(update={"start_line": 10, "end_line": 11})
    assert log.task_id(first) == f"{log.run_id}/f"
    assert log.task_id(second) == f"{log.run_id}/f@10"
    log.finish("finished", [first, second])
    lines = (log.path / "parts.jsonl").read_text().splitlines()
    assert [json.loads(line)["task_id"] for line in lines] == [
        f"{log.run_id}/f",
        f"{log.run_id}/f@10",
    ]
