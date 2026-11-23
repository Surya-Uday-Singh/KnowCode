"""Tests for the `knowcode` command: required inputs, reading files and
snippets, output files, progress, exit codes. A scripted FakeLLMClient
stands in for DeepSeek."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from knowcode.cli import main
from knowcode.config import Settings
from knowcode.llm import FakeLLMClient, LLMSetupError
from knowcode.prompts import ROLES, Role

SAMPLE = str(Path(__file__).parent / "fixtures" / "sample_cart.py")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("KNOWCODE_", "DEEPSEEK_")):
            monkeypatch.delenv(name)


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
    verdict = json.dumps({"score": 5, "pass": True, "critique": ""})
    return [summary, verdict, verdict]


def run(args, script=(), stdin=""):
    out, err = io.StringIO(), io.StringIO()
    client = FakeLLMClient(script)
    code = main(
        args,
        stdin=io.StringIO(stdin),
        stdout=out,
        stderr=err,
        client_factory=lambda: client,
        settings=Settings(_env_file=None),
    )
    return code, out.getvalue(), err.getvalue(), client


ROUND_PRICE = ["summarize", SAMPLE, "--chunk", "round_price"]


# --- Required inputs ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, missing",
    [
        (["summarize", SAMPLE, "--granularity", "file"], "--role"),
        (["summarize", SAMPLE, "--role", "qa"], "--granularity"),
        (["summarize", "--role", "qa", "--granularity", "file"], "path"),
    ],
)
def test_role_granularity_and_path_are_required(args, missing):
    code, out, err, _ = run(args)
    assert code == 1
    assert "required" in err and missing in err
    assert "usage: knowcode summarize" in err
    assert out == ""


@pytest.mark.parametrize("flag, value", [("--role", "manager"), ("--granularity", "class")])
def test_unknown_role_or_granularity_is_rejected(flag, value):
    args = ["summarize", SAMPLE, "--role", "qa", "--granularity", "file"]
    args[args.index(flag) + 1] = value
    code, _, err, _ = run(args)
    assert code == 1 and "invalid choice" in err


def test_a_command_is_required():
    code, _, err, _ = run([])
    assert code == 1 and "required" in err


# --- Reading input ------------------------------------------------------------------------


def test_missing_file_is_bad_input():
    code, _, err, client = run(
        ["summarize", "nope.py", "--role", "qa", "--granularity", "function"]
    )
    assert code == 1 and "cannot read nope.py" in err
    assert client.calls == []


def test_syntax_error_is_bad_input(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n    pass\n")
    code, _, err, client = run(["summarize", str(bad), "--role", "qa", "--granularity", "function"])
    assert code == 1 and "Syntax error (line 1" in err
    assert client.calls == []  # no LLM call for invalid code


def test_pasted_snippet_is_read_from_standard_input():
    snippet = "    def add(a, b):\n        return a + b\n"  # indented, as if copied from a class
    code, out, _, _ = run(
        ["summarize", "-", "--role", "developer", "--granularity", "function"],
        passing(Role.DEVELOPER, "add"),
        stdin=snippet,
    )
    assert code == 0
    assert out.startswith("# KnowCode summary: snippet\n")
    assert "## `add` · function · lines 1-2" in out


def test_chunk_option_summarizes_only_that_part():
    code, out, _, client = run(
        [*ROUND_PRICE, "--role", "qa", "--granularity", "function"], passing(Role.QA, "round_price")
    )
    assert code == 0
    assert len(client.calls) == 3
    assert "round_price" in out and "CartItem" not in out


def test_unknown_chunk_is_bad_input():
    code, _, err, _ = run(
        ["summarize", SAMPLE, "--role", "qa", "--granularity", "file", "--chunk", "nope"]
    )
    assert code == 1 and "no part named 'nope'" in err


# --- Output ----------------------------------------------------------------------------------


def test_markdown_goes_to_stdout_and_progress_to_stderr():
    code, out, err, _ = run(
        [*ROUND_PRICE, "--role", "qa", "--granularity", "file"], passing(Role.QA, "round_price")
    )
    assert code == 0
    assert out.startswith("# KnowCode file overview:")
    assert "[1/1] round_price ... ✅ passed (5/5) · 3 calls" in err
    assert "Done: 1 part (1 passed) · 3 LLM calls" in err
    assert "[1/1]" not in out and "Done:" not in out


def test_json_format():
    code, out, _, _ = run(
        [*ROUND_PRICE, "--role", "pm", "--granularity", "function", "--format", "json"],
        passing(Role.PM, "round_price"),
    )
    assert code == 0
    data = json.loads(out)
    assert data["role"] == "pm" and data["parts"][0]["status"] == "passed"


def test_output_option_writes_a_file_and_prints_nothing_else(tmp_path):
    target = tmp_path / "qa.md"
    code, out, err, _ = run(
        [*ROUND_PRICE, "--role", "qa", "--granularity", "file", "--output", str(target)],
        passing(Role.QA, "round_price"),
    )
    assert code == 0 and out == ""
    assert target.read_text().startswith("# KnowCode file overview:")
    assert f"Written to {target}" in err


def test_unwritable_output_is_bad_input(tmp_path):
    target = tmp_path / "missing-dir" / "qa.md"
    code, _, err, _ = run(
        [*ROUND_PRICE, "--role", "qa", "--granularity", "file", "--output", str(target)],
        passing(Role.QA, "round_price"),
    )
    assert code == 1 and "cannot write" in err


# --- Exit codes for the run ---------------------------------------------------------------------


def test_run_finishes_with_0_even_when_parts_fail():
    code, out, err, _ = run([*ROUND_PRICE, "--role", "qa", "--granularity", "function"], ["x", "y"])
    assert code == 0
    assert "⛔ Not summarized" in out
    assert "1 format failure" in err


def test_setup_error_stops_the_run_with_2():
    code, out, err, _ = run(
        [*ROUND_PRICE, "--role", "qa", "--granularity", "function"],
        [LLMSetupError("HTTP 402: the DeepSeek account has no balance left.", attempts=1)],
    )
    assert code == 2
    assert "run stopped: HTTP 402" in err
    assert out == ""


def test_missing_api_key_stops_the_run_with_2():
    out, err = io.StringIO(), io.StringIO()
    code = main(
        [*ROUND_PRICE, "--role", "qa", "--granularity", "function"],
        stdout=out,
        stderr=err,
        settings=Settings(_env_file=None),  # no key, and no client_factory: the real client
    )
    assert code == 2 and "DEEPSEEK_API_KEY is not set" in err.getvalue()


def test_python_dash_m_knowcode_runs_the_command():
    result = subprocess.run(
        [sys.executable, "-m", "knowcode", "summarize", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--granularity {function,file}" in result.stdout
