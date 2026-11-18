"""Tests for the preprocessor: syntax checking, chunking and size limits."""

import ast
import textwrap
from pathlib import Path

import pytest

from knowcode.preprocessor import (
    CodeSyntaxError,
    InputError,
    NothingToSummarizeError,
    estimate_tokens,
    preprocess,
    preprocess_file,
)

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cart.py"

# A limit large enough that nothing gets split, for tests that aren't about size.
NO_LIMIT = 100_000


def code(text: str) -> str:
    """Allow test code to be written as an indented triple-quoted block."""
    return textwrap.dedent(text).lstrip("\n")


# --- Syntax check -----------------------------------------------------------


def test_syntax_error_is_reported_with_its_line_number():
    with pytest.raises(CodeSyntaxError) as info:
        preprocess("def broken(:\n    pass\n", NO_LIMIT)
    assert info.value.line == 1
    # Syntax errors are classified as input failures, not LLM failures.
    assert isinstance(info.value, InputError)
    assert info.value.category == "input"


# --- Messy real-world input -------------------------------------------------


def test_snippet_pasted_with_indentation_is_accepted():
    # A method copied out of the middle of a class keeps its leading spaces.
    snippet = "    def total(self):\n        return self.price * self.qty\n"
    result = preprocess(snippet, NO_LIMIT)
    assert [c.name for c in result.chunks] == ["total"]
    assert result.chunks[0].source.startswith("def total(self):")
    assert result.chunks[0].start_line == 1


def test_leading_byte_order_mark_in_text_is_ignored():
    result = preprocess("﻿def f():\n    return 1\n", NO_LIMIT)
    assert [c.name for c in result.chunks] == ["f"]


def test_file_with_byte_order_mark_is_read(tmp_path):
    path = tmp_path / "bom.py"
    path.write_bytes(b"\xef\xbb\xbfdef f():\n    return 1\n")
    assert [c.name for c in preprocess_file(path, NO_LIMIT).chunks] == ["f"]


def test_file_with_latin1_encoding_declaration_is_read(tmp_path):
    path = tmp_path / "latin1.py"
    text = "# -*- coding: latin-1 -*-\ndef greet():\n    return 'café'\n"
    path.write_bytes(text.encode("latin-1"))
    [chunk] = preprocess_file(path, NO_LIMIT).chunks
    assert "café" in chunk.source


def test_messy_formatting_is_accepted():
    # Tabs, odd spacing, trailing comments, one-line defs and Windows line endings.
    source = "import os,sys\r\n\r\ndef  f( a,b ):\r\n\treturn a+b   # add\r\nclass   X : pass\r\n"
    result = preprocess(source, NO_LIMIT)
    assert [c.name for c in result.chunks] == ["f", "X"]


def test_empty_source_has_nothing_to_summarize():
    with pytest.raises(NothingToSummarizeError):
        preprocess("", NO_LIMIT)


def test_comment_only_source_has_nothing_to_summarize():
    with pytest.raises(NothingToSummarizeError):
        preprocess("# just a comment\n\n# and another\n", NO_LIMIT)


# --- Chunking ---------------------------------------------------------------


def test_top_level_definitions_become_chunks_in_file_order():
    result = preprocess_file(SAMPLE, NO_LIMIT)
    assert [(c.name, c.kind) for c in result.chunks] == [
        ("round_price", "function"),
        ("CartItem", "class"),
        ("Cart", "class"),
        ("export_cart", "function"),  # async functions are chunked like normal ones
    ]
    assert result.failures == []


def test_imports_are_collected_including_ones_inside_try_blocks():
    result = preprocess_file(SAMPLE, NO_LIMIT)
    assert result.imports == [
        "import math",
        "from dataclasses import dataclass",
        "import json",
    ]


def test_chunk_includes_decorators_and_correct_line_numbers():
    result = preprocess_file(SAMPLE, NO_LIMIT)
    cart_item = next(c for c in result.chunks if c.name == "CartItem")

    assert cart_item.source.startswith("@dataclass\nclass CartItem:")

    # The recorded line range should point at the decorator and the last line
    # of the class in the original file.
    file_lines = SAMPLE.read_text().splitlines()
    assert file_lines[cart_item.start_line - 1] == "@dataclass"
    assert file_lines[cart_item.end_line - 1].strip() == (
        "return round_price(self.unit_price * self.quantity)"
    )


def test_nested_function_stays_inside_its_parent_chunk():
    result = preprocess_file(SAMPLE, NO_LIMIT)
    assert "apply_discount" not in [c.name for c in result.chunks]

    cart = next(c for c in result.chunks if c.name == "Cart")
    assert "def apply_discount" in cart.source


def test_module_level_constants_become_context_not_chunks():
    result = preprocess_file(SAMPLE, NO_LIMIT)
    assert all("MAX_ITEMS = 50" not in c.source for c in result.chunks)
    assert result.constants == ["MAX_ITEMS = 50", "DISCOUNT_THRESHOLD = 100.0"]


def test_every_chunk_is_valid_python_on_its_own():
    for chunk in preprocess_file(SAMPLE, NO_LIMIT).chunks:
        ast.parse(chunk.source)  # raises SyntaxError if the chunk was cut badly


# --- Size limits ------------------------------------------------------------


def test_token_estimate_is_characters_divided_by_four_rounded_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_class_over_the_limit_is_split_into_its_methods():
    source = code(
        """
        class Big:
            def first(self):
                return 1

            def second(self):
                return 2
        """
    )
    # One token under the class's size: the class is too big, each method fits.
    result = preprocess(source, estimate_tokens(source) - 1)

    assert [(c.name, c.kind, c.parent) for c in result.chunks] == [
        ("Big.first", "function", "Big"),
        ("Big.second", "function", "Big"),
    ]
    # Methods are dedented so they can be read (and parsed) on their own.
    assert result.chunks[0].source.startswith("def first(self):")
    assert result.failures == []


def test_function_over_the_limit_is_reported_not_truncated():
    source = "def huge():\n" + "    x = 1\n" * 50
    result = preprocess(source, max_chunk_tokens=10)

    assert result.chunks == []
    [failure] = result.failures
    assert failure.name == "huge"
    assert failure.reason == "too_large"
    assert "over the limit" in failure.message


def test_split_class_keeps_small_methods_and_reports_oversized_one():
    source = (
        code(
            """
        class Mixed:
            def small(self):
                return 1

            def large(self):
        """
        )
        + "        y = 2\n" * 40
    )
    result = preprocess(source, max_chunk_tokens=20)

    assert [c.name for c in result.chunks] == ["Mixed.small"]
    assert [f.name for f in result.failures] == ["Mixed.large"]


def test_oversized_class_with_no_methods_is_reported():
    # A big class made only of attributes has nothing to split into.
    source = "class Table:\n" + "".join(f"    A{i} = {i}\n" for i in range(60))
    result = preprocess(source, max_chunk_tokens=20)

    assert result.chunks == []
    assert result.failures[0].name == "Table"
    assert result.failures[0].kind == "class"


# --- Script code (statements outside any function or class) ----------------


def test_file_with_only_script_code_becomes_one_script_chunk():
    source = code(
        """
        import csv
        for row in csv.reader(open("data.csv")):
            print(row)
        """
    )
    result = preprocess(source, NO_LIMIT)

    [chunk] = result.chunks
    assert chunk.kind == "script"
    assert chunk.name == "script (lines 2-3)"
    # The import is file context, so the chunk starts at the loop.
    assert chunk.source.startswith("for row in")
    assert result.imports == ["import csv"]


def test_mixed_file_gets_function_and_script_chunks_in_order():
    source = code(
        """
        import csv
        MAX_ROWS = 1000

        def clean(row):
            return [c.strip() for c in row]

        rows = []
        with open("data.csv") as f:
            for row in csv.reader(f):
                rows.append(clean(row))
        print(len(rows), "rows loaded")
        """
    )
    result = preprocess(source, NO_LIMIT)

    assert [(c.name, c.kind) for c in result.chunks] == [
        ("clean", "function"),
        ("script (lines 7-11)", "script"),
    ]
    # The setup lines at the top don't get a summary, but the constant is
    # shared as context so `clean` and the script are summarized knowing it.
    assert result.constants == ["MAX_ROWS = 1000"]
    # `rows = []` is part of the work below it, so it stays in the script chunk
    # instead of being treated as a setup constant.
    assert result.chunks[1].source.startswith("rows = []")


def test_main_guard_block_becomes_a_script_chunk():
    source = code(
        """
        def main():
            return 0

        if __name__ == "__main__":
            main()
        """
    )
    result = preprocess(source, NO_LIMIT)
    assert [c.kind for c in result.chunks] == ["function", "script"]
    assert result.chunks[1].source.startswith('if __name__ == "__main__":')


def test_computed_assignments_count_as_work_not_setup():
    # A data script made only of assignments is still doing real work,
    # because the values come from function calls rather than literals.
    source = code(
        """
        import pandas as pd
        df = pd.read_csv("sales.csv")
        totals = df.groupby("region").sum()
        """
    )
    result = preprocess(source, NO_LIMIT)
    assert [c.kind for c in result.chunks] == ["script"]
    assert result.constants == []


def test_file_with_only_setup_lines_is_summarized_as_a_whole():
    # e.g. a settings file. It has no "work", but it shouldn't be rejected.
    source = code(
        """
        \"\"\"Default settings.\"\"\"
        import os
        TIMEOUT = 30
        RETRIES = 3
        """
    )
    result = preprocess(source, NO_LIMIT)

    [chunk] = result.chunks
    assert chunk.kind == "script"
    assert "TIMEOUT = 30" in chunk.source
    # The constants are already inside the chunk, so they aren't repeated as context.
    assert result.constants == []


def test_try_import_fallback_counts_as_setup():
    source = code(
        """
        try:
            import json
        except ImportError:
            json = None

        def dump(x):
            return json.dumps(x)
        """
    )
    result = preprocess(source, NO_LIMIT)
    assert [c.name for c in result.chunks] == ["dump"]


def test_long_script_is_split_between_statements():
    source = "".join(f"print('step {i}')\n" for i in range(30))
    limit = estimate_tokens(source) // 3  # forces several pieces
    result = preprocess(source, limit)

    assert len(result.chunks) > 1
    assert result.failures == []
    # Pieces cover every line exactly once, in order, with no gaps or overlaps.
    covered = [n for c in result.chunks for n in range(c.start_line, c.end_line + 1)]
    assert covered == list(range(1, 31))
    for chunk in result.chunks:
        assert chunk.estimated_tokens <= limit
        ast.parse(chunk.source)


def test_script_statement_too_large_on_its_own_is_reported():
    loop = "for i in range(10):\n" + "    total = i * 2\n" * 40
    result = preprocess(loop, max_chunk_tokens=20)

    assert result.chunks == []
    [failure] = result.failures
    assert failure.kind == "script"
    assert failure.reason == "too_large"
