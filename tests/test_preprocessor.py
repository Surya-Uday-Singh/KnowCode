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


def test_empty_source_has_nothing_to_summarize():
    with pytest.raises(NothingToSummarizeError):
        preprocess("", NO_LIMIT)


def test_script_with_no_functions_or_classes_has_nothing_to_summarize():
    with pytest.raises(NothingToSummarizeError):
        preprocess("import os\nx = 1\nprint(x)\n", NO_LIMIT)


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


def test_module_level_constants_are_not_included_in_chunks():
    result = preprocess_file(SAMPLE, NO_LIMIT)
    assert all("MAX_ITEMS = 50" not in c.source for c in result.chunks)


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
