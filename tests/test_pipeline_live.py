"""The whole loop against the real DeepSeek API. Costs a little money, so it
only runs when asked for: pytest -m live (skipped without DEEPSEEK_API_KEY)."""

from pathlib import Path

import pytest

from knowcode.config import Settings
from knowcode.llm import DeepSeekClient
from knowcode.pipeline import summarize_chunk
from knowcode.preprocessor import preprocess_file
from knowcode.prompts import Role

pytestmark = pytest.mark.live

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cart.py"


def test_round_price_qa_summary_ends_with_a_kept_summary():
    settings = Settings()
    if settings.deepseek_api_key is None or not settings.deepseek_api_key.get_secret_value():
        pytest.skip("DEEPSEEK_API_KEY is not set")
    result = preprocess_file(SAMPLE)
    chunk = next(c for c in result.chunks if c.name == "round_price")

    outcome = summarize_chunk(chunk, Role.QA, result.context, DeepSeekClient(settings), settings)

    assert outcome.status in {"passed", "passed_with_rule_errors", "below_threshold"}
    assert outcome.summary is not None and outcome.summary.name == "round_price"
    assert 1 <= len(outcome.drafts) <= 2
    assert all(len(draft.reads) == settings.critic_runs for draft in outcome.drafts if draft.usable)
