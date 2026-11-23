"""Real calls to the DeepSeek API. These cost a little money, so they only run
when asked for explicitly:

    pytest -m live

They read DEEPSEEK_API_KEY from .env and are skipped if it isn't set.
"""

import json
from pathlib import Path

import pytest

from knowcode.config import Settings
from knowcode.llm import DeepSeekClient
from knowcode.preprocessor import preprocess_file
from knowcode.prompts import Role, build_critic_messages, build_generator_messages

pytestmark = pytest.mark.live

SAMPLE = Path(__file__).parent / "fixtures" / "sample_cart.py"


@pytest.fixture(scope="module")
def client():
    settings = Settings()
    if settings.deepseek_api_key is None or not settings.deepseek_api_key.get_secret_value():
        pytest.skip("DEEPSEEK_API_KEY is not set")
    return DeepSeekClient(settings)


def test_generator_then_critic_return_json(client):
    result = preprocess_file(SAMPLE)
    chunk = next(c for c in result.chunks if c.name == "round_price")

    draft = client.complete(
        build_generator_messages(chunk, Role.QA, result.imports, result.constants), "generator"
    )
    assert not draft.truncated
    summary = json.loads(draft.content)
    assert summary["name"] == "round_price"
    assert set(summary["highlights"]) == set(
        ("expected_behavior", "boundary_values", "failure_modes")
    )

    verdict_reply = client.complete(
        build_critic_messages(chunk, Role.QA, draft.content, result.imports, result.constants),
        "critic",
    )
    assert not verdict_reply.truncated
    verdict = json.loads(verdict_reply.content)
    assert verdict["score"] in range(1, 6)
    assert isinstance(verdict["pass"], bool)
