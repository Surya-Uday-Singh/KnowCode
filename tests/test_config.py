"""Tests for configuration loading.

Every test passes _env_file=None so that a developer's real .env file
never affects the results.
"""

import pytest
from pydantic import ValidationError

from knowcode.config import MissingApiKeyError, Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clear KnowCode/DeepSeek variables from the environment so a developer's
    shell settings can't change test results. Runs before every test."""
    for name in [
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_CRITIC_MODEL",
        "KNOWCODE_GENERATOR_THINKING",
        "KNOWCODE_CRITIC_THINKING",
        "KNOWCODE_REQUEST_TIMEOUT_SECONDS",
        "KNOWCODE_RETRY_BASE_DELAY_SECONDS",
        "KNOWCODE_GENERATOR_TEMPERATURE",
        "KNOWCODE_CRITIC_TEMPERATURE",
        "KNOWCODE_GENERATOR_MAX_TOKENS",
        "KNOWCODE_CRITIC_MAX_TOKENS",
        "KNOWCODE_MAX_TECHNICAL_RETRIES",
        "KNOWCODE_MAX_QUALITY_RETRIES",
        "KNOWCODE_PASS_THRESHOLD",
        "KNOWCODE_MAX_CHUNK_TOKENS",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_default_values():
    s = Settings(_env_file=None)
    assert s.generator_temperature == 0.3
    assert s.critic_temperature == 0.0
    assert s.generator_max_tokens == 1000
    assert s.critic_max_tokens == 8000
    assert s.generator_thinking is False
    assert s.critic_thinking is True
    assert s.max_technical_retries == 3
    assert s.max_quality_retries == 1
    assert s.pass_threshold == 4
    assert s.max_chunk_tokens == 6000
    assert s.request_timeout_seconds == 120.0
    assert s.retry_base_delay_seconds == 1.0
    assert s.deepseek_base_url == "https://api.deepseek.com"
    assert s.deepseek_model == "deepseek-flash"


def test_critic_uses_the_main_model_unless_one_is_set(monkeypatch):
    assert Settings(_env_file=None).critic_model == "deepseek-flash"
    monkeypatch.setenv("DEEPSEEK_CRITIC_MODEL", "")  # an empty line in .env
    assert Settings(_env_file=None).critic_model == "deepseek-flash"
    monkeypatch.setenv("DEEPSEEK_CRITIC_MODEL", "deepseek-v4-pro")
    assert Settings(_env_file=None).critic_model == "deepseek-v4-pro"


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("KNOWCODE_PASS_THRESHOLD", "5")
    monkeypatch.setenv("DEEPSEEK_MODEL", "some-other-model")
    s = Settings(_env_file=None)
    assert s.pass_threshold == 5
    assert s.deepseek_model == "some-other-model"


def test_invalid_value_fails_fast(monkeypatch):
    monkeypatch.setenv("KNOWCODE_PASS_THRESHOLD", "9")  # must be 1-5
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_missing_api_key_gives_clear_error():
    s = Settings(_env_file=None)
    with pytest.raises(MissingApiKeyError, match="DEEPSEEK_API_KEY"):
        s.require_api_key()


def test_api_key_is_hidden_when_printed(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-super-secret")
    s = Settings(_env_file=None)
    assert s.require_api_key() == "sk-super-secret"
    assert "sk-super-secret" not in repr(s)
