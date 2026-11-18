"""Application settings.

Every tunable value (API connection, model parameters, retry limits, size
limits) lives in the `Settings` class below. Values are read from environment
variables, or from a `.env` file in the directory the program is run from.

pydantic validates each value on load, so a bad setting such as
KNOWCODE_PASS_THRESHOLD=abc stops the program at startup with a clear error
instead of causing a confusing failure later.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingApiKeyError(RuntimeError):
    """Raised when an LLM call is attempted but DEEPSEEK_API_KEY is not set."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # other variables in .env are allowed and simply ignored
        populate_by_name=True,  # lets tests write Settings(pass_threshold=3) directly
    )

    # --- DeepSeek connection -------------------------------------------------
    # The key is optional here so that everything that doesn't talk to the LLM
    # (parsing, prompt building, tests) works without one. Code that makes a
    # real API call asks for it through require_api_key().
    # SecretStr masks the value when settings are printed or logged.
    deepseek_api_key: SecretStr | None = Field(default=None, validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL"
    )
    deepseek_model: str = Field(default="deepseek-chat", validation_alias="DEEPSEEK_MODEL")

    # --- Model parameters ----------------------------------------------------
    # The generator gets a little randomness for more natural writing; the
    # critic runs at 0.0 so the same draft gets the same score every time.
    generator_temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, validation_alias="KNOWCODE_GENERATOR_TEMPERATURE"
    )
    critic_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, validation_alias="KNOWCODE_CRITIC_TEMPERATURE"
    )
    # Output length caps. Summaries need more room than the critic's short verdict.
    generator_max_tokens: int = Field(
        default=1000, gt=0, validation_alias="KNOWCODE_GENERATOR_MAX_TOKENS"
    )
    critic_max_tokens: int = Field(default=500, gt=0, validation_alias="KNOWCODE_CRITIC_MAX_TOKENS")

    # --- Retries and scoring -------------------------------------------------
    # Technical retries resend the exact same request after a network/API error
    # or an empty reply. They don't count against the quality retry budget.
    max_technical_retries: int = Field(
        default=3, ge=0, validation_alias="KNOWCODE_MAX_TECHNICAL_RETRIES"
    )
    # Quality retries ask the generator to rewrite a draft using the critic's
    # feedback. Kept at 1: one revision fixes most problems, and a hard cap
    # guarantees the generate/critique loop always terminates.
    max_quality_retries: int = Field(
        default=1, ge=0, validation_alias="KNOWCODE_MAX_QUALITY_RETRIES"
    )
    # A draft passes when the critic scores it at least this high (scale 1-5).
    pass_threshold: int = Field(default=4, ge=1, le=5, validation_alias="KNOWCODE_PASS_THRESHOLD")

    # --- Preprocessing -------------------------------------------------------
    # Largest chunk of source code we'll send in one request, in estimated
    # tokens (characters / 4). A class over the limit is split into its
    # methods; a single function over the limit is reported, not truncated.
    max_chunk_tokens: int = Field(default=6000, gt=0, validation_alias="KNOWCODE_MAX_CHUNK_TOKENS")

    def require_api_key(self) -> str:
        """Return the API key, or raise a clear error if it is not set."""
        if self.deepseek_api_key is None or not self.deepseek_api_key.get_secret_value():
            raise MissingApiKeyError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.deepseek_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Load settings once and return the same object on every later call."""
    return Settings()
