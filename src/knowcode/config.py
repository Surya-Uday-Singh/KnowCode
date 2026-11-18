"""Configuration for KnowCode.

All tunable values live here, in one typed object. They are read from
environment variables or from a `.env` file in the current directory.
The defaults match the design doc (model parameters on p.9, retry rules on p.15-17).

pydantic checks every value when it loads. A typo such as
KNOWCODE_PASS_THRESHOLD=abc fails immediately with a clear message,
not halfway through a run.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingApiKeyError(RuntimeError):
    """Raised when a live LLM call is attempted without DEEPSEEK_API_KEY."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unrelated variables in .env are not an error
        populate_by_name=True,  # allows Settings(pass_threshold=3) in tests
    )

    # --- DeepSeek connection -------------------------------------------------
    # The key is optional at load time so that parsing, prompts and tests work
    # without it. It becomes required only when an LLM call is made (require_api_key).
    # SecretStr keeps the key out of printed output and logs.
    deepseek_api_key: SecretStr | None = Field(default=None, validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL"
    )
    deepseek_model: str = Field(default="deepseek-chat", validation_alias="DEEPSEEK_MODEL")

    # --- Model parameters (doc p.9) -----------------------------------------
    generator_temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, validation_alias="KNOWCODE_GENERATOR_TEMPERATURE"
    )
    critic_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, validation_alias="KNOWCODE_CRITIC_TEMPERATURE"
    )
    generator_max_tokens: int = Field(
        default=1000, gt=0, validation_alias="KNOWCODE_GENERATOR_MAX_TOKENS"
    )
    critic_max_tokens: int = Field(default=500, gt=0, validation_alias="KNOWCODE_CRITIC_MAX_TOKENS")

    # --- Retry and scoring rules (doc p.15-17) --------------------------------
    # Technical retries: resend the same request after API errors / empty replies.
    max_technical_retries: int = Field(
        default=3, ge=0, validation_alias="KNOWCODE_MAX_TECHNICAL_RETRIES"
    )
    # Quality retries: critic-driven rewrite with feedback. The doc fixes this at 1.
    max_quality_retries: int = Field(
        default=1, ge=0, validation_alias="KNOWCODE_MAX_QUALITY_RETRIES"
    )
    # Minimum critic score (1-5) for a draft to pass.
    pass_threshold: int = Field(default=4, ge=1, le=5, validation_alias="KNOWCODE_PASS_THRESHOLD")

    def require_api_key(self) -> str:
        """Return the API key, or raise a clear error if it is not set."""
        if self.deepseek_api_key is None or not self.deepseek_api_key.get_secret_value():
            raise MissingApiKeyError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.deepseek_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them for the rest of the program."""
    return Settings()
