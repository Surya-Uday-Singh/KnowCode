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
    deepseek_model: str = Field(default="deepseek-flash", validation_alias="DEEPSEEK_MODEL")
    # Optional separate model for the critic (e.g. a stronger one). Empty means
    # the critic uses deepseek_model too.
    deepseek_critic_model: str | None = Field(
        default=None, validation_alias="DEEPSEEK_CRITIC_MODEL"
    )

    # --- Model parameters ----------------------------------------------------
    # Thinking mode makes the model reason step by step before answering. The
    # critic uses it, because checking every claim against the code is a
    # reasoning task; the generator doesn't need it. With thinking on, the API
    # ignores temperature, and the reasoning counts towards max_tokens.
    generator_thinking: bool = Field(default=False, validation_alias="KNOWCODE_GENERATOR_THINKING")
    critic_thinking: bool = Field(default=True, validation_alias="KNOWCODE_CRITIC_THINKING")
    # Temperatures only apply when thinking is off. The generator gets a little
    # randomness for more natural writing; 0.0 makes a non-thinking critic give
    # the same draft the same score every time.
    generator_temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, validation_alias="KNOWCODE_GENERATOR_TEMPERATURE"
    )
    critic_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, validation_alias="KNOWCODE_CRITIC_TEMPERATURE"
    )
    # Output length caps. The critic's verdict is short, but its reasoning
    # counts towards its cap, so it needs more room than the generator.
    generator_max_tokens: int = Field(
        default=1000, gt=0, validation_alias="KNOWCODE_GENERATOR_MAX_TOKENS"
    )
    critic_max_tokens: int = Field(
        default=8000, gt=0, validation_alias="KNOWCODE_CRITIC_MAX_TOKENS"
    )

    # --- Requests ------------------------------------------------------------
    # How long to wait for one reply before treating it as a timeout. Critic
    # calls in thinking mode have taken 20-80 seconds, so this is generous.
    request_timeout_seconds: float = Field(
        default=120.0, gt=0, validation_alias="KNOWCODE_REQUEST_TIMEOUT_SECONDS"
    )
    # Wait before the first technical retry; each later wait doubles (1s, 2s, 4s).
    retry_base_delay_seconds: float = Field(
        default=1.0, ge=0, validation_alias="KNOWCODE_RETRY_BASE_DELAY_SECONDS"
    )

    # --- Retries and scoring -------------------------------------------------
    # Technical retries resend the exact same request after a network/API error
    # or an empty reply. They don't count against the quality retry budget.
    # 3 retries means up to 4 attempts in total.
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

    @property
    def critic_model(self) -> str:
        """The critic's model: DEEPSEEK_CRITIC_MODEL if set, else DEEPSEEK_MODEL."""
        # An empty line in .env ("DEEPSEEK_CRITIC_MODEL=") counts as not set.
        return self.deepseek_critic_model or self.deepseek_model

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
