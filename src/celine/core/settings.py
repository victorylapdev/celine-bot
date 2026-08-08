"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Celine backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CELINE_",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    ai_provider: str = "openai"
    openai_api_key: str | None = None
    database_url: str | None = None
    redis_url: str | None = None
    mqtt_url: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return cached settings for the process lifetime."""
    return Settings()
