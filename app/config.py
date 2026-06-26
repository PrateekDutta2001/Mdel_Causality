"""Application configuration via pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings for the agentic selector service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_api_key: str | None = Field(
        default=None,
        description="Fallback LLM API key; prefer X-LLM-API-Key header per request.",
    )
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-latest"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "experiment_logs"

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "agentic-rag-causal-selector"

    embedding_backend: Literal["sentence-transformers", "openai"] = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"

    experiment_logs_dir: str = "data/experiment_logs"
    causal_confidence_threshold: float = 0.5
    hpo_n_trials: int = 20
    evaluator_confidence_threshold: float = 0.6
    evaluator_max_retries: int = 2

    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
