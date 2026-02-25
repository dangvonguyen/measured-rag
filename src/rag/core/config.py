from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Schema constant - change this requires a new migration
EMBEDDING_DIMENSION = 1536


class RAGSettings(BaseSettings):
    """RAG-specific settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_MAX_TOKENS: int = 8191


@lru_cache(maxsize=1)
def get_rag_settings() -> RAGSettings:
    """
    Return the singleton RAG settings instance.
    """
    return RAGSettings()
