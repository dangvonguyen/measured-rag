from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseSettings):
    """Chat-specific settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.1

    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_THRESHOLD: float = 0.5

    # Conversation history - last N user+assistant pairs (= 2N messages)
    HISTORY_MAX_TURNS: int = 10


@lru_cache(maxsize=1)
def get_chat_settings() -> ChatSettings:
    return ChatSettings()
