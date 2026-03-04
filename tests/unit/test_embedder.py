import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
import tiktoken

from rag.services.embedder import OpenAIEmbeddingService


@pytest.fixture
def dimension() -> int:
    return 1536


@pytest.fixture
def mock_vector(dimension: int) -> list[float]:
    return [0.1] * dimension


@pytest.fixture
def mock_client(mock_vector: list[float]) -> MagicMock:
    embedding_item = MagicMock()
    embedding_item.embedding = mock_vector
    embedding_item.index = 0

    response = MagicMock()
    response.data = [embedding_item]

    client = MagicMock()
    client.embeddings.create = AsyncMock(return_value=response)
    return client


@pytest.mark.unit
class TestEmptyInputGuard:
    """ValueError raised for empty / whitespace input."""

    async def test_empty_string_raises_before_api_call(
        self, tokenizer: tiktoken.Encoding, mock_client: MagicMock
    ) -> None:
        service = OpenAIEmbeddingService(client=mock_client, tokenizer=tokenizer)

        with pytest.raises(ValueError, match="empty"):
            await service.embed_text("")

        mock_client.embeddings.create.assert_not_called()

    async def test_whitespace_only_raises_before_api_call(
        self, tokenizer: tiktoken.Encoding, mock_client: MagicMock
    ) -> None:
        service = OpenAIEmbeddingService(client=mock_client, tokenizer=tokenizer)

        with pytest.raises(ValueError, match="empty"):
            await service.embed_text("   \t\n")

        mock_client.embeddings.create.assert_not_called()


@pytest.mark.unit
class TestMaxTokenTruncation:
    """Oversized text is truncated, warning logged, vector returned."""

    async def test_truncation_calls_api_with_trimmed_text(
        self,
        tokenizer: tiktoken.Encoding,
        mock_vector: list[float],
        mock_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        max_tokens = 10
        long_text = "hello world " * 20

        service = OpenAIEmbeddingService(
            client=mock_client, tokenizer=tokenizer, max_tokens=max_tokens
        )

        with caplog.at_level(logging.WARNING, logger="rag.services.embedder"):
            result = await service.embed_text(long_text)

        # API was called exactly once
        mock_client.embeddings.create.assert_called_once()

        # The text passed must fit within max_tokens
        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        actual_tokens = len(tokenizer.encode(call_kwargs["input"]))
        assert actual_tokens <= max_tokens, (
            f"API received {actual_tokens} tokens; expected ≤ {max_tokens}"
        )

        # A warning must be emitted
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("truncat" in msg.lower() for msg in warning_messages), (
            "Expected a truncation warning but none was logged"
        )

        assert result == mock_vector

    async def test_text_within_limit_is_not_truncated(
        self,
        tokenizer: tiktoken.Encoding,
        mock_vector: list[float],
        mock_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """No warning and no truncation when input fits inside max_tokens."""
        max_tokens = 100
        short_text = "This is a short sentence."

        service = OpenAIEmbeddingService(
            client=mock_client, tokenizer=tokenizer, max_tokens=max_tokens
        )

        with caplog.at_level(logging.WARNING, logger="rag.services.embedder"):
            result = await service.embed_text(short_text)

        # API receives the original text unmodified
        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["input"] == short_text

        # No warning emitted
        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert not any("truncat" in msg.lower() for msg in warning_messages)

        assert result == mock_vector
