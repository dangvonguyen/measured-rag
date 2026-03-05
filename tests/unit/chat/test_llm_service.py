from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from chat.core.schemas import ChatMessage
from chat.services.llm import OpenAILLMService


def _make_chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content
    return chunk


async def _mock_stream(tokens: list[str | None]) -> AsyncIterator[MagicMock]:
    for token in tokens:
        yield _make_chunk(token)


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def llm(mock_client: AsyncMock) -> OpenAILLMService:
    return OpenAILLMService(client=mock_client, model="gpt-4o-mini")


@pytest.fixture
def messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Say hello."),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
class TestLLMServiceStream:
    async def test_yields_text_tokens(
        self, llm: OpenAILLMService, messages: list[ChatMessage], mock_client: AsyncMock
    ) -> None:
        tokens = ["Hello", ", ", "world", "!"]
        mock_client.chat.completions.create.return_value = _mock_stream([*tokens, None])

        result = []
        async for token in llm.complete_stream(messages):
            result.append(token)

        assert result == tokens
        mock_client.chat.completions.create.assert_called_once()

    async def test_skips_none_delta(
        self, llm: OpenAILLMService, messages: list[ChatMessage], mock_client: AsyncMock
    ) -> None:
        # Stream containing None and empty strings
        tokens = ["Hello", None, "", " world"]
        mock_client.chat.completions.create.return_value = _mock_stream(tokens)

        result = []
        async for token in llm.complete_stream(messages):
            result.append(token)

        # Only truthy strings should be yielded
        assert result == ["Hello", " world"]

    async def test_passes_correct_params_to_api(
        self, llm: OpenAILLMService, messages: list[ChatMessage], mock_client: AsyncMock
    ) -> None:
        mock_client.chat.completions.create.return_value = _mock_stream(["ok"])

        async for _ in llm.complete_stream(messages):
            pass

        # Verify call arguments directly using built-in mock methods
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["stream"] is True
        assert kwargs["temperature"] is not None

        # Verify message serialization
        assert kwargs["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello."},
        ]

    async def test_handles_api_error(
        self, llm: OpenAILLMService, messages: list[ChatMessage], mock_client: AsyncMock
    ) -> None:
        """
        Ensure the service raises and logs on API failure.
        """
        mock_client.chat.completions.create.side_effect = Exception("API Down")

        with pytest.raises(Exception, match="API Down"):
            async for _ in llm.complete_stream(messages):
                pass
