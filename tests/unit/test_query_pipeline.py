import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from rag.core.config import RAGSettings
from rag.core.models import ChatMessage, Message, RAGResponseLog, RetrievedChunk
from rag.pipeline.query import QueryPipeline


def _msg(
    content: str, role: str = "user", conversation_id: uuid.UUID | None = None
) -> Message:
    return Message(
        id=uuid.uuid4(),
        conversation_id=conversation_id or uuid.uuid4(),
        role=role,
        content=content,
        created_at=datetime.now(UTC),
    )


def _chunk(chunk_id: str = "c1", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        source_name="test.md",
        chunk_index=0,
        text="Some context.",
        score=score,
        metadata={},
    )


async def _stream(*tokens: str) -> AsyncIterator[str]:
    for token in tokens:
        yield token


async def _error_stream() -> AsyncIterator[str]:
    yield "partial"
    raise RuntimeError("LLM exploded")


@pytest.fixture
def settings() -> RAGSettings:
    return RAGSettings(
        LLM_MODEL="gpt-4o-mini",
        LLM_TEMPERATURE=0.1,
        LLM_MAX_TOKENS=2048,
        RETRIEVAL_TOP_K=5,
        RETRIEVAL_THRESHOLD=0.5,
        HISTORY_MAX_TURNS=10,
    )


@pytest.fixture
def embedder() -> AsyncMock:
    mock = AsyncMock()
    mock.embed_text.return_value = [0.1] * 5
    return mock


@pytest.fixture
def retriever() -> AsyncMock:
    mock = AsyncMock()
    mock.retrieve.return_value = [_chunk()]
    return mock


@pytest.fixture
def prompt_builder() -> MagicMock:
    mock = MagicMock()
    mock.build.return_value = [
        ChatMessage(role="system", content="Be helpful."),
        ChatMessage(role="user", content="Question: hello"),
    ]
    return mock


@pytest.fixture
def llm() -> MagicMock:
    mock = MagicMock()
    mock.complete_stream.return_value = _stream("Hello", " world")
    return mock


@pytest.fixture
def conv_repo() -> AsyncMock:
    mock = AsyncMock()
    mock.get_history.return_value = []
    mock.add_message.side_effect = [
        _msg("hello", role="user"),
        _msg("Hello world", role="assistant"),
    ]
    return mock


@pytest.fixture
def pipeline(
    embedder: AsyncMock,
    retriever: AsyncMock,
    prompt_builder: MagicMock,
    llm: MagicMock,
    conv_repo: AsyncMock,
    settings: RAGSettings,
) -> QueryPipeline:
    return QueryPipeline(
        embedder=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        conversation_repo=conv_repo,
        settings=settings,
    )


@pytest.mark.unit
class TestQueryPipelineStream:
    async def test_run_stream_success(
        self,
        pipeline: QueryPipeline,
        embedder: AsyncMock,
        retriever: AsyncMock,
        prompt_builder: MagicMock,
        llm: MagicMock,
        conv_repo: AsyncMock,
        settings: RAGSettings,
    ) -> None:
        """
        Tests the complete happy-path execution of the query pipeline stream.
        """
        conv_id = uuid.uuid4()
        query = "hello"
        history = [_msg("prev", role="user")]
        conv_repo.get_history.return_value = history

        token_seen: list[str] = []
        user_msg_saved_before_stream = False

        # Run stream and track intermediate state
        async for token in pipeline.run_stream(conv_id, query):
            if not token_seen:
                # Assert user message is persisted before yielding any tokens
                conv_repo.add_message.assert_called_once()
                assert conv_repo.add_message.call_args_list[0][0][1] == "user"
                user_msg_saved_before_stream = True
            token_seen.append(token)

        # Verify stream yields all tokens correctly
        assert token_seen == ["Hello", " world"]
        assert user_msg_saved_before_stream

        # Verify query embedding
        embedder.embed_text.assert_called_once_with(query)

        # Verify retrieval with the correct settings
        retriever.retrieve.assert_called_once_with(
            embedder.embed_text.return_value,
            top_k=settings.RETRIEVAL_TOP_K,
            threshold=settings.RETRIEVAL_THRESHOLD,
        )

        # Verify history fetched and passed to prompt builder
        conv_repo.get_history.assert_called_once_with(
            conv_id, limit=settings.HISTORY_MAX_TURNS * 2
        )
        assert prompt_builder.build.call_args[0][0] == history

        # Verify assistant message persisted with full content
        assert conv_repo.add_message.call_count == 2
        assert conv_repo.add_message.call_args_list[1] == call(
            conv_id, "assistant", "Hello world"
        )

        # Verify RAG response logging
        conv_repo.log_rag_response.assert_called_once()
        _, log = conv_repo.log_rag_response.call_args[0]
        assert isinstance(log, RAGResponseLog)
        assert log.query == query
        assert log.latency_ms >= 0
        assert log.prompt_tokens > 0
        assert log.completion_tokens > 0
        assert len(log.retrieved_chunks) == 1
        assert log.retrieved_chunks[0]["chunk_id"] == "c1"

    async def test_run_stream_handles_error(
        self, pipeline: QueryPipeline, llm: MagicMock, conv_repo: AsyncMock
    ) -> None:
        """
        Tests that a partial assistant message is saved if the LLM crashes mid-stream.
        """
        llm.complete_stream.return_value = _error_stream()
        conv_repo.add_message.side_effect = [
            _msg("hello", role="user"),
            _msg("partial", role="assistant"),
        ]

        with pytest.raises(RuntimeError, match="LLM exploded"):
            async for _ in pipeline.run_stream(uuid.uuid4(), "hello"):
                pass

        assert conv_repo.add_message.call_count == 2
        assert conv_repo.add_message.call_args_list[1][0][1] == "assistant"
        assert conv_repo.add_message.call_args_list[1][0][2] == "partial"
