import logging
from unittest.mock import AsyncMock

import pytest

from rag.core.schemas import RetrievedChunk
from rag.retrieval.retriever import RetrieverService


@pytest.fixture
def query_vector() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-0",
        document_id="doc-0",
        source_name="test.md",
        chunk_index=0,
        text="Sample text.",
        score=0.9,
        metadata={},
    )


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    store = AsyncMock()
    store.search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def retriever(mock_vector_store: AsyncMock) -> RetrieverService:
    return RetrieverService(mock_vector_store)


@pytest.mark.unit
class TestRetrieveDelegation:
    async def test_delegates_with_strict_signature(
        self,
        retriever: RetrieverService,
        mock_vector_store: AsyncMock,
        query_vector: list[float],
    ) -> None:
        """
        Verifies arguments are passed strictly as kwargs.
        """
        await retriever.retrieve(query_vector, top_k=5, threshold=0.75)

        mock_vector_store.search.assert_awaited_once_with(
            query_vector,
            top_k=5,
            threshold=0.75,
            filters=None,
        )

    async def test_delegates_filters(
        self,
        retriever: RetrieverService,
        mock_vector_store: AsyncMock,
        query_vector: list[float],
    ) -> None:
        """
        Verifies that filters are forwarded to the store unchanged.
        """
        filters = {"source_name": "docs/a.md"}

        await retriever.retrieve(query_vector, top_k=5, threshold=0.0, filters=filters)

        assert mock_vector_store.search.call_args.kwargs["filters"] == filters


@pytest.mark.unit
class TestRetrieveReturnValue:
    async def test_returns_chunks_unchanged(
        self,
        retriever: RetrieverService,
        mock_vector_store: AsyncMock,
        query_vector: list[float],
        sample_chunk: RetrievedChunk,
    ) -> None:
        """
        Verifies that the retriever passes chunks from the store to the caller as-is.
        """
        mock_vector_store.search.return_value = [sample_chunk]

        result = await retriever.retrieve(query_vector, top_k=1, threshold=0.0)

        assert result == [sample_chunk]

    async def test_propagates_store_exceptions(
        self,
        retriever: RetrieverService,
        mock_vector_store: AsyncMock,
        query_vector: list[float],
    ) -> None:
        """
        Ensure upstream errors (DB connection, timeout) are not swallowed silently.
        """
        mock_vector_store.search.side_effect = ConnectionError("DB down")

        with pytest.raises(ConnectionError):
            await retriever.retrieve(query_vector, top_k=1, threshold=0.0)


@pytest.mark.unit
class TestRetrieveLogging:
    async def test_logs_retrieval_metadata(
        self,
        retriever: RetrieverService,
        mock_vector_store: AsyncMock,
        query_vector: list[float],
        sample_chunk: RetrievedChunk,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Verifies observability: count, top_score, and latency presence.
        """
        mock_vector_store.search.return_value = [sample_chunk]

        with caplog.at_level(logging.INFO, logger="rag.retrieval.retriever"):
            await retriever.retrieve(query_vector, top_k=1, threshold=0.0)

        assert len(caplog.records) > 0
        log_text = caplog.text

        # Verify key metrics are present
        assert "chunks=1" in log_text
        assert "top_score=0.9" in log_text
        assert "latency_ms=" in log_text

    async def test_logs_none_score_on_empty(
        self,
        retriever: RetrieverService,
        mock_vector_store: AsyncMock,
        query_vector: list[float],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Verifies top_score is logged as 'none' when the store returns no chunks.
        """
        mock_vector_store.search.return_value = []

        with caplog.at_level(logging.INFO, logger="rag.retrieval.retriever"):
            await retriever.retrieve(query_vector, top_k=1, threshold=0.0)

        assert "top_score=none" in caplog.text
