from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.core.models import DocumentChunk, RetrievedChunk
from rag.services.vector_store import PGVectorStore


def make_chunk(
    chunk_id: str = "chunk-0",
    text: str = "Sample text.",
    chunk_index: int = 0,
    source_name: str = "test.md",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        token_count=10,
        metadata={"chunk_index": chunk_index, "source_name": source_name},
    )


def make_embedding(value: float = 0.1, dim: int = 1536) -> list[float]:
    return [value] * dim


def make_db_row(
    chunk_id: str = "chunk-0",
    document_id: str = "doc-1",
    source_name: str = "test.md",
    chunk_index: int = 0,
    text: str = "Sample text.",
    meta: dict | None = None,
    score: float = 0.9,
) -> MagicMock:
    row = MagicMock()
    row.id = chunk_id
    row.document_id = document_id
    row.source_name = source_name
    row.chunk_index = chunk_index
    row.text = text
    row.meta = meta or {}
    row.score = score
    return row


def mock_execute_result(rows: list) -> MagicMock:
    result = MagicMock()
    result.all.return_value = rows
    return result


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.merge = AsyncMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock(return_value=mock_execute_result([]))
    return session


@pytest.fixture
def vector_store(mock_session: MagicMock) -> PGVectorStore:
    return PGVectorStore(mock_session)


@pytest.mark.unit
class TestIngest:
    """ingest() maps DocumentChunks + embeddings to DB rows and persists them."""

    async def test_ingest_merges_each_chunk(
        self, vector_store: PGVectorStore, mock_session: MagicMock
    ) -> None:
        chunks = [make_chunk(f"chunk-{i}", chunk_index=i) for i in range(3)]
        embeddings = [make_embedding() for _ in range(3)]

        await vector_store.ingest("doc-1", chunks, embeddings)

        assert mock_session.merge.await_count == 3

    async def test_ingest_maps_chunk_fields_to_row(
        self, vector_store: PGVectorStore, mock_session: MagicMock
    ) -> None:
        chunk = make_chunk(
            "chunk-0", text="Hello world", chunk_index=0, source_name="doc.md"
        )
        embedding = make_embedding(value=0.5)

        await vector_store.ingest("doc-1", [chunk], [embedding])

        row = mock_session.merge.call_args[0][0]
        assert row.id == "chunk-0"
        assert row.document_id == "doc-1"
        assert row.source_name == "doc.md"
        assert row.chunk_index == 0
        assert row.text == "Hello world"
        assert row.embedding_vector == embedding

    async def test_ingest_empty_chunks_skips_db_calls(
        self, vector_store: PGVectorStore, mock_session: MagicMock
    ) -> None:
        await vector_store.ingest("doc-1", [], [])

        mock_session.merge.assert_not_awaited()
        mock_session.flush.assert_not_awaited()

    async def test_ingest_mismatched_lengths_raises(
        self, vector_store: PGVectorStore
    ) -> None:
        chunks = [make_chunk("chunk-0")]
        embeddings = [make_embedding(), make_embedding()]  # 2 != 1

        with pytest.raises(
            ValueError, match=r"[Cc]hunks.*[Ee]mbedding|[Ee]mbedding.*[Cc]hunk"
        ):
            await vector_store.ingest("doc-1", chunks, embeddings)


@pytest.mark.unit
class TestSearch:
    """search() returns RetrievedChunks sorted by similarity descending."""

    async def test_search_returns_retrieved_chunk_instances(
        self, vector_store: PGVectorStore, mock_session: MagicMock
    ) -> None:
        mock_session.execute.return_value = mock_execute_result([make_db_row()])

        results = await vector_store.search(make_embedding(), top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)

    async def test_search_maps_all_fields(
        self, vector_store: PGVectorStore, mock_session: MagicMock
    ) -> None:
        db_row = make_db_row(
            chunk_id="chunk-5",
            document_id="doc-42",
            source_name="policy.md",
            chunk_index=5,
            text="Important clause.",
            meta={"author": "Alice"},
            score=0.75,
        )
        mock_session.execute.return_value = mock_execute_result([db_row])

        results = await vector_store.search(make_embedding(), top_k=1)

        r = results[0]
        assert r.chunk_id == "chunk-5"
        assert r.document_id == "doc-42"
        assert r.source_name == "policy.md"
        assert r.chunk_index == 5
        assert r.text == "Important clause."
        assert r.metadata == {"author": "Alice"}
        assert r.score == 0.75

    async def test_search_empty_db_returns_empty_list(
        self, vector_store: PGVectorStore, mock_session: MagicMock
    ) -> None:
        mock_session.execute.return_value = mock_execute_result([])

        results = await vector_store.search(make_embedding(), top_k=10)

        assert results == []

    async def test_search_calls_execute_once(
        self, vector_store: PGVectorStore, mock_session: MagicMock
    ) -> None:
        await vector_store.search(make_embedding(), top_k=5)

        mock_session.execute.assert_awaited_once()
