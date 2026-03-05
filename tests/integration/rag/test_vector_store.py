import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.core.schemas import DocumentChunk, RetrievedChunk
from rag.db.models import DocumentChunkRow
from rag.db.repositories.vector_store import VectorStore

DIMENSION = 1536


def make_chunk(
    chunk_id: str = "chunk-0",
    text: str = "Sample text.",
    chunk_index: int = 0,
    source_name: str = "test.md",
    document_id: str = "doc-0",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_name=source_name,
        chunk_index=chunk_index,
        text=text,
        token_count=10,
        metadata={},
    )


def make_embedding(index: int) -> list[float]:
    """Return a unit vector with 1.0 at position `index`, 0.0 elsewhere."""
    v = [0.0] * DIMENSION
    v[index] = 1.0
    return v


@pytest.mark.integration
class TestIngestIntegration:
    async def test_ingest_persists_rows_to_db(
        self, vector_store: VectorStore, db_session: AsyncSession
    ) -> None:
        chunks = [make_chunk(f"chunk-{i}", chunk_index=i) for i in range(2)]
        embeddings = [make_embedding(i) for i in range(2)]

        await vector_store.ingest(chunks, embeddings)

        result = await db_session.execute(
            select(DocumentChunkRow).where(DocumentChunkRow.document_id == "doc-0")
        )
        rows = result.scalars().all()
        assert len(rows) == 2
        assert {r.id for r in rows} == {"chunk-0", "chunk-1"}

    async def test_ingest_upsert_updates_existing_chunk(
        self, vector_store: VectorStore, db_session: AsyncSession
    ) -> None:
        """Re-ingesting the same chunk_id replaces the row, not duplicates it."""
        chunk = make_chunk("chunk-0", text="Original text")
        updated_chunk = make_chunk("chunk-0", text="Updated text")
        embedding = make_embedding(0)

        await vector_store.ingest([chunk], [embedding])
        await vector_store.ingest([updated_chunk], [embedding])

        result = await db_session.execute(
            select(DocumentChunkRow).where(DocumentChunkRow.id == "chunk-0")
        )
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].text == "Updated text"


@pytest.mark.integration
class TestSearchIntegration:
    async def test_search_returns_nearest_vector(
        self, vector_store: VectorStore
    ) -> None:
        """The chunk whose embedding is closest to the query ranks first."""
        chunks = [make_chunk(f"chunk-{i}", chunk_index=i) for i in range(3)]
        embeddings = [make_embedding(i) for i in range(3)]
        await vector_store.ingest(chunks, embeddings)

        # Query identical to chunk-2's embedding, chunk-2 should rank first
        results = await vector_store.search(make_embedding(2), top_k=3)

        assert len(results) > 0
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].chunk_id == "chunk-2"

    async def test_search_respects_score_threshold(
        self, vector_store: VectorStore
    ) -> None:
        """Chunks below the cosine similarity threshold are excluded."""
        chunks = [make_chunk(f"chunk-{i}", chunk_index=i) for i in range(3)]
        embeddings = [make_embedding(i) for i in range(3)]
        await vector_store.ingest(chunks, embeddings)

        # Query = e_0, similarity to chunk-0 ≈ 1.0
        results = await vector_store.search(make_embedding(0), top_k=10, threshold=0.9)

        ids = [r.chunk_id for r in results]
        assert "chunk-0" in ids
        assert "chunk-1" not in ids
        assert "chunk-2" not in ids

    async def test_search_limits_results_to_top_k(
        self, vector_store: VectorStore
    ) -> None:
        chunks = [make_chunk(f"chunk-tk-{i}", chunk_index=i) for i in range(5)]
        embeddings = [make_embedding(i) for i in range(5)]
        await vector_store.ingest(chunks, embeddings)

        results = await vector_store.search(make_embedding(0), top_k=2)

        assert len(results) <= 2

    async def test_search_with_metadata_filter(self, vector_store: VectorStore) -> None:
        """filters kwarg restricts results to matching source_name."""
        chunk_a = make_chunk("chunk-0", source_name="docs/a.md", chunk_index=0)
        chunk_b = make_chunk("chunk-1", source_name="docs/b.md", chunk_index=1)
        embeddings = [make_embedding(0), make_embedding(0)]  # same direction
        await vector_store.ingest([chunk_a, chunk_b], embeddings)

        results = await vector_store.search(
            make_embedding(0), top_k=10, filters={"source_name": "docs/a.md"}
        )

        ids = [r.chunk_id for r in results]
        assert "chunk-0" in ids
        assert "chunk-1" not in ids
