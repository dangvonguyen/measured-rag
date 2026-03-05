from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.db.models import DocumentChunkRow
from rag.db.repositories.vector_store import VectorStore
from rag.ingestion.chunker import SentenceChunker
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader()


@pytest.fixture
def chunker() -> SentenceChunker:
    return SentenceChunker(chunk_size=32, chunk_overlap=0)


@pytest.fixture
def mock_embedder() -> AsyncMock:
    """Fake EmbeddingService returning deterministic unit vectors (no API call)."""
    embedder = AsyncMock()
    embedder.embed_batch = AsyncMock(
        side_effect=lambda texts: [[0.1] * 1536 for _ in texts]
    )
    return embedder


@pytest.fixture
def pipeline(
    loader: DocumentLoader,
    chunker: SentenceChunker,
    mock_embedder: AsyncMock,
    vector_store: VectorStore,
) -> IngestionPipeline:
    return IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedder=mock_embedder,
        vector_store=vector_store,
    )


@pytest.mark.integration
class TestIngestionWorkflows:
    async def test_idempotent_reingestion(
        self, pipeline: IngestionPipeline, db_session: AsyncSession, documents_dir: Path
    ) -> None:
        doc_name = "terms_of_service.md"
        doc_path = documents_dir / doc_name

        # First pass
        stats_1 = await pipeline.ingest(doc_path)
        count_1 = stats_1[doc_name]

        # Second pass
        stats_2 = await pipeline.ingest(doc_path)
        count_2 = stats_2[doc_name]

        assert count_2 == count_1

        # Verify database state (no duplicate rows)
        stmt = select(DocumentChunkRow).where(DocumentChunkRow.source_name == doc_name)
        rows = (await db_session.execute(stmt)).scalars().all()
        assert len(rows) == count_1

    async def test_partial_failure_rolls_back(
        self,
        loader: DocumentLoader,
        chunker: SentenceChunker,
        vector_store: VectorStore,
        db_session: AsyncSession,
        documents_dir: Path,
    ) -> None:
        """A failure during embedding rolls back the deletion of existing chunks."""
        doc_name = "terms_of_service.md"
        doc_path = documents_dir / doc_name

        # Successful initial ingestion
        ok_embedder = AsyncMock()
        ok_embedder.embed_batch = AsyncMock(
            side_effect=lambda texts: [[0.1] * 1536 for _ in texts]
        )
        p_ok = IngestionPipeline(loader, chunker, ok_embedder, vector_store)

        result = await p_ok.ingest(doc_path)
        initial_count = result[doc_name]

        # Re-ingest with failing embedder
        fail_embedder = AsyncMock()
        fail_embedder.embed_batch = AsyncMock(
            side_effect=RuntimeError("Simulated embed failure")
        )
        p_fail = IngestionPipeline(loader, chunker, fail_embedder, vector_store)

        with pytest.raises(RuntimeError, match="Simulated embed failure"):
            await p_fail.ingest(doc_path)

        # Verify original chunks were restored (not deleted)
        stmt = select(DocumentChunkRow).where(DocumentChunkRow.source_name == doc_name)
        rows_after = (await db_session.execute(stmt)).scalars().all()
        assert len(rows_after) == initial_count


@pytest.mark.integration
class TestMetadataValidation:
    """IT-ING-04: Every chunk produced carries all required metadata fields."""

    async def test_chunk_metadata_completeness(
        self, loader: DocumentLoader, chunker: SentenceChunker, documents_dir: Path
    ) -> None:
        """Verify that chunker propagates all required metadata"""
        docs = loader.load(documents_dir / "terms_of_service.md")
        doc = docs[0]

        chunks = chunker.chunk(
            doc.text,
            metadata={"document_id": doc.document_id, "source_name": doc.source_name},
        )

        for chunk in chunks:
            assert chunk.document_id
            assert chunk.source_name
            assert chunk.chunk_index >= 0
            assert chunk.token_count > 0
