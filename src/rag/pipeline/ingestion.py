import logging
from pathlib import Path

from rag.core.interfaces import (
    DocumentChunker,
    DocumentLoader,
    EmbeddingService,
    VectorStore,
)
from rag.core.models import LoadedDocument

logger = logging.getLogger(__name__)

IngestionResult = dict[str, int]


class IngestionPipeline:
    """Offline ingestion orchestrator."""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: DocumentChunker,
        embedder: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    async def ingest(self, path: Path) -> IngestionResult:
        """Load, chunk, embed, and store all documents at path.

        Returns {source_name: chunk_count} for each processed document.
        """
        docs = self._loader.load(path)

        if not docs:
            logger.warning("No documents found at %s", path)
            return {}

        # Begin transaction for entire batch
        async with self._vector_store.transaction():
            results: IngestionResult = {}

            for doc in docs:
                chunk_count = await self._ingest_document(doc)
                results[doc.source_name] = chunk_count

            return results

    async def _ingest_document(self, doc: LoadedDocument) -> int:
        """
        Atomically replace chunks for a single document.
        Must be called within an active transaction.
        """
        # Delete existing chunks
        deleted = await self._vector_store.delete_by_source(doc.source_name)

        chunks = self._chunker.chunk(
            doc.text,
            metadata={
                "document_id": doc.document_id,
                "source_name": doc.source_name,
                "page_number": None,
            },
        )

        if not chunks:
            logger.warning(
                "Document '%s' produced zero chunks; skipping.", doc.source_name
            )
            return 0

        # Batch embed and store
        embeddings = await self._embedder.embed_batch([c.text for c in chunks])

        await self._vector_store.ingest(doc.document_id, chunks, embeddings)

        logger.info(
            "Ingested '%s': %d chunks (replaced %d old)",
            doc.source_name,
            len(chunks),
            deleted,
        )
        return len(chunks)
