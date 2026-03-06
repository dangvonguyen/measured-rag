from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from common.schemas import RetrievedChunk
from rag.core.interfaces import IVectorStore
from rag.core.schemas import DocumentChunk
from rag.db.models import DocumentChunkRow

# Define whitelist for top-level schema columns to prevent attribute injection
_ALLOWED_FILTER_COLS = frozenset({"document_id", "source_name", "chunk_index"})


class VectorStore(IVectorStore):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def ingest(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings must have the same length: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        if not chunks:
            return

        # Prepare records for bulk insert
        records = [
            {
                "id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_name": chunk.source_name,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "embedding_vector": embedding,
                "meta": chunk.metadata,
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        # Use postgresql dialect insert for bulk upsert
        stmt = insert(DocumentChunkRow).values(records)

        # Define fields to update if the chunk_id already exists
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=[DocumentChunkRow.id],
            set_={
                "text": stmt.excluded.text,
                "embedding_vector": stmt.excluded.embedding_vector,
                "metadata": stmt.excluded.metadata,
                "source_name": stmt.excluded.source_name,
                "chunk_index": stmt.excluded.chunk_index,
            },
        )

        await self._session.execute(upsert_stmt)
        await self._session.flush()

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        distance = DocumentChunkRow.embedding_vector.cosine_distance(query_vector)
        score_col = (1 - distance).label("score")

        where_clauses = [score_col >= threshold]

        if filters:
            for key, value in filters.items():
                if key in _ALLOWED_FILTER_COLS:
                    col = getattr(DocumentChunkRow, key)
                    where_clauses.append(col == value)
                else:
                    if not isinstance(key, str):
                        raise TypeError(
                            f"Filter key must be str, got {type(key).__name__!r}"
                        )
                    where_clauses.append(DocumentChunkRow.meta[key] == value)

        stmt = (
            select(
                DocumentChunkRow.id,
                DocumentChunkRow.document_id,
                DocumentChunkRow.source_name,
                DocumentChunkRow.chunk_index,
                DocumentChunkRow.text,
                DocumentChunkRow.meta,
                score_col,
            )
            .where(*where_clauses)
            .order_by(distance)
            .limit(top_k)
        )

        result = await self._session.execute(stmt)
        rows = result.all()

        return [
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                source_name=row.source_name,
                chunk_index=row.chunk_index,
                text=row.text,
                score=row.score,
                metadata=row.meta,
            )
            for row in rows
        ]

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None]:
        """Wrap a unit of work in a savepoint."""
        async with self._session.begin_nested():
            yield

    async def delete_by_source(self, source_name: str) -> int:
        """Delete all chunks with this source_name and return the deleted row count."""
        stmt = delete(DocumentChunkRow).where(
            DocumentChunkRow.source_name == source_name
        )
        result = cast(CursorResult[Any], await self._session.execute(stmt))
        await self._session.flush()
        return result.rowcount
