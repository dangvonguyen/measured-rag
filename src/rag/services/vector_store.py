from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.core.interfaces import VectorStore
from rag.core.models import DocumentChunk, RetrievedChunk
from rag.db.models import DocumentChunkRow


class PGVectorStore(VectorStore):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def ingest(
        self,
        document_id: str,
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

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            row = DocumentChunkRow(
                id=chunk.chunk_id,
                document_id=document_id,
                source_name=chunk.metadata["source_name"],
                chunk_index=chunk.metadata["chunk_index"],
                text=chunk.text,
                embedding_vector=embedding,
                meta=chunk.metadata,
            )
            await self._session.merge(row)

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
                col = getattr(DocumentChunkRow, key, None)
                if col is not None:
                    where_clauses.append(col == value)
                else:
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
