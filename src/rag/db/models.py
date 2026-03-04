from typing import Any

from pgvector.sqlalchemy import Vector  # type:ignore
from sqlalchemy import Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from common.db.base import Base, CreatedAtMixin
from rag.core.config import EMBEDDING_DIMENSION


class DocumentChunkRow(Base, CreatedAtMixin):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk"),
        {"schema": "rag"},
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_name: Mapped[str] = mapped_column(String, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_vector: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION), nullable=False
    )
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=lambda: {}, nullable=False
    )
