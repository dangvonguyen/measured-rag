import uuid
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector  # type:ignore
from sqlalchemy import (
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from common.db.base import Base, CreatedAtMixin, TimestampMixin
from rag.core.config import EMBEDDING_DIMENSION


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class DocumentChunkRow(Base, CreatedAtMixin):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk"),
        Index(
            "ix_rag_document_chunks_embedding_hnsw",
            "embedding_vector",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding_vector": "vector_cosine_ops"},
        ),
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


class ConversationRow(Base, TimestampMixin):
    __tablename__ = "conversations"
    __table_args__ = {
        "schema": "rag",
    }

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=lambda: {}, nullable=False
    )


class MessageRow(Base, CreatedAtMixin):
    __tablename__ = "messages"
    __table_args__ = {
        "schema": "rag",
    }

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("rag.conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, schema="rag"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)


class RAGResponseRow(Base, CreatedAtMixin):
    __tablename__ = "rag_responses"
    __table_args__ = (
        UniqueConstraint("message_id", name="uq_rag_response_message"),
        {"schema": "rag"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("rag.messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunks: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False
    )
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
