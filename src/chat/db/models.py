import uuid
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Enum,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from common.db.base import Base, CreatedAtMixin, TimestampMixin


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationRow(Base, TimestampMixin):
    __tablename__ = "conversations"
    __table_args__ = {
        "schema": "chat",
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
        "schema": "chat",
    }

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat.conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, schema="chat", name="message_role"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)


class RAGResponseRow(Base, CreatedAtMixin):
    __tablename__ = "rag_responses"
    __table_args__ = (
        UniqueConstraint("message_id", name="uq_rag_response_message"),
        {"schema": "chat"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat.messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunks: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False
    )
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
