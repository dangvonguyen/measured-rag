from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chat.core.interfaces import IConversationRepository
from chat.core.schemas import Conversation, Message, RAGResponseLog
from chat.db.models import ConversationRow, MessageRole, MessageRow, RAGResponseRow


class PGConversationRepository(IConversationRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, metadata: dict[str, Any] | None = None) -> Conversation:
        row = ConversationRow(meta=metadata or {})
        self._session.add(row)
        await self._session.flush()
        return self._row_to_conversation(row)

    async def get(self, conversation_id: UUID) -> Conversation | None:
        row = await self._session.get(ConversationRow, conversation_id)
        return self._row_to_conversation(row) if row else None

    async def get_history(self, conversation_id: UUID, limit: int) -> list[Message]:
        stmt = (
            select(MessageRow)
            .where(MessageRow.conversation_id == conversation_id)
            .order_by(MessageRow.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        # Reverse to restore chronological order (oldest first)
        return [self._row_to_message(row) for row in reversed(rows)]

    async def add_message(
        self, conversation_id: UUID, role: str, content: str
    ) -> Message:
        row = MessageRow(
            conversation_id=conversation_id,
            role=MessageRole(role),
            content=content,
        )
        self._session.add(row)
        await self._session.flush()
        return self._row_to_message(row)

    async def log_rag_response(self, message_id: UUID, log: RAGResponseLog) -> None:
        row = RAGResponseRow(
            message_id=message_id,
            query=log.query,
            retrieved_chunks=log.retrieved_chunks,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            latency_ms=log.latency_ms,
        )
        self._session.add(row)

    def _row_to_conversation(self, row: ConversationRow) -> Conversation:
        return Conversation(
            id=row.id,
            created_at=row.created_at,
            metadata=row.meta,
        )

    def _row_to_message(self, row: MessageRow) -> Message:
        return Message(
            id=row.id,
            conversation_id=row.conversation_id,
            role=str(row.role),
            content=row.content,
            created_at=row.created_at,
        )
