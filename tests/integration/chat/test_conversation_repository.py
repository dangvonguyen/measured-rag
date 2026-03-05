import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from chat.core.schemas import Message, RAGResponseLog
from chat.db.models import MessageRow
from chat.db.repositories.conversation import ConversationRepository


@pytest.fixture
def repo(db_session: AsyncSession) -> ConversationRepository:
    return ConversationRepository(db_session)


async def _stamp(session: AsyncSession, *messages: Message) -> None:
    """Assign distinct created_at values to enforce stable ordering in tests."""
    t0 = datetime.now(UTC)
    for i, msg in enumerate(messages):
        await session.execute(
            update(MessageRow)
            .where(MessageRow.id == msg.id)
            .values(created_at=t0 + timedelta(seconds=i))
        )
    await session.flush()


@pytest.mark.integration
class TestCreateAndGet:
    async def test_create_and_get(self, repo: ConversationRepository) -> None:
        created = await repo.create({"key": "value"})
        found = await repo.get(created.id)

        assert found is not None
        assert found.id == created.id
        assert found.metadata["key"] == "value"

    async def test_get_missing_id(self, repo: ConversationRepository) -> None:
        result = await repo.get(uuid.uuid4())
        assert result is None


@pytest.mark.integration
class TestMessages:
    async def test_add_message(self, repo: ConversationRepository) -> None:
        conv = await repo.create({})
        msg = await repo.add_message(conv.id, "user", "Hello!")

        assert msg.id is not None
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.conversation_id == conv.id

    async def test_get_history_ordering(
        self, repo: ConversationRepository, db_session: AsyncSession
    ) -> None:
        conv = await repo.create({})
        m1 = await repo.add_message(conv.id, "user", "first")
        m2 = await repo.add_message(conv.id, "assistant", "second")
        m3 = await repo.add_message(conv.id, "user", "third")
        await _stamp(db_session, m1, m2, m3)

        history = await repo.get_history(conv.id, limit=10)

        assert [m.content for m in history] == ["first", "second", "third"]

    async def test_get_limited_history(
        self, repo: ConversationRepository, db_session: AsyncSession
    ) -> None:
        conv = await repo.create({})
        msgs = [await repo.add_message(conv.id, "user", f"msg{i}") for i in range(5)]
        await _stamp(db_session, *msgs)

        history = await repo.get_history(conv.id, limit=2)
        assert len(history) == 2
        assert [m.content for m in history] == ["msg3", "msg4"]

    async def test_get_empty_history(self, repo: ConversationRepository) -> None:
        conv = await repo.create({})
        history = await repo.get_history(conv.id, limit=10)
        assert history == []


@pytest.mark.integration
class TestRAGResponseLogging:
    async def test_log_rag_response_succeeds(
        self, repo: ConversationRepository
    ) -> None:
        conv = await repo.create({})
        msg = await repo.add_message(conv.id, "assistant", "The answer is 42.")
        log = RAGResponseLog(
            query="What is the answer?",
            retrieved_chunks=[
                {"chunk_id": "c1", "score": 0.9, "source_name": "doc.md"}
            ],
            prompt_tokens=100,
            completion_tokens=20,
            latency_ms=250.5,
        )

        await repo.log_rag_response(msg.id, log)
