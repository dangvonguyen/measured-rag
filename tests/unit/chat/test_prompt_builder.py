from datetime import UTC, datetime
from uuid import uuid4

import pytest

from chat.core.schemas import ChatMessage, Message
from chat.utils.prompt_builder import PromptBuilder
from common.schemas import RetrievedChunk


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            source_name="guide.md",
            chunk_index=0,
            text="Apples are red.",
            score=0.9,
            metadata={},
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="d1",
            source_name="guide.md",
            chunk_index=1,
            text="Bananas are yellow.",
            score=0.8,
            metadata={},
        ),
    ]


@pytest.fixture
def sample_history() -> list[Message]:
    conv_id = uuid4()
    now = datetime.now(UTC)
    return [
        Message(
            id=uuid4(),
            conversation_id=conv_id,
            role="user",
            content="What color is an apple?",
            created_at=now,
        ),
        Message(
            id=uuid4(),
            conversation_id=conv_id,
            role="assistant",
            content="Apples are red.",
            created_at=now,
        ),
    ]


@pytest.mark.unit
class TestPromptBuilderStructure:
    def test_returns_chat_messages(
        self, builder: PromptBuilder, sample_chunks: list[RetrievedChunk]
    ) -> None:
        msgs = builder.build([], sample_chunks, "test")
        assert all(isinstance(m, ChatMessage) for m in msgs)

    def test_first_message_is_system(
        self, builder: PromptBuilder, sample_chunks: list[RetrievedChunk]
    ) -> None:
        msgs = builder.build([], sample_chunks, "test")
        assert msgs[0].role == "system"

    def test_last_message_is_user(
        self, builder: PromptBuilder, sample_chunks: list[RetrievedChunk]
    ) -> None:
        msgs = builder.build([], sample_chunks, "What color?")
        assert msgs[-1].role == "user"

    def test_user_message_contains_query(
        self, builder: PromptBuilder, sample_chunks: list[RetrievedChunk]
    ) -> None:
        query = "What color is a banana?"
        msgs = builder.build([], sample_chunks, query)
        assert query in msgs[-1].content

    def test_context_contains_chunk_texts(
        self, builder: PromptBuilder, sample_chunks: list[RetrievedChunk]
    ) -> None:
        msgs = builder.build([], sample_chunks, "test")
        user_content = msgs[-1].content
        assert "Apples are red." in user_content
        assert "Bananas are yellow." in user_content

    def test_context_uses_numbered_citations(
        self, builder: PromptBuilder, sample_chunks: list[RetrievedChunk]
    ) -> None:
        msgs = builder.build([], sample_chunks, "test")
        user_content = msgs[-1].content
        assert "[1]" in user_content
        assert "[2]" in user_content

    def test_no_history_returns_two_messages(
        self, builder: PromptBuilder, sample_chunks: list[RetrievedChunk]
    ) -> None:
        msgs = builder.build([], sample_chunks, "test")
        assert len(msgs) == 2  # system + user

    def test_empty_chunks_builds_prompt_without_context(
        self, builder: PromptBuilder
    ) -> None:
        msgs = builder.build([], [], "What is 2+2?")
        assert len(msgs) == 2
        assert "2+2" in msgs[-1].content


@pytest.mark.unit
class TestPromptBuilderHistory:
    def test_history_appended_between_system_and_user(
        self,
        builder: PromptBuilder,
        sample_chunks: list[RetrievedChunk],
        sample_history: list[Message],
    ) -> None:
        msgs = builder.build(sample_history, sample_chunks, "Follow-up?")
        roles = [m.role for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]

    def test_history_content_preserved(
        self,
        builder: PromptBuilder,
        sample_chunks: list[RetrievedChunk],
        sample_history: list[Message],
    ) -> None:
        msgs = builder.build(sample_history, sample_chunks, "test")
        assert msgs[1].content == "What color is an apple?"
        assert msgs[2].content == "Apples are red."

    def test_custom_system_prompt(self, sample_chunks: list[RetrievedChunk]) -> None:
        custom = "Custom system instructions."
        builder = PromptBuilder(system_prompt=custom)
        msgs = builder.build([], sample_chunks, "test")
        assert msgs[0].content == custom
