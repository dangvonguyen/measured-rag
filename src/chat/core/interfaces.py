from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from chat.core.schemas import ChatMessage, Conversation, Message, RAGResponseLog
from common.schemas import RetrievedChunk


class ILLMService(ABC):
    @abstractmethod
    def complete_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...


class IPromptBuilder(ABC):
    @abstractmethod
    def build(
        self, history: list[Message], chunks: list[RetrievedChunk], query: str
    ) -> list[ChatMessage]: ...


class IConversationRepository(ABC):
    @abstractmethod
    async def create(self, metadata: dict[str, Any]) -> Conversation: ...

    @abstractmethod
    async def get(self, conversation_id: UUID) -> Conversation | None: ...

    @abstractmethod
    async def get_history(self, conversation_id: UUID, limit: int) -> list[Message]: ...

    @abstractmethod
    async def add_message(
        self, conversation_id: UUID, role: str, content: str
    ) -> Message: ...

    @abstractmethod
    async def log_rag_response(self, message_id: UUID, log: RAGResponseLog) -> None: ...
