from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from rag.core.models import (
    ChatMessage,
    DocumentChunk,
    LoadedDocument,
    Message,
    RetrievedChunk,
)


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> list[LoadedDocument]: ...


class DocumentChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[DocumentChunk]: ...


class EmbeddingService(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class VectorStore(ABC):
    @abstractmethod
    async def ingest(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None: ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    async def delete_by_source(self, source_name: str) -> int: ...

    @abstractmethod
    def transaction(self) -> AbstractAsyncContextManager[None]:
        """Async context manager that wraps a unit of work in a savepoint.

        Commits on success; rolls back to it on exception.
        """
        ...


class RetrieverService(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query_vector: list[float],
        top_k: int,
        threshold: float,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...


class PromptBuilder(ABC):
    @abstractmethod
    def build(
        self,
        history: list[Message],
        chunks: list[RetrievedChunk],
        query: str,
    ) -> list[ChatMessage]: ...


class LLMService(ABC):
    @abstractmethod
    def complete_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...
