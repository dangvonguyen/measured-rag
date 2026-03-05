from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from rag.core.schemas import DocumentChunk, LoadedDocument, RetrievedChunk


class IDocumentLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> list[LoadedDocument]: ...


class IDocumentChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[DocumentChunk]: ...


class IEmbeddingService(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class IVectorStore(ABC):
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


class IRetrieverService(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int,
        threshold: float,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...
