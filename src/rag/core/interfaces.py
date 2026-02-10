from abc import ABC, abstractmethod
from typing import Any

from rag.core.models import DocumentChunk


class DocumentChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[DocumentChunk]: ...
