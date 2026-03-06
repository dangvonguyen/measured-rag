from abc import ABC, abstractmethod
from typing import Any

from common.schemas import RetrievedChunk


class IRetrieverService(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int,
        threshold: float,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...
