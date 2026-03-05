import logging
import time
from typing import Any

from rag.core.interfaces import IRetrieverService, IVectorStore
from rag.core.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class RetrieverService(IRetrieverService):
    def __init__(self, vector_store: IVectorStore) -> None:
        self._vector_store = vector_store

    async def retrieve(
        self,
        query_vector: list[float],
        top_k: int,
        threshold: float,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()

        results = await self._vector_store.search(
            query_vector,
            top_k=top_k,
            threshold=threshold,
            filters=filters,
        )

        if logger.isEnabledFor(logging.INFO):
            latency_ms = (time.perf_counter() - start) * 1000
            top_score = f"{results[0].score:.4f}" if results else "none"

            logger.info(
                "chunks=%d top_score=%s latency_ms=%.1f",
                len(results),
                top_score,
                latency_ms,
            )

        return results
