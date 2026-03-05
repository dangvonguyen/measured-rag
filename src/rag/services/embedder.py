import logging

import tiktoken
from openai import AsyncOpenAI

from common.config import get_settings
from rag.core.config import EMBEDDING_DIMENSION, get_rag_settings
from rag.core.interfaces import IEmbeddingService

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService(IEmbeddingService):
    """OpenAI-backed embedding service."""

    def __init__(
        self,
        model: str | None = None,
        dimension: int | None = None,
        max_tokens: int | None = None,
        client: AsyncOpenAI | None = None,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> None:
        rag = get_rag_settings()
        self._model = model or rag.EMBEDDING_MODEL
        self._dimension = dimension or EMBEDDING_DIMENSION
        self._max_tokens = max_tokens or rag.EMBEDDING_MAX_TOKENS
        self._client = client or AsyncOpenAI(
            api_key=get_settings().OPENAI_API_KEY.get_secret_value()
        )
        self._tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        trimmed_text = self._truncate(text)
        response = await self._client.embeddings.create(
            model=self._model,
            input=trimmed_text,
            dimensions=self._dimension,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in a single API call."""
        if not texts:
            return []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Empty text at index {i} in batch")

        trimmed_texts = [self._truncate(t) for t in texts]
        response = await self._client.embeddings.create(
            model=self._model,
            input=trimmed_texts,
            dimensions=self._dimension,
        )
        # OpenAi may return items out of insertion order
        ordered = sorted(response.data, key=lambda item: item.index)
        return [d.embedding for d in ordered]

    @property
    def dimension(self) -> int:
        return self._dimension

    def _truncate(self, text: str) -> str:
        """Ensure text is within the token limit, truncating if needed."""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self._max_tokens:
            return text

        logger.warning(
            "Input text truncated from %d to %d tokens before embedding",
            len(tokens),
            self._max_tokens,
        )
        return self._tokenizer.decode(tokens[: self._max_tokens])
