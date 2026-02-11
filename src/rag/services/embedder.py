import logging

import tiktoken
from openai import AsyncOpenAI

from rag.core.interfaces import EmbeddingService

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIMENSION = 1536
_DEFAULT_MAX_TOKENS = 8191


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI-backed embedding service."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        dimension: int = _DEFAULT_DIMENSION,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        client: AsyncOpenAI | None = None,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._max_tokens = max_tokens
        self._client = client or AsyncOpenAI()
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
