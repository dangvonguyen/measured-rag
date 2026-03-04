import logging
from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from common.config import get_settings
from rag.core.config import get_rag_settings
from rag.core.interfaces import LLMService
from rag.core.models import ChatMessage

logger = logging.getLogger(__name__)


class OpenAILLMService(LLMService):
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        rag = get_rag_settings()

        self._model = model or rag.LLM_MODEL
        self._temperature = (
            temperature if temperature is not None else rag.LLM_TEMPERATURE
        )
        self._max_tokens = max_tokens or rag.LLM_MAX_TOKENS

        self._client = client or AsyncOpenAI(
            api_key=get_settings().OPENAI_API_KEY.get_secret_value()
        )

        self._validate_config()

    async def complete_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        try:
            formatted_messages = cast(
                list[ChatCompletionMessageParam],
                [m.model_dump(include={"role", "content"}) for m in messages],
            )

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=formatted_messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True,
            )

            async for chunk in response:
                content = self._extract_content(chunk)
                if content:
                    yield content

        except Exception as e:
            logger.exception(f"OpenAI Stream Error: {e}")
            raise

    def _validate_config(self) -> None:
        """
        Validate LLM configuration parameters.
        """
        if not self._model:
            raise ValueError("LLM model must be configured.")

        if not (0.0 <= self._temperature <= 2.0):
            raise ValueError("Temperature must be in [0.0, 2.0].")

        if self._max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

    def _extract_content(self, chunk: ChatCompletionChunk) -> str | None:
        """
        Helper to safely extract content from a chunk.
        """
        if not chunk.choices:
            return None
        return chunk.choices[0].delta.content
