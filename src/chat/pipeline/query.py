import time
from collections.abc import AsyncGenerator
from uuid import UUID

import tiktoken

from chat.core.interfaces import IConversationRepository, ILLMService, IPromptBuilder
from chat.core.schemas import RAGResponseLog
from rag.core.config import RAGSettings
from rag.core.interfaces import IRetrieverService


class QueryPipeline:
    """Online query orchestrator."""

    def __init__(
        self,
        retriever: IRetrieverService,
        prompt_builder: IPromptBuilder,
        llm: ILLMService,
        conversation_repo: IConversationRepository,
        settings: RAGSettings,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._conv_repo = conversation_repo
        self._settings = settings

    async def run_stream(
        self, conversation_id: UUID, query: str
    ) -> AsyncGenerator[str]:
        """
        Stream an LLM response grounded in retrieved context.

        Orchestration order:
        1. Load conversation history
        2. Retrieve chunks
        3. Build prompt
        4. Persist user message
        5. Stream LLM
        6. Persist assistant message + log RAGResponse
        """
        history = await self._conv_repo.get_history(
            conversation_id, limit=self._settings.HISTORY_MAX_TURNS * 2
        )

        chunks = await self._retriever.retrieve(
            query,
            top_k=self._settings.RETRIEVAL_TOP_K,
            threshold=self._settings.RETRIEVAL_THRESHOLD,
        )
        messages = self._prompt_builder.build(history, chunks, query)

        await self._conv_repo.add_message(conversation_id, "user", query)

        accumulated = ""
        start = time.perf_counter()
        try:
            async for token in self._llm.complete_stream(messages):
                accumulated += token
                yield token
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            assistant_msg = await self._conv_repo.add_message(
                conversation_id, "assistant", accumulated
            )

            enc = tiktoken.encoding_for_model(self._settings.LLM_MODEL)
            prompt_tokens = sum(len(enc.encode(m.content)) for m in messages)
            completion_tokens = len(enc.encode(accumulated)) if accumulated else 0

            log = RAGResponseLog(
                query=query,
                retrieved_chunks=[
                    {
                        "chunk_id": c.chunk_id,
                        "score": c.score,
                        "source_name": c.source_name,
                    }
                    for c in chunks
                ],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
            await self._conv_repo.log_rag_response(assistant_msg.id, log)
