from chat.core.interfaces import IPromptBuilder
from chat.core.schemas import ChatMessage, Message
from common.schemas import RetrievedChunk

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer only from the provided context. "
    "Cite sources inline using their reference number, e.g. [1], [2]. "
    "If the context is insufficient, say so clearly."
)


class PromptBuilder(IPromptBuilder):
    def __init__(self, system_prompt: str | None = None) -> None:
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def build(
        self,
        history: list[Message],
        chunks: list[RetrievedChunk],
        query: str,
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self._system_prompt)
        ]

        for msg in history:
            messages.append(ChatMessage(role=msg.role, content=msg.content))

        context_lines = [
            f"[{i + 1}] {chunk.source_name}: {chunk.text}"
            for i, chunk in enumerate(chunks)
        ]

        if context_lines:
            context_block = "\n".join(context_lines)
            user_content = f"Context:\n{context_block}\n\nQuestion: {query}"
        else:
            user_content = f"Question: {query}"

        messages.append(ChatMessage(role="user", content=user_content))
        return messages
