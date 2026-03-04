from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Conversation(BaseModel):
    id: UUID
    created_at: datetime
    metadata: dict[str, Any]


class Message(BaseModel):
    id: UUID
    conversation_id: UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class RAGResponseLog(BaseModel):
    query: str
    retrieved_chunks: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
