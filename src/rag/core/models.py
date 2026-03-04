from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    text: str = Field(min_length=1)
    document_id: str
    source_name: str
    file_path: str
    file_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    chunk_index: int
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any]


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
