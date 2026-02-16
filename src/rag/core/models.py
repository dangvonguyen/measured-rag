from typing import Any

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
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    metadata: dict[str, Any]


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any]
