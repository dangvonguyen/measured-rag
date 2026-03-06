from typing import Any

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any]
