from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    metadata: dict[str, Any]
