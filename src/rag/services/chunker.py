import uuid
from typing import Any

import tiktoken
from llama_index.core.node_parser import SentenceSplitter

from rag.core.interfaces import DocumentChunker
from rag.core.models import DocumentChunk


class SentenceChunker(DocumentChunker):
    def __init__(
        self,
        tokenizer: tiktoken.Encoding | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap cannot exceed chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=self._tokenizer.encode,
        )

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[DocumentChunk]:
        if not text or not text.strip():
            return []

        chunk_texts = self._splitter.split_text(text)
        chunks: list[DocumentChunk] = []
        for idx, chunk_text in enumerate(chunk_texts):
            token_count = len(self._tokenizer.encode(chunk_text))
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=str(metadata.get("document_id", "")),
                source_name=str(metadata.get("source_name", "")),
                chunk_index=idx,
                text=chunk_text,
                token_count=token_count,
                metadata={"page_number": metadata.get("page_number")},
            )
            chunks.append(chunk)

        return chunks
