from typing import Any

import pytest
import tiktoken

from rag.core.schemas import DocumentChunk
from rag.ingestion.chunker import SentenceChunker


@pytest.mark.unit
class TestChunkTokenLimits:
    """All chunks must respect chunk_size."""

    def test_all_chunks_within_limit(
        self,
        tokenizer: tiktoken.Encoding,
        sample_tos_text: str,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        All chunks must not exceed the specified chunk_size.
        """
        chunk_size = 32
        chunker = SentenceChunker(
            tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=8
        )
        chunks = chunker.chunk(sample_tos_text, sample_metadata)

        for chunk in chunks:
            assert chunk.token_count <= chunk_size, (
                f"Chunk {chunk.chunk_index} has {chunk.token_count} "
                f"tokens, exceeds limit of {chunk_size}"
            )

    def test_token_count_accuracy(
        self,
        tokenizer: tiktoken.Encoding,
        sample_tos_text: str,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        Stored token_count must match actual tokenizer count.
        """
        chunker = SentenceChunker(tokenizer=tokenizer, chunk_size=32, chunk_overlap=8)
        chunks = chunker.chunk(sample_tos_text, sample_metadata)

        for chunk in chunks:
            actual = len(tokenizer.encode(chunk.text))
            assert chunk.token_count == actual, (
                f"Chunk {chunk.chunk_index}: stored={chunk.token_count}, "
                f"actual={actual}"
            )


@pytest.mark.unit
class TestContentPreservation:
    """No content loss after chunking."""

    def test_full_source_coverage(
        self,
        tokenizer: tiktoken.Encoding,
        sample_tos_text: str,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        Concatenating chunks with no overlap reconstructs original text.
        """
        chunker = SentenceChunker(tokenizer=tokenizer, chunk_size=32, chunk_overlap=0)
        chunks = chunker.chunk(sample_tos_text, sample_metadata)

        reconstructed = "".join(c.text for c in chunks)

        # Compare text content ignoring whitespace differences
        assert "".join(sample_tos_text.split()) == "".join(reconstructed.split()), (
            "Content mismatch after chunking (ignoring whitespace)"
        )


@pytest.mark.unit
class TestOverlap:
    """Overlap tokens between consecutive chunks."""

    def test_overlap_produces_duplicate_tokens(
        self,
        tokenizer: tiktoken.Encoding,
        sample_tos_text: str,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        When chunk_overlap > 0, the total token count across all chunks must
        exceed the source token count, proving that tokens are shared between
        consecutive chunks.
        """
        overlap = 5
        chunker = SentenceChunker(
            tokenizer=tokenizer, chunk_size=32, chunk_overlap=overlap
        )
        chunks = chunker.chunk(sample_tos_text, sample_metadata)

        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")

        source_token_count = len(tokenizer.encode(sample_tos_text))
        total_chunk_tokens = sum(len(tokenizer.encode(c.text)) for c in chunks)

        assert total_chunk_tokens > source_token_count, (
            "Total tokens across chunks should exceed source token count when "
            f"overlap > 0; source={source_token_count}, chunks={total_chunk_tokens}"
        )


@pytest.mark.unit
class TestMetadataPropagation:
    """Every chunk carries parent document metadata."""

    def test_metadata_on_all_chunks(
        self,
        tokenizer: tiktoken.Encoding,
        sample_tos_text: str,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        All chunks inherit parent document metadata.
        """
        chunker = SentenceChunker(tokenizer=tokenizer, chunk_size=32, chunk_overlap=8)
        chunks = chunker.chunk(sample_tos_text, sample_metadata)

        for chunk in chunks:
            assert chunk.document_id == sample_metadata["document_id"]
            assert chunk.source_name == sample_metadata["source_name"]
            assert chunk.metadata["page_number"] == sample_metadata["page_number"]

    def test_chunk_index_sequential(
        self,
        tokenizer: tiktoken.Encoding,
        sample_tos_text: str,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        Chunk indices are sequential starting from 0.
        """
        chunker = SentenceChunker(tokenizer=tokenizer, chunk_size=32, chunk_overlap=8)
        chunks = chunker.chunk(sample_tos_text, sample_metadata)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


@pytest.mark.unit
class TestEdgeCases:
    """Edge cases."""

    def test_empty_string_returns_empty(
        self,
        tokenizer: tiktoken.Encoding,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        Empty input returns empty list, no exception.
        """
        chunker = SentenceChunker(tokenizer=tokenizer, chunk_size=32, chunk_overlap=8)
        assert chunker.chunk("", sample_metadata) == []
        assert chunker.chunk("   ", sample_metadata) == []
        assert chunker.chunk("\n\n", sample_metadata) == []

    def test_single_sentence_returns_one_chunk(
        self,
        tokenizer: tiktoken.Encoding,
        sample_single_sentence: str,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        Short document returns exactly one chunk.
        """
        chunker = SentenceChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=0)
        chunks = chunker.chunk(sample_single_sentence, sample_metadata)
        assert len(chunks) == 1
        assert chunks[0].text == sample_single_sentence

    def test_unicode_handling(
        self,
        tokenizer: tiktoken.Encoding,
        sample_unicode_text: str,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        Unicode text handled correctly.
        """
        chunker = SentenceChunker(tokenizer=tokenizer, chunk_size=32, chunk_overlap=8)
        chunks = chunker.chunk(sample_unicode_text, sample_metadata)

        assert len(chunks) >= 1
        for chunk in chunks:
            chunk.text.encode("utf-8")
            actual = len(tokenizer.encode(chunk.text))
            assert chunk.token_count == actual

    def test_returns_document_chunk_type(
        self,
        tokenizer: tiktoken.Encoding,
        sample_tos_text: str,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        All returned chunks are DocumentChunk instances.
        """
        chunker = SentenceChunker(tokenizer=tokenizer, chunk_size=32, chunk_overlap=8)
        chunks = chunker.chunk(sample_tos_text, sample_metadata)
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)

    def test_chunk_ids_are_unique(
        self,
        tokenizer: tiktoken.Encoding,
        sample_tos_text: str,
        sample_metadata: dict[str, Any],
    ) -> None:
        """
        Each chunk has a unique id.
        """
        chunker = SentenceChunker(tokenizer=tokenizer, chunk_size=32, chunk_overlap=8)
        chunks = chunker.chunk(sample_tos_text, sample_metadata)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_configuration_validation(self, tokenizer: tiktoken.Encoding) -> None:
        """
        Validate configuration parameters.
        """
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SentenceChunker(tokenizer=tokenizer, chunk_size=0, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SentenceChunker(tokenizer=tokenizer, chunk_size=-10, chunk_overlap=5)

        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            SentenceChunker(tokenizer=tokenizer, chunk_size=50, chunk_overlap=-5)

        with pytest.raises(ValueError, match=r"chunk_overlap cannot exceed chunk_size"):
            SentenceChunker(tokenizer=tokenizer, chunk_size=50, chunk_overlap=100)
