import pytest

from rag.services.embedder import OpenAIEmbeddingService


@pytest.fixture(scope="module")
def sample_text() -> str:
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture(scope="module")
def expected_dimension(embedder: OpenAIEmbeddingService) -> int:
    return embedder.dimension


@pytest.mark.integration
class TestEmbedderIntegration:
    async def test_embed_text_returns_correct_dimension(
        self,
        embedder: OpenAIEmbeddingService,
        sample_text: str,
        expected_dimension: int,
    ) -> None:
        """
        A single string returns a vector of the correct length and type.
        """
        vector = await embedder.embed_text(sample_text)

        assert len(vector) == expected_dimension, (
            f"Expected dimension {expected_dimension}, got {len(vector)}"
        )
        assert all(isinstance(v, float) for v in vector), "Vector must contain floats"

    async def test_embed_text_is_deterministic(
        self, embedder: OpenAIEmbeddingService, sample_text: str
    ) -> None:
        """
        Same input produces the same output vector.
        """
        vector_a = await embedder.embed_text(sample_text)
        vector_b = await embedder.embed_text(sample_text)

        # Use approx for float comparison
        assert vector_a == pytest.approx(vector_b, abs=1e-9)

    async def test_embed_batch_single_matches_embed_text(
        self, embedder: OpenAIEmbeddingService, sample_text: str
    ) -> None:
        """
        Batch processing yields identical results to single processing.
        """
        single = await embedder.embed_text(sample_text)
        batch = await embedder.embed_batch([sample_text])

        assert batch[0] == single

    async def test_embed_batch_handles_empty_list(
        self, embedder: OpenAIEmbeddingService
    ) -> None:
        """
        An empty batch returns an empty result without error.
        """
        batch = await embedder.embed_batch([])
        assert batch == []

    @pytest.mark.parametrize("invalid_input", ["", "   \t\n"])
    async def test_embed_text_validates_input(
        self, embedder: OpenAIEmbeddingService, invalid_input: str
    ) -> None:
        """
        Verifies behavior on empty/whitespace strings.
        """
        with pytest.raises(ValueError, match="empty"):
            await embedder.embed_text(invalid_input)
