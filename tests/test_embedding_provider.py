"""Tests for embedding provider ABC and SentenceTransformerProvider."""

from __future__ import annotations

import numpy as np
import pytest

from ddharmon.embedding.provider import EmbeddingProvider, _embedding_dimension


class MockProvider(EmbeddingProvider):
    """Mock provider implementing EmbeddingProvider for interface compliance."""

    def __init__(self, model_name: str = "mock-model", dimension: int = 128):
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(texts), self._dimension)).astype(np.float32)
        # L2-normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


class TestEmbeddingProviderABC:
    """Tests for EmbeddingProvider abstract base class."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_mock_provider_satisfies_interface(self) -> None:
        """A mock provider implementing EmbeddingProvider satisfies the interface."""
        provider = MockProvider()
        assert isinstance(provider, EmbeddingProvider)
        assert provider.model_name == "mock-model"
        assert provider.dimension == 128
        result = provider.embed(["hello"])
        assert result.shape == (1, 128)
        assert result.dtype == np.float32


# All SentenceTransformerProvider tests require sentence-transformers
st = pytest.importorskip("sentence_transformers")


@pytest.mark.integration
class TestSentenceTransformerProvider:
    """Tests for SentenceTransformerProvider with the real model.

    Marked ``integration`` because constructing the provider downloads the encoder
    (~440MB BioLORD-2023, plus MiniLM) from the HuggingFace Hub. The fast per-PR CI
    gate deselects these via ``-m "not integration"`` so a stalled HF download can
    never hang the pipeline; the real encoder is still exercised weekly by the
    benchmark gate (``.github/workflows/benchmarks.yml``).
    """

    @pytest.fixture(scope="class")
    def provider(self):
        from ddharmon.embedding.provider import SentenceTransformerProvider

        return SentenceTransformerProvider()

    @pytest.fixture(scope="class")
    def minilm_provider(self):
        from ddharmon.embedding.provider import SentenceTransformerProvider

        return SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")

    def test_model_name_returns_configured_string(self, provider) -> None:
        """SentenceTransformerProvider.model_name returns the configured default (BioLORD-2023)."""
        assert provider.model_name == "FremyCompany/BioLORD-2023"

    def test_dimension_returns_768(self, provider) -> None:
        """SentenceTransformerProvider.dimension returns 768 for the default BioLORD-2023 encoder."""
        assert provider.dimension == 768

    def test_embed_returns_correct_shape(self, provider) -> None:
        """embed(["hello", "world"]) returns ndarray shape (2, 768) with float32 dtype."""
        result = provider.embed(["hello", "world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 768)
        assert result.dtype == np.float32

    def test_embeddings_are_l2_normalized(self, provider) -> None:
        """Returned embeddings are L2-normalized (norm ~= 1.0 for each row)."""
        result = provider.embed(["hello", "world", "test sentence"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_custom_model_name_dimension(self, minilm_provider) -> None:
        """Custom model name accepted (all-MiniLM-L6-v2 returns dimension 384)."""
        assert minilm_provider.model_name == "all-MiniLM-L6-v2"
        assert minilm_provider.dimension == 384

    def test_custom_model_embed_shape(self, minilm_provider) -> None:
        """Custom model embeds to correct dimension."""
        result = minilm_provider.embed(["test"])
        assert result.shape == (1, 384)


class TestEmbeddingDimensionAcrossTheRename:
    """`_embedding_dimension` must survive the sentence-transformers 5.x -> 6.x rename.

    sentence-transformers 6.0 renamed `get_sentence_embedding_dimension` to
    `get_embedding_dimension` and warns on the old name, so the old name will eventually stop
    working. Real installs are covered on both sides of the rename with test doubles rather than a
    version pin — the `pyproject.toml` floor is `>=3.0.0`, which spans both APIs.
    """

    def test_prefers_the_new_name(self) -> None:
        class New:
            def get_embedding_dimension(self) -> int:
                return 768

        assert _embedding_dimension(New()) == 768

    def test_falls_back_to_the_pre_6_0_name(self) -> None:
        class Old:
            def get_sentence_embedding_dimension(self) -> int:
                return 384

        assert _embedding_dimension(Old()) == 384

    def test_new_name_wins_when_both_exist(self) -> None:
        """6.x keeps the old name as a warning shim, so both are present — take the un-deprecated one."""

        class Both:
            def get_embedding_dimension(self) -> int:
                return 768

            def get_sentence_embedding_dimension(self) -> int:  # pragma: no cover - must not be called
                raise AssertionError("the deprecated getter must not be preferred")

        assert _embedding_dimension(Both()) == 768

    def test_raises_when_neither_name_exists(self) -> None:
        """A future rename must fail loudly, not silently produce a wrong width."""
        with pytest.raises(AttributeError, match="neither get_embedding_dimension"):
            _embedding_dimension(object())

    def test_raises_rather_than_caching_a_none_width(self) -> None:
        """The width is written into the embedding cache schema, so None must not propagate."""

        class NoFixedWidth:
            def get_embedding_dimension(self) -> None:
                return None

        with pytest.raises(ValueError, match="no fixed output width"):
            _embedding_dimension(NoFixedWidth())
