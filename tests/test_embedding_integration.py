"""Integration tests for the embedding layer with real model on TwinsUK data.

Tests embed_dictionary() end-to-end with the SentenceTransformerProvider,
verifying caching, determinism, and semantic similarity results.

Skips automatically if sentence-transformers or TwinsUK CSV is unavailable.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

st = pytest.importorskip("sentence_transformers")

from ddharmon.embedding import EmbeddedDictionary, SentenceTransformerProvider, embed_dictionary, find_similar
from ddharmon.ingestion import load_dictionary
from ddharmon.models import DataDictionary

TWINSUK_DEMOGRAPHICS = Path(__file__).parent.parent / "data" / "examples" / "TwinsUK" / "TwinsUK_demographics.csv"


@pytest.fixture(scope="module")
def _shared_provider() -> SentenceTransformerProvider:
    """Shared provider to avoid model reload per test."""
    return SentenceTransformerProvider()


@pytest.fixture
def twinsuk_demographics() -> DataDictionary:
    """Load TwinsUK demographics dictionary -- skips if file not available."""
    if not TWINSUK_DEMOGRAPHICS.exists():
        pytest.skip("TwinsUK demographics CSV not available")
    return load_dictionary(
        TWINSUK_DEMOGRAPHICS,
        cohort_name="TwinsUK",
        variable_name="Historical_ID",
        description="Phenotype_Description",
        category="Data_Type",
    )


class TestEmbeddingIntegration:
    """End-to-end tests with real SentenceTransformer model."""

    def test_embed_twinsuk_demographics(
        self, twinsuk_demographics: DataDictionary, tmp_path: Path, _shared_provider: SentenceTransformerProvider
    ) -> None:
        """Load TwinsUK demographics, embed with default provider, verify field count."""
        result = embed_dictionary(twinsuk_demographics, provider=_shared_provider, cache_dir=tmp_path)

        assert isinstance(result, EmbeddedDictionary)
        assert len(result.embeddings) == twinsuk_demographics.field_count
        assert result.model_name == "all-mpnet-base-v2"

        # Verify vector dimensions
        matrix = result.get_all_vectors()
        assert matrix.shape[0] == twinsuk_demographics.field_count
        assert matrix.shape[1] == 768  # all-mpnet-base-v2 dimension

    def test_cache_hit_fast(
        self, twinsuk_demographics: DataDictionary, tmp_path: Path, _shared_provider: SentenceTransformerProvider
    ) -> None:
        """Re-embed same dictionary from cache -- completes in < 1 second."""
        # First run: embeds and caches
        embed_dictionary(twinsuk_demographics, provider=_shared_provider, cache_dir=tmp_path)

        # Second run: should be pure cache hit (no model loading or embedding)
        start = time.monotonic()
        result = embed_dictionary(twinsuk_demographics, provider=_shared_provider, cache_dir=tmp_path)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"Cache re-embed took {elapsed:.2f}s, expected < 1.0s"
        assert len(result.embeddings) == twinsuk_demographics.field_count

    def test_find_similar_self_match(
        self, twinsuk_demographics: DataDictionary, tmp_path: Path, _shared_provider: SentenceTransformerProvider
    ) -> None:
        """find_similar() on a field returns itself as top result with score ~1.0."""
        result = embed_dictionary(twinsuk_demographics, provider=_shared_provider, cache_dir=tmp_path)

        # Pick the first field as query
        names = result.get_variable_names()
        query_var = names[0]
        query_vec = result.embeddings[query_var]
        matrix = result.get_all_vectors()
        top_results = find_similar(query_vec, matrix, top_k=5)

        # The query itself should be the top result (similarity ~1.0)
        top_idx, top_score = top_results[0]
        assert names[top_idx] == query_var
        assert top_score > 0.99

    def test_deterministic_embeddings(
        self, twinsuk_demographics: DataDictionary, tmp_path: Path, _shared_provider: SentenceTransformerProvider
    ) -> None:
        """Embeddings are deterministic -- same input produces same vectors across two runs."""
        cache_dir_1 = tmp_path / "run1"
        cache_dir_2 = tmp_path / "run2"

        result1 = embed_dictionary(twinsuk_demographics, provider=_shared_provider, cache_dir=cache_dir_1)
        result2 = embed_dictionary(twinsuk_demographics, provider=_shared_provider, cache_dir=cache_dir_2)

        for var_name in result1.embeddings:
            np.testing.assert_array_almost_equal(
                result1.embeddings[var_name],
                result2.embeddings[var_name],
                decimal=6,
                err_msg=f"Non-deterministic embedding for {var_name}",
            )
