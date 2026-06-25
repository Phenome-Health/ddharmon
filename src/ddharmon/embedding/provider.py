"""Embedding provider ABC and implementations.

Defines the abstract interface for embedding providers and provides
the default SentenceTransformerProvider for local CPU-based embedding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding generation goes through this interface. Concrete
    implementations must provide model_name, dimension, and embed().
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique model identifier, used as cache key component."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension (e.g., 768 for BioLORD-2023)."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            Array of shape (N, dimension) with float32 dtype.
            Embeddings are L2-normalized (cosine similarity = dot product).
        """
        ...


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence-transformers provider.

    Uses CPU-only inference with FremyCompany/BioLORD-2023 (768d) as default.
    Model is loaded once on construction and reused for all embed() calls.

    Note: First use downloads the model from HuggingFace Hub (~440MB).

    Default — FremyCompany/BioLORD-2023 (768d): a concept<->definition contrastive encoder. Selected over
    mpnet because survey-field-description<->CDE-definition matching is exactly its training objective. It
    is the only encoder of seven swept (mpnet, BioLORD, MedEmbed, PubMedBERT, E5, BGE-large) that wins BOTH
    the CDEMapper retrieval benchmark (hybrid recall@5 0.637->0.679) AND the held-out PhenX cross-cohort
    co-clustering separability (Δ 0.536->0.611) — retrieval-contrastive encoders top dense recall but
    collapse held-out separability. Pair with BM25+RRF (ddharmon.matching.hybrid_topk); do NOT ensemble with
    mpnet (it dilutes the better ranking). See benchmarks/README.md.

    Model options (pass as model_name):
        Biomedical-specialized:
            FremyCompany/BioLORD-2023           768d, concept<->definition contrastive (default)
            pritamdeka/S-PubMedBert-MS-MARCO    768d, PubMedBERT fine-tuned for retrieval

        General-purpose:
            all-mpnet-base-v2       768d, prior default; strong general-purpose baseline
            all-MiniLM-L6-v2        384d, 5x faster, good for prototyping

        Validate any swap on BOTH standing benchmarks (held-out PhenX, not just DEV recall) before adopting.
    """

    def __init__(self, model_name: str = "FremyCompany/BioLORD-2023") -> None:
        # Lazy import so the module can be imported without sentence-transformers installed
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        print(f"Embedding model loaded: {model_name} ({self._dimension}d)")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed texts using sentence-transformers.

        Args:
            texts: List of strings to embed.

        Returns:
            L2-normalized float32 array of shape (len(texts), dimension).
        """
        result: NDArray[np.float32] = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return result
