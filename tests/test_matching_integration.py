"""Integration tests for the full matching pipeline with real data and LLM calls.

Tests load real TwinsUK + Arivale dictionaries, embed them, and run
match_dictionaries() end-to-end. Validates embedding recall quality for
known canary pairs and LLM structured output parsing.

Marked as integration: requires API keys, may incur costs, skipped in CI.
Skips gracefully if data files or API keys are unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

st = pytest.importorskip("sentence_transformers")

from ddharmon.embedding import EmbeddedDictionary, SentenceTransformerProvider, embed_dictionary
from ddharmon.ingestion import load_dictionary
from ddharmon.llm import CandidateJudgment, get_client
from ddharmon.llm.base import BaseLLMClient
from ddharmon.matching import MatchingConfig, match_dictionaries, retrieve_candidates
from ddharmon.matching.reranker import rerank_candidates
from ddharmon.models import DataDictionary
from ddharmon.models.enums import Relation
from ddharmon.models.mapping import MappingResult

logger = logging.getLogger(__name__)

# --- Paths ---
DATA_DIR = Path(__file__).parent.parent / "data" / "examples"
TWINSUK_DEMOGRAPHICS = DATA_DIR / "TwinsUK" / "TwinsUK_demographics.csv"
ARIVALE_DEMOGRAPHICS = DATA_DIR / "arivale" / "demographics_metadata.tsv"
CANARY_PAIRS_PATH = Path(__file__).parent.parent / "data" / "fixtures" / "canary_pairs.json"


# --- Module-scoped fixtures (avoid model reload / re-embedding overhead) ---


@pytest.fixture(scope="module")
def _shared_provider() -> SentenceTransformerProvider:
    """Shared embedding provider to avoid model reload per test."""
    return SentenceTransformerProvider()


@pytest.fixture(scope="module")
def twinsuk_demographics() -> DataDictionary:
    """Load TwinsUK demographics dictionary -- skips if file unavailable."""
    if not TWINSUK_DEMOGRAPHICS.exists():
        pytest.skip("TwinsUK demographics CSV not available")
    return load_dictionary(
        TWINSUK_DEMOGRAPHICS,
        cohort_name="TwinsUK",
        variable_name="Historical_ID",
        description="Phenotype_Description",
        category="Data_Type",
    )


@pytest.fixture(scope="module")
def arivale_demographics() -> DataDictionary:
    """Load Arivale demographics dictionary -- skips if file unavailable."""
    if not ARIVALE_DEMOGRAPHICS.exists():
        pytest.skip("Arivale demographics TSV not available")
    return load_dictionary(
        ARIVALE_DEMOGRAPHICS,
        cohort_name="Arivale",
        variable_name="Column Name",
        description="Description",
        category="Category",
        data_type="Variable Type",
        units="Units",
    )


@pytest.fixture(scope="module")
def embedded_dictionaries(
    twinsuk_demographics: DataDictionary,
    arivale_demographics: DataDictionary,
    _shared_provider: SentenceTransformerProvider,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[EmbeddedDictionary, EmbeddedDictionary]:
    """Embed both dictionaries (module-scoped to avoid re-embedding)."""
    cache_dir = tmp_path_factory.mktemp("embedding_cache")
    logger.info("Embedding TwinsUK demographics (%d fields)...", twinsuk_demographics.field_count)
    src = embed_dictionary(twinsuk_demographics, provider=_shared_provider, cache_dir=cache_dir)
    logger.info("Embedding Arivale demographics (%d fields)...", arivale_demographics.field_count)
    tgt = embed_dictionary(arivale_demographics, provider=_shared_provider, cache_dir=cache_dir)
    return src, tgt


@pytest.fixture(scope="module")
def llm_client() -> BaseLLMClient:
    """Get an LLM client -- tries Anthropic first, then OpenAI. Skips if no API key."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Using Anthropic LLM client")
        return get_client("anthropic")
    elif os.environ.get("OPENAI_API_KEY"):
        logger.info("Using OpenAI LLM client")
        return get_client("openai")
    else:
        pytest.skip("No LLM API key available (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")


@pytest.fixture(scope="module")
def canary_pairs() -> dict:
    """Load canary pair fixture."""
    if not CANARY_PAIRS_PATH.exists():
        pytest.skip("Canary pairs fixture not found")
    with open(CANARY_PAIRS_PATH) as f:
        return json.load(f)


# --- Helper functions ---


def _find_field_containing(dictionary: DataDictionary, substring: str) -> str | None:
    """Find a field whose variable_name or description contains the substring (case-insensitive)."""
    lower = substring.lower()
    for var_name, field in dictionary.fields.items():
        if lower in var_name.lower() or lower in field.description.lower():
            return var_name
    return None


# --- Integration tests ---


@pytest.mark.integration
class TestFullPipelineSmoke:
    """End-to-end pipeline smoke test with real data and real LLM."""

    def test_full_pipeline_smoke(
        self,
        embedded_dictionaries: tuple[EmbeddedDictionary, EmbeddedDictionary],
        llm_client: BaseLLMClient,
    ) -> None:
        """Load, embed, match end-to-end. Verify MappingResult shape and constraints."""
        source_emb, target_emb = embedded_dictionaries

        config = MatchingConfig(top_k=5, cosine_threshold=0.3)
        result = match_dictionaries(source_emb, target_emb, client=llm_client, config=config)

        # Structural assertions
        assert isinstance(result, MappingResult)
        assert result.source_name == "TwinsUK"
        assert result.target_name == "Arivale"
        assert len(result.mappings) > 0, "Expected at least one mapping from demographics data"

        # All mappings have valid Relation values
        valid_relations = set(Relation)
        for fm in result.mappings:
            assert fm.relation in valid_relations, f"Invalid relation: {fm.relation}"
            assert 0.0 <= fm.confidence <= 1.0, f"Confidence out of range: {fm.confidence}"

        # Log summary stats
        logger.info(
            "Pipeline result: %d mappings, %d auto_approved, %d pending, %d auto_rejected, "
            "%d source_unmapped, %d target_unmapped",
            len(result.mappings),
            len(result.auto_approved),
            len(result.pending_review),
            len(result.auto_rejected),
            len(result.source_unmapped),
            len(result.target_unmapped),
        )


@pytest.mark.integration
class TestEmbeddingRecallCanary:
    """Validate that cosine retrieval finds known matching pairs in top-5."""

    def test_embedding_recall_canary(
        self,
        embedded_dictionaries: tuple[EmbeddedDictionary, EmbeddedDictionary],
        canary_pairs: dict,
    ) -> None:
        """For each canary pair, check the expected target appears in top-5 candidates."""
        source_emb, target_emb = embedded_dictionaries
        demo_pairs = canary_pairs.get("twinsuk_arivale_demographics", {}).get("pairs", [])

        if not demo_pairs:
            pytest.skip("No canary pairs defined for twinsuk_arivale_demographics")

        # Retrieve candidates (embedding-only, no LLM)
        candidates = retrieve_candidates(source_emb, target_emb, top_k=5, cosine_threshold=0.0)

        hits = 0
        total_checked = 0

        for pair in demo_pairs:
            src_var = _find_field_containing(source_emb.dictionary, pair["source_contains"])
            tgt_var = _find_field_containing(target_emb.dictionary, pair["target_contains"])

            if src_var is None:
                logger.warning("Canary source '%s' not found in TwinsUK dictionary", pair["source_contains"])
                continue
            if tgt_var is None:
                logger.warning("Canary target '%s' not found in Arivale dictionary", pair["target_contains"])
                continue

            total_checked += 1
            src_candidates = candidates.get(src_var, [])
            candidate_names = [c[0] for c in src_candidates]

            if tgt_var in candidate_names:
                hits += 1
                logger.info(
                    "CANARY HIT: '%s' -> '%s' (rank %d)",
                    pair["source_contains"],
                    pair["target_contains"],
                    candidate_names.index(tgt_var) + 1,
                )
            else:
                logger.warning(
                    "CANARY MISS: '%s' -> '%s' not in top-5. Got: %s",
                    pair["source_contains"],
                    pair["target_contains"],
                    candidate_names[:5],
                )

        if total_checked == 0:
            pytest.skip("No canary pairs could be resolved to actual fields")

        recall = hits / total_checked
        logger.info("Embedding recall: %d/%d = %.1f%%", hits, total_checked, recall * 100)

        # Hard gate: embedding quality must be sufficient
        assert recall >= 0.8, (
            f"Embedding recall {recall:.1%} ({hits}/{total_checked}) below 80% threshold. "
            "Embedding model may need tuning or text composition needs improvement."
        )


@pytest.mark.integration
class TestLLMStructuredOutput:
    """Validate that LLM returns valid structured output for reranking."""

    def test_llm_structured_output(
        self,
        embedded_dictionaries: tuple[EmbeddedDictionary, EmbeddedDictionary],
        llm_client: BaseLLMClient,
    ) -> None:
        """Take one source field with top-5 candidates, rerank via LLM, validate response."""
        source_emb, target_emb = embedded_dictionaries

        # Retrieve candidates for all fields
        candidates = retrieve_candidates(source_emb, target_emb, top_k=5, cosine_threshold=0.3)

        # Pick the first source field that has candidates
        src_var = None
        src_candidates = []
        for var_name in sorted(candidates.keys()):
            if candidates[var_name]:
                src_var = var_name
                src_candidates = candidates[var_name]
                break

        assert src_var is not None, "No source field had candidates above cosine threshold"

        # Build field/cosine pairs for reranker
        src_field = source_emb.dictionary.fields[src_var]
        candidate_fields_with_scores: list[tuple] = []
        for tgt_var, cosine_score in src_candidates:
            tgt_field = target_emb.dictionary.fields[tgt_var]
            candidate_fields_with_scores.append((tgt_field, cosine_score))

        logger.info(
            "Reranking %d candidates for source field '%s' (%s)",
            len(candidate_fields_with_scores),
            src_var,
            src_field.description[:80],
        )

        # Call reranker directly
        judgments = rerank_candidates(
            llm_client,
            src_field,
            source_emb.dictionary,
            candidate_fields_with_scores,
            target_emb.dictionary,
        )

        assert len(judgments) > 0, "LLM returned no judgments"

        valid_relations = {"exact", "broader", "narrower", "composite", "derivable", "no_match"}
        for judgment, cosine_score in judgments:
            assert isinstance(judgment, CandidateJudgment)
            assert judgment.relation in valid_relations, f"Invalid relation: {judgment.relation}"
            assert 0.0 <= judgment.confidence <= 1.0, f"Confidence out of range: {judgment.confidence}"
            assert judgment.rationale, f"Empty rationale for candidate {judgment.candidate_variable}"
            logger.info(
                "  %s: relation=%s confidence=%.2f cosine=%.3f rationale='%s'",
                judgment.candidate_variable,
                judgment.relation,
                judgment.confidence,
                cosine_score,
                judgment.rationale[:60],
            )
