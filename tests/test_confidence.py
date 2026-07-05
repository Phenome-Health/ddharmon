"""Tests for confidence scoring and triage."""

from __future__ import annotations

import pytest

from ddharmon.models.enums import ReviewStatus


class TestConfidenceConfig:
    """Tests for ConfidenceConfig dataclass."""

    def test_default_weights(self):
        from ddharmon.matching.confidence import ConfidenceConfig

        config = ConfidenceConfig()
        assert config.llm_weight == 0.6
        assert config.cosine_weight == 0.4
        assert config.auto_approve_threshold == 0.9
        assert config.auto_reject_threshold == 0.3

    def test_weights_must_sum_to_one(self):
        from ddharmon.matching.confidence import ConfidenceConfig

        with pytest.raises(ValueError, match="sum to 1.0"):
            ConfidenceConfig(llm_weight=0.5, cosine_weight=0.3)

    def test_custom_thresholds(self):
        from ddharmon.matching.confidence import ConfidenceConfig

        config = ConfidenceConfig(
            auto_approve_threshold=0.95,
            auto_reject_threshold=0.2,
        )
        assert config.auto_approve_threshold == 0.95
        assert config.auto_reject_threshold == 0.2


class TestScoreMapping:
    """Tests for score_mapping()."""

    def test_weighted_composite(self):
        from ddharmon.matching.confidence import score_mapping

        # 0.6*0.95 + 0.4*0.85 = 0.57 + 0.34 = 0.91
        score = score_mapping(llm_confidence=0.95, cosine_similarity=0.85)
        assert abs(score - 0.91) < 1e-6

    def test_clamped_to_zero(self):
        from ddharmon.matching.confidence import score_mapping

        score = score_mapping(llm_confidence=-0.5, cosine_similarity=-0.5)
        assert score == 0.0

    def test_clamped_to_one(self):
        from ddharmon.matching.confidence import score_mapping

        score = score_mapping(llm_confidence=1.5, cosine_similarity=1.5)
        assert score == 1.0

    def test_custom_config(self):
        from ddharmon.matching.confidence import ConfidenceConfig, score_mapping

        config = ConfidenceConfig(llm_weight=0.5, cosine_weight=0.5)
        # 0.5*0.8 + 0.5*0.6 = 0.4 + 0.3 = 0.7
        score = score_mapping(llm_confidence=0.8, cosine_similarity=0.6, config=config)
        assert abs(score - 0.7) < 1e-6


class TestTriageMapping:
    """Tests for triage_mapping()."""

    def test_auto_approved(self):
        from ddharmon.matching.confidence import triage_mapping

        assert triage_mapping(0.91) == ReviewStatus.AUTO_APPROVED

    def test_pending_review(self):
        from ddharmon.matching.confidence import triage_mapping

        assert triage_mapping(0.5) == ReviewStatus.PENDING_REVIEW

    def test_auto_rejected(self):
        from ddharmon.matching.confidence import triage_mapping

        assert triage_mapping(0.2) == ReviewStatus.AUTO_REJECTED

    def test_boundary_approve(self):
        from ddharmon.matching.confidence import triage_mapping

        # Exactly at threshold = approved
        assert triage_mapping(0.9) == ReviewStatus.AUTO_APPROVED

    def test_boundary_reject(self):
        from ddharmon.matching.confidence import triage_mapping

        # Exactly at threshold = rejected
        assert triage_mapping(0.3) == ReviewStatus.AUTO_REJECTED

    def test_custom_thresholds(self):
        from ddharmon.matching.confidence import ConfidenceConfig, triage_mapping

        config = ConfidenceConfig(auto_approve_threshold=0.95, auto_reject_threshold=0.1)
        # 0.91 would normally be auto_approved, but with 0.95 threshold it's pending
        assert triage_mapping(0.91, config=config) == ReviewStatus.PENDING_REVIEW
        # 0.2 would normally be auto_rejected, but with 0.1 threshold it's pending
        assert triage_mapping(0.2, config=config) == ReviewStatus.PENDING_REVIEW


class TestScoreTransformSpec:
    """C2 units-driven transform-spec confidence + triage routing."""

    def _spec(self, **kw):
        from ddharmon.harmonization.models import TransformKind, TransformSpec

        base = {"source_variable": "C:v", "target_cde_id": "CDE", "kind": TransformKind.UNIT}
        base.update(kw)
        return TransformSpec(**base)

    def test_none_scores_zero_and_rejects(self):
        from ddharmon.harmonization.models import TransformKind
        from ddharmon.matching.confidence import score_transform_spec, triage_mapping

        s = self._spec(kind=TransformKind.NONE)
        assert score_transform_spec(s) == 0.0
        assert triage_mapping(score_transform_spec(s)) == ReviewStatus.AUTO_REJECTED

    def test_identity_is_high(self):
        from ddharmon.harmonization.models import TransformKind
        from ddharmon.matching.confidence import score_transform_spec

        assert score_transform_spec(self._spec(kind=TransformKind.IDENTITY)) == 0.9

    def test_unit_known_conversion_is_high_and_approves(self):
        from ddharmon.matching.confidence import score_transform_spec, triage_mapping

        s = self._spec(factor=2.2046, offset=0.0, source_unit="kg", target_unit="lb")
        assert score_transform_spec(s) == 0.9
        assert triage_mapping(score_transform_spec(s)) == ReviewStatus.AUTO_APPROVED

    def test_unit_missing_units_is_low_and_reviews_not_rejects(self):
        from ddharmon.matching.confidence import score_transform_spec, triage_mapping

        s = self._spec(needs_units=True)
        assert score_transform_spec(s) == 0.4  # low band, but ABOVE the 0.3 reject cutoff
        assert triage_mapping(score_transform_spec(s)) == ReviewStatus.PENDING_REVIEW

    def test_unit_inferred_is_medium(self):
        from ddharmon.matching.confidence import score_transform_spec

        # a UNIT spec with no recognized units/factor -> inferred -> medium
        assert score_transform_spec(self._spec()) == 0.6

    def test_arithmetic_unverified_is_medium(self):
        from ddharmon.harmonization.models import TransformKind
        from ddharmon.matching.confidence import score_transform_spec

        s = self._spec(kind=TransformKind.ARITHMETIC, formula="source / 12")
        assert score_transform_spec(s) == 0.6

    def test_arithmetic_verified_keeps_higher_inline_confidence(self):
        from ddharmon.harmonization.models import TransformKind
        from ddharmon.matching.confidence import score_transform_spec

        s = self._spec(kind=TransformKind.ARITHMETIC, formula="source / 12", confidence=0.95)
        assert score_transform_spec(s) == 0.95

    def test_data_dependent_is_low(self):
        from ddharmon.harmonization.models import TransformKind
        from ddharmon.matching.confidence import score_transform_spec

        assert score_transform_spec(self._spec(kind=TransformKind.DATA_DEPENDENT)) == 0.4

    def test_categorical_respects_inline_confidence(self):
        from ddharmon.harmonization.models import TransformKind
        from ddharmon.matching.confidence import score_transform_spec

        s = self._spec(kind=TransformKind.CATEGORICAL, confidence=0.83, coverage=0.9)
        assert score_transform_spec(s) == 0.83  # C1 owns categorical confidence

    def test_categorical_fallback_to_coverage_band(self):
        from ddharmon.harmonization.models import TransformKind
        from ddharmon.matching.confidence import score_transform_spec

        full = self._spec(kind=TransformKind.CATEGORICAL, confidence=0.0, coverage=1.0)
        partial = self._spec(kind=TransformKind.CATEGORICAL, confidence=0.0, coverage=0.5)
        empty = self._spec(kind=TransformKind.CATEGORICAL, confidence=0.0, coverage=0.0)
        assert score_transform_spec(full) == 0.9
        assert score_transform_spec(partial) == 0.6
        assert score_transform_spec(empty) == 0.4

    def test_custom_bands(self):
        from ddharmon.harmonization.models import TransformKind
        from ddharmon.matching.confidence import TransformConfidenceConfig, score_transform_spec

        cfg = TransformConfidenceConfig(high=0.8, medium=0.5, low=0.35)
        assert score_transform_spec(self._spec(kind=TransformKind.IDENTITY), cfg) == 0.8
