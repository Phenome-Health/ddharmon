"""Confidence scoring and triage for field mappings.

Computes weighted composite scores from LLM confidence and cosine similarity,
then triages into auto_approved / pending_review / auto_rejected buckets.

Also hosts the **units-driven transform-spec confidence** (C2): :func:`score_transform_spec` maps a
:class:`~ddharmon.harmonization.models.TransformSpec` to a confidence band so transform specs get the same
auto-approve / review / reject triage as field mappings. (``harmonization.models`` is imported lazily
inside the function — a module-level import would close an import cycle through ``matching.__init__``.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ddharmon.models.enums import ReviewStatus

if TYPE_CHECKING:
    from ddharmon.harmonization.models import TransformSpec


@dataclass
class ConfidenceConfig:
    """Configuration for confidence scoring and triage thresholds.

    Weights must sum to 1.0 (within floating-point tolerance).

    Attributes:
        llm_weight: Weight for LLM confidence in composite score.
        cosine_weight: Weight for cosine similarity in composite score.
        auto_approve_threshold: Minimum composite score for auto-approval.
        auto_reject_threshold: Maximum composite score for auto-rejection.
    """

    llm_weight: float = 0.6
    cosine_weight: float = 0.4
    auto_approve_threshold: float = 0.9
    auto_reject_threshold: float = 0.3

    def __post_init__(self) -> None:
        weight_sum = self.llm_weight + self.cosine_weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.4f}")


_DEFAULT_CONFIG = ConfidenceConfig()


def score_mapping(
    llm_confidence: float,
    cosine_similarity: float,
    config: ConfidenceConfig | None = None,
) -> float:
    """Compute weighted composite confidence score.

    Args:
        llm_confidence: LLM-assigned confidence (0.0-1.0).
        cosine_similarity: Cosine similarity from embedding retrieval.
        config: Optional custom config (uses defaults if None).

    Returns:
        Composite score clamped to [0.0, 1.0].
    """
    if config is None:
        config = _DEFAULT_CONFIG

    raw = config.llm_weight * llm_confidence + config.cosine_weight * cosine_similarity
    return max(0.0, min(1.0, raw))


def triage_mapping(
    confidence: float,
    config: ConfidenceConfig | None = None,
) -> ReviewStatus:
    """Triage a mapping into review status based on confidence thresholds.

    Args:
        confidence: Composite confidence score (0.0-1.0).
        config: Optional custom config (uses defaults if None).

    Returns:
        ReviewStatus: AUTO_APPROVED, PENDING_REVIEW, or AUTO_REJECTED.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    if confidence >= config.auto_approve_threshold:
        return ReviewStatus.AUTO_APPROVED
    elif confidence <= config.auto_reject_threshold:
        return ReviewStatus.AUTO_REJECTED
    else:
        return ReviewStatus.PENDING_REVIEW


@dataclass
class TransformConfidenceConfig:
    """Confidence bands for transform specs (C2), aligned to the mapping triage thresholds.

    The bands are deliberately chosen to land on the right side of the default triage cutoffs
    (auto_approve 0.9 / auto_reject 0.3):

    - ``high`` (0.9) — auto-approvable: a deterministic known-unit conversion, identity, or a
      full-coverage categorical recode.
    - ``medium`` (0.6) — pending review: units inferred, an unverified LLM arithmetic formula, partial
      categorical coverage.
    - ``low`` (0.4) — pending review (NOT rejected): units missing/ambiguous (``needs_units``) or a
      data-dependent N3 spec (``needs_data``) — valid but it needs a human / row data, never auto-approve.

    A :class:`TransformKind.NONE <ddharmon.harmonization.models.TransformKind>` spec scores 0.0 →
    auto-rejected (no usable spec was authored).
    """

    high: float = 0.9
    medium: float = 0.6
    low: float = 0.4


_DEFAULT_TRANSFORM_CONFIG = TransformConfidenceConfig()


def score_transform_spec(
    spec: TransformSpec,
    config: TransformConfidenceConfig | None = None,
) -> float:
    """Units-driven transform-spec confidence in [0, 1] (C2).

    Pure: reads the spec's kind + flags and returns a band; the generators assign the result to
    ``spec.confidence`` and route it through :func:`triage_mapping`. Categorical (C1) keeps its inline
    coverage+LLM confidence — this respects it (returns the existing value, falling back to a coverage
    band only if unset).

    Args:
        spec: The transform spec to score.
        config: Optional band overrides (uses defaults if None).

    Returns:
        Confidence score in [0.0, 1.0].
    """
    from ddharmon.harmonization.models import TransformKind

    cfg = config or _DEFAULT_TRANSFORM_CONFIG
    kind = spec.kind

    if kind == TransformKind.NONE:
        return 0.0
    if kind == TransformKind.IDENTITY:
        return cfg.high  # source already aligned — nothing to recode
    if kind == TransformKind.CATEGORICAL:
        if spec.confidence > 0:
            return spec.confidence  # C1 owns categorical confidence (coverage + LLM)
        return cfg.high if spec.coverage >= 1.0 else (cfg.medium if spec.coverage > 0 else cfg.low)
    if kind == TransformKind.UNIT:
        if spec.needs_units:
            return cfg.low  # units missing/ambiguous — needs a human, not a rejection
        if spec.factor is not None and spec.source_unit and spec.target_unit:
            return cfg.high  # known conversion, units recognized on both sides
        return cfg.medium  # units present but conversion inferred/uncertain
    if kind == TransformKind.ARITHMETIC:
        if spec.needs_data:
            return cfg.low
        # LLM-proposed formula; a verifier that confirmed it on test inputs leaves a higher inline value.
        return max(cfg.medium, spec.confidence)
    if kind == TransformKind.DATA_DEPENDENT:
        return cfg.low  # N3 — values computed at apply-time, needs row data by construction
    return cfg.low
