"""Data models for sub-cluster-anchored CDE harmonization (v1).

Plain dataclasses (not Pydantic), with __post_init__ validation. The v1 pipeline:

    semantic cluster  ->  value sub-cluster  ->  CDE anchor  ->  adopt/refine/novel

These models carry the result of each stage so the orchestrator
(``harmonization.pipeline``) and downstream consumers (EITL export) can work
without re-deriving anything from the notebook globals the logic used to live in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ddharmon.models.cluster import FieldReference
from ddharmon.models.data_dictionary import Field

# Verdict vocabularies, by gate mode. ``harmonize`` mode (cluster has response-
# option data) emits the full three-way verdict; ``kg_only`` mode (concept-only,
# no machine-readable encodings) emits a concept-level two-way verdict.
HARMONIZE_VERDICTS = ("adopt", "refine", "novel")
KGONLY_VERDICTS = ("adopt", "unaligned")

# v2 lean head/tail pipeline: the fused-assign verdict, and the two routes it
# produces (adopt/refine -> CDE assignment; novel -> GenCDE/clustering residual).
LEANB_VERDICTS = ("adopt", "refine", "novel")
ROUTE_ASSIGNED = "assigned"
ROUTE_RESIDUAL = "gencde_residual"


@dataclass
class AnchorResult:
    """CDE anchor recommendation for one sub-cluster.

    The anchor is the in-sub-cluster CDE most central to the members (ranked by
    similarity to the medoid, then canonicalness, then metadata richness).
    ``has_cde`` is False when no CDE landed in the sub-cluster — a GenCDE is
    needed. ``alternate_cdes`` are the runner-up CDEs as ``(ref, field, sim)``.
    """

    has_cde: bool
    anchor_ref: FieldReference | None = None
    anchor_field: Field | None = None
    medoid_ref: FieldReference | None = None
    medoid_sim: float | None = None
    alternate_cdes: list[tuple[FieldReference, Field, float]] = field(default_factory=list)


@dataclass
class HarmonizationVerdict:
    """The adopt/refine/novel recommendation for one sub-cluster.

    This is the v1 deliverable per sub-cluster — routed to EITL for human
    verification. No transformation spec is authored (deferred to v1.1+).
    """

    sub_cluster_id: str  # f"{parent_topic_id}:{sub_label}"
    parent_topic_id: int
    sub_label: int
    mode: str  # harmonize | kg_only | single_cohort | cde_only | noise
    verdict: str  # adopt | refine | novel | unaligned | "" (no LLM call)
    parent_cde_id: str | None = None
    confidence: float | None = None
    evidence: str = ""
    label: str = ""  # derived (c-TF-IDF) sub-cluster label
    cohorts: list[str] = field(default_factory=list)
    n_fields: int = 0  # non-CDE cohort fields
    encoded_fraction: float = 0.0
    anchor_designation: str | None = None
    decided_by: str = "llm"  # llm | deterministic
    raw: dict = field(default_factory=dict)


@dataclass
class LeanBRecord:
    """One harmonization decision from the v2 lean head/tail pipeline — per concept-GROUP.

    A semantic cluster is grouped by an embedding that ignores the variable name, so it can pool more
    than one distinct concept. The split-aware pipeline therefore emits one record per concept-GROUP,
    not one per cluster (adopt-with-context): distinct concepts in one cluster get distinct records, and
    a shared CDE id across groups of one cluster is allowed but the groups stay distinct records —
    never silently pooled. Each group is assigned to a CDE (``adopt``/``refine`` -> ``route="assigned"``)
    or routed to the GenCDE/clustering residual (``novel`` -> ``route="gencde_residual"``). The decision
    is made by a generate-ideal call (the independent coverage anchor, ``ideal_cde``), a split call (the
    partition into groups), and a per-group re-retrieve + rank+verdict call over candidates retrieved for
    that group. ``coverage_gap`` is a transparent diagnostic (a ``novel`` whose nearest candidate is below
    ``COVERAGE_GAP_TAU`` cosine), never a decision gate. ``n_members`` is the GROUP's member count.
    """

    cluster_id: str
    verdict: str  # adopt | refine | novel | "" (no/unparseable LLM response)
    route: str  # assigned | gencde_residual
    group_id: str = ""  # f"{cluster_id}#g{idx}" — distinct concept-group within the cluster
    concept: str = ""  # the group's concept label (from the split stage)
    cde_id: str | None = None  # chosen CDE designation (adopt/refine)
    cde_external_id: str | None = None  # external/catalog id of the chosen CDE, if available
    ideal_cde: str = ""  # the independently-generated ideal (the coverage anchor)
    ranking: list[int] = field(default_factory=list)  # candidate numbers, best-first (LLM)
    rationale: str = ""
    top1_cos: float | None = None  # nearest-candidate dense cosine (retrieval signal)
    chosen_cos: float | None = None  # dense cosine of the CHOSEN candidate (the match's geometric support)
    coverage_gap: bool = False  # diagnostic: novel & top1_cos < COVERAGE_GAP_TAU
    floored: bool = False  # the retrieval floor downgraded an adopt/refine -> novel (chosen_cos < floor)
    member_variable_names: list[str] = field(default_factory=list)  # this group's members as "cohort:var"
    cohorts: list[str] = field(default_factory=list)
    cross_cohort: bool = False
    n_members: int = 0  # the GROUP's member count
    decided_by: str = "llm"  # llm | deterministic
    raw: dict = field(default_factory=dict)
