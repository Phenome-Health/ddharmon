"""Data models for sub-cluster-anchored CDE harmonization.

Dataclasses for the harmonization pipeline. The sub-cluster-anchored pipeline:

    semantic cluster  ->  value sub-cluster  ->  CDE anchor  ->  adopt/refine/novel

These models carry the result of each stage so the orchestrator
(``harmonization.pipeline``) and downstream consumers (EITL export) can work
without re-deriving anything from the notebook globals the logic used to live in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ddharmon.models.cluster import FieldReference
from ddharmon.models.data_dictionary import Field, ResponseOption

# Verdict vocabularies, by gate mode. ``harmonize`` mode (cluster has response-
# option data) emits the full three-way verdict; ``kg_only`` mode (concept-only,
# no machine-readable encodings) emits a concept-level two-way verdict.
HARMONIZE_VERDICTS = ("adopt", "refine", "novel")
KGONLY_VERDICTS = ("adopt", "unaligned")

# Lean head/tail pipeline: the fused-assign verdict, and the two routes it
# produces (adopt/refine -> CDE assignment; novel -> GenCDE/clustering residual).
LEANB_VERDICTS = ("adopt", "refine", "novel")
ROUTE_ASSIGNED = "assigned"
ROUTE_RESIDUAL = "gencde_residual"

# How a refine-derived element relates to the CDE it was derived from. SSSOM/SKOS predicates, because the
# NIH CDE model carries no "refines" relation of its own (see the GenCDE docstring). These are the same
# predicates the pairwise SSSOM export and the Kraken concept-layer bridge use, so a refinement is
# expressible to the outside world without inventing vocabulary.
RELATION_NARROWER = "skos:narrowMatch"  # the refinement specializes the parent (adds a qualifier/scope)
RELATION_BROADER = "skos:broadMatch"  # the refinement generalizes the parent (parent was too narrow)
RELATION_CLOSE = "skos:closeMatch"  # same concept, changed representation or value domain
RELATION_RELATED = "skos:relatedMatch"  # related but neither strictly narrower nor broader
REFINEMENT_RELATIONS = (RELATION_NARROWER, RELATION_BROADER, RELATION_CLOSE, RELATION_RELATED)


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

    This is the sub-cluster-anchored deliverable per sub-cluster — routed to EITL for human
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


class TransformKind(StrEnum):
    """How a source field's values map onto the matched CDE's value set (the transform layer).

    For 1.0: CATEGORICAL is authored (C1); UNIT (N1) + ARITHMETIC (N2) are authored in C2; DATA_DEPENDENT
    (N3) is detected + parameterized but its values are computed at apply-time, never here.
    """

    IDENTITY = "identity"  # source already aligned — no recode needed
    CATEGORICAL = "categorical"  # source code -> target code recode
    UNIT = "unit"  # N1: linear unit/scale conversion (target = source * factor + offset)
    ARITHMETIC = "arithmetic"  # N2: deterministic formula derivation
    DATA_DEPENDENT = "data_dependent"  # N3: detected + parameterized; needs row data (not authored here)
    WIDE_TO_LONG = "wide_to_long"  # structural: N numbered occurrence columns -> ONE repeated field
    NONE = "none"  # could not author a spec


@dataclass
class TransformSpec:
    """A metadata-level recipe for converting ONE source field's values to a target CDE's value set.

    Per the harmonization boundary this is EMITTED, never executed on row data. One TransformSpec per
    source member of an adopt/refine group — i.e. one Sankey edge (source var -> CDE). ``coverage`` is the
    verification signal: the fraction of source codes the recode actually maps. Low coverage on a refine
    sets ``needs_review`` but never overrides the assign verdict.
    """

    source_variable: str  # "cohort:var" — the edge this recode is for
    target_cde_id: str
    kind: TransformKind
    confidence: float = 0.0
    coverage: float = 0.0  # fraction of source codes mapped (verification signal)
    needs_units: bool = False
    needs_data: bool = False
    needs_review: bool = False
    rationale: str = ""
    generated_by: str = "llm"  # llm | rule
    # categorical (C1)
    code_map: dict[str, str] = field(default_factory=dict)  # source code -> target code
    unmapped_source_codes: list[str] = field(default_factory=list)
    # unit / N1 (C2):  target = source * factor + offset
    factor: float | None = None
    offset: float | None = None
    source_unit: str | None = None
    target_unit: str | None = None
    # arithmetic / N2 (C2)
    formula: str | None = None
    inputs: list[str] = field(default_factory=list)
    # data-dependent / N3 (C3)
    method: str | None = None
    params: dict = field(default_factory=dict)


@dataclass
class CandidateCDE:
    """One retrieved+ranked CDE candidate the assign stage evaluated, persisted for the review UI.

    The sub-cluster-anchored pipeline discarded these (only the chosen ``cde_id`` + orphan ``ranking`` indices survived);
    the candidate-review workbench needs the ranked set to render. Minimal fields taken from the
    assign-stage retrieval context; permissible values / steward are intentionally omitted for now (this
    stays a metadata-level record — the chosen CDE's value set flows via ``TransformSpec``).
    """

    rank: int  # 1-based, best-first (LLM ranking order; retrieval order as fallback)
    cde_id: str  # designation / variable name
    cde_external_id: str | None = None  # external/catalog id (tinyId / standard code) for link-out
    definition: str = ""  # rich candidate text (definition/question context) shown in the panel
    cosine: float = 0.0  # dense cosine of this candidate to the group centroid
    is_chosen: bool = False  # the adopted/refined candidate for this group
    llm_suggested: bool = False  # the LLM's top-ranked candidate


@dataclass
class LeanBRecord:
    """One harmonization decision from the lean head/tail pipeline — per concept-GROUP.

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
    adopt_demoted: bool = (
        False  # M5 adopt_floor demoted a weak-support adopt -> refine (retrieval_floor<=cos<adopt_floor)
    )
    coherence_gap: bool = False  # M3: coded edges mostly unmappable (NONE) -> over-broad match, flagged/demoted
    concept_mismatch: bool = False  # M7: assigned CDE fails the concept-match gate (right values, wrong concept)
    # Set by the coherence judge, which is not part of this release; refine's mis-assigned gate reads it.
    # Absent that judge it stays False, so the gate simply never fires on this condition — the point is to
    # never author a refinement for a group already judged over-merged.
    incoherent: bool = False
    member_variable_names: list[str] = field(default_factory=list)  # this group's members as "cohort:var"
    cohorts: list[str] = field(default_factory=list)
    cross_cohort: bool = False
    n_members: int = 0  # the GROUP's member count
    transforms: list[TransformSpec] = field(default_factory=list)  # per-source-var recodes (C1+); one per edge
    candidates: list[CandidateCDE] = field(default_factory=list)  # ranked CDE candidates the assign stage saw
    gencde: GenCDE | None = None  # novel -> synthesized GenCDE (the tail's harmonization target)
    decided_by: str = "llm"  # llm | deterministic
    raw: dict = field(default_factory=dict)


class RefinementAxis(StrEnum):
    """What about a matched CDE has to change for it to fit the concept group (the ``refine`` verdict).

    A ``refine`` says "this CDE is close but not right". These are the ways it is wrong, derived from the
    axes the assign stage's own rationales actually name. The axis is classified DETERMINISTICALLY from
    evidence already on the record (:func:`~ddharmon.harmonization.refine.classify_refinement_axis`), so
    the LLM is only asked to author the delta, never to pick the category.
    """

    VALUE_DOMAIN = "value_domain"  # the parent's permissible values cannot express the source's codes
    QUALIFIER = "qualifier"  # a qualifier is missing (body site, method, person, laterality, time window)
    REPRESENTATION = "representation"  # datatype/unit mismatch (banded vs continuous, IU vs mcg)
    STRUCTURAL = "structural"  # repeating measure / component decomposition (one element -> N occurrences)
    SCOPE = "scope"  # the parent is narrower than the concept the group actually measures


@dataclass
class GenCDE:
    """A generated Common Data Element synthesized for a ``novel`` concept group — the tail's target.

    "GenCDE" is DataTecnica/FAIRkit's term (Long et al., npj Digit Med 2026, DOI 10.1038/s41746-026-02795-z):
    an LLM-authored, human-reviewable CDE that mirrors the standard NIH/CDE metadata structure (the "Gen"
    denotes AI-generated + expert-reviewed provenance and scale, not a new schema). FAIRkit generates one
    CDE from ONE sparse dictionary entry (generate-from-template); this inverts that — a ``novel`` record is
    already a cross-cohort cluster of fields measuring one concept, so the GenCDE is synthesized from the
    POOLED empirical evidence (the members' reconciled answer options, units, question texts) —
    generate-from-cluster-empirics. Metadata-level: emitted + routed to review, never executed on row data.

    Fields map to the Long-et-al. generation schema for benchmark comparability: ``preferred_name`` =
    variable_name, ``definition`` = short_description, ``question_text`` = preferred_question_text,
    ``data_type`` = value_format, ``units`` = UOM, ``aliases`` = synonyms.

    ``value_coverage`` is the verification signal (fraction of answer-concepts observed across cohorts that
    the synthesized ``permissible_values`` represents); low coverage or low ``confidence`` sets
    ``needs_review`` but never changes the ``novel`` verdict (flag-not-gate, like TransformSpec.coverage).
    It is ``None`` (N/A) for a NUMERIC concept — there are no observed answer-labels to cover, so a coverage
    number is undefined; a numeric GenCDE's confidence rests on the LLM confidence + numeric-domain
    completeness (units / bounds) instead, and ``None`` must not be read as "0% covered".

    **Two provenances, one model.** A GenCDE is either synthesized FROM SCRATCH (a ``novel`` group reached no
    existing CDE) or DERIVED FROM A REAL CDE (a ``refine`` group matched a CDE that is close but not right).
    The derived case is the same artifact carrying a parent pointer plus a typed, minimal delta — so both
    routes share one authoring path, one review panel, and one NIH-conformant serializer
    (:mod:`ddharmon.export.cde_json`). ``parent_cde_id`` is the semantic test for which one this is; the
    ``gencde_id`` prefix (``GENCDE:`` vs ``REFCDE:``) makes it legible at a glance in exports and the UI.

    The refinement RELATION is expressed as an SSSOM/SKOS mapping predicate rather than in the CDE
    standard's own vocabulary, because the NIH CDE model has no "refines" pointer: its one self-referential
    field, ``derivationRules``, is a score-AGGREGATION slot (``ruleType: "score"``) populated on 2 of 22,743
    public CDEs, and the ISO 11179 qualifier slots it would otherwise ride in (``objectClass`` /
    ``property`` / ``dataElementConcept`` concepts) are empty on ~96% of them.
    """

    gencde_id: str  # deterministic id for the concept group, e.g. "GENCDE:<cluster_id>#g<idx>"
    preferred_name: str = ""  # canonical snake_case variable name
    title: str = ""  # expanded descriptive label
    definition: str = ""  # short clinical definition
    question_text: str = ""  # the acquisition question this element answers
    data_type: str = ""  # numeric | categorical | binary | date | text (value_format)
    permissible_values: list[ResponseOption] = field(default_factory=list)  # reconciled categorical domain
    units: str | None = None  # unit of measure (UCUM-style), for numeric concepts
    minimum_value: float | None = None  # numeric lower bound
    maximum_value: float | None = None  # numeric upper bound
    aliases: list[str] = field(default_factory=list)  # synonyms / cross-vocab names (free text)
    # provenance — the empirical basis (generate-from-cluster-empirics)
    source_variables: list[str] = field(default_factory=list)  # member edges, "cohort:var"
    source_cohorts: list[str] = field(default_factory=list)
    ideal_seed: str = ""  # the generate-ideal free-text anchor this GenCDE was grown from
    related_cdes: list[str] = field(default_factory=list)  # near-miss candidate names (SSSOM broader/related)
    # verification / review (flag-not-gate)
    value_coverage: float | None = None  # fraction of observed answer-concepts represented; None = N/A (numeric)
    uncovered_labels: list[str] = field(default_factory=list)  # observed answer-concepts the domain missed
    confidence: float = 0.0
    needs_review: bool = False
    rationale: str = ""
    generated_by: str = "llm"  # llm | rule
    # ── derivation (a ``refine``-derived element; all empty/zero => synthesized from scratch) ──────────
    parent_cde_id: str | None = None  # the matched CDE's designation — the element this refines
    parent_cde_external_id: str | None = None  # the parent's external/catalog id (tinyId) for link-out
    relation: str = ""  # SSSOM/SKOS predicate vs the parent (see RELATION_* below)
    refinement_axis: str = ""  # RefinementAxis — what about the parent had to change
    # The delta itself: the review payload, and the evidence the minimality check reads. The materialized
    # RESULT of applying it (parent ∪ delta) lives in the ordinary fields above — permissible_values,
    # units, data_type — so a consumer that ignores derivation still sees a complete, usable element.
    added_permissible_values: list[ResponseOption] = field(default_factory=list)
    relabeled_values: dict[str, str] = field(default_factory=dict)  # parent code -> revised label
    deprecated_values: list[str] = field(default_factory=list)  # parent codes the concept does not use
    qualifier_added: str = ""  # the qualifier the parent lacked ("right carotid bulb", "from waveform")
    changed_fields: list[str] = field(default_factory=list)  # parent fields the delta CONTRADICTS
    # Parent fields that were EMPTY and this element supplies. Not a change — the public catalog is
    # sparse (90% of matched parents carry no question_text, 37% no definition), so filling a blank is
    # metadata completion. Tracked for review, excluded from `delta_size` so minimality means what it says.
    completed_fields: list[str] = field(default_factory=list)
    delta_size: float = 0.0  # fraction of what the parent DID assert that this element changes
    over_refined: bool = False  # the delta rewrites rather than refines -> the honest verdict is `novel`
