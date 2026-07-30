"""CDE harmonization.

**Lean head/tail (split-aware) — the default pipeline** — assignment-first for the covered head,
GenCDE/clustering for the tail; one record per distinct concept-GROUP within a cluster::

    cluster -> hybrid retrieve -> generate-ideal -> split-assign -> per-group re-assign -> route

    harmonize_leanb()        -- full pipeline (generate -> split -> per-group assign -> route)
    prepare_leanb()          -- retrieve + build generate-ideal prompts from precomputed clusters
    prepare_split()          -- build split-assign prompts (partition members into concept groups)
    prepare_group_assign()   -- parse split groups, re-retrieve per group, build per-group assign prompts
    assemble_leanb()         -- parse per-group responses into routed LeanBRecord decisions
    LeanBResult / LeanBRecord / CdeBackbone / export_leanb_eitl_queue()

**Sub-cluster-anchored — retained**:: semantic cluster -> value sub-cluster -> CDE anchor ->
classify (adopt/refine/novel) -> EITL.

    harmonize_dictionaries() -- full sub-cluster-anchored pipeline
    prepare_from_clusters() / assemble_verdicts() / find_anchor_cde()
    HarmonizationResult / HarmonizationVerdict / AnchorResult / SubClusterResult

**Post-run layers** — read a finished run's concepts and say what they enable::

    generate_analysis_ideas() -- downstream cross-cohort analyses the harmonization unlocks (suggest-only)
    derive_composite()        -- can this run support a published composite score (frailty index, PHQ-9
                                 sum, intrinsic capacity)? which concepts compose it, and how
    fetch_source()            -- the composite builder's document front door (paste / URL / DOI / repo / PDF)
"""

from ddharmon.harmonization.analysis_ideas import (
    AnalysisIdea,
    AnalysisIdeasResult,
    build_concept_digest,
    generate_analysis_ideas,
)
from ddharmon.harmonization.anchor import (
    CDE_COHORT,
    build_field_lookup,
    canonicalness_score,
    field_richness,
    find_anchor_cde,
)
from ddharmon.harmonization.composite import (
    CodingKind,
    CohortCoverage,
    ComponentCoding,
    ComponentMatch,
    CompositeKind,
    CompositeResult,
    CompositeSpec,
    ConceptEntry,
    DerivationStep,
    FeasibilityReport,
    ScoreComponent,
    ScoreDefinition,
    assess_feasibility,
    build_composite_spec,
    build_concept_index,
    derive_composite,
    extract_score_definition,
    match_components,
    records_from_payload,
    shortlist_concepts,
    spec_to_dict,
    spec_to_json,
)
from ddharmon.harmonization.gencde import (
    assemble_gencde,
    observed_answer_labels,
    prepare_gencde,
)
from ddharmon.harmonization.leanb import (
    CdeBackbone,
    LeanBResult,
    assemble_leanb,
    export_leanb_eitl_queue,
    harmonize_leanb,
    prepare_group_assign,
    prepare_leanb,
    prepare_split,
    recover_outlier_clusters,
    write_records_json,
)
from ddharmon.harmonization.merge import (
    assemble_merge,
    merge_candidate_pairs,
    prepare_merge,
)
from ddharmon.harmonization.models import (
    AnchorResult,
    CandidateCDE,
    GenCDE,
    HarmonizationVerdict,
    LeanBRecord,
    TransformKind,
    TransformSpec,
)
from ddharmon.harmonization.parse import parse_verdict_payload
from ddharmon.harmonization.pipeline import (
    HarmonizationResult,
    PromptRecord,
    assemble_verdicts,
    export_eitl_queue,
    harmonize_dictionaries,
    prepare_from_clusters,
    write_buckets,
    write_prompts_jsonl,
)
from ddharmon.harmonization.positional import detect_enumerated_family, detect_positional_enumeration
from ddharmon.harmonization.score_sources import (
    ScoreSource,
    fetch_source,
    from_pdf,
    from_text,
    from_url,
    html_to_text,
    pdf_to_text,
)
from ddharmon.harmonization.substrate import (
    ClusteringSubstrate,
    build_substrate,
    cluster_content_id,
    clusters_from_substrate,
    load_substrate,
    save_substrate,
)
from ddharmon.harmonization.transform import (
    apply_coherence_gate,
    assemble_arith_specgen,
    assemble_concept_gate,
    assemble_gencde_arith_specgen,
    assemble_gencde_specgen,
    assemble_specgen,
    eval_formula,
    export_transform_review,
    generate_gencde_unit_specs,
    generate_unit_specs,
    generate_wide_to_long_specs,
    prepare_arith_specgen,
    prepare_concept_gate,
    prepare_gencde_arith_specgen,
    prepare_gencde_specgen,
    prepare_specgen,
    verify_formula,
)
from ddharmon.models.cluster import SubClusterResult

__all__ = [
    "CDE_COHORT",
    "AnalysisIdea",
    "AnalysisIdeasResult",
    "AnchorResult",
    "CandidateCDE",
    "CdeBackbone",
    "ClusteringSubstrate",
    "CodingKind",
    "CohortCoverage",
    "ComponentCoding",
    "ComponentMatch",
    "CompositeKind",
    "CompositeResult",
    "CompositeSpec",
    "ConceptEntry",
    "DerivationStep",
    "FeasibilityReport",
    "GenCDE",
    "HarmonizationResult",
    "HarmonizationVerdict",
    "LeanBRecord",
    "LeanBResult",
    "PromptRecord",
    "ScoreComponent",
    "ScoreDefinition",
    "ScoreSource",
    "SubClusterResult",
    "TransformKind",
    "TransformSpec",
    "apply_coherence_gate",
    "assemble_arith_specgen",
    "assess_feasibility",
    "assemble_concept_gate",
    "assemble_gencde",
    "assemble_gencde_arith_specgen",
    "assemble_gencde_specgen",
    "assemble_leanb",
    "assemble_merge",
    "assemble_specgen",
    "assemble_verdicts",
    "build_composite_spec",
    "build_concept_digest",
    "build_concept_index",
    "build_field_lookup",
    "build_substrate",
    "canonicalness_score",
    "cluster_content_id",
    "clusters_from_substrate",
    "derive_composite",
    "detect_enumerated_family",
    "detect_positional_enumeration",
    "merge_candidate_pairs",
    "prepare_merge",
    "eval_formula",
    "extract_score_definition",
    "fetch_source",
    "from_pdf",
    "from_text",
    "from_url",
    "html_to_text",
    "export_eitl_queue",
    "export_leanb_eitl_queue",
    "export_transform_review",
    "field_richness",
    "find_anchor_cde",
    "generate_analysis_ideas",
    "generate_gencde_unit_specs",
    "generate_unit_specs",
    "generate_wide_to_long_specs",
    "harmonize_dictionaries",
    "harmonize_leanb",
    "load_substrate",
    "observed_answer_labels",
    "match_components",
    "parse_verdict_payload",
    "pdf_to_text",
    "records_from_payload",
    "shortlist_concepts",
    "spec_to_dict",
    "spec_to_json",
    "prepare_arith_specgen",
    "prepare_concept_gate",
    "prepare_from_clusters",
    "prepare_gencde",
    "prepare_gencde_arith_specgen",
    "prepare_gencde_specgen",
    "prepare_group_assign",
    "prepare_leanb",
    "prepare_specgen",
    "prepare_split",
    "recover_outlier_clusters",
    "save_substrate",
    "verify_formula",
    "write_buckets",
    "write_prompts_jsonl",
    "write_records_json",
]
