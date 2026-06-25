"""CDE harmonization.

**v2 (lean head/tail, split-aware)** — assignment-first for the covered head, GenCDE/clustering for the
tail; one record per distinct concept-GROUP within a cluster::

    cluster -> hybrid retrieve -> generate-ideal -> split-assign -> per-group re-assign -> route

    harmonize_leanb()        -- full v2 pipeline (generate -> split -> per-group assign -> route)
    prepare_leanb()          -- retrieve + build generate-ideal prompts from precomputed clusters
    prepare_split()          -- build split-assign prompts (partition members into concept groups)
    prepare_group_assign()   -- parse split groups, re-retrieve per group, build per-group assign prompts
    assemble_leanb()         -- parse per-group responses into routed LeanBRecord decisions
    LeanBResult / LeanBRecord / CdeBackbone / export_leanb_eitl_queue()

**v1 (sub-cluster-anchored)** — retained:: semantic cluster -> value sub-cluster -> CDE anchor ->
classify (adopt/refine/novel) -> EITL.

    harmonize_dictionaries() -- full v1 pipeline
    prepare_from_clusters() / assemble_verdicts() / find_anchor_cde()
    HarmonizationResult / HarmonizationVerdict / AnchorResult / SubClusterResult
"""

from ddharmon.harmonization.anchor import (
    CDE_COHORT,
    build_field_lookup,
    canonicalness_score,
    field_richness,
    find_anchor_cde,
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
    write_records_json,
)
from ddharmon.harmonization.models import (
    AnchorResult,
    HarmonizationVerdict,
    LeanBRecord,
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
from ddharmon.models.cluster import SubClusterResult

__all__ = [
    "CDE_COHORT",
    "AnchorResult",
    "CdeBackbone",
    "HarmonizationResult",
    "HarmonizationVerdict",
    "LeanBRecord",
    "LeanBResult",
    "PromptRecord",
    "SubClusterResult",
    "assemble_leanb",
    "assemble_verdicts",
    "build_field_lookup",
    "canonicalness_score",
    "export_eitl_queue",
    "export_leanb_eitl_queue",
    "field_richness",
    "find_anchor_cde",
    "harmonize_dictionaries",
    "harmonize_leanb",
    "parse_verdict_payload",
    "prepare_from_clusters",
    "prepare_group_assign",
    "prepare_leanb",
    "prepare_split",
    "write_buckets",
    "write_prompts_jsonl",
    "write_records_json",
]
