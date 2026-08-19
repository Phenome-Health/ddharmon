"""Lean head/tail CDE harmonization pipeline (split-aware, 3-stage) — the default pipeline.

Supersedes the sub-cluster-anchored pipeline (``harmonize_dictionaries``). Where that approach anchored
each value sub-cluster to the most-central in-cluster CDE, this one leads with **assignment to the given
CDE backbone** for the covered head and routes the uncovered tail to GenCDE/clustering — the division of
labor established empirically through benchmark experiments. A semantic cluster is grouped by an embedding
that ignores the variable name, so one cluster can pool MORE THAN ONE distinct concept; the pipeline is therefore
SPLIT-AWARE and emits one record per concept-GROUP. Per concept cluster::

    hybrid retrieve (BM25 lexical + dense centroid, RRF) top-k CDE candidates
      -> generate-ideal      (LLM, no candidates -> a qualifier-faithful coverage anchor)
      -> split-assign         (LLM, partition members into distinct-concept groups + rank+verdict each)
      -> per-group re-assign  (LLM, RE-RETRIEVE per group, then rank candidates + adopt/refine/novel)
      -> route: adopt/refine -> CDE assignment ;  novel -> GenCDE / clustering residual (tail)

Three LLM stages. The pipeline is split so each stage is testable without an LLM and can run
inline *or* be exported for the offline Batch API and assembled later: ``prepare_leanb`` builds the
generate prompts, ``prepare_split`` builds the split prompts from the generated ideals,
``prepare_group_assign`` parses the split groups and builds one re-retrieved assign prompt per group, and
``assemble_leanb`` parses the per-group responses into per-group records. Each prompt record carries the
context needed to build the next stage / assemble its record.

The adopt/refine/novel cutoff is intentionally strict (the ideal anchors the novel decision); its final
calibration is expected to come from human (expert-in-the-loop) review of the routed output.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ddharmon.embedding.service import EmbeddedDictionary
from ddharmon.harmonization.anchor import CDE_COHORT, build_field_lookup
from ddharmon.harmonization.leanb_prompts import (
    ASSIGN_SCHEMA,
    COHERENCE_SCHEMA,
    IDEAL_SCHEMA,
    KINDS_TOOL_NAME,
    KINDS_TOOL_SCHEMA,
    SPLIT_SCHEMA,
    SPLIT_TOOL_NAME,
    SPLIT_TOOL_SCHEMA,
    SYS_COHERENCE,
    SYS_KINDS,
    SYS_READJUDICATE,
    build_coherence_user_prompt,
    build_group_assign_user_prompt,
    build_ideal_user_prompt,
    build_kinds_user_prompt,
    build_readjudicate_user_prompt,
    build_split_user_prompt,
    generate_ideal_system_prompt,
    group_reassign_system_prompt,
    split_system_prompt,
)
from ddharmon.harmonization.models import CandidateCDE, ConceptGroup, GenCDE, LeanBRecord
from ddharmon.harmonization.parse import extract_json
from ddharmon.harmonization.pipeline import PromptRecord
from ddharmon.harmonization.substrate import (
    ClusteringSubstrate,
    build_substrate,
    cluster_content_id,
    clusters_from_substrate,
)
from ddharmon.matching import BM25, hybrid_topk, tokenize
from ddharmon.models.cluster import FieldCluster, FieldReference
from ddharmon.models.data_dictionary import Field
from ddharmon.text_hygiene import CDE_TEXT_BOILERPLATE, is_sentinel_label, strip_sentinel_encodings

logger = logging.getLogger(__name__)

# Prompt-hygiene ablation hook: the SHIPPED default is ON (drop missing/refused/DK sentinels from the
# concept-identity prompt route; the ingestion admin-text strip is separately default-on). Set
# DDHARMON_PROMPT_HYGIENE=0 ONLY to reproduce pre-hygiene behavior for a validation A/B — production
# never sets it. Read once at import; the calibration/subset harness reads the same var for admin-text.
_PROMPT_HYGIENE = os.environ.get("DDHARMON_PROMPT_HYGIENE", "1") != "0"

DEFAULT_MODEL_TAG = "claude-sonnet-4-6"
DEFAULT_TOP_K = 20  # wide pool for the assignment engine
COVERAGE_GAP_TAU = 0.70  # diagnostic only (NOT a decision gate)
# Retrieval floor (#1): downgrade an adopt/refine -> novel when the CHOSEN candidate's dense cosine is below
# this — i.e. the engine force-fit the least-bad candidate when nothing is actually close. A BOTTOM floor
# (fires only when nothing is near), not a mid routing threshold; on the CDEMapper gold τ=0.30 holds head
# accuracy (0.521) while trimming the spurious tail. Set 0.0 to disable.
DEFAULT_RETRIEVAL_FLOOR = 0.30
# Adopt-specific floor (M5): demote a weak-support adopt (retrieval_floor <= chosen cos < this) -> refine, so
# an exact-equivalence claim needs genuine support. Validated on the held-out full-5 (2026-07-04): demoted 44
# weak adopts (all sensibly weak — study-ID, borderline BP), a precision move. Set None to disable.
DEFAULT_ADOPT_FLOOR = 0.55
MAX_SHOW = 45  # members shown to the split LLM in one call (~the reliable-enumeration limit; M2 Phase 2a
# raised this from 22 — the old cap truncated 68% of clusters before the LLM ever saw them. Clusters larger
# than this are chunked by `chunk_oversized` so every unit is shown in full; tunable in the 2b A/B run.
SPLIT_ENFORCE_MAX_TOKENS = 4096  # M15: output budget for the ENFORCED split tool-call; the structured JSON
# for a full partition of a large cluster overflows the 1024 stage default (observed empty {} at 1024 on the
# 25-member BP cluster). 4096 covers a chunk_cap(45)-member partition with headroom.
_SAMPLE_MEMBERS = 5  # members in the generate-ideal / per-group sample line
_CAND_TRUNC = 170
_MEMBER_TRUNC = 200  # retrieval member text (lean, concept-only — value codes are noise in BM25/dense)
_VALUE_ENC_TRUNC = 240  # cap the source value-set string fed into a prompt (CLSA country lists etc. are long)
_MEMBER_PROMPT_TRUNC = 420  # prompt-side member text: concept + symbolic value metadata (richer than retrieval)

# Dual-sample coherence judge. k1 centroid-CLOSEST members anchor the summary; k2 centroid-FURTHEST
# members verify it (DISJOINT — the fix for the self-fulfilling same-sample verification of the published
# cluster-refinement method this stage builds on).
COHERENCE_K1 = 5  # fixed core sample (centroid-closest) for the summary
COHERENCE_K2_FLOOR = 5  # min periphery sample
COHERENCE_K2_CEILING = 20  # max periphery sample (token budget + diminishing returns)
COHERENCE_MIN_MEMBERS = COHERENCE_K1 + 1  # groups smaller than this are trivially coherent -> not judged
# The VOLUNTARY stop boundaries `harmonize_leanb(stop_after=...)` accepts. Each name means "this stage
# COMPLETED — stop before the next one", which is the opposite reading from the unset-callable early
# returns (those are named for the stage that has NOT run and hand back its prompts). Deliberately ONE
# name: a string parameter can learn another accepted value without breaking a caller, so a boundary is
# added when a consumer actually needs it. An unrecognised value RAISES — a silent no-op would run the
# whole paid pipeline past the stop the caller asked for.
STOP_AFTER_BOUNDARIES: tuple[str, ...] = ("gencde",)
_COH_TEXT_TRUNC = 1000  # per-member text shown to the judge. Raised from 200: the old cap cut long
# descriptions mid-sentence (instrument-administration preambles are routinely longer than that) before the
# judge ever saw them.

# M5 index hygiene (opt-in): generic survey/CDE instruction boilerplate is now defined once in
# :mod:`ddharmon.text_hygiene` (:data:`CDE_TEXT_BOILERPLATE`, imported above and re-exported here for
# back-compat) so ingestion, the prompts, and the calibration tooling share one cohort-agnostic list.
# An opaque catalog code (leading designation token) — mostly caps/digits/underscores, no lowercase word.
_OPAQUE_CODE_RE = re.compile(r"^[A-Z0-9][A-Z0-9_\-.]{2,}$")
_BOILERPLATE_RE = re.compile("|".join(re.escape(p) for p in CDE_TEXT_BOILERPLATE), re.IGNORECASE)


def _clean_cde_text(text: str) -> str:
    """M5 index hygiene: strip generic instruction boilerplate + a leading opaque code from CDE rich text.

    Removes survey/CDE boilerplate phrases (:data:`CDE_TEXT_BOILERPLATE`) that add BM25 noise and clutter the
    candidate block, and drops a leading opaque snake_case/UPPER catalog code (e.g. ``PHX0001``) when the
    remaining text is non-empty — so the concept text, not the code, drives lexical retrieval. Whitespace is
    collapsed. Empty/whitespace-only results fall back to the original text (never blank out a candidate).
    """
    if not text or not text.strip():
        return text
    cleaned = _BOILERPLATE_RE.sub(" ", text)
    toks = cleaned.split()
    if len(toks) >= 2 and _OPAQUE_CODE_RE.match(toks[0]):
        rest = " ".join(toks[1:]).strip()
        if rest:  # only drop the opaque lead when real concept text remains
            cleaned = rest
    cleaned = " ".join(cleaned.split())
    return cleaned or " ".join(text.split())


def _norm(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    n = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.where(n == 0, 1.0, n)).astype(np.float32)


def _humanize(var_name: str) -> str:
    """employmentworkaddress_zipcode / personOneAddressZip -> 'employment work address zipcode' tokens."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", var_name)
    return re.sub(r"[^A-Za-z0-9]+", " ", s).strip().lower()


def _base_member_text(fld: Field | None, ref: FieldReference) -> str:
    """Human-readable base text for one member field (question > description > short label > name)."""
    if fld is not None:
        for cand in (fld.question_text, fld.description, fld.short_label, fld.variable_name):
            if cand and cand.strip():
                return " ".join(cand.split())
    return ref.variable_name


def _aug_text(var_name: str, base: str) -> str:
    """Base text augmented with the humanized variable NAME when it carries WORD-LIKE tokens the base lacks.

    I.e. a qualifier the generic question/description omits (work/home address, self/contact, laterality).
    Skips opaque codes (e.g. 'Field 5374', 'PMARRY') by requiring >=2 new alpha tokens. The LLM is the
    arbiter (down-weights noise); kept out of the geometric vector where opaque names provably hurt.
    """
    humanized = _humanize(var_name)
    name_toks = [t for t in humanized.split() if len(t) > 2 and t.isalpha()]
    base_toks = set(re.sub(r"[^a-z0-9]+", " ", base.lower()).split())
    extra = [t for t in name_toks if t not in base_toks]
    return f"{base} [field: {humanized}]" if len(extra) >= 2 else base


def _member_text(fld: Field | None, ref: FieldReference) -> str:
    """Gap-1 augmented member text: base text + humanized name when the name adds a qualifier.

    This is the RETRIEVAL text (BM25 + dense). Kept lean (concept-only) on purpose — source value
    codes are noise in the geometric/lexical space. For the prompt-side, value-aware
    rendering see :func:`_member_prompt_text`.
    """
    return _aug_text(ref.variable_name, _base_member_text(fld, ref))


def _value_set_text(fld: Field, *, drop_sentinels: bool = False) -> str:
    """Readable source value set for a prompt: the raw encoding, else reconstructed from parsed options.

    ``drop_sentinels`` (default False) removes missing/refused/don't-know sentinel options (e.g.
    ``-9=MISSING``, ``-3=Prefer not to answer``) so a field whose ONLY encoding is a sentinel renders
    empty (read as numeric, not a single-option categorical). It is ON for the concept-identity route
    (:func:`_member_prompt_text` → judge / gen-ideal / split / assign) and OFF for spec-gen
    (``transform.py``), which needs the missing codes to author value-recode transforms.
    """
    raw = (fld.value_encoding_raw or "").strip()
    if raw:
        return strip_sentinel_encodings(raw) if drop_sentinels else raw
    if fld.response_options:
        opts = fld.response_options
        if drop_sentinels:
            opts = [ro for ro in opts if not is_sentinel_label(ro.label)]
        return "|".join(f"{ro.code}={ro.label}" for ro in opts)
    return ""


def _member_prompt_text(fld: Field | None, ref: FieldReference) -> str:
    """Member text for LLM PROMPTS (gen-ideal / split / assign / spec-gen) — NOT retrieval.

    Augments the lean concept text (:func:`_member_text`) with the source field's SYMBOLIC value
    metadata: response options (value_encoding), units, and data_type. This lets the harmonizability
    judgment, GenCDE answer-authoring, and spec-gen stages SEE the source variable's answer options —
    which the retrieval text deliberately omits. (Symbolic signals belong in prompts, not the vector.)
    """
    base = _member_text(fld, ref)
    if fld is None:
        return base
    extras: list[str] = []
    if fld.data_type and fld.data_type.strip():
        extras.append(f"type {fld.data_type.strip()}")
    if fld.units and fld.units.strip():
        extras.append(f"units {fld.units.strip()}")
    # drop_sentinels: missing/refused/DK codes are noise for the concept-identity stages this text feeds
    # (gen-ideal / split / assign / coherence judge) — a numeric field encoded only as `-9=MISSING` should
    # read as numeric, not a single-option categorical. spec-gen sources values separately (keeps them).
    # Gated on the _PROMPT_HYGIENE ablation hook (shipped ON; =0 only to reproduce pre-hygiene for an A/B).
    values = _value_set_text(fld, drop_sentinels=_PROMPT_HYGIENE)
    if values:
        extras.append(f"values: {values[:_VALUE_ENC_TRUNC]}")
    return f"{base} [{'; '.join(extras)}]" if extras else base


def _cde_rich_text(fld: Field) -> str:
    """Richest text for a CDE field: designation + question + definition + permissible values."""
    parts = [fld.variable_name, fld.question_text, fld.description, fld.value_encoding_raw]
    return " ".join(p.strip() for p in parts if p and p.strip())


def _cde_external_id(fld: Field) -> str:
    """Best-effort external/catalog id for a CDE (field_id, else first standard code)."""
    if fld.field_id:
        return fld.field_id
    for codes in (fld.standard_codes or {}).values():  # standard_codes: dict[str, list[str]]
        if codes:
            return str(codes[0])
    return ""


@dataclass
class CdeBackbone:
    """The CDE catalog as searchable arrays: ids, L2-normalized vectors, rich text + a BM25 index."""

    ids: list[str]  # CDE designations (variable names), aligned to ``vectors``/``rich_texts``
    vectors: NDArray[np.float32]  # (n_cde, d), L2-normalized
    rich_texts: list[str]
    external_ids: list[str]
    bm25: BM25 = field(repr=False, default=None)  # type: ignore[assignment]

    @classmethod
    def from_embedded(
        cls, cde_dict: EmbeddedDictionary, cde_fields: dict[str, Field], *, clean_text: bool = False
    ) -> CdeBackbone:
        ids = list(cde_dict.get_variable_names())
        vectors = _norm(np.asarray(cde_dict.get_all_vectors(), dtype=np.float32))
        rich = [_cde_rich_text(cde_fields[i]) if i in cde_fields else i for i in ids]
        if clean_text:  # M5 index hygiene: drop boilerplate + opaque lead codes from BM25 corpus + display
            rich = [_clean_cde_text(t) for t in rich]
        ext = [_cde_external_id(cde_fields[i]) if i in cde_fields else "" for i in ids]
        return cls(ids=ids, vectors=vectors, rich_texts=rich, external_ids=ext, bm25=BM25(rich))


@dataclass
class LeanBResult:
    """Harmonization records plus the prompts that produced (or will produce) them.

    ``ideal_prompts`` are populated when the generate stage has not run inline (export for Batch);
    ``split_prompts`` when generate has run but split has not; ``group_assign_prompts`` when split has run
    but the per-group assign has not. ``records`` are the final per-group decisions.

    ``concept_groups`` are the post-split groups those per-group assign prompts describe — the shape that
    exists once ``split`` has run and before ``classify`` has decided anything. They carry the coherence
    judge's verdicts (stamped pre-assign), so a caller that stopped at the ``classify=None`` boundary can
    render each group's coherence state without paying for assign.
    """

    records: list[LeanBRecord] = field(default_factory=list)
    ideal_prompts: list[PromptRecord] = field(default_factory=list)
    split_prompts: list[PromptRecord] = field(default_factory=list)
    group_assign_prompts: list[PromptRecord] = field(default_factory=list)
    concept_groups: list[ConceptGroup] = field(default_factory=list)  # post-split groups, judged pre-assign
    merge_prompts: list[PromptRecord] = field(default_factory=list)  # M2 cross-record merge — Batch export
    specgen_prompts: list[PromptRecord] = field(default_factory=list)  # stage 4 (transform specs) — Batch export
    gencde_prompts: list[PromptRecord] = field(default_factory=list)  # novel -> GenCDE synthesis — Batch export
    concept_gate_prompts: list[PromptRecord] = field(default_factory=list)  # M7 concept-match gate — Batch export
    coherence_prompts: list[PromptRecord] = field(default_factory=list)  # step-2 coherence judge — Batch export
    kinds_prompts: list[PromptRecord] = field(default_factory=list)  # R2 distinct-KINDS discriminator — Batch export
    refine_prompts: list[PromptRecord] = field(default_factory=list)  # refine -> derived CDE — Batch export
    substrate: ClusteringSubstrate | None = None  # the frozen clustering partition (save it to replay cheaply)

    def buckets(self) -> dict[str, list[LeanBRecord]]:
        out: dict[str, list[LeanBRecord]] = defaultdict(list)
        for r in self.records:
            out[r.verdict or "unclassified"].append(r)
        return dict(out)


def _candidate(backbone: CdeBackbone, i: int, cos: float) -> dict:
    return {
        "designation": backbone.ids[i],
        "cos": float(cos),
        "text": backbone.rich_texts[i][:_CAND_TRUNC],
        "external_id": backbone.external_ids[i],
    }


def _retrieve(
    rows: list[int],
    member_texts: list[str],
    embeddings: NDArray[np.float32],
    backbone: CdeBackbone,
    top_k: int,
) -> tuple[list[dict], float]:
    """Hybrid (dense centroid ⊕ BM25 lexical, RRF) top-k CDE candidates for a set of member rows.

    Returns ``(candidates, top1_cos)`` where each candidate is a context dict
    ``{designation, cos, text, external_id}``.
    """
    if not rows:
        return [], 0.0
    centroid = _norm(embeddings[rows].mean(axis=0))
    dense = backbone.vectors @ centroid
    lexical = backbone.bm25.scores(tokenize(" ".join(member_texts[:8])))
    idx = hybrid_topk(dense, lexical, top_k)
    cands = [_candidate(backbone, i, dense[i]) for i in idx]
    top1 = float(dense[idx[0]]) if idx else 0.0
    return cands, top1


def _cluster_base_context(cluster: FieldCluster, cde_cohort: str) -> dict:
    members = [m for m in cluster.members if m.dictionary_name != cde_cohort]
    cohorts = sorted({m.dictionary_name for m in members})
    return {
        # content-addressed cluster id (the non-CDE member set), NOT the ephemeral HDBSCAN ordinal — this is
        # what makes the leanb prompt ids (and so the Batch response cache) stable across runs of a frozen
        # substrate. Flows into every downstream prompt id + the record's cluster_id/group_id.
        "cluster_id": cluster_content_id([(m.dictionary_name, m.variable_name) for m in members]),
        "cohorts": cohorts,
        "cross_cohort": len(cohorts) >= 2,
        "n_members": len(members),
    }


def _build_backbone(
    embedded_dicts: list[EmbeddedDictionary],
    field_lookup: dict[tuple[str, str], Field],
    cde_cohort: str,
    cde_dict: EmbeddedDictionary | None,
    *,
    clean_text: bool = False,
) -> CdeBackbone:
    cde_embedded = cde_dict or _find_cde_dict(embedded_dicts, cde_cohort)
    return CdeBackbone.from_embedded(
        cde_embedded, {k[1]: v for k, v in field_lookup.items() if k[0] == cde_cohort}, clean_text=clean_text
    )


def prepare_leanb(
    clusters: list[FieldCluster],
    embedded_dicts: list[EmbeddedDictionary],
    embeddings: NDArray[np.float32],
    field_refs: list[FieldReference],
    *,
    cde_cohort: str = CDE_COHORT,
    cde_dict: EmbeddedDictionary | None = None,
    top_k: int = DEFAULT_TOP_K,
    model_tag: str = DEFAULT_MODEL_TAG,
    clean_cde_text: bool = False,
    measurand_split: bool = False,
    debias_ideal: bool = False,
) -> list[PromptRecord]:
    """Retrieve candidates and build the stage-1 generate-ideal prompts (one per non-empty cluster).

    Each returned record's ``context`` carries (a) the FULL ordered non-CDE member list — each member as
    ``{member_id, dictionary_name, variable_name, text, row}`` with the Gap-1 augmented text and its
    embedding-row index — and (b) the cluster-level retrieved candidates + metadata, so
    :func:`prepare_split` can build the split prompt without re-retrieving, and
    :func:`prepare_group_assign` can map split groups back to members and re-retrieve per group.

    ``clean_cde_text`` (M5) applies :func:`_clean_cde_text` index hygiene to the CDE candidate pool.
    ``measurand_split`` (M11) appends the measurand-enumeration clause to the generate-ideal prompt so the
    seed does not bundle distinct measurands (systolic/diastolic/pulse) into one concept.
    ``debias_ideal`` (M13) selects the de-biased generate-ideal variant (drops the "one concept" presumption;
    enumerate distinct measurands/concepts as distinct ideals). Takes precedence over ``measurand_split``.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    row_of = {(r.dictionary_name, r.variable_name): i for i, r in enumerate(field_refs)}
    backbone = _build_backbone(embedded_dicts, field_lookup, cde_cohort, cde_dict, clean_text=clean_cde_text)

    records: list[PromptRecord] = []
    for cluster in clusters:
        base = _cluster_base_context(cluster, cde_cohort)
        if base["n_members"] == 0:  # CDE-only cluster — nothing to harmonize
            continue
        non_cde = [m for m in cluster.members if m.dictionary_name != cde_cohort]
        members: list[dict] = []
        for k, m in enumerate(non_cde, 1):
            key = (m.dictionary_name, m.variable_name)
            if key not in row_of:
                continue
            fld = field_lookup.get(key)
            text = _member_text(fld, m)[:_MEMBER_TRUNC]
            members.append(
                {
                    "member_id": f"m{k}",
                    "dictionary_name": m.dictionary_name,
                    "variable_name": m.variable_name,
                    "text": text,  # lean — retrieval (BM25/dense)
                    "prompt_text": _member_prompt_text(fld, m)[:_MEMBER_PROMPT_TRUNC],  # value-aware — prompts
                    "row": row_of[key],
                }
            )
        if not members:
            continue
        rows = [mem["row"] for mem in members]
        member_texts = [mem["text"] for mem in members]  # retrieval
        prompt_lines = [mem["prompt_text"] for mem in members]  # value-aware prompt
        cands, top1 = _retrieve(rows, member_texts, embeddings, backbone, top_k)
        if not cands:
            continue
        ctx = {
            **base,
            "members": members,
            "candidates": cands,
            "top1_cos": round(top1, 4),
            "sample_lines": member_texts[:_SAMPLE_MEMBERS],
        }
        records.append(
            PromptRecord(
                id=f"leanb:ideal:{base['cluster_id']}",
                system_prompt=generate_ideal_system_prompt(measurand_split, debias_ideal),
                user_prompt=build_ideal_user_prompt(prompt_lines[:_SAMPLE_MEMBERS]),
                schema=IDEAL_SCHEMA,
                model_tag=model_tag,
                context=ctx,
            )
        )
    logger.info("prepare_leanb: %d clusters -> %d generate-ideal prompts", len(clusters), len(records))
    return records


def prepare_split(
    ideal_records: list[PromptRecord],
    ideal_responses: dict[str, object],
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    max_show: int = MAX_SHOW,
    representation_refine: bool = False,
    measurand_split: bool = False,
    bundle_guard: bool = False,
    enforce_schema: bool = False,
) -> list[PromptRecord]:
    """Build the stage-2 split-assign prompts from the generated ideals + the carried members/candidates.

    The split prompt shows the cluster members (up to ``max_show``), each prefixed with its ``[mK]`` id,
    and asks the model to PARTITION them into distinct-concept groups + decide each. With M2 chunking on,
    every unit is ``<= max_show`` so no member is truncated before the LLM.

    ``representation_refine`` (M4) appends the representation-mismatch clause so a same-concept candidate in a
    different encoding (banding/flag/composite/unit) routes to refine, not novel.
    ``measurand_split`` (M11) appends the measurand-axis clause so distinct quantities sharing one object
    (systolic vs diastolic BP vs pulse) are partitioned into separate groups instead of fused.
    ``bundle_guard`` (M14) appends the bundling-candidate guard so a candidate bundling several measurands/
    concepts does not license a shared adopt over a heterogeneous group (split first).
    ``enforce_schema`` (M15) issues the split as a FORCED tool call (structurally guaranteed ``{groups:[…]}``
    wrapper) instead of a soft text schema — eliminates the ~35% wrapper-drop that dropped residual members.
    """
    sys_prompt = split_system_prompt(representation_refine, measurand_split, bundle_guard)
    records: list[PromptRecord] = []
    for rec in ideal_records:
        ctx = rec.context
        ideal_cde = _parse_ideal(ideal_responses.get(rec.id))
        members = ctx["members"]
        numbered = [(mem["member_id"], mem["prompt_text"]) for mem in members[:max_show]]
        cand_block = _numbered_candidate_block(ctx["candidates"])
        records.append(
            PromptRecord(
                id=f"leanb:split:{ctx['cluster_id']}",
                system_prompt=sys_prompt,
                user_prompt=build_split_user_prompt(ideal_cde, numbered, cand_block),
                schema=SPLIT_SCHEMA,
                model_tag=model_tag,
                context={**ctx, "ideal_cde": ideal_cde},
                tool_schema=SPLIT_TOOL_SCHEMA if enforce_schema else None,
                tool_name=SPLIT_TOOL_NAME if enforce_schema else None,
                # Enforced tool-call JSON for a large cluster (up to chunk_cap members) overflows the 1024
                # stage default -> truncated tool input -> 0 members (observed on the 25-member BP cluster).
                max_tokens=SPLIT_ENFORCE_MAX_TOKENS if enforce_schema else None,
            )
        )
    logger.info("prepare_split: %d ideal prompts -> %d split prompts", len(ideal_records), len(records))
    return records


def prepare_group_assign(
    split_records: list[PromptRecord],
    split_responses: dict[str, object],
    embedded_dicts: list[EmbeddedDictionary],
    embeddings: NDArray[np.float32],
    field_refs: list[FieldReference],
    *,
    cde_cohort: str = CDE_COHORT,
    cde_dict: EmbeddedDictionary | None = None,
    top_k: int = DEFAULT_TOP_K,
    model_tag: str = DEFAULT_MODEL_TAG,
    clean_cde_text: bool = False,
    representation_refine: bool = False,
    bundle_guard: bool = False,
) -> list[PromptRecord]:
    """Parse the split groups and build one per-group, re-retrieved single-concept assign prompt.

    For each parsed group: map ``member_ids`` (m1, m2, …) back to the cluster's carried members,
    RE-RETRIEVE a per-group hybrid top-k (the group-member centroid ⊕ BM25 over the group's own member
    text), and build a per-group single-concept assign prompt. The prompt's ``context`` carries everything
    :func:`assemble_leanb` needs for the group's record (the per-group candidates, cohorts, members, …).

    ``clean_cde_text`` (M5) cleans the per-group candidate pool; ``representation_refine`` (M4) appends the
    representation-mismatch clause to the per-group assign prompt.
    ``bundle_guard`` (M14) appends the bundling-candidate guard to the per-group assign prompt.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    backbone = _build_backbone(embedded_dicts, field_lookup, cde_cohort, cde_dict, clean_text=clean_cde_text)
    sys_prompt = group_reassign_system_prompt(representation_refine, bundle_guard)
    row_of = {(r.dictionary_name, r.variable_name): i for i, r in enumerate(field_refs)}

    records: list[PromptRecord] = []
    for rec in split_records:
        ctx = rec.context
        members = ctx["members"]
        by_id = {mem["member_id"]: mem for mem in members}
        groups = _parse_split_groups(split_responses.get(rec.id))
        # No usable split -> fall back to a single group over ALL members (no over-split, no drop).
        if not groups:
            groups = [{"member_ids": [mem["member_id"] for mem in members], "concept": "", "verdict": "", "raw": {}}]
        else:
            # Residual completion for ANY partial coverage (M1). The split step drops members on LARGE
            # clusters in two ways: the wrapper-drop (a single bare group, ~35% of clusters) AND well-formed
            # MULTI-group splits that enumerate only a SUBSET of members (e.g. a 438-member food cluster ->
            # 3 valid groups covering 22 -> 416 dropped). Union the coverage over ALL groups and sweep every
            # uncovered member into one residual group so it still flows through assign instead of vanishing.
            # A split that already covers every member is a no-op (no spurious residual). This subsumes the
            # earlier len==1-only gate.
            all_ids = [mem["member_id"] for mem in members]
            covered = {"m" + re.sub(r"\D", "", str(mid)) for g in groups for mid in g.get("member_ids", [])}
            residual = [mid for mid in all_ids if "m" + re.sub(r"\D", "", str(mid)) not in covered]
            if residual and len(residual) < len(all_ids):
                groups.append({"member_ids": residual, "concept": "", "verdict": "", "raw": {}})
        for gi, group in enumerate(groups):
            grp_members = _group_members(group.get("member_ids", []), by_id, members, fallback=len(groups) == 1)
            if not grp_members:
                continue
            rows = [mem["row"] for mem in grp_members if (mem["dictionary_name"], mem["variable_name"]) in row_of]
            member_texts = [mem["text"] for mem in grp_members]  # retrieval
            prompt_lines = [mem["prompt_text"] for mem in grp_members]  # value-aware prompt
            cands, top1 = _retrieve(rows, member_texts, embeddings, backbone, top_k)
            concept = str(group.get("concept", ""))
            cohorts = sorted({mem["dictionary_name"] for mem in grp_members})
            var_names = [f"{mem['dictionary_name']}:{mem['variable_name']}" for mem in grp_members]
            grp_ctx = {
                "cluster_id": ctx["cluster_id"],
                "group_idx": gi,
                "group_id": f"{ctx['cluster_id']}#g{gi}",
                "concept": concept,
                "ideal_cde": ctx.get("ideal_cde", ""),
                "candidates": cands,
                "top1_cos": round(top1, 4),
                "member_variable_names": var_names,
                "cohorts": cohorts,
                "cross_cohort": len(cohorts) >= 2,
                "n_members": len(grp_members),
                "split_raw": group.get("raw", {}),
            }
            records.append(
                PromptRecord(
                    id=f"leanb:groupassign:{ctx['cluster_id']}:{gi}",
                    system_prompt=sys_prompt,
                    user_prompt=build_group_assign_user_prompt(
                        concept or ctx.get("ideal_cde", ""),
                        prompt_lines[:_SAMPLE_MEMBERS],
                        [c["text"] for c in cands],
                    ),
                    schema=ASSIGN_SCHEMA,
                    model_tag=model_tag,
                    context=grp_ctx,
                )
            )
    logger.info("prepare_group_assign: %d split prompts -> %d per-group prompts", len(split_records), len(records))
    return records


def assemble_leanb(
    group_assign_records: list[PromptRecord],
    responses: dict[str, object],
    *,
    retrieval_floor: float = DEFAULT_RETRIEVAL_FLOOR,
    adopt_floor: float | None = None,
) -> LeanBResult:
    """Parse per-group assign responses into routed :class:`LeanBRecord` decisions (one per group).

    The verdict + chosen ``cde_id`` are parsed against the PER-GROUP candidates. ``retrieval_floor`` (#1):
    if a verdict adopts/refines a candidate whose dense cosine is below the floor, downgrade it to
    ``novel`` (the engine force-fit the least-bad candidate when nothing was close). 0 disables.

    ``adopt_floor`` (M5, opt-in): an ADOPT claims exact equivalence, which needs stronger geometric support
    than a refine. When set and the chosen cosine sits in ``[retrieval_floor, adopt_floor)``, the adopt is
    demoted to ``refine`` (still ``assigned`` and eligible for a transform spec, but no longer an exact
    adopt; ``adopt_demoted=True``). ``refine`` verdicts are untouched; below ``retrieval_floor`` the record
    has already gone ``novel``. ``None`` (default) leaves adopt handling unchanged.
    """
    records: list[LeanBRecord] = []
    for rec in group_assign_records:
        ctx = rec.context
        cands = ctx["candidates"]
        payload = _parse_assign(responses.get(rec.id))
        verdict = str(payload.get("verdict", "")) if payload else ""
        des = ext = chosen_cos = None
        if verdict in ("adopt", "refine"):
            i = _candidate_index(payload.get("cde_id") if payload else None)
            if i is None or not (0 <= i < len(cands)):
                # The model commonly conveys its pick via `ranking` (best-first) and leaves `cde_id` null;
                # fall back to its top-ranked candidate so the chosen CDE + cosine resolve. Without this,
                # chosen_cos stays None and the retrieval floor wrongly downgrades EVERY adopt/refine to novel.
                rk = _parse_ranking(payload.get("ranking") if payload else None, len(cands))
                i = rk[0] if rk else None
            if i is not None and 0 <= i < len(cands):
                des = cands[i]["designation"]
                ext = cands[i]["external_id"] or None
                chosen_cos = cands[i].get("cos")
        # #1 retrieval floor: force novel when the chosen candidate is geometrically far (or unresolved)
        floored = bool(
            verdict in ("adopt", "refine")
            and retrieval_floor > 0
            and (chosen_cos is None or chosen_cos < retrieval_floor)
        )
        if floored:
            verdict, des, ext = "novel", None, None
        # M5 adopt-specific floor: a weak-support ADOPT (retrieval_floor <= cos < adopt_floor) can't support
        # its exact-equivalence claim -> demote to refine. Runs after the retrieval floor, so a survivor here
        # always has chosen_cos >= retrieval_floor. refine is left as-is.
        adopt_demoted = bool(
            verdict == "adopt"
            and adopt_floor is not None
            and adopt_floor > 0
            and chosen_cos is not None
            and chosen_cos < adopt_floor
        )
        if adopt_demoted:
            verdict = "refine"
        route = "assigned" if verdict in ("adopt", "refine") else "gencde_residual"
        top1 = ctx.get("top1_cos")
        rk = _parse_ranking(payload.get("ranking") if payload else None, len(cands))
        # Persist the ranked candidate set for the review UI (the sub-cluster-anchored pipeline discarded it). Best-first by the LLM
        # ranking, then any un-ranked candidates in retrieval order.
        cand_order = rk + [j for j in range(len(cands)) if j not in rk]
        llm_top = rk[0] if rk else None
        candidates = [
            CandidateCDE(
                rank=pos + 1,
                cde_id=cands[j]["designation"],
                cde_external_id=cands[j].get("external_id") or None,
                definition=str(cands[j].get("text", "")),
                cosine=round(float(cands[j].get("cos", 0.0)), 4),
                is_chosen=des is not None and cands[j]["designation"] == des,
                llm_suggested=j == llm_top,
            )
            for pos, j in enumerate(cand_order)
        ]
        records.append(
            LeanBRecord(
                cluster_id=ctx["cluster_id"],
                verdict=verdict,
                route=route,
                group_id=ctx.get("group_id", ""),
                concept=ctx.get("concept", ""),
                cde_id=des,
                cde_external_id=ext,
                ideal_cde=ctx.get("ideal_cde", ""),
                ranking=rk,
                candidates=candidates,
                rationale=str(payload.get("rationale", "")) if payload else "",
                top1_cos=top1,
                chosen_cos=round(chosen_cos, 4) if chosen_cos is not None else None,
                coverage_gap=bool(verdict == "novel" and top1 is not None and top1 < COVERAGE_GAP_TAU),
                floored=floored,
                adopt_demoted=adopt_demoted,
                member_variable_names=ctx.get("member_variable_names", []),
                cohorts=ctx.get("cohorts", []),
                cross_cohort=ctx.get("cross_cohort", False),
                n_members=ctx.get("n_members", 0),
                decided_by="llm",
                raw=payload or {},
            )
        )
    return LeanBResult(records=records, group_assign_prompts=group_assign_records)


def concept_groups_from_prompts(group_assign_records: list[PromptRecord]) -> list[ConceptGroup]:
    """Materialize the post-split concept groups the per-group assign prompts describe — $0, no LLM.

    One :class:`~.models.ConceptGroup` per assign prompt, read straight out of the prompt's own context.
    This is the shape that exists between ``split`` and ``classify``: the split stage has partitioned every
    cluster into distinct concepts, but nothing has assigned any of them a CDE yet. It is what the coherence
    judge's verdict pass runs on, and what a caller pausing at the ``classify=None`` boundary renders.

    Deterministic and order-preserving: group ``i`` here corresponds to ``group_assign_records[i]``, hence to
    ``assemble_leanb``'s record ``i``. :func:`transfer_coherence_verdicts` relies only on the ids, not the
    order, so a merge or a residual sweep between the two cannot mis-pair them.
    """
    groups: list[ConceptGroup] = []
    for rec in group_assign_records:
        ctx = rec.context
        groups.append(
            ConceptGroup(
                cluster_id=ctx["cluster_id"],
                group_id=ctx.get("group_id", ""),
                concept=ctx.get("concept", ""),
                ideal_cde=ctx.get("ideal_cde", ""),
                top1_cos=ctx.get("top1_cos"),
                member_variable_names=list(ctx.get("member_variable_names", [])),
                cohorts=list(ctx.get("cohorts", [])),
                cross_cohort=bool(ctx.get("cross_cohort", False)),
                n_members=int(ctx.get("n_members", 0)),
            )
        )
    return groups


# ── step 2: dual-sample coherence judge (post-SPLIT, pre-assign; read-only) ───────────────────────
#
# The judge runs in TWO passes, and the order matters:
#
#   1. the VERDICT pass (:func:`assemble_coherence_verdicts`) — after ``split``, before ``classify``, at
#      the judge's native post-split concept-group granularity. It stamps the verdict fields AND the hard
#      ``incoherent`` flag, both of which are pure functions of the judge response with no dependence on
#      anything the assign stage produces. A caller that pauses at the ``classify=None`` boundary can
#      therefore render a group's coherence state without having paid for assign.
#   2. the PROPAGATION pass (:func:`propagate_coherence_review`) — after specgen / gencde, because
#      ``needs_review`` lands on ``rec.transforms`` and ``rec.gencde``, and neither exists before then.
#
# :func:`transfer_coherence_verdicts` bridges the two: it copies pass 1's verdicts from the pre-record
# :class:`~.models.ConceptGroup` shapes onto the real records the moment :func:`assemble_leanb` builds
# them. :func:`assemble_coherence` survives as a thin wrapper running both passes back to back.
#
# The R2 distinct-KINDS discriminator (:func:`prepare_kinds`) reads ``LeanBRecord`` fields and therefore
# stays where it is, AFTER :func:`assemble_leanb` — it cannot move early with the verdict pass.


@runtime_checkable
class CoherenceTarget(Protocol):
    """The structural shape the coherence judge reads and stamps.

    Satisfied by both :class:`~.models.ConceptGroup` (the pre-record, post-split group the verdict pass
    actually runs on) and :class:`~.models.LeanBRecord` (the post-assign record the shipped
    :func:`assemble_coherence` wrapper still accepts). Declared structurally rather than as a concrete
    class so the verdict pass is testable against a hand-built shape and so the pass provably cannot
    touch a record's route or assignment verdict — neither is in this protocol.
    """

    cluster_id: str
    group_id: str
    member_variable_names: list[str]
    n_members: int
    matrix_suspect: bool
    coherent: bool
    coherence_verdict: str
    coherence_summary: str
    coherence_axis: str
    coherence_distinct_values: list[str]
    coherence_outliers: list[str]
    incoherent: bool


def _adaptive_k2(n: int) -> int:
    """Periphery sample size: 20% of the group clipped to [floor, ceiling]; small groups take the remainder.

    ``(n + 4) // 5 == ceil(0.2 * n)`` (integer, no float).
    """
    if n < 10:
        return max(1, n - COHERENCE_K1)
    return min(COHERENCE_K2_CEILING, max(COHERENCE_K2_FLOOR, (n + 4) // 5))


def _dual_sample(rows: list[int], embeddings: NDArray[np.float32]) -> tuple[list[int], list[int]] | None:
    """Split a group's member rows into (k1 centroid-closest, k2 centroid-furthest) DISJOINT local indices.

    "Centroid" is the cosine MEDOID (the member most similar to all others) — robust to the off-concept
    boundary members we are trying to surface. Returns local indices into ``rows``, or ``None`` when the
    group is too small to judge (< :data:`COHERENCE_MIN_MEMBERS`). The disjoint k1/k2 sampling is the fix
    for the self-fulfilling verification of the published method (same closest-sample summarizes AND verifies).
    """
    if len(rows) < COHERENCE_MIN_MEMBERS:
        return None
    embs = embeddings[rows]
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.where(norms == 0, 1.0, norms)
    sim = embs_n @ embs_n.T
    medoid = int(np.argmax(sim.sum(axis=1)))
    order = np.argsort(1.0 - sim[medoid]).tolist()  # closest-to-medoid first
    k1_idx = order[:COHERENCE_K1]
    k1_set = set(k1_idx)
    k2_idx = [i for i in reversed(order) if i not in k1_set][: _adaptive_k2(len(rows))]
    return k1_idx, k2_idx


def _matrix_skeleton(tokens: list[str], df: dict[str, int], rare_cut: int) -> tuple[str, int]:
    """Collapse a member's tokens to a TEMPLATE skeleton (contiguous varying-slot tokens -> one '·' marker).

    Returns ``(skeleton, n_template_tokens)``. A token is a varying SLOT if it is a digit or appears in
    ``<= rare_cut`` members (document frequency); everything else is stable template.
    """
    sk: list[str] = []
    kept = 0
    prev_slot = False
    for t in tokens:
        if t.isdigit() or df.get(t, 0) <= rare_cut:
            if not prev_slot:
                sk.append("·")
            prev_slot = True
        else:
            sk.append(t)
            kept += 1
            prev_slot = False
    return " ".join(sk), kept


def _matrix_suspect(texts: list[str], *, min_template_coverage: float = 0.45, min_members: int = 2) -> bool:
    """$0 deterministic pre-filter: True when ``>= min_members`` texts collapse to one template skeleton.

    Vocabulary-agnostic frequent-template/rare-slot detector for matrix groups (one question template ×
    many entity fillers — "seeing a provider for {condition}", "reading {N}"). High precision, cheap; an
    OPTIONAL pre-filter / triage sort, NOT the primary coherence mechanism (that is the LLM judge).
    """
    texts = [t for t in texts if t]
    if len(texts) < min_members:
        return False
    tok = [re.findall(r"[a-z]+|\d+", t.lower()) for t in texts]
    df: dict[str, int] = defaultdict(int)
    for ts in tok:
        for w in set(ts):
            df[w] += 1
    rare_cut = max(1, len(texts) // 10)  # a slot filler appears in <= 10% of members
    skel: dict[str, int] = defaultdict(int)
    for ts in tok:
        if not ts:
            continue
        sk, kept = _matrix_skeleton(ts, df, rare_cut)
        if kept / len(ts) >= min_template_coverage:  # ignore mostly-slot members
            skel[sk] += 1
    return any(c >= min_members for c in skel.values())


def prepare_coherence(
    records: Sequence[CoherenceTarget],
    embedded_dicts: list[EmbeddedDictionary],
    embeddings: NDArray[np.float32],
    field_refs: list[FieldReference],
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    pre_filter: bool = True,
) -> list[PromptRecord]:
    """Build one read-only coherence-judge prompt per group large enough to judge (>= 6 members).

    Accepts anything satisfying :class:`CoherenceTarget` — the pre-record
    :class:`~.models.ConceptGroup` (the post-split shape the pipeline judges) or a
    :class:`~.models.LeanBRecord` (the shipped post-assign call path). It reads only member names, member
    count and the group/cluster ids, all of which the split stage has already produced, so the judge needs
    no assign-stage data.

    Dual-samples each group's members (k1 centroid-closest core / k2 centroid-furthest periphery) from the
    frozen ``embeddings`` and renders each sampled member as ``cohort:var — <concept text>`` (the lean,
    value-free retrieval text — value codes are geometric noise). Groups below
    :data:`COHERENCE_MIN_MEMBERS` get NO prompt and are left EXPLICITLY UNJUDGED (``coherence_verdict``
    stays ``""`` — never ``"single"``). When ``pre_filter`` is set, the deterministic matrix detector
    is stamped on ``matrix_suspect`` here (a $0 signal independent of the LLM).
    :func:`assemble_coherence_verdicts` folds the LLM verdicts back on.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    row_of = {(r.dictionary_name, r.variable_name): i for i, r in enumerate(field_refs)}

    def member_key(fid: str) -> tuple[str, str]:
        cohort, _, var = fid.partition(":")
        return cohort, var

    prompts: list[PromptRecord] = []
    for rec in records:
        resolved = [(f, row_of[member_key(f)]) for f in rec.member_variable_names if member_key(f) in row_of]
        rows = [row for _, row in resolved]
        fids = [f for f, _ in resolved]
        texts = [
            _member_text(field_lookup.get(member_key(f)), field_refs[row])[:_COH_TEXT_TRUNC] for f, row in resolved
        ]
        if pre_filter:
            rec.matrix_suspect = _matrix_suspect(texts)
        sample = _dual_sample(rows, embeddings)
        if sample is None:
            continue  # group too small to judge -> stays coherent by default
        k1_idx, k2_idx = sample
        core = [f"{fids[i]} — {texts[i]}" for i in k1_idx]
        periphery = [f"{fids[i]} — {texts[i]}" for i in k2_idx]
        prompts.append(
            PromptRecord(
                id=f"leanb:coherence:{rec.group_id or rec.cluster_id}",
                system_prompt=SYS_COHERENCE,
                user_prompt=build_coherence_user_prompt(rec.n_members, core, periphery),
                schema=COHERENCE_SCHEMA,
                model_tag=model_tag,
                context={
                    "group_id": rec.group_id,
                    "cluster_id": rec.cluster_id,
                    "periphery_fids": [fids[i] for i in k2_idx],
                },
            )
        )
    logger.info(
        "prepare_coherence: %d records -> %d judge prompts (groups >= %d members)",
        len(records),
        len(prompts),
        COHERENCE_MIN_MEMBERS,
    )
    return prompts


def _parse_coherence(resp: object) -> dict | None:
    if resp is None:
        return None
    try:
        payload = extract_json(resp if isinstance(resp, str) else json.dumps(resp))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _coherence_key(cluster_id: str, group_id: str) -> str:
    """The judge's fold key: the concept group when there is one, else the cluster.

    Mirrors the prompt id (``leanb:coherence:{group_id or cluster_id}``) so the fold cannot miss an item
    the prompt was built for. The predecessor keyed on ``group_id`` alone and SILENTLY DROPPED any item
    carrying only ``cluster_id`` — the verdict vanished with no error. The judge's operating granularity is
    still the post-split concept group; this fallback is a no-silent-drop guard, not a second granularity.
    """
    return group_id or cluster_id


def assemble_coherence_verdicts(
    coherence_records: list[PromptRecord],
    responses: dict[str, object],
    targets: Sequence[CoherenceTarget],
) -> Sequence[CoherenceTarget]:
    """Fold the dual-sample judge verdicts onto the judged groups — FLAG, never gate.

    Pass 1 of 2 (see the section header). Runs after ``split`` and BEFORE ``classify``, on the post-split
    :class:`~.models.ConceptGroup` shapes, so a caller pausing at the ``classify=None`` boundary already has
    every field below. Sets ``coherent`` / ``coherence_verdict`` / ``coherence_summary`` /
    ``coherence_axis`` / ``coherence_distinct_values`` / ``coherence_outliers`` on each judged group.

    The HARD flag ``incoherent`` (→ ``needs_review`` on the recodes / GenCDE, surfaced for human
    re-adjudication) is stamped HERE, by the verdict pass — it is a pure function of the verdict with no
    dependence on post-assign state, and a reviewer screen that renders before assign must be able to show
    it. Deferring it to :func:`propagate_coherence_review` would make every pre-assign row read as coherent.

    It fires ONLY on ``verdict == "split"`` — a group too varied to be one concept even as a one-slot
    template. Calibrated on the held-out 5-cohort run: treating ``qualify`` as a
    hard flag over-fired 46% (coherent concepts that merely carry a value/qualifier axis — "milk
    consumption *by fat content*", PHQ/GAD items); ``split`` alone is the precise 14% over-merge signal.
    ``qualify`` is therefore an ADVISORY — its ``axis`` + ``distinct_values`` ARE the CDE × qualifier
    value-set hint, recorded but NOT ``needs_review``. ``coherent`` false + the flagged
    ``coherence_outliers`` stay a secondary signal.

    A group with no response, an unparseable response or a response carrying no usable verdict is left
    EXPLICITLY UNJUDGED (``coherence_verdict == ""``) and never resolves to ``"single"`` — the same state a
    sub-threshold group is left in, so "not judged" is never rendered as "judged coherent". Parse failures
    degrade rather than raise, so a malformed judge response cannot unwind a paid run.

    The group's route and assignment verdict are LEFT UNCHANGED — neither is even reachable through
    :class:`CoherenceTarget`. The pipeline NEVER silently re-splits an over-merged group (the cure is the
    human loop).
    """
    by_group = {_coherence_key(t.cluster_id, t.group_id): t for t in targets}
    for cr in coherence_records:
        payload = _parse_coherence(responses.get(cr.id))
        if payload is None:
            continue
        gid = cr.context.get("group_id") or cr.context.get("cluster_id")
        rec = by_group.get(gid) if isinstance(gid, str) else None
        if rec is None:
            continue
        gran = payload.get("granularity")
        gran = gran if isinstance(gran, dict) else {}
        verdict = str(gran.get("verdict", "")).strip().lower()
        rec.coherent = bool(payload.get("coherent", True))
        rec.coherence_summary = str(payload.get("summary", ""))
        rec.coherence_verdict = verdict if verdict in ("single", "qualify", "split") else ""
        axis = gran.get("axis")
        rec.coherence_axis = "" if axis in (None, "null", "") else str(axis)
        dv = gran.get("distinct_values")
        rec.coherence_distinct_values = [str(x) for x in dv] if isinstance(dv, list) else []
        periphery_fids = cr.context.get("periphery_fids", [])
        positions: list[int] = []
        for x in payload.get("outliers", []) if isinstance(payload.get("outliers"), list) else []:
            try:
                positions.append(int(x))
            except (TypeError, ValueError):
                continue
        rec.coherence_outliers = [periphery_fids[i - 1] for i in positions if 1 <= i <= len(periphery_fids)]
        # RECALIBRATED on the held-out 5-cohort run: ONLY `split` is a hard flag. `qualify` is advisory (over-fired 46%
        # on coherent-with-qualifier groups); `coherent`/`coherence_outliers` are a secondary signal.
        rec.incoherent = rec.coherence_verdict == "split"
    return targets


_COHERENCE_VERDICT_FIELDS = (
    "coherent",
    "coherence_verdict",
    "coherence_summary",
    "coherence_axis",
    "coherence_distinct_values",
    "coherence_outliers",
    "incoherent",
    "matrix_suspect",
)


def transfer_coherence_verdicts(
    groups: Sequence[CoherenceTarget],
    records: list[LeanBRecord],
) -> list[LeanBRecord]:
    """Copy the early verdict pass's output from the pre-record groups onto the assembled records.

    The bridge between the two judge passes. :func:`assemble_coherence_verdicts` runs before ``classify``,
    so it stamps :class:`~.models.ConceptGroup` shapes; :func:`assemble_leanb` then builds the real
    :class:`~.models.LeanBRecord` objects from the same per-group assign prompts. Without this step the
    verdicts would be stranded on the groups and every record would read as unjudged — which is exactly the
    silent "unjudged rendered as coherent" failure the judge's own contract forbids.

    Matched on :func:`_coherence_key` (group id, cluster id as fallback), so a one-identifier item is not
    dropped. Copies the verdict fields and the $0 ``matrix_suspect`` pre-filter, and NOTHING else — it does
    not propagate ``needs_review`` (that is :func:`propagate_coherence_review`'s job, after specgen) and it
    does not touch the record's verdict or route. Idempotent: a replay re-copies the same values.
    """
    by_group = {_coherence_key(g.cluster_id, g.group_id): g for g in groups}
    for rec in records:
        grp = by_group.get(_coherence_key(rec.cluster_id, rec.group_id))
        if grp is None:
            continue
        for name in _COHERENCE_VERDICT_FIELDS:
            setattr(rec, name, getattr(grp, name))
    return records


def propagate_coherence_review(records: list[LeanBRecord]) -> list[LeanBRecord]:
    """Land the hard coherence flag on the artifacts that only exist after assign — pass 2 of 2.

    This is the ONLY part of the judge that depends on post-assign state: ``rec.transforms`` is populated
    by the transform-spec stage and ``rec.gencde`` by the GenCDE stage, so neither exists when the verdict
    pass runs. An ``incoherent`` record's recodes and GenCDE are marked ``needs_review`` — a recode into a
    group that is not one concept is meaningless until a human re-adjudicates it.

    IDEMPOTENT, because a resumed run may replay it: setting ``needs_review`` twice is the same as setting
    it once, and a coherent record is never touched. Also safe to call when the judge never ran — no record
    carries ``incoherent``, so it is a no-op. FLAG, never gate: verdict and route are left alone, and
    nothing here re-groups anything.
    """
    for rec in records:
        if not rec.incoherent:
            continue
        for t in rec.transforms:
            t.needs_review = True
        if rec.gencde is not None:
            rec.gencde.needs_review = True
    return records


def assemble_coherence(
    coherence_records: list[PromptRecord],
    responses: dict[str, object],
    records: list[LeanBRecord],
) -> list[LeanBRecord]:
    """Both judge passes back to back, on post-assign records — the shipped single-call entry point.

    Retained verbatim in behaviour for the Batch/driver path and the notebooks, which fold the judge in one
    step over finished records. New code inside the pipeline calls the two halves separately, because the
    verdict pass has to run before ``classify`` and the propagation pass cannot
    (:func:`assemble_coherence_verdicts`, :func:`propagate_coherence_review`).
    """
    assemble_coherence_verdicts(coherence_records, responses, records)
    return propagate_coherence_review(records)


# ── R2: distinct-KINDS discriminator over `qualify` groups (opt-in, ADDITIVE over the R0 split flag) ──
# R0 (shipped default, set in assemble_coherence) flags only `split`. R2 additionally flags a `qualify`
# group when this cheap second LLM read calls it `distinct_kinds` (genuinely different measurands sharing an
# axis label) rather than `values_of_one_property` (one concept + a qualifier value-set). Validated on the
# returned human pairwise gold — 100% recall on human-confirmed over-merges.
_INT_RE = re.compile(r"^\s*-?\d+\s*$")


def _is_positional(distinct_values: list[str]) -> bool:
    """True when a majority of the judge's ``distinct_values`` are bare integers — an occurrence index /
    repeating measure (numbered trials, Minnesota T-wave 1..8): one concept measured repeatedly, NEVER an
    over-merge. R2 excludes these (reproduces the ``exclude_positional`` guard the rule was validated with;
    discovered-not-hardcoded — keys on value SHAPE, no cohort/vocabulary specifics)."""
    if len(distinct_values) < 2:
        return False
    bare = sum(1 for v in distinct_values if _INT_RE.match(str(v)))
    return bare >= (len(distinct_values) + 1) // 2


def prepare_kinds(records: list[LeanBRecord]) -> list[PromptRecord]:
    """R2 — build a distinct-KINDS discriminator prompt for each ``qualify`` record.

    Only ``qualify`` groups need it (``single`` is coherent; ``split`` is already R0-flagged). Positional /
    repeating-measure groups are skipped ($0 exclusion → never flagged, and no wasted call). Reads the
    coherence judge's own outputs (summary / axis / distinct_values) — so it MUST run after
    :func:`assemble_coherence` has stamped them.
    """
    prompts: list[PromptRecord] = []
    for rec in records:
        if rec.coherence_verdict != "qualify" or _is_positional(rec.coherence_distinct_values):
            continue
        prompts.append(
            PromptRecord(
                id=f"leanb:kinds:{rec.group_id or rec.cluster_id}",
                system_prompt=SYS_KINDS,
                user_prompt=build_kinds_user_prompt(
                    rec.coherence_summary, rec.coherence_axis, rec.coherence_distinct_values
                ),
                schema=json.dumps(KINDS_TOOL_SCHEMA),
                model_tag=DEFAULT_MODEL_TAG,
                context={"group_id": rec.group_id, "cluster_id": rec.cluster_id},
                tool_schema=KINDS_TOOL_SCHEMA,
                tool_name=KINDS_TOOL_NAME,
            )
        )
    logger.info("prepare_kinds: %d qualify groups -> distinct-KINDS discriminator prompts (R2)", len(prompts))
    return prompts


def assemble_kinds(
    kinds_prompts: list[PromptRecord],
    responses: dict[str, object],
    records: list[LeanBRecord],
) -> list[LeanBRecord]:
    """Fold the distinct-KINDS discriminator verdicts onto the records — R2 = R0 ∪ (qualify ∧ distinct_kinds).

    A ``distinct_kinds`` verdict HARD-flags the qualify group (``incoherent=True`` + ``needs_review`` on its
    recodes / GenCDE — the SAME escalation as a ``split``); ``values_of_one_property`` leaves it coherent.
    ADDITIVE — never clears an existing R0 (``split``) flag. Stamps ``coherence_kind`` for audit. FLAG, never
    gate: verdict/route are unchanged (the cure is the human re-adjudication loop, as with R0).

    Folds on :func:`_coherence_key`, matching the prompt id ``leanb:kinds:{group_id or cluster_id}``. The
    predecessor keyed on ``group_id`` alone while the prompt id already fell back to ``cluster_id``, so a
    record carrying only ``cluster_id`` got a prompt built and paid for, and then had its verdict SILENTLY
    discarded — the same no-silent-drop defect fixed in the coherence fold.
    """
    by_group = {_coherence_key(r.cluster_id, r.group_id): r for r in records}
    for cr in kinds_prompts:
        payload = _parse_coherence(responses.get(cr.id))  # generic tolerant JSON/tool-call parse
        if payload is None:
            continue
        gid = cr.context.get("group_id") or cr.context.get("cluster_id")
        rec = by_group.get(gid) if isinstance(gid, str) else None
        if rec is None:
            continue
        kind = str(payload.get("kind", "")).strip().lower()
        rec.coherence_kind = kind if kind in ("values_of_one_property", "distinct_kinds") else ""
        if rec.coherence_kind == "distinct_kinds":
            rec.incoherent = True
            for t in rec.transforms:
                t.needs_review = True
            if rec.gencde is not None:
                rec.gencde.needs_review = True
    return records


# ── re-adjudication: human-triggered re-split of a flagged over-merged group ──────────────────────


def prepare_readjudicate(
    flagged_records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    embeddings: NDArray[np.float32],
    field_refs: list[FieldReference],
    *,
    cde_cohort: str = CDE_COHORT,
    cde_dict: EmbeddedDictionary | None = None,
    top_k: int = DEFAULT_TOP_K,
    model_tag: str = DEFAULT_MODEL_TAG,
    clean_cde_text: bool = False,
    enforce_schema: bool = True,
    desired_n: dict[str, int] | None = None,
) -> list[PromptRecord]:
    """Build one re-split prompt per FLAGGED over-merged record, to re-partition it into coherent concepts.

    Reconstructs each flagged group's members (``member_variable_names`` -> embedding rows + text),
    re-retrieves cluster-level CDE candidates, and builds a split-shaped :class:`PromptRecord` whose
    ``context`` matches what :func:`prepare_group_assign` consumes — so re-adjudication reuses the normal
    split -> per-group-assign machinery verbatim. The prompt carries the coherence judge's ``axis`` +
    ``distinct_values`` as a hint and an optional per-group ``desired_n`` target. Each child's ids are
    namespaced under the PARENT ``group_id`` (``context['cluster_id'] = parent.group_id``) so re-split
    children are unique and traceable to the group they were carved from.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    backbone = _build_backbone(embedded_dicts, field_lookup, cde_cohort, cde_dict, clean_text=clean_cde_text)
    row_of = {(r.dictionary_name, r.variable_name): i for i, r in enumerate(field_refs)}
    desired_n = desired_n or {}

    records: list[PromptRecord] = []
    for rec in flagged_records:
        members: list[dict] = []
        for k, fid in enumerate(rec.member_variable_names, 1):
            cohort, _, var = fid.partition(":")
            key = (cohort, var)
            if key not in row_of:
                continue
            fld = field_lookup.get(key)
            ref = field_refs[row_of[key]]
            members.append(
                {
                    "member_id": f"m{k}",
                    "dictionary_name": cohort,
                    "variable_name": var,
                    "text": _member_text(fld, ref)[:_MEMBER_TRUNC],
                    "prompt_text": _member_prompt_text(fld, ref)[:_MEMBER_PROMPT_TRUNC],
                    "row": row_of[key],
                }
            )
        if len(members) < 2:
            continue  # nothing to re-partition
        rows = [m["row"] for m in members]
        cands, top1 = _retrieve(rows, [m["text"] for m in members], embeddings, backbone, top_k)
        if not cands:
            continue
        parent_gid = rec.group_id or rec.cluster_id
        numbered = [(m["member_id"], m["prompt_text"]) for m in members[:MAX_SHOW]]
        ctx = {
            "cluster_id": parent_gid,  # namespaces the child records under the parent group (traceable + unique)
            "cohorts": rec.cohorts,
            "cross_cohort": rec.cross_cohort,
            "n_members": len(members),
            "members": members,
            "candidates": cands,
            "top1_cos": round(top1, 4),
            "ideal_cde": rec.ideal_cde,
        }
        records.append(
            PromptRecord(
                id=f"leanb:readjudicate:{parent_gid}",
                system_prompt=SYS_READJUDICATE,
                user_prompt=build_readjudicate_user_prompt(
                    numbered,
                    _numbered_candidate_block(cands),
                    axis=rec.coherence_axis,
                    distinct_values=rec.coherence_distinct_values,
                    desired_n=desired_n.get(parent_gid),
                ),
                schema=SPLIT_SCHEMA,
                model_tag=model_tag,
                context=ctx,
                tool_schema=SPLIT_TOOL_SCHEMA if enforce_schema else None,
                tool_name=SPLIT_TOOL_NAME if enforce_schema else None,
                max_tokens=SPLIT_ENFORCE_MAX_TOKENS if enforce_schema else None,
            )
        )
    logger.info("prepare_readjudicate: %d flagged records -> %d re-split prompts", len(flagged_records), len(records))
    return records


def readjudicate(
    result: LeanBResult,
    embedded_dicts: list[EmbeddedDictionary],
    embeddings: NDArray[np.float32],
    field_refs: list[FieldReference],
    *,
    split: Callable[[list[PromptRecord]], dict[str, object]],
    classify: Callable[[list[PromptRecord]], dict[str, object]],
    group_ids: list[str] | None = None,
    desired_n: dict[str, int] | None = None,
    cde_cohort: str = CDE_COHORT,
    cde_dict: EmbeddedDictionary | None = None,
    top_k: int = DEFAULT_TOP_K,
    model_tag: str = DEFAULT_MODEL_TAG,
    clean_cde_text: bool = True,
    representation_refine: bool = True,
    enforce_schema: bool = True,
    retrieval_floor: float = DEFAULT_RETRIEVAL_FLOOR,
    adopt_floor: float | None = DEFAULT_ADOPT_FLOOR,
) -> LeanBResult:
    """Human-triggered re-adjudication: re-split flagged over-merged groups and splice children back in.

    Selects the records to re-adjudicate (``group_ids`` if given, else every record with ``incoherent``),
    re-splits each into distinct concepts, then re-retrieves + re-assigns each child (reusing
    :func:`prepare_group_assign` + :func:`assemble_leanb`) and REPLACES each flagged parent in
    ``result.records`` with its children (tagged ``readjudicated_from``). The pipeline never re-splits
    automatically — the coherence flag is a suggestion; this pass runs only when a caller invokes it (the
    workbench action / the driver), so ``split`` + ``classify`` are supplied by the caller (mirror
    harmonize_leanb's stage callables). A parent that re-splits to a single group is effectively unchanged
    (one child replaces it). The GenCDE/spec tail is NOT re-run here — the caller re-runs the standard
    gencde/specgen stages over the updated ``result.records`` (the new novel children are picked up there).
    """
    selected = (
        [r for r in result.records if r.incoherent]
        if group_ids is None
        else [r for r in result.records if r.group_id in set(group_ids)]
    )
    if not selected:
        return result
    readj_prompts = prepare_readjudicate(
        selected,
        embedded_dicts,
        embeddings,
        field_refs,
        cde_cohort=cde_cohort,
        cde_dict=cde_dict,
        top_k=top_k,
        model_tag=model_tag,
        clean_cde_text=clean_cde_text,
        enforce_schema=enforce_schema,
        desired_n=desired_n,
    )
    if not readj_prompts:
        return result
    child_assign_prompts = prepare_group_assign(
        readj_prompts,
        split(readj_prompts),
        embedded_dicts,
        embeddings,
        field_refs,
        cde_cohort=cde_cohort,
        cde_dict=cde_dict,
        top_k=top_k,
        model_tag=model_tag,
        clean_cde_text=clean_cde_text,
        representation_refine=representation_refine,
    )
    children = assemble_leanb(
        child_assign_prompts, classify(child_assign_prompts), retrieval_floor=retrieval_floor, adopt_floor=adopt_floor
    ).records
    for c in children:
        c.readjudicated_from = c.cluster_id  # prepare_readjudicate set ctx cluster_id = the parent group_id
    reworked = {r.group_id for r in selected}
    result.records = [r for r in result.records if r.group_id not in reworked] + children
    logger.info("readjudicate: %d flagged group(s) -> %d re-split children (spliced)", len(selected), len(children))
    return result


def recover_outlier_clusters(
    clusters: list[FieldCluster],
    substrate: ClusteringSubstrate,
    embeddings: NDArray[np.float32],
    field_refs: list[FieldReference],
    *,
    min_cluster_size: int = 8,
) -> tuple[list[FieldCluster], ClusteringSubstrate]:
    """M10 — recover the substrate's HDBSCAN outliers as extra clusters via an isolated residual re-cluster.

    The main clustering drops ~18% of fields as noise, but a large share are coherent SUB-THRESHOLD families
    (smaller than the main ``min_cluster_size``): whole cancer-type sets, short scales, afford-care items.
    :func:`~ddharmon.clustering.topic_engine.recluster_residual` finds them at a lower density (a
    recall-favoring coarse pass designed to FEED the split-aware stage — which then partitions any over-merge).
    Recovered clusters are appended to ``clusters`` (so they flow through the normal ideal->split->assign
    pipeline) and folded into a NEW substrate (recovered members move from ``outlier`` into ``clusters``) so a
    later frozen-substrate replay reproduces the recovery exactly — the residual re-cluster uses UMAP and is
    not bit-reproducible, so it must be frozen. A no-op (returns the inputs unchanged) when the substrate has
    no outliers or none map to a field row.
    """
    from ddharmon.clustering.topic_engine import recluster_residual

    if not substrate.outlier:
        return clusters, substrate
    row_of = {(r.dictionary_name, r.variable_name): i for i, r in enumerate(field_refs)}
    rows = [row_of[k] for k in substrate.outlier if k in row_of]
    if not rows:
        return clusters, substrate
    recovered, _leftover = recluster_residual(embeddings, field_refs, rows, min_cluster_size=min_cluster_size)
    if not recovered:
        return clusters, substrate
    recovered_keys = {(m.dictionary_name, m.variable_name) for cl in recovered for m in cl.members}
    new_substrate = ClusteringSubstrate(
        clusters=substrate.clusters + [[(m.dictionary_name, m.variable_name) for m in cl.members] for cl in recovered],
        min_cluster_size=substrate.min_cluster_size,
        n_fields=substrate.n_fields,
        outlier=[k for k in substrate.outlier if k not in recovered_keys],
    )
    logger.info(
        "recover_outlier_clusters: recovered %d clusters (%d fields) from %d outliers; %d still noise",
        len(recovered),
        len(recovered_keys),
        len(rows),
        len(new_substrate.outlier),
    )
    return clusters + recovered, new_substrate


def harmonize_leanb(
    embedded_dicts: list[EmbeddedDictionary],
    *,
    generate: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    split: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    classify: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    merge: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    specgen: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    concept_gate: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    coherence: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    distinct_kinds: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    gencde: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    refine: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    cde_cohort: str = CDE_COHORT,
    min_cluster_size: int = 15,
    top_k: int = DEFAULT_TOP_K,
    retrieval_floor: float = DEFAULT_RETRIEVAL_FLOOR,
    adopt_floor: float | None = DEFAULT_ADOPT_FLOOR,
    model_tag: str = DEFAULT_MODEL_TAG,
    substrate: ClusteringSubstrate | None = None,
    chunk_cap: int | None = MAX_SHOW,
    chunk_skip_enumerated: bool = True,
    max_show: int = MAX_SHOW,
    coherence_gate: bool = True,
    clean_cde_text: bool = True,
    representation_refine: bool = True,
    measurand_split: bool = False,
    gencde_specgen: bool = False,
    refine_cdes: bool = False,
    recover_outliers: bool = True,
    residual_min_cluster_size: int = 8,
    max_clusters: int | None = None,
    stop_after: str | None = None,
) -> LeanBResult:
    """Run the full pipeline: cluster -> retrieve -> generate-ideal -> split -> per-group assign -> route.

    ``generate`` (stage 1), ``split`` (stage 2), and ``classify`` (stage 3, per-group assign) each map
    prompt records to ``{id: response}``. At each ``None`` boundary the result carries the prepared
    prompts to export for the Batch API: ``generate=None`` -> ``ideal_prompts``; ``generate`` set but
    ``split=None`` -> ``split_prompts``; ``split`` set but ``classify=None`` -> ``group_assign_prompts``.
    With all three set the LLM runs inline and ``records`` are populated. ``specgen`` (stage 4) then
    generates categorical transform specs for adopt/refine records; when set, recodes are attached to
    ``records[].transforms``; either way ``specgen_prompts`` is exposed for the Batch API path.
    ``gencde`` (opt-in) mirrors this for the tail: it synthesizes a :class:`~.models.GenCDE` for each
    ``novel`` record (attached to ``records[].gencde``) so the residual has a harmonization target;
    ``gencde_prompts`` is always exposed for the Batch API path.
    ``retrieval_floor`` downgrades far-cosine adopt/refine to novel (see :func:`assemble_leanb`).

    ``substrate``: pass a frozen :class:`~ddharmon.harmonization.substrate.ClusteringSubstrate` to SKIP
    the non-reproducible UMAP+HDBSCAN clustering and reload that exact partition (deterministic embeddings
    + field refs via ``collect_inputs``). The clustering otherwise runs fresh; either way the substrate the
    run used is returned on ``LeanBResult.substrate`` (save it to replay later — see :mod:`.substrate`).

    The M2/M3/M4/M5/M10 quality mods below are ON by default — they were validated together on a held-out
    5-cohort A/B (fields reaching a record
    62->98%, real-concept grouping 29.5->60.7%, assignment 25->42%). Each can be turned OFF individually via
    its argument for ablation or a lean run.

    M2 grouping (default on): ``chunk_cap`` chunks oversized clusters into coherence-aware sub-units
    ``<= chunk_cap`` so the split LLM sees every member (defaults to ``max_show``; set to ``None`` to disable);
    ``chunk_skip_enumerated`` keeps a detected enumerated-entity family whole instead of chunking it;
    ``max_show`` caps the members shown per split call; ``merge`` (a callable, like ``split``/``classify``)
    runs the cross-record merge that reunites same-concept records after split+chunking, before stage 4;
    ``coherence_gate`` runs the M3 NONE-fraction gate over the assembled specs (set ``False`` to disable).

    Matching quality (default on): ``clean_cde_text`` (M5) applies index hygiene to the CDE candidate pool
    (drops boilerplate + opaque lead codes); ``representation_refine`` (M4) appends the representation-mismatch
    clause to the split + per-group assign prompts so a same-concept candidate in a different encoding routes
    to refine, not novel; ``adopt_floor`` (M5) demotes a weak-support adopt to refine (set ``None`` to disable).
    ``measurand_split`` (M11, OPT-IN / default off, pending an A/B): appends a measurand-axis clause to the
    generate-ideal + split prompts so distinct quantities sharing one object/encounter (systolic vs diastolic
    BP vs pulse) are partitioned into separate groups instead of fused into one over-merged concept.
    ``gencde_specgen`` (M12, OPT-IN / default off): generate C1 categorical member->GenCDE recodes for
    ``novel`` records that synthesized a categorical GenCDE, so the tail carries transform specs like the
    adopt/refine path (prompts join ``specgen_prompts``; assembled when ``specgen`` is set). Gate on group
    coherence — a recode into an incoherent GenCDE is meaningless.
    ``refine_cdes`` (OPT-IN / default off): give the ``refine`` bucket a harmonization target of its own —
    a GenCDE DERIVED from the matched CDE (parent + a typed, minimal delta), attached to
    ``records[].gencde`` with ``parent_cde_id`` set, so a ``refine`` stops being an ``adopt`` with a caveat.
    The ``refine`` callable (like ``gencde``) authors the deltas a rule cannot derive; the unit and
    structural deltas are computed deterministically either way, and ``refine_prompts`` is always exposed
    for the Batch path. Runs LAST — the triage gate reads the M7 concept-match flag and the coherence
    verdict, so a match those stages already doubt is never dressed up as a refinement — then re-points the
    transform specs at the refined element and mechanically closes the recodes the parent could not express.
    NOTE (frozen-substrate cache): these change prompt/candidate TEXT, not the content-addressed prompt ids —
    a replay on a frozen substrate reuses cached split/assign responses and will NOT reflect them; delete the
    affected stages' ``responses_*.jsonl`` to force a re-run.

    ``recover_outliers`` (M10, default on): re-cluster the substrate's HDBSCAN outliers in isolation
    (:func:`recover_outlier_clusters`, ``residual_min_cluster_size``) to recover the coherent sub-threshold
    families global clustering dropped as noise, appending them as extra clusters that flow through the
    normal pipeline. The recovered partition is folded into the returned substrate (the residual re-cluster
    uses UMAP, not bit-reproducible, so it must be frozen for an exact replay). No-op when there are no
    outliers; set ``False`` to disable.

    ``max_clusters`` (default ``None`` = no cap) is the CLI's cost cap: keep only the largest
    ``max_clusters`` split units so a bounded run harmonizes the highest-coverage concepts first.

    ``stop_after`` (default ``None`` = run to completion): stop VOLUNTARILY at a named stage boundary and
    return the partial :class:`LeanBResult` produced so far. It reads as "this stage COMPLETED — stop
    before the next one", and it is a SEPARATE mechanism from the ``None``-callable early returns above:
    those fire because a stage's callable is unset, so they hand back that stage's *prepared prompts* and
    are named for the stage that has NOT run. Here the callable IS set, the stage HAS run, and its output
    is on the result.

    Exactly one boundary name is accepted (:data:`STOP_AFTER_BOUNDARIES`); anything else raises
    ``ValueError`` before a single stage runs, because a silently-ignored typo would run the entire paid
    pipeline past the stop the caller asked for. The vocabulary is deliberately minimal — a string
    parameter can learn another accepted value without breaking any caller, so names are added when a
    consumer needs one, not in advance.

    - ``"gencde"`` — after ``classify`` / ``assemble`` / ``merge`` / ``gencde``, before the transform-spec
      stage. This is the stop where a reviewer commits to the assignments before ``specgen`` is paid for.

    The staged review flow's other pause points need no name here: the pause before assign is the shipped
    ``classify=None`` early return (with the coherence judge's verdict pass already run, so the groups
    carry their verdicts), and the pause after the last paid stage needs no boundary at all because the
    pipeline has simply finished. ``generate=None`` now serves a $0 preview run rather than a review pause.

    The returned partial result is a RESUME INPUT, not a preview: like every early return it carries
    ``substrate``, so a resumed run replays the identical clustering partition instead of re-clustering
    non-deterministically and stranding the decisions the reviewer already made against the old one.
    """
    from ddharmon.clustering.topic_engine import collect_inputs, topic_model_dictionaries

    if stop_after is not None and stop_after not in STOP_AFTER_BOUNDARIES:
        raise ValueError(
            f"stop_after={stop_after!r} is not a recognised stage boundary. "
            f"Accepted: {', '.join(repr(b) for b in STOP_AFTER_BOUNDARIES)}, or None to run to completion."
        )

    if substrate is None:
        tm = topic_model_dictionaries(embedded_dicts, min_cluster_size=min_cluster_size)
        clusters, embeddings, field_refs = tm.clusters, tm.embeddings, tm.field_refs
        substrate = build_substrate(
            clusters, min_cluster_size=min_cluster_size, outlier=tm.outlier_cluster, n_fields=len(field_refs)
        )
    else:  # replay: reload the frozen partition instead of re-clustering (cache hits downstream)
        _docs, embeddings, field_refs, _cohorts = collect_inputs(embedded_dicts)
        clusters = clusters_from_substrate(substrate, field_refs)

    # M10 (opt-in): recover the substrate's HDBSCAN outliers as extra clusters (sub-threshold families that
    # global clustering dropped). Runs before chunking so recovered giants are chunked too; folds the recovered
    # partition into `substrate` so a later replay reproduces it (the residual re-cluster is non-reproducible).
    if recover_outliers:
        clusters, substrate = recover_outlier_clusters(
            clusters, substrate, embeddings, field_refs, min_cluster_size=residual_min_cluster_size
        )

    # M2 (opt-in): chunk oversized clusters into coherence-aware sub-units <= chunk_cap so the split LLM sees
    # every member (the substrate keeps the ORIGINAL partition — chunking is a deterministic, cache-safe
    # function of the frozen members + cached embeddings, applied after substrate capture). Off by default.
    if chunk_cap:
        from ddharmon.harmonization.chunk import chunk_oversized

        clusters = chunk_oversized(
            clusters, embeddings, field_refs, cap=chunk_cap, skip_enumerated=chunk_skip_enumerated
        )

    cde_dict = _find_cde_dict(embedded_dicts, cde_cohort)
    ideal_prompts = prepare_leanb(
        clusters,
        embedded_dicts,
        embeddings,
        field_refs,
        cde_cohort=cde_cohort,
        cde_dict=cde_dict,
        top_k=top_k,
        model_tag=model_tag,
        clean_cde_text=clean_cde_text,
        measurand_split=measurand_split,
    )
    # Cost cap (used by the CLI): keep only the largest ``max_clusters`` units (most members first) so a
    # bounded run harmonizes the highest-coverage concepts first. Applied after chunk/recover so it caps the
    # actual split units. ``None`` = no cap.
    if max_clusters is not None:
        ideal_prompts = sorted(ideal_prompts, key=lambda r: len(r.context["members"]), reverse=True)[:max_clusters]
    if generate is None:
        return LeanBResult(ideal_prompts=ideal_prompts, substrate=substrate)

    split_prompts = prepare_split(
        ideal_prompts,
        generate(ideal_prompts),
        model_tag=model_tag,
        max_show=max_show,
        representation_refine=representation_refine,
        measurand_split=measurand_split,
    )
    if split is None:
        return LeanBResult(split_prompts=split_prompts, substrate=substrate)

    group_assign_prompts = prepare_group_assign(
        split_prompts,
        split(split_prompts),
        embedded_dicts,
        embeddings,
        field_refs,
        cde_cohort=cde_cohort,
        cde_dict=cde_dict,
        top_k=top_k,
        model_tag=model_tag,
        clean_cde_text=clean_cde_text,
        representation_refine=representation_refine,
    )
    # Step-2 dual-sample coherence judge, VERDICT pass — post-split, PRE-assign, read-only. Flags
    # over-merged/incoherent GROUPS (the BP systolic+diastolic+pulse fusion, the arthritis semantic
    # umbrella) for human re-adjudication — NEVER auto-split. It runs
    # HERE, at the judge's native post-split concept-group granularity and before anything has been
    # assigned, because a caller that pauses at the `classify=None` boundary below must be able to render
    # each group's coherence state; a verdict stamped after `classify` does not exist when that caller
    # needs it. Nothing in this pass depends on assign-stage output. Prompts are always prepared ($0
    # dual-sampling; the $0 deterministic matrix pre-filter is stamped on ConceptGroup.matrix_suspect too);
    # the LLM judge applies only when `coherence` is set. `propagate_coherence_review` lands the resulting
    # flag on the recodes / GenCDE much later, once those artifacts exist.
    concept_groups = concept_groups_from_prompts(group_assign_prompts)
    coherence_prompts = prepare_coherence(concept_groups, embedded_dicts, embeddings, field_refs, model_tag=model_tag)
    if coherence is not None and coherence_prompts:
        assemble_coherence_verdicts(coherence_prompts, coherence(coherence_prompts), concept_groups)

    if classify is None:
        return LeanBResult(
            group_assign_prompts=group_assign_prompts,
            concept_groups=concept_groups,
            coherence_prompts=coherence_prompts,
            substrate=substrate,
        )

    result = assemble_leanb(
        group_assign_prompts, classify(group_assign_prompts), retrieval_floor=retrieval_floor, adopt_floor=adopt_floor
    )
    result.substrate = substrate
    result.concept_groups = concept_groups
    result.coherence_prompts = coherence_prompts
    # Carry the pre-assign verdicts onto the records the assign stage just built. Without this the
    # verdicts stay stranded on the groups and every record reads as UNJUDGED — the "not judged rendered
    # as coherent" failure the judge's contract forbids. Matched on group id (cluster id as fallback), so
    # it is order-independent and a one-identifier group is not dropped.
    transfer_coherence_verdicts(concept_groups, result.records)

    # M2 cross-record merge (opt-in): reunite same-concept records that split + chunking left separate.
    # Runs BEFORE stage 4 so specs are generated for the final grouping. merge_prompts is always exposed
    # (deterministic candidate gen) for the Batch/driver path; the merge only applies when `merge` is set.
    from ddharmon.harmonization.merge import assemble_merge, prepare_merge

    result.merge_prompts = prepare_merge(result.records, embedded_dicts, embeddings, field_refs, model_tag=model_tag)
    if merge is not None and result.merge_prompts:
        result.records = assemble_merge(result.records, result.merge_prompts, merge(result.merge_prompts))

    # GenCDE synthesis (opt-in): author a generated CDE for each `novel` record so the tail has a
    # harmonization TARGET (novels reached no existing CDE — otherwise the verdict is a dead end). Runs
    # AFTER merge so reunited same-concept novels synthesize once. gencde_prompts is always exposed for the
    # Batch/driver path; the `gencde` callable applies it inline when set. This inverts FAIRkit's
    # generate-from-template: the GenCDE is synthesized from the group's POOLED cross-cohort evidence.
    from ddharmon.harmonization.gencde import assemble_gencde, prepare_gencde

    result.gencde_prompts = prepare_gencde(result.records, embedded_dicts, model_tag=model_tag)
    if gencde is not None and result.gencde_prompts:
        result.records = assemble_gencde(result.gencde_prompts, gencde(result.gencde_prompts), result.records)

    # `stop_after="gencde"` — the one named VOLUNTARY boundary. The assignments (and their GenCDEs) are
    # decided and on `result`; the transform-spec stage below has not been paid for. `result.substrate` was
    # set right after assemble_leanb, so this partial is a resume input: a continuation replays the exact
    # same partition rather than re-clustering and stranding the decisions made against the old one.
    if stop_after == "gencde":
        return result

    # stage 4: transform-spec generation (verifying post-pass over adopt/refine records).
    # Local import avoids a module-level cycle (transform imports leanb). N1 unit specs are deterministic
    # (no LLM) and always run; categorical (C1) recodes and N2 arithmetic formulas need the Batch LLM —
    # specgen=None leaves them unattached but still exposes specgen_prompts for the Batch API path.
    from ddharmon.harmonization.transform import (
        apply_coherence_gate,
        assemble_arith_specgen,
        assemble_concept_gate,
        assemble_gencde_arith_specgen,
        assemble_gencde_specgen,
        assemble_specgen,
        generate_gencde_unit_specs,
        generate_unit_specs,
        generate_wide_to_long_specs,
        prepare_arith_specgen,
        prepare_concept_gate,
        prepare_gencde_arith_specgen,
        prepare_gencde_specgen,
        prepare_specgen,
    )

    cde_fields = dict(cde_dict.dictionary.fields) if cde_dict is not None else {}
    # deterministic ($0) pre-passes: wide->long first (claims repeating-measure records so N1/C1 skip them),
    # then N1 unit specs (leaves needs_units residuals for the N2 LLM path).
    generate_wide_to_long_specs(result.records, embedded_dicts, cde_fields)
    generate_unit_specs(result.records, embedded_dicts, cde_fields)  # N1 (deterministic) — leaves residuals
    cat_prompts = prepare_specgen(result.records, embedded_dicts, cde_fields, model_tag=model_tag)  # C1
    arith_prompts = prepare_arith_specgen(result.records, embedded_dicts, cde_fields, model_tag=model_tag)  # N2
    # GenCDE tail (opt-in, M12): member->GenCDE recodes for novel records so the novel path carries transform
    # specs too (not just the adopt/refine path). Categorical GenCDEs get C1-style value recodes; NUMERIC
    # GenCDEs get N1 deterministic unit conversions + N2 arithmetic formulas (mirrors the CDE N1/N2 path, but
    # targeting the synthesized GenCDE's units/bounds). Gated on gencde_specgen (default OFF) — a GenCDE recode
    # is only meaningful once its group is coherent (see the over-merge / granularity work).
    gencde_cat_prompts: list[PromptRecord] = []
    gencde_arith_prompts: list[PromptRecord] = []
    if gencde_specgen:
        gencde_cat_prompts = prepare_gencde_specgen(result.records, embedded_dicts, model_tag=model_tag)  # C1
        generate_gencde_unit_specs(result.records, embedded_dicts)  # N1 (deterministic) — leaves residuals
        gencde_arith_prompts = prepare_gencde_arith_specgen(result.records, embedded_dicts, model_tag=model_tag)  # N2
    result.specgen_prompts = cat_prompts + arith_prompts + gencde_cat_prompts + gencde_arith_prompts
    if specgen is not None and result.specgen_prompts:
        responses = specgen(result.specgen_prompts)
        assemble_specgen(cat_prompts, responses, result.records)
        assemble_arith_specgen(arith_prompts, responses, result.records)
        if gencde_cat_prompts:
            assemble_gencde_specgen(gencde_cat_prompts, responses, result.records)
        if gencde_arith_prompts:
            assemble_gencde_arith_specgen(gencde_arith_prompts, responses, result.records)
        # M3 (opt-in): flag/demote records whose coded edges are mostly unmappable (over-broad matches).
        # Runs only when specs were assembled inline; the Batch/driver path calls apply_coherence_gate itself.
        if coherence_gate:
            apply_coherence_gate(result.records)

    # M7 concept-match gate (opt-in LLM stage): flag adopt/refine records whose assigned CDE fails a
    # same-concept check (the full-coverage-wrong-concept failure M3 can't see). Prompts are always exposed
    # for the Batch/driver path; the gate applies only when `concept_gate` is set. Runs after specs so the
    # flag can set needs_review on the record's recodes.
    result.concept_gate_prompts = prepare_concept_gate(result.records, embedded_dicts, cde_fields, model_tag=model_tag)
    if concept_gate is not None and result.concept_gate_prompts:
        assemble_concept_gate(result.concept_gate_prompts, concept_gate(result.concept_gate_prompts), result.records)

    # R2 (opt-in, `distinct_kinds`): second read over `qualify` groups → upgrades the R0 split-only flag to
    # R2 (also flag qualify∧distinct_kinds). It reads the coherence judge's OWN outputs off LeanBRecord, so
    # it cannot move early with the verdict pass — it stays here, after assemble_leanb has built the records
    # and transfer_coherence_verdicts has carried the verdicts onto them. The Batch/driver path calls
    # prepare_kinds/assemble_kinds itself; kinds_prompts is always exposed for it. Default
    # (distinct_kinds=None) leaves the shipped R0 behavior unchanged.
    if coherence is not None and result.coherence_prompts:
        result.kinds_prompts = prepare_kinds(result.records)
        if distinct_kinds is not None and result.kinds_prompts:
            assemble_kinds(result.kinds_prompts, distinct_kinds(result.kinds_prompts), result.records)

    # Step-2 coherence judge, PROPAGATION pass — the genuinely post-assign half. `needs_review` lands on
    # `rec.transforms` and `rec.gencde`, and neither existed when the verdict pass ran before `classify`.
    # UNCONDITIONAL and idempotent: it acts only on records already flagged `incoherent`, so it is a no-op
    # when the judge never ran, and it still fires for a resumed run whose verdicts came from persisted
    # state rather than from a `coherence` callable in this process.
    propagate_coherence_review(result.records)

    # Refinement authoring (opt-in, `refine_cdes`): give the `refine` bucket a real harmonization TARGET —
    # a GenCDE DERIVED from the matched CDE (parent + a typed, minimal delta) — mirroring what `gencde`
    # does for `novel`. Runs LAST on purpose: the axis triage and its mis-assigned gate read the M7
    # concept-match flag and the coherence judge's verdict, both of which are only set above, and a
    # refinement must never be authored for a match those stages already doubt.
    #
    # Three sub-steps: the $0 deterministic deltas (unit/structural) are attached first so the LLM is
    # never paid for an answer a rule can derive; `refine_prompts` is then always exposed for the
    # Batch/driver path; finally `retarget_refined_specs` repoints the transform specs at the refined
    # element and mechanically closes the recodes the parent's value domain could not express.
    if refine_cdes:
        from ddharmon.harmonization.refine import (
            apply_deterministic_refinements,
            assemble_refine,
            prepare_refine,
            retarget_refined_specs,
        )

        apply_deterministic_refinements(result.records, cde_fields)
        result.refine_prompts = prepare_refine(result.records, embedded_dicts, cde_fields, model_tag=model_tag)
        if refine is not None and result.refine_prompts:
            assemble_refine(result.refine_prompts, refine(result.refine_prompts), result.records, cde_fields)
        retarget_refined_specs(result.records, embedded_dicts)
    return result


# ── parse helpers ───────────────────────────────────────────────


def _numbered_candidate_block(candidates: list[dict]) -> str:
    """The ``[N] (cos=…) text`` block shown to the split LLM."""
    return "\n".join(f"  [{i + 1}] (cos={c['cos']:.3f}) {c['text']}" for i, c in enumerate(candidates))


def _parse_ideal(resp: object) -> str:
    if resp is None:
        return ""
    try:
        payload = extract_json(resp if isinstance(resp, str) else json.dumps(resp))
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    return str(payload.get("ideal_cde", "")) if isinstance(payload, dict) else ""


def _parse_split_groups(resp: object) -> list[dict]:
    """Parse a split response into a list of group dicts (each with ``member_ids``/``concept``/…).

    Returns ``[]`` on a missing/unparseable response or one with no usable ``groups`` list.
    """
    if resp is None:
        return []
    try:
        payload = extract_json(resp if isinstance(resp, str) else json.dumps(resp))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list):
        # Tolerant parse: the split Batch schema is SOFT (appended as text, not enforced), so on ~35% of
        # clusters the model drops the ``{"groups": [...]}`` wrapper and returns a single bare group object
        # with ``member_ids`` at the top level. Recover it instead of discarding the split, which silently
        # collapses the cluster to one un-split group and inflates 'novel'. (See leanb-stages-batch-schema-is-soft;
        # mirrors _parse_specgen.) Residual completion in prepare_group_assign then catches the members this
        # single bare group omitted.
        if isinstance(payload.get("member_ids"), list):
            raw_groups = [payload]
        else:
            return []
    groups: list[dict] = []
    for g in raw_groups:
        if not isinstance(g, dict):
            continue
        member_ids = [str(m) for m in g.get("member_ids", []) if isinstance(g.get("member_ids"), list)]
        groups.append(
            {
                "member_ids": member_ids,
                "concept": str(g.get("concept", "")),
                "verdict": str(g.get("verdict", "")),
                "raw": g,
            }
        )
    return groups


def _group_members(
    member_ids: list[str], by_id: dict[str, dict], all_members: list[dict], *, fallback: bool
) -> list[dict]:
    """Map a group's ``member_ids`` (m1, m2, …) back to the carried member dicts.

    Unknown ids are dropped. When the partition resolves to nothing and ``fallback`` is set (a single
    group), use all members rather than drop the cluster.
    """
    resolved: list[dict] = []
    seen: set[str] = set()
    for mid in member_ids:
        key = "m" + re.sub(r"\D", "", str(mid))
        mem = by_id.get(key) or by_id.get(str(mid))
        if mem is not None and mem["member_id"] not in seen:
            resolved.append(mem)
            seen.add(mem["member_id"])
    if not resolved and fallback:
        return list(all_members)
    return resolved


def _parse_assign(resp: object) -> dict | None:
    if resp is None:
        return None
    try:
        payload = extract_json(resp if isinstance(resp, str) else json.dumps(resp))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) and "verdict" in payload else None


def _candidate_index(raw: object) -> int | None:
    """The LLM returns the chosen candidate's 1-based NUMBER; map to a 0-based index."""
    m = re.search(r"\d+", str(raw if raw is not None else ""))
    return int(m.group()) - 1 if m else None


def _parse_ranking(raw: object, n: int) -> list[int]:
    out: list[int] = []
    for x in raw if isinstance(raw, list) else []:
        m = re.search(r"\d+", str(x))
        if m:
            i = int(m.group()) - 1
            if 0 <= i < n and i not in out:
                out.append(i)
    return out


def _find_cde_dict(embedded_dicts: list[EmbeddedDictionary], cde_cohort: str) -> EmbeddedDictionary:
    for ed in embedded_dicts:
        dd = getattr(ed, "dictionary", None)
        name = getattr(dd, "cohort_name", None) or getattr(dd, "name", None)
        if name == cde_cohort:
            return ed
    raise ValueError(f"no embedded dictionary named {cde_cohort!r} among {len(embedded_dicts)} dicts")


# ── export helpers ──────────────────────────────────────────────


def write_prompts_jsonl(prompt_records: list[PromptRecord], path: str | Path) -> int:
    """Write prompt records as JSONL for the Batch API workflow. Returns count."""
    path = Path(path)
    with open(path, "w") as f:
        for rec in prompt_records:
            f.write(json.dumps(rec.to_jsonl_record()) + "\n")
    return len(prompt_records)


def export_leanb_eitl_queue(result: LeanBResult, path: str | Path) -> int:
    """Write a TSV review queue for expert-in-the-loop verification. Returns row count.

    One row per concept-GROUP record, ordered so the decisions most needing review (refine, then novel,
    then adopt) surface first; within a verdict, lower nearest-cosine first.
    """
    path = Path(path)
    cols = [
        "cluster_id",
        "group_id",
        "concept",
        "verdict",
        "route",
        "cde_id",
        "cde_external_id",
        "top1_cos",
        "chosen_cos",
        "coverage_gap",
        "floored",
        "coherence_gap",
        "incoherent",
        "coherence_verdict",
        "coherence_axis",
        "matrix_suspect",
        "cross_cohort",
        "n_members",
        "cohorts",
        "members",
        "ideal_cde",
        "gencde_name",
        "gencde_definition",
        "gencde_data_type",
        "gencde_permissible_values",
        "gencde_units",
        "gencde_value_coverage",
        "gencde_needs_review",
        "rationale",
    ]
    rank = {"refine": 0, "novel": 1, "adopt": 2, "": 3}
    rows = sorted(result.records, key=lambda r: (rank.get(r.verdict, 3), r.top1_cos if r.top1_cos is not None else 0.0))

    def clean(s: str) -> str:
        return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")

    def gencde_cells(g: GenCDE | None) -> list[str]:
        """The synthesized-target cells for a novel (empty when the record has no GenCDE)."""
        if g is None:
            return ["", "", "", "", "", "", ""]
        pv = ";".join(f"{o.code}={o.label}" if o.code else o.label for o in g.permissible_values)
        return [
            clean(g.preferred_name),
            clean(g.definition),
            g.data_type,
            clean(pv),
            clean(g.units or ""),
            "n/a" if g.value_coverage is None else f"{g.value_coverage:.2f}",
            str(g.needs_review),
        ]

    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write(
                "\t".join(
                    [
                        r.cluster_id,
                        r.group_id,
                        clean(r.concept),
                        r.verdict,
                        r.route,
                        r.cde_id or "",
                        r.cde_external_id or "",
                        "" if r.top1_cos is None else f"{r.top1_cos:.3f}",
                        "" if r.chosen_cos is None else f"{r.chosen_cos:.3f}",
                        str(r.coverage_gap),
                        str(r.floored),
                        str(r.coherence_gap),
                        str(r.incoherent),
                        r.coherence_verdict,
                        clean(r.coherence_axis),
                        str(r.matrix_suspect),
                        str(r.cross_cohort),
                        str(r.n_members),
                        ";".join(r.cohorts),
                        clean(";".join(r.member_variable_names)),
                        clean(r.ideal_cde),
                        *gencde_cells(r.gencde),
                        clean(r.rationale),
                    ]
                )
                + "\n"
            )
    return len(rows)


def write_records_json(result: LeanBResult, path: str | Path) -> int:
    """Write all decision records as a JSON array. Returns count."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in result.records], f, indent=2)
    return len(result.records)
