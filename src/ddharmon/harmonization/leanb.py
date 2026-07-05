"""Lean head/tail CDE harmonization pipeline (split-aware, 3-stage) — the default pipeline.

Supersedes the sub-cluster-anchored pipeline (``harmonize_dictionaries``). Where that approach anchored
each value sub-cluster to the most-central in-cluster CDE, this one leads with **assignment to the given
CDE backbone** for the covered head and routes the uncovered tail to GenCDE/clustering — the division of
labor that the research harness settled. A semantic cluster is grouped by an embedding that ignores the
variable name, so one cluster can pool MORE THAN ONE distinct concept; the pipeline is therefore
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
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ddharmon.embedding.service import EmbeddedDictionary
from ddharmon.harmonization.anchor import CDE_COHORT, build_field_lookup
from ddharmon.harmonization.leanb_prompts import (
    ASSIGN_SCHEMA,
    IDEAL_SCHEMA,
    SPLIT_SCHEMA,
    SYS_GENERATE_IDEAL,
    build_group_assign_user_prompt,
    build_ideal_user_prompt,
    build_split_user_prompt,
    group_reassign_system_prompt,
    split_system_prompt,
)
from ddharmon.harmonization.models import CandidateCDE, LeanBRecord
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

logger = logging.getLogger(__name__)

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
_SAMPLE_MEMBERS = 5  # members in the generate-ideal / per-group sample line
_CAND_TRUNC = 170
_MEMBER_TRUNC = 200  # retrieval member text (lean, concept-only — value codes are noise in BM25/dense)
_VALUE_ENC_TRUNC = 240  # cap the source value-set string fed into a prompt (CLSA country lists etc. are long)
_MEMBER_PROMPT_TRUNC = 420  # prompt-side member text: concept + symbolic value metadata (richer than retrieval)

# M5 index hygiene (opt-in): generic survey/CDE instruction boilerplate that carries no concept signal but
# pollutes BM25 (spurious lexical hits) and the candidate block shown to the assign LLM. These are UNIVERSAL
# data-collection artifacts (interviewer/skip-logic/multi-select instructions), NOT cohort-specific — the
# audit's worst case was ethnicity matching a CDE whose only distinctive text was "READ IF NECESSARY" @0.45.
# Matched case-insensitively as whole phrases; extend generically, do not add cohort-specific terms.
CDE_TEXT_BOILERPLATE = (
    "read if necessary",
    "do not read",
    "read out",
    "read all that apply",
    "select all that apply",
    "check all that apply",
    "mark all that apply",
    "choose all that apply",
    "select one",
    "if necessary",
    "for office use only",
    "office use only",
    "see instructions",
    "please specify",
)
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
    codes are noise in the geometric/lexical space (Run 016/048). For the prompt-side, value-aware
    rendering see :func:`_member_prompt_text`.
    """
    return _aug_text(ref.variable_name, _base_member_text(fld, ref))


def _value_set_text(fld: Field) -> str:
    """Readable source value set for a prompt: the raw encoding, else reconstructed from parsed options."""
    raw = (fld.value_encoding_raw or "").strip()
    if raw:
        return raw
    if fld.response_options:
        return "|".join(f"{ro.code}={ro.label}" for ro in fld.response_options)
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
    values = _value_set_text(fld)
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
    """

    records: list[LeanBRecord] = field(default_factory=list)
    ideal_prompts: list[PromptRecord] = field(default_factory=list)
    split_prompts: list[PromptRecord] = field(default_factory=list)
    group_assign_prompts: list[PromptRecord] = field(default_factory=list)
    merge_prompts: list[PromptRecord] = field(default_factory=list)  # M2 cross-record merge — Batch export
    specgen_prompts: list[PromptRecord] = field(default_factory=list)  # stage 4 (transform specs) — Batch export
    concept_gate_prompts: list[PromptRecord] = field(default_factory=list)  # M7 concept-match gate — Batch export
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
) -> list[PromptRecord]:
    """Retrieve candidates and build the stage-1 generate-ideal prompts (one per non-empty cluster).

    Each returned record's ``context`` carries (a) the FULL ordered non-CDE member list — each member as
    ``{member_id, dictionary_name, variable_name, text, row}`` with the Gap-1 augmented text and its
    embedding-row index — and (b) the cluster-level retrieved candidates + metadata, so
    :func:`prepare_split` can build the split prompt without re-retrieving, and
    :func:`prepare_group_assign` can map split groups back to members and re-retrieve per group.

    ``clean_cde_text`` (M5) applies :func:`_clean_cde_text` index hygiene to the CDE candidate pool.
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
                system_prompt=SYS_GENERATE_IDEAL,
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
) -> list[PromptRecord]:
    """Build the stage-2 split-assign prompts from the generated ideals + the carried members/candidates.

    The split prompt shows the cluster members (up to ``max_show``), each prefixed with its ``[mK]`` id,
    and asks the model to PARTITION them into distinct-concept groups + decide each. With M2 chunking on,
    every unit is ``<= max_show`` so no member is truncated before the LLM.

    ``representation_refine`` (M4) appends the representation-mismatch clause so a same-concept candidate in a
    different encoding (banding/flag/composite/unit) routes to refine, not novel.
    """
    sys_prompt = split_system_prompt(representation_refine)
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
) -> list[PromptRecord]:
    """Parse the split groups and build one per-group, re-retrieved single-concept assign prompt.

    For each parsed group: map ``member_ids`` (m1, m2, …) back to the cluster's carried members,
    RE-RETRIEVE a per-group hybrid top-k (the group-member centroid ⊕ BM25 over the group's own member
    text), and build a per-group single-concept assign prompt. The prompt's ``context`` carries everything
    :func:`assemble_leanb` needs for the group's record (the per-group candidates, cohorts, members, …).

    ``clean_cde_text`` (M5) cleans the per-group candidate pool; ``representation_refine`` (M4) appends the
    representation-mismatch clause to the per-group assign prompt.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    backbone = _build_backbone(embedded_dicts, field_lookup, cde_cohort, cde_dict, clean_text=clean_cde_text)
    sys_prompt = group_reassign_system_prompt(representation_refine)
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
    recover_outliers: bool = True,
    residual_min_cluster_size: int = 8,
    max_clusters: int | None = None,
) -> LeanBResult:
    """Run the full pipeline: cluster -> retrieve -> generate-ideal -> split -> per-group assign -> route.

    ``generate`` (stage 1), ``split`` (stage 2), and ``classify`` (stage 3, per-group assign) each map
    prompt records to ``{id: response}``. At each ``None`` boundary the result carries the prepared
    prompts to export for the Batch API: ``generate=None`` -> ``ideal_prompts``; ``generate`` set but
    ``split=None`` -> ``split_prompts``; ``split`` set but ``classify=None`` -> ``group_assign_prompts``.
    With all three set the LLM runs inline and ``records`` are populated. ``specgen`` (stage 4) then
    generates categorical transform specs for adopt/refine records; when set, recodes are attached to
    ``records[].transforms``; either way ``specgen_prompts`` is exposed for the Batch API path.
    ``retrieval_floor`` downgrades far-cosine adopt/refine to novel (see :func:`assemble_leanb`).

    ``substrate``: pass a frozen :class:`~ddharmon.harmonization.substrate.ClusteringSubstrate` to SKIP
    the non-reproducible UMAP+HDBSCAN clustering and reload that exact partition (deterministic embeddings
    + field refs via ``collect_inputs``). The clustering otherwise runs fresh; either way the substrate the
    run used is returned on ``LeanBResult.substrate`` (save it to replay later — see :mod:`.substrate`).

    The M2/M3/M4/M5/M10 quality mods below are ON by default — they were validated together on a held-out
    full-5 A/B (2026-07-04, .planning/experiments/full5-stack-run-2026-07-04.md: fields reaching a record
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
    NOTE (frozen-substrate cache): these change prompt/candidate TEXT, not the content-addressed prompt ids —
    a replay on a frozen substrate reuses cached split/assign responses and will NOT reflect them; delete the
    affected stages' ``responses_*.jsonl`` to force a re-run.

    ``recover_outliers`` (M10, default on): re-cluster the substrate's HDBSCAN outliers in isolation
    (:func:`recover_outlier_clusters`, ``residual_min_cluster_size``) to recover the coherent sub-threshold
    families global clustering dropped as noise, appending them as extra clusters that flow through the
    normal pipeline. The recovered partition is folded into the returned substrate (the residual re-cluster
    uses UMAP, not bit-reproducible, so it must be frozen for an exact replay). No-op when there are no
    outliers; set ``False`` to disable.
    """
    from ddharmon.clustering.topic_engine import collect_inputs, topic_model_dictionaries

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
    if classify is None:
        return LeanBResult(group_assign_prompts=group_assign_prompts, substrate=substrate)

    result = assemble_leanb(
        group_assign_prompts, classify(group_assign_prompts), retrieval_floor=retrieval_floor, adopt_floor=adopt_floor
    )
    result.substrate = substrate

    # M2 cross-record merge (opt-in): reunite same-concept records that split + chunking left separate.
    # Runs BEFORE stage 4 so specs are generated for the final grouping. merge_prompts is always exposed
    # (deterministic candidate gen) for the Batch/driver path; the merge only applies when `merge` is set.
    from ddharmon.harmonization.merge import assemble_merge, prepare_merge

    result.merge_prompts = prepare_merge(result.records, embedded_dicts, embeddings, field_refs, model_tag=model_tag)
    if merge is not None and result.merge_prompts:
        result.records = assemble_merge(result.records, result.merge_prompts, merge(result.merge_prompts))

    # stage 4: transform-spec generation (verifying post-pass over adopt/refine records).
    # Local import avoids a module-level cycle (transform imports leanb). N1 unit specs are deterministic
    # (no LLM) and always run; categorical (C1) recodes and N2 arithmetic formulas need the Batch LLM —
    # specgen=None leaves them unattached but still exposes specgen_prompts for the Batch API path.
    from ddharmon.harmonization.transform import (
        apply_coherence_gate,
        assemble_arith_specgen,
        assemble_concept_gate,
        assemble_specgen,
        generate_unit_specs,
        generate_wide_to_long_specs,
        prepare_arith_specgen,
        prepare_concept_gate,
        prepare_specgen,
    )

    cde_fields = dict(cde_dict.dictionary.fields) if cde_dict is not None else {}
    # deterministic ($0) pre-passes: wide->long first (claims repeating-measure records so N1/C1 skip them),
    # then N1 unit specs (leaves needs_units residuals for the N2 LLM path).
    generate_wide_to_long_specs(result.records, embedded_dicts, cde_fields)
    generate_unit_specs(result.records, embedded_dicts, cde_fields)  # N1 (deterministic) — leaves residuals
    cat_prompts = prepare_specgen(result.records, embedded_dicts, cde_fields, model_tag=model_tag)  # C1
    arith_prompts = prepare_arith_specgen(result.records, embedded_dicts, cde_fields, model_tag=model_tag)  # N2
    result.specgen_prompts = cat_prompts + arith_prompts
    if specgen is not None and result.specgen_prompts:
        responses = specgen(result.specgen_prompts)
        assemble_specgen(cat_prompts, responses, result.records)
        assemble_arith_specgen(arith_prompts, responses, result.records)
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
        "cross_cohort",
        "n_members",
        "cohorts",
        "members",
        "ideal_cde",
        "rationale",
    ]
    rank = {"refine": 0, "novel": 1, "adopt": 2, "": 3}
    rows = sorted(result.records, key=lambda r: (rank.get(r.verdict, 3), r.top1_cos if r.top1_cos is not None else 0.0))

    def clean(s: str) -> str:
        return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")

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
                        str(r.cross_cohort),
                        str(r.n_members),
                        ";".join(r.cohorts),
                        clean(";".join(r.member_variable_names)),
                        clean(r.ideal_cde),
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
