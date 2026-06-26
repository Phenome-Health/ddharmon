"""v2 lean head/tail CDE harmonization pipeline (split-aware, 3-stage).

Supersedes the v1 sub-cluster-anchored pipeline (``harmonize_dictionaries``). Where v1 anchored each
value sub-cluster to the most-central in-cluster CDE, v2 leads with **assignment to the given CDE
backbone** for the covered head and routes the uncovered tail to GenCDE/clustering — the division of
labor that the research harness settled. A semantic cluster is grouped by an embedding that ignores the
variable name, so one cluster can pool MORE THAN ONE distinct concept; v2 is therefore SPLIT-AWARE and
emits one record per concept-GROUP. Per concept cluster::

    hybrid retrieve (BM25 lexical + dense centroid, RRF) top-k CDE candidates
      -> generate-ideal      (LLM, no candidates -> a qualifier-faithful coverage anchor)
      -> split-assign         (LLM, partition members into distinct-concept groups + rank+verdict each)
      -> per-group re-assign  (LLM, RE-RETRIEVE per group, then rank candidates + adopt/refine/novel)
      -> route: adopt/refine -> CDE assignment ;  novel -> GenCDE / clustering residual (tail)

Three LLM stages. As in v1 the pipeline is split so each stage is testable without an LLM and can run
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
    SYS_GROUP_REASSIGN,
    SYS_SPLIT,
    build_group_assign_user_prompt,
    build_ideal_user_prompt,
    build_split_user_prompt,
)
from ddharmon.harmonization.models import LeanBRecord
from ddharmon.harmonization.parse import extract_json
from ddharmon.harmonization.pipeline import PromptRecord
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
MAX_SHOW = 22  # members shown to the split LLM (sample for big repeating-measure clusters)
_SAMPLE_MEMBERS = 5  # members in the generate-ideal / per-group sample line
_CAND_TRUNC = 170
_MEMBER_TRUNC = 200


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
    """Gap-1 augmented member text: base text + humanized name when the name adds a qualifier."""
    return _aug_text(ref.variable_name, _base_member_text(fld, ref))


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
    def from_embedded(cls, cde_dict: EmbeddedDictionary, cde_fields: dict[str, Field]) -> CdeBackbone:
        ids = list(cde_dict.get_variable_names())
        vectors = _norm(np.asarray(cde_dict.get_all_vectors(), dtype=np.float32))
        rich = [_cde_rich_text(cde_fields[i]) if i in cde_fields else i for i in ids]
        ext = [_cde_external_id(cde_fields[i]) if i in cde_fields else "" for i in ids]
        return cls(ids=ids, vectors=vectors, rich_texts=rich, external_ids=ext, bm25=BM25(rich))


@dataclass
class LeanBResult:
    """v2 harmonization records plus the prompts that produced (or will produce) them.

    ``ideal_prompts`` are populated when the generate stage has not run inline (export for Batch);
    ``split_prompts`` when generate has run but split has not; ``group_assign_prompts`` when split has run
    but the per-group assign has not. ``records`` are the final per-group decisions.
    """

    records: list[LeanBRecord] = field(default_factory=list)
    ideal_prompts: list[PromptRecord] = field(default_factory=list)
    split_prompts: list[PromptRecord] = field(default_factory=list)
    group_assign_prompts: list[PromptRecord] = field(default_factory=list)

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
        "cluster_id": str(cluster.cluster_id),
        "cohorts": cohorts,
        "cross_cohort": len(cohorts) >= 2,
        "n_members": len(members),
    }


def _build_backbone(
    embedded_dicts: list[EmbeddedDictionary],
    field_lookup: dict[tuple[str, str], Field],
    cde_cohort: str,
    cde_dict: EmbeddedDictionary | None,
) -> CdeBackbone:
    cde_embedded = cde_dict or _find_cde_dict(embedded_dicts, cde_cohort)
    return CdeBackbone.from_embedded(cde_embedded, {k[1]: v for k, v in field_lookup.items() if k[0] == cde_cohort})


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
) -> list[PromptRecord]:
    """Retrieve candidates and build the stage-1 generate-ideal prompts (one per non-empty cluster).

    Each returned record's ``context`` carries (a) the FULL ordered non-CDE member list — each member as
    ``{member_id, dictionary_name, variable_name, text, row}`` with the Gap-1 augmented text and its
    embedding-row index — and (b) the cluster-level retrieved candidates + metadata, so
    :func:`prepare_split` can build the split prompt without re-retrieving, and
    :func:`prepare_group_assign` can map split groups back to members and re-retrieve per group.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    row_of = {(r.dictionary_name, r.variable_name): i for i, r in enumerate(field_refs)}
    backbone = _build_backbone(embedded_dicts, field_lookup, cde_cohort, cde_dict)

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
            text = _member_text(field_lookup.get(key), m)[:_MEMBER_TRUNC]
            members.append(
                {
                    "member_id": f"m{k}",
                    "dictionary_name": m.dictionary_name,
                    "variable_name": m.variable_name,
                    "text": text,
                    "row": row_of[key],
                }
            )
        if not members:
            continue
        rows = [mem["row"] for mem in members]
        member_texts = [mem["text"] for mem in members]
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
                user_prompt=build_ideal_user_prompt(member_texts[:_SAMPLE_MEMBERS]),
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
) -> list[PromptRecord]:
    """Build the stage-2 split-assign prompts from the generated ideals + the carried members/candidates.

    The split prompt shows all cluster members (up to :data:`MAX_SHOW`), each prefixed with its ``[mK]``
    id, and asks the model to PARTITION them into distinct-concept groups + decide each.
    """
    records: list[PromptRecord] = []
    for rec in ideal_records:
        ctx = rec.context
        ideal_cde = _parse_ideal(ideal_responses.get(rec.id))
        members = ctx["members"]
        numbered = [(mem["member_id"], mem["text"]) for mem in members[:MAX_SHOW]]
        cand_block = _numbered_candidate_block(ctx["candidates"])
        records.append(
            PromptRecord(
                id=f"leanb:split:{ctx['cluster_id']}",
                system_prompt=SYS_SPLIT,
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
) -> list[PromptRecord]:
    """Parse the split groups and build one per-group, re-retrieved single-concept assign prompt.

    For each parsed group: map ``member_ids`` (m1, m2, …) back to the cluster's carried members,
    RE-RETRIEVE a per-group hybrid top-k (the group-member centroid ⊕ BM25 over the group's own member
    text), and build a per-group single-concept assign prompt. The prompt's ``context`` carries everything
    :func:`assemble_leanb` needs for the group's record (the per-group candidates, cohorts, members, …).
    """
    field_lookup = build_field_lookup(embedded_dicts)
    backbone = _build_backbone(embedded_dicts, field_lookup, cde_cohort, cde_dict)
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
        for gi, group in enumerate(groups):
            grp_members = _group_members(group.get("member_ids", []), by_id, members, fallback=len(groups) == 1)
            if not grp_members:
                continue
            rows = [mem["row"] for mem in grp_members if (mem["dictionary_name"], mem["variable_name"]) in row_of]
            member_texts = [mem["text"] for mem in grp_members]
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
                    system_prompt=SYS_GROUP_REASSIGN,
                    user_prompt=build_group_assign_user_prompt(
                        concept or ctx.get("ideal_cde", ""),
                        member_texts[:_SAMPLE_MEMBERS],
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
) -> LeanBResult:
    """Parse per-group assign responses into routed :class:`LeanBRecord` decisions (one per group).

    The verdict + chosen ``cde_id`` are parsed against the PER-GROUP candidates. ``retrieval_floor`` (#1):
    if a verdict adopts/refines a candidate whose dense cosine is below the floor, downgrade it to
    ``novel`` (the engine force-fit the least-bad candidate when nothing was close). 0 disables.
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
        route = "assigned" if verdict in ("adopt", "refine") else "gencde_residual"
        top1 = ctx.get("top1_cos")
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
                ranking=_parse_ranking(payload.get("ranking") if payload else None, len(cands)),
                rationale=str(payload.get("rationale", "")) if payload else "",
                top1_cos=top1,
                chosen_cos=round(chosen_cos, 4) if chosen_cos is not None else None,
                coverage_gap=bool(verdict == "novel" and top1 is not None and top1 < COVERAGE_GAP_TAU),
                floored=floored,
                member_variable_names=ctx.get("member_variable_names", []),
                cohorts=ctx.get("cohorts", []),
                cross_cohort=ctx.get("cross_cohort", False),
                n_members=ctx.get("n_members", 0),
                decided_by="llm",
                raw=payload or {},
            )
        )
    return LeanBResult(records=records, group_assign_prompts=group_assign_records)


def harmonize_leanb(
    embedded_dicts: list[EmbeddedDictionary],
    *,
    generate: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    split: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    classify: Callable[[list[PromptRecord]], dict[str, object]] | None = None,
    cde_cohort: str = CDE_COHORT,
    min_cluster_size: int = 15,
    top_k: int = DEFAULT_TOP_K,
    retrieval_floor: float = DEFAULT_RETRIEVAL_FLOOR,
    model_tag: str = DEFAULT_MODEL_TAG,
    max_clusters: int | None = None,
) -> LeanBResult:
    """Run the full v2 pipeline: cluster -> retrieve -> generate-ideal -> split -> per-group assign -> route.

    ``generate`` (stage 1), ``split`` (stage 2), and ``classify`` (stage 3, per-group assign) each map
    prompt records to ``{id: response}``. At each ``None`` boundary the result carries the prepared
    prompts to export for the Batch API: ``generate=None`` -> ``ideal_prompts``; ``generate`` set but
    ``split=None`` -> ``split_prompts``; ``split`` set but ``classify=None`` -> ``group_assign_prompts``.
    With all three set the LLM runs inline and ``records`` are populated. ``retrieval_floor`` downgrades
    far-cosine adopt/refine to novel (see :func:`assemble_leanb`). ``max_clusters`` caps how many clusters
    are harmonized (largest first) to bound LLM cost — ``None`` processes the whole corpus.
    """
    from ddharmon.clustering.topic_engine import topic_model_dictionaries

    cde_dict = _find_cde_dict(embedded_dicts, cde_cohort)
    # Cluster the COHORT fields only — the CDE backbone is the retrieval target, not a clustered cohort.
    # (prepare_leanb still receives the full embedded_dicts so retrieval can reach the backbone.)
    cohort_dicts = [
        ed
        for ed in embedded_dicts
        if (getattr(ed.dictionary, "cohort_name", None) or getattr(ed.dictionary, "name", None)) != cde_cohort
    ]
    result = topic_model_dictionaries(cohort_dicts or embedded_dicts, min_cluster_size=min_cluster_size)
    ideal_prompts = prepare_leanb(
        result.clusters,
        embedded_dicts,
        result.embeddings,
        result.field_refs,
        cde_cohort=cde_cohort,
        cde_dict=cde_dict,
        top_k=top_k,
        model_tag=model_tag,
    )
    if max_clusters is not None:
        ideal_prompts = sorted(ideal_prompts, key=lambda r: len(r.context["members"]), reverse=True)[:max_clusters]
    if generate is None:
        return LeanBResult(ideal_prompts=ideal_prompts)

    split_prompts = prepare_split(ideal_prompts, generate(ideal_prompts), model_tag=model_tag)
    if split is None:
        return LeanBResult(split_prompts=split_prompts)

    group_assign_prompts = prepare_group_assign(
        split_prompts,
        split(split_prompts),
        embedded_dicts,
        result.embeddings,
        result.field_refs,
        cde_cohort=cde_cohort,
        cde_dict=cde_dict,
        top_k=top_k,
        model_tag=model_tag,
    )
    if classify is None:
        return LeanBResult(group_assign_prompts=group_assign_prompts)

    return assemble_leanb(group_assign_prompts, classify(group_assign_prompts), retrieval_floor=retrieval_floor)


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
