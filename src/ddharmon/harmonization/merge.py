"""Cross-record redundancy MERGE (M2 Phase 2b) — reunite same-concept records after split + chunking.

Coherence-aware chunking bounds how many members reach the split LLM, but it can leave the SAME concept
spread across sibling records (the chunks of one enumerated family, or the split-out group + its M1
residual, or a cross-cluster duplicate). This stage reunites them.

Ported from the nb05 research probe (Islam 2026, arXiv 2604.07562 "Reasoning-Based Refinement of
Unsupervised Text Clusters with LLMs", stage ii = redundancy adjudication; Run-017 silhouette 0.35->0.48):

  1. Candidate generation ($0, deterministic, recall-oriented — the LLM makes the call):
     (a) record-centroid cosine >= TAU  (Islam grid-searched τ∈{.75,.80,.85,.90}, chose .85; a tunable
         DIAGNOSTIC, not a gate — on L2-normalized vectors cosine ranking suffices);
     (b) shared digit-stripped SIGNATURE — reunites a numbered family the slot index fragmented across
         records (medication 1..20 + 21..40), even below τ.
  2. LLM adjudication (conservative): per candidate pair, representative member texts of BOTH records;
     MERGE only if the SAME underlying concept (guards granularity loss / lumping distinct concepts).
  3. Union-find transitive closure over the confirmed links, so >2-way splits reunite into one record.

Batch-compatible: ``prepare_merge`` -> ``merge(prompts)`` -> ``assemble_merge`` (like the other leanb
stages); the schema is soft so ``_parse_merge`` is tolerant. Candidate generation + union-find + the merged
record construction are deterministic and testable without an LLM. Runs BEFORE stage-4 spec-gen so specs are
generated for the final (merged) grouping. Cohort-agnostic (nothing keys on cohort identity).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from ddharmon.harmonization.anchor import build_field_lookup
from ddharmon.harmonization.models import LeanBRecord
from ddharmon.harmonization.parse import extract_json
from ddharmon.harmonization.pipeline import PromptRecord
from ddharmon.harmonization.positional import signature
from ddharmon.models.data_dictionary import Field

logger = logging.getLogger(__name__)

DEFAULT_MERGE_TAU = 0.85  # centroid-cosine candidate threshold (Islam 2026; tunable diagnostic, not a gate)
_N_REPR = 6  # representative members shown per record in the adjudication prompt
_REPR_TRUNC = 140
_SIG_DOMINANT = 0.70  # a record's dominant digit-stripped signature must cover >= this fraction of members
_VERDICT_RANK = {"adopt": 0, "refine": 0, "novel": 1, "": 2}  # which record leads a merged group

SYS_MERGE = (
    "You decide whether two clusters of biomedical data-dictionary fields are the SAME underlying concept "
    "and should be MERGED into one. Merge ONLY if they measure the same thing (a split of one concept, or "
    "two encodings of it). Do NOT merge concepts that are merely related, adjacent, or in the same topic — "
    "over-merging destroys harmonization granularity. A family of the same measure recorded across numbered "
    "occurrences (e.g. medication 1..20 and 21..40) IS one concept -> merge. Judge on meaning, not wording. "
    "Return JSON only."
)
MERGE_SCHEMA = json.dumps({"merge": "true|false", "reason": "<one sentence>"})


def _record_id(rec: LeanBRecord) -> str:
    return rec.group_id or rec.cluster_id


def _pair(a: str, b: str) -> tuple[str, str]:
    """A canonical, order-independent 2-tuple key for a record pair."""
    return (a, b) if a <= b else (b, a)


def _label(fld: Field | None) -> str:
    if fld is None:
        return ""
    for cand in (fld.question_text, fld.short_label, fld.variable_name):
        if cand and cand.strip():
            return cand.strip()
    return ""


def _norm(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    # np.where's 1.0 literal is float64, so the division upcasts; keep the float32 contract (and the
    # embedding dtype) explicit — pyright >=1.1.410 enforces the ndarray dtype covariance.
    return (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def _member_rows(rec: LeanBRecord, row_of: dict[tuple[str, str], int]) -> list[int]:
    rows = []
    for sv in rec.member_variable_names:
        cohort, _, var = sv.partition(":")
        r = row_of.get((cohort, var))
        if r is not None:
            rows.append(r)
    return rows


def build_merge_user_prompt(reps_a: list[str], reps_b: list[str]) -> str:
    a = "\n  - ".join(reps_a) or "(none)"
    b = "\n  - ".join(reps_b) or "(none)"
    return (
        f"Cluster A members (sample):\n  - {a}\n\n"
        f"Cluster B members (sample):\n  - {b}\n\n"
        "Are A and B the SAME underlying concept (should be merged)? Return JSON."
    )


def _reps(rec: LeanBRecord, embeddings: NDArray[np.float32], row_of, text_of, k: int = _N_REPR) -> list[str]:
    """The ``k`` members closest to the record centroid (most representative) as readable text."""
    members = rec.member_variable_names
    rows = _member_rows(rec, row_of)
    if not rows:
        return [text_of.get(_key(sv), sv.partition(":")[2])[:_REPR_TRUNC] for sv in members[:k]]
    centroid = _norm(embeddings[np.asarray(rows, dtype=np.intp)].mean(axis=0))
    scored = []
    for sv in members:
        cohort, _, var = sv.partition(":")
        r = row_of.get((cohort, var))
        cos = float(_norm(embeddings[r]) @ centroid) if r is not None else -1.0
        scored.append((cos, sv))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [text_of.get(_key(sv), sv.partition(":")[2])[:_REPR_TRUNC] for _, sv in scored[:k]]


def _key(sv: str) -> tuple[str, str]:
    cohort, _, var = sv.partition(":")
    return (cohort, var)


def _dominant_signature(rec: LeanBRecord, text_of) -> str | None:
    """The record's dominant digit-stripped member-label signature (containing a digit), if one dominates."""
    sigs = [signature(text_of.get(_key(sv), "")) for sv in rec.member_variable_names]
    sigs = [s for s in sigs if s]
    if not sigs:
        return None
    dom = max(set(sigs), key=sigs.count)
    if "#" in dom and sigs.count(dom) / len(sigs) >= _SIG_DOMINANT:
        return dom
    return None


def merge_candidate_pairs(
    records: list[LeanBRecord],
    embeddings: NDArray[np.float32],
    row_of: dict[tuple[str, str], int],
    text_of: dict[tuple[str, str], str],
    *,
    tau: float = DEFAULT_MERGE_TAU,
) -> dict[tuple[str, str], dict]:
    """Deterministic recall-oriented candidate pairs: centroid cos >= tau (general) ∪ shared signature.

    Keyed by a sorted ``(id_a, id_b)`` record-id pair. Records with no member embeddings are skipped from
    the centroid pass (they can still pair via signature).
    """
    ids: list[str] = []
    centroids = []
    for rec in records:
        rows = _member_rows(rec, row_of)
        if not rows:
            continue
        ids.append(_record_id(rec))
        centroids.append(_norm(embeddings[np.asarray(rows, dtype=np.intp)].mean(axis=0)))
    pairs: dict[tuple[str, str], dict] = {}
    if len(ids) >= 2:
        matrix = np.stack(centroids)
        sims = matrix @ matrix.T
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if sims[i, j] >= tau:
                    pairs[_pair(ids[i], ids[j])] = {"centroid_cos": round(float(sims[i, j]), 3), "via": "centroid"}
    cen_of = dict(zip(ids, centroids, strict=True))
    by_sig: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        dom = _dominant_signature(rec, text_of)
        if dom is not None:
            by_sig[dom].append(_record_id(rec))
    for sig, group in by_sig.items():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                key = _pair(group[a], group[b])
                if key in pairs:
                    pairs[key]["via"] = "centroid+signature"
                elif key[0] in cen_of and key[1] in cen_of:
                    pairs[key] = {
                        "centroid_cos": round(float(cen_of[key[0]] @ cen_of[key[1]]), 3),
                        "via": "signature",
                        "signature": sig,
                    }
                else:
                    pairs[key] = {"centroid_cos": None, "via": "signature", "signature": sig}
    return pairs


def prepare_merge(
    records: list[LeanBRecord],
    embedded_dicts,
    embeddings: NDArray[np.float32],
    field_refs,
    *,
    tau: float = DEFAULT_MERGE_TAU,
    model_tag: str = "claude-sonnet-4-6",
) -> list[PromptRecord]:
    """Build one adjudication prompt per candidate merge pair (representative members of both records)."""
    row_of = {(r.dictionary_name, r.variable_name): i for i, r in enumerate(field_refs)}
    field_lookup = build_field_lookup(embedded_dicts)
    text_of = {k: _label(v) for k, v in field_lookup.items()}
    by_id = {_record_id(rec): rec for rec in records}
    pairs = merge_candidate_pairs(records, embeddings, row_of, text_of, tau=tau)

    prompts: list[PromptRecord] = []
    for (a, b), meta in sorted(pairs.items()):
        if a not in by_id or b not in by_id:
            continue
        reps_a = _reps(by_id[a], embeddings, row_of, text_of)
        reps_b = _reps(by_id[b], embeddings, row_of, text_of)
        prompts.append(
            PromptRecord(
                id=f"leanb:merge:{a}|{b}",
                system_prompt=SYS_MERGE,
                user_prompt=build_merge_user_prompt(reps_a, reps_b),
                schema=MERGE_SCHEMA,
                model_tag=model_tag,
                context={"a": a, "b": b, **meta},
            )
        )
    logger.info("prepare_merge: %d records -> %d candidate merge pairs", len(records), len(prompts))
    return prompts


def _parse_merge(resp: object) -> bool:
    if resp is None:
        return False
    try:
        payload = extract_json(resp if isinstance(resp, str) else json.dumps(resp))
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    return str(payload.get("merge", "")).strip().lower() in ("true", "yes", "1")


def union_find(links: list[tuple[str, str]], all_ids: list[str]) -> list[list[str]]:
    """Transitive closure of merge links -> connected-component groups (each an id list, order-stable)."""
    parent = {u: u for u in all_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = defaultdict(list)
    for u in all_ids:  # preserve input order within each group
        groups[find(u)].append(u)
    return list(groups.values())


def _merge_group(recs: list[LeanBRecord]) -> LeanBRecord:
    """Fold a set of same-concept records into one — the highest-priority record leads, members unioned."""
    primary = min(recs, key=lambda r: (_VERDICT_RANK.get(r.verdict, 3), -r.n_members, _record_id(r)))
    members: list[str] = []
    seen: set[str] = set()
    for rec in recs:
        for sv in rec.member_variable_names:
            if sv not in seen:
                seen.add(sv)
                members.append(sv)
    members.sort()
    cohorts = sorted({sv.partition(":")[0] for sv in members})
    merged_from = sorted(_record_id(r) for r in recs)
    note = f"[merged {len(recs)} records: {', '.join(merged_from)}]"
    return LeanBRecord(
        cluster_id=primary.cluster_id,
        verdict=primary.verdict,
        route=primary.route,
        group_id=primary.group_id,
        concept=primary.concept,
        cde_id=primary.cde_id,
        cde_external_id=primary.cde_external_id,
        ideal_cde=primary.ideal_cde,
        ranking=primary.ranking,
        rationale=(primary.rationale + " " if primary.rationale else "") + note,
        top1_cos=primary.top1_cos,
        chosen_cos=primary.chosen_cos,
        coverage_gap=primary.coverage_gap,
        floored=primary.floored,
        coherence_gap=primary.coherence_gap,
        member_variable_names=members,
        cohorts=cohorts,
        cross_cohort=len(cohorts) >= 2,
        n_members=len(members),
        candidates=primary.candidates,
        decided_by=primary.decided_by,
        raw={**primary.raw, "merged_from": merged_from},
    )


def assemble_merge(
    records: list[LeanBRecord], merge_prompts: list[PromptRecord], responses: dict[str, object]
) -> list[LeanBRecord]:
    """Apply the LLM merge verdicts: union-find the confirmed pairs, fold each group into one record.

    Records not in any confirmed pair pass through unchanged. Output order follows first appearance of each
    group's representative in ``records``.
    """
    by_id = {_record_id(rec): rec for rec in records}
    links = [(mp.context["a"], mp.context["b"]) for mp in merge_prompts if _parse_merge(responses.get(mp.id))]
    all_ids = [_record_id(rec) for rec in records]
    groups = union_find(links, all_ids)
    out: list[LeanBRecord] = []
    for g in groups:
        if len(g) == 1:
            out.append(by_id[g[0]])
        else:
            out.append(_merge_group([by_id[u] for u in g]))
    logger.info(
        "assemble_merge: %d records, %d confirmed links -> %d records (%d merged groups)",
        len(records),
        len(links),
        len(out),
        sum(1 for g in groups if len(g) > 1),
    )
    return out
