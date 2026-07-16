"""C1: categorical transform-spec generation — a verifying post-stage over leanb records.

For each adopt/refine :class:`LeanBRecord` (a concept group assigned to a CDE), build ONE schema-enforced
prompt showing the target CDE's value set + each source member's value set, and emit one categorical
recode per source variable (source code -> CDE code). Metadata-level: emit the spec, never touch row data.

The retrieval/assign stages already ran value-aware (``leanb._member_prompt_text``), so a ``refine`` here
is an honest "needs transform". ``coverage`` (mapped / total source codes) is the verification signal: a
low-coverage recode is flagged ``needs_review`` but never overrides the assign verdict.

Run via the Batch API (schema-enforced), like the other leanb stages — never inline ``complete()``.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import operator
import re
from pathlib import Path

from ddharmon.embedding.service import EmbeddedDictionary
from ddharmon.export.eitl import cde_url, labeled, pack, qtext, write_csv
from ddharmon.harmonization.anchor import build_field_lookup
from ddharmon.harmonization.leanb import DEFAULT_MODEL_TAG, _value_set_text
from ddharmon.harmonization.models import LeanBRecord, TransformKind, TransformSpec
from ddharmon.harmonization.parse import extract_json
from ddharmon.harmonization.pipeline import PromptRecord
from ddharmon.harmonization.positional import detect_positional_enumeration, signature
from ddharmon.harmonization.substrate import content_token
from ddharmon.matching.confidence import TransformConfidenceConfig, score_transform_spec
from ddharmon.models.data_dictionary import Field, ResponseOption
from ddharmon.values.response_parser import parse_value_encoding
from ddharmon.values.units import UnitCanonicalizer, is_identity_conversion

logger = logging.getLogger(__name__)

REVIEW_COVERAGE = 0.8  # below this fraction of source codes mapped -> needs_review
REVIEW_CONFIDENCE = 0.6  # below this LLM confidence -> needs_review

SYS_SPECGEN = (
    "You write a value-recode specification that harmonizes ONE survey/clinical variable to a Common Data "
    "Element (CDE). You are given a TARGET CDE with its permissible answer values and ONE SOURCE variable "
    "with its coded answer options. Return a code_map that maps the SOURCE codes to the CDE's TARGET codes, "
    "mapping only where the meaning aligns. List any source codes you cannot confidently map (missing-data "
    "sentinels, options with no CDE equivalent) in 'unmapped'. Never invent CDE codes — every target code "
    "MUST appear in the CDE's value set. Return a confidence in [0,1]."
)

SPECGEN_SCHEMA = json.dumps(
    {
        "code_map": {"<source_code>": "<cde_code>"},
        "unmapped": ["<source_code>"],
        "confidence": "<0.0-1.0>",
        "notes": "<short rationale>",
    }
)


def _codes_of(value_set: str) -> list[str]:
    """Source codes from a value-encoding string ('1=Yes|2=No' -> ['1', '2'])."""
    return [ro.code for ro in parse_value_encoding(value_set)] if value_set else []


def _as_float(v: object) -> float:
    try:
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _spec_confidence(coverage: float, llm_confidence: float) -> float:
    """C1 categorical spec confidence: coverage + LLM confidence, evenly weighted.

    Units-driven scoring for numeric N1/N2 lives in ``matching.confidence`` (added with C2).
    """
    return max(0.0, min(1.0, 0.5 * coverage + 0.5 * llm_confidence))


def build_specgen_user_prompt(cde_id: str, cde_value_set: str, source_value_set: str) -> str:
    """Render the spec-gen prompt for one ``(CDE, source-encoding)`` pair.

    The recode is a value-label -> value-label mapping, so it's a pure function of the CDE concept + the
    two value sets — the source variable's NAME is deliberately not shown, which lets every edge sharing a
    ``(cde_id, source value set)`` reuse one prompt (see :func:`prepare_specgen` dedup).
    """
    return "\n".join(
        [
            f"TARGET CDE: {cde_id}",
            f"CDE values: {cde_value_set}",
            "",
            f"SOURCE values: {source_value_set}",
            "",
            "Return the code_map (source code -> CDE code), any unmapped source codes, and a confidence.",
        ]
    )


def prepare_specgen(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    cde_fields: dict[str, Field],
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
) -> list[PromptRecord]:
    """One categorical spec-gen prompt per unique ``(cde_id, source value set)`` across adopt/refine edges.

    The recode for a coded source-var -> CDE edge is a value-label -> value-label mapping, so it depends
    only on the CDE concept + the two value sets, not the source variable name. Two failure modes drove
    this shape:

    - **Under-return:** an earlier design batched all of a group's members into one prompt asking for an
      N-element list; at scale the LLM answered only a subset of large groups (output-length ceiling),
      silently NONE'ing the rest (measured: 11% matched / 88.7% spurious-NONE on a full run).
    - **String-mismatch:** recodes were bound back to members by an echoed ``source_variable`` string the
      model often paraphrased -> 40% of the recodes it DID emit were discarded.

    So each prompt covers one edge's worth of recode, bound back by ``context`` (not a string). On top of
    that, edges sharing the same ``(cde_id, source value set)`` — e.g. many cohort items with an identical
    Likert scale assigned to the same CDE — collapse to ONE prompt whose recode is fanned out to every
    such edge in :func:`assemble_specgen`. That cuts a large fraction of categorical LLM calls AND gives
    identical encodings an identical, consistent recode.

    Novel/no-CDE records and edges whose CDE or source variable lacks a coded value set are skipped
    (numeric / no-encoding cases -> C2's N1 path).
    """
    field_lookup = build_field_lookup(embedded_dicts)
    # group coded edges by (cde_id, source value set); each unique signature -> one prompt fanned to its edges
    sigs: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for rec in records:
        if rec.verdict not in ("adopt", "refine") or not rec.cde_id:
            continue
        cde_fld = cde_fields.get(rec.cde_id)
        cde_vs = _value_set_text(cde_fld) if cde_fld else ""
        if not cde_vs:
            continue  # CDE has no coded value set -> categorical recode N/A (numeric -> C2)
        if any(t.kind == TransformKind.WIDE_TO_LONG for t in rec.transforms):
            continue  # repeating measure -> one structural wide->long spec, not per-column recodes
        key = rec.group_id or rec.cluster_id
        for sv in rec.member_variable_names:
            cohort, _, var = sv.partition(":")
            fld = field_lookup.get((cohort, var))
            vs = _value_set_text(fld) if fld else ""
            if not vs:
                continue  # numeric / no-encoding edge -> N1 (generate_unit_specs) handles it
            sig = (rec.cde_id, vs)
            entry = sigs.get(sig)
            if entry is None:
                entry = {"cde_id": rec.cde_id, "cde_value_set": cde_vs, "source_value_set": vs, "edges": []}
                sigs[sig] = entry
                order.append(sig)
            entry["edges"].append((key, sv))
    out: list[PromptRecord] = []
    for sig in order:
        e = sigs[sig]
        out.append(
            PromptRecord(
                # content-addressed by the (cde_id, source-encoding) signature -> stable across runs (L2)
                id=f"leanb:specgen:{content_token(e['cde_id'], e['source_value_set'])}",
                system_prompt=SYS_SPECGEN,
                user_prompt=build_specgen_user_prompt(e["cde_id"], e["cde_value_set"], e["source_value_set"]),
                schema=SPECGEN_SCHEMA,
                model_tag=model_tag,
                context={
                    "cde_id": e["cde_id"],
                    "cde_value_set": e["cde_value_set"],
                    "source_value_set": e["source_value_set"],
                    "edges": e["edges"],  # [(record_key, source_variable), ...] sharing this recode
                },
            )
        )
    n_edges = sum(len(sigs[s]["edges"]) for s in order)
    logger.info(
        "prepare_specgen: %d coded edges -> %d unique (cde, source-encoding) prompts (%d collapsed)",
        n_edges,
        len(out),
        n_edges - len(out),
    )
    return out


def _parse_specgen(resp: object) -> dict:
    """The single recode object from a per-edge spec-gen response, tolerant of the shapes the LLM returns.

    The schema asks for a bare ``{code_map, unmapped, confidence, notes}`` object (one edge per prompt),
    but the Batch schema is a soft instruction, so the model sometimes still wraps it as
    ``{"recodes": [ {...} ]}`` or returns a bare ``[ {...} ]`` list. Accept all three and take the single
    recode; anything else -> ``{}`` (the edge becomes a NONE spec, flagged for review).
    """
    if resp is None:
        return {}
    try:
        payload = extract_json(resp if isinstance(resp, str) else json.dumps(resp))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    if isinstance(payload, list):  # bare list (wrapper kept the array but dropped the key)
        objs = [r for r in payload if isinstance(r, dict)]
        return objs[0] if objs else {}
    if not isinstance(payload, dict):
        return {}
    recodes = payload.get("recodes")
    if isinstance(recodes, list):  # legacy wrapper retained -> take the first recode
        objs = [r for r in recodes if isinstance(r, dict)]
        return objs[0] if objs else {}
    return payload  # bare recode object (the expected shape)


def _int_code(code: str) -> int | None:
    """Parse an ordinal code as an integer, else ``None`` (a non-numeric option is not an ordinal scale)."""
    try:
        return int(str(code).strip())
    except (TypeError, ValueError):
        return None


def monotonic_ordinal_fill(
    src_codes: list[str], cde_codes: list[str], code_map: dict[str, str]
) -> dict[str, str] | None:
    """Complete a partial recode between two EQUAL-LENGTH ordinal scales by monotonic position (M9).

    When source and target are equal-length integer ordinal scales (e.g. two 1..5 Likert scales) and every
    code the LLM DID map is rank-aligned (the k-th source level -> the k-th target level), the unmapped
    interior codes are deterministically fillable: align the two scales by sorted position. Returns the full
    positional map, or ``None`` when the shape doesn't qualify (unequal length, non-integer codes, a
    non-monotonic/misaligned partial map, an already-complete map, or fewer than two anchor mappings).

    Conservative by construction: a reverse-coded or non-trivially-permuted scale fails the rank-alignment
    check (its anchors land at mismatched ranks) and is left to the LLM map + review, never auto-filled.
    """
    n = len(src_codes)
    if n < 3 or n != len(cde_codes) or not code_map:
        return None
    if n - len(code_map) < 1:
        return None  # already complete — nothing to fill
    src_ints = [_int_code(c) for c in src_codes]
    cde_ints = [_int_code(c) for c in cde_codes]
    if any(v is None for v in src_ints) or any(v is None for v in cde_ints):
        return None  # not an integer ordinal scale on both sides
    if len(set(src_ints)) != n or len(set(cde_ints)) != n:
        return None  # duplicate codes -> not a clean ordinal sequence
    src_sorted = sorted(src_codes, key=lambda c: _int_code(c))  # type: ignore[arg-type,return-value]
    cde_sorted = sorted(cde_codes, key=lambda c: _int_code(c))  # type: ignore[arg-type,return-value]
    src_rank = {c: i for i, c in enumerate(src_sorted)}
    cde_rank = {c: i for i, c in enumerate(cde_sorted)}
    anchors = 0
    for s, t in code_map.items():
        if s not in src_rank or t not in cde_rank or src_rank[s] != cde_rank[t]:
            return None  # a mapped pair is off the monotonic diagonal -> don't fill
        anchors += 1
    if anchors < 2:
        return None  # need >= 2 aligned anchors to trust the positional alignment
    return {src_sorted[i]: cde_sorted[i] for i in range(n)}


def _compute_recode(ctx: dict, response: object) -> dict:
    """Parse a spec-gen response + the prompt's two value sets into a target-id-agnostic recode.

    Shared by the CDE (:func:`assemble_specgen`) and GenCDE (:func:`assemble_gencde_specgen`) assemble
    passes — the recode (code_map, coverage, kind, confidence, review) is a pure function of the response +
    the prompt's ``(cde_value_set, source_value_set)`` and does NOT depend on which target the edge carries.
    Hallucinated target codes (not in the target value set) are dropped; a purely identity map -> ``IDENTITY``,
    an empty one -> ``NONE``.
    """
    vs = str(ctx.get("source_value_set", ""))
    tgt_codes_ordered = _codes_of(ctx.get("cde_value_set", ""))
    tgt_codes = set(tgt_codes_ordered)
    payload = _parse_specgen(response)
    code_map = {str(k): str(val) for k, val in (payload.get("code_map") or {}).items()}
    # enforce the schema rule: drop any target not in the target value set (no invented codes)
    if tgt_codes:
        code_map = {k: val for k, val in code_map.items() if val in tgt_codes}
    src_codes = _codes_of(vs)
    # M9: complete a rank-aligned partial recode between two equal-length integer ordinal scales by
    # monotonic position (fills interior codes the LLM skipped). A no-op unless the shape qualifies.
    filled = monotonic_ordinal_fill(src_codes, tgt_codes_ordered, code_map)
    ordinal_filled = filled is not None
    if filled is not None:
        code_map = filled
    unmapped = sorted(set(src_codes) - set(code_map.keys()))
    coverage = round(len(code_map) / len(src_codes) if src_codes else 0.0, 3)
    llm_conf = _as_float(payload.get("confidence"))
    if not code_map:
        kind = TransformKind.NONE
    elif all(k == val for k, val in code_map.items()):
        kind = TransformKind.IDENTITY
    else:
        kind = TransformKind.CATEGORICAL
    notes = str(payload.get("notes", "") or "")
    if ordinal_filled:
        notes = (notes + " " if notes else "") + "[ordinal fill: interior codes completed by monotonic position]"
    return {
        "kind": kind,
        "code_map": code_map,
        "unmapped": unmapped,
        "coverage": coverage,
        "confidence": round(_spec_confidence(coverage, llm_conf), 3),
        "needs_review": coverage < REVIEW_COVERAGE or llm_conf < REVIEW_CONFIDENCE or not code_map,
        "notes": notes,
    }


def assemble_specgen(
    specgen_records: list[PromptRecord],
    responses: dict[str, object],
    records: list[LeanBRecord],
) -> list[LeanBRecord]:
    """Parse each spec-gen response into a recode and fan it out to every edge that shares its prompt.

    A prompt covers one unique ``(cde_id, source value set)`` (see :func:`prepare_specgen`); its single
    recode (:func:`_compute_recode`) is attached as a :class:`TransformSpec` to each
    ``(record, source_variable)`` edge in the prompt's ``context["edges"]`` — so identical encodings get an
    identical, consistent recode and no edge is bound by a fragile string nor dropped because a sibling went
    unanswered. ``needs_review`` fires on low coverage / low LLM confidence / empty map — the assign verdict
    is never changed.
    """
    by_key = {(r.group_id or r.cluster_id): r for r in records}
    for sgr in specgen_records:
        ctx = sgr.context
        r = _compute_recode(ctx, responses.get(sgr.id))
        for record_key, sv in ctx.get("edges", []):
            rec = by_key.get(record_key)
            if rec is None:
                continue
            rec.transforms.append(
                TransformSpec(
                    source_variable=sv,
                    target_cde_id=rec.cde_id or "",
                    kind=r["kind"],
                    code_map=dict(r["code_map"]),  # per-edge copy (no shared mutable across specs)
                    unmapped_source_codes=list(r["unmapped"]),
                    coverage=r["coverage"],
                    confidence=r["confidence"],
                    needs_review=r["needs_review"],
                    rationale=r["notes"],
                    generated_by="llm",
                )
            )
    return records


def _gencde_value_set_text(gencde: object) -> str:
    """Render a GenCDE's synthesized categorical domain as a ``code=label|…`` value set (like _value_set_text)."""
    pv = getattr(gencde, "permissible_values", None) or []
    return "|".join(f"{ro.code}={ro.label}" for ro in pv)


def prepare_gencde_specgen(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
) -> list[PromptRecord]:
    """C1 categorical recode prompts for the tail: map each novel member's codes into its GenCDE's domain.

    Mirrors :func:`prepare_specgen` but the TARGET is the record's synthesized :class:`~.models.GenCDE`
    (its ``permissible_values``) instead of a catalog CDE — so a ``novel`` record gets member->target recodes
    just like an adopt/refine record does, completing the harmonization of the novel path. Only categorical
    GenCDEs (non-empty ``permissible_values``) and coded source edges qualify; numeric GenCDEs (units/bounds)
    are the unit/arith path, and wide->long records are skipped (one structural spec, not per-column recodes).
    Each prompt covers one ``(gencde_id, source value set)`` and is fanned back to its edges in
    :func:`assemble_gencde_specgen`. Since a GenCDE id is unique per record, a prompt's edges belong to one
    record — but identical source encodings within it still share one recode.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    sigs: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for rec in records:
        g = rec.gencde
        if g is None:
            continue
        tgt_vs = _gencde_value_set_text(g)
        if not tgt_vs:
            continue  # numeric / no coded domain -> unit/arith path, not a categorical recode
        if any(t.kind == TransformKind.WIDE_TO_LONG for t in rec.transforms):
            continue
        gencde_id = g.gencde_id
        key = rec.group_id or rec.cluster_id
        for sv in rec.member_variable_names:
            cohort, _, var = sv.partition(":")
            fld = field_lookup.get((cohort, var))
            vs = _value_set_text(fld) if fld else ""
            if not vs:
                continue  # numeric / no-encoding edge
            sig = (gencde_id, vs)
            entry = sigs.get(sig)
            if entry is None:
                entry = {"gencde_id": gencde_id, "target_value_set": tgt_vs, "source_value_set": vs, "edges": []}
                sigs[sig] = entry
                order.append(sig)
            entry["edges"].append((key, sv))
    out: list[PromptRecord] = []
    for sig in order:
        e = sigs[sig]
        out.append(
            PromptRecord(
                id=f"leanb:gencde_specgen:{content_token(e['gencde_id'], e['source_value_set'])}",
                system_prompt=SYS_SPECGEN,
                user_prompt=build_specgen_user_prompt(e["gencde_id"], e["target_value_set"], e["source_value_set"]),
                schema=SPECGEN_SCHEMA,
                model_tag=model_tag,
                context={
                    "gencde_id": e["gencde_id"],
                    "cde_value_set": e["target_value_set"],  # keep the key _compute_recode reads
                    "source_value_set": e["source_value_set"],
                    "edges": e["edges"],
                },
            )
        )
    n_edges = sum(len(sigs[s]["edges"]) for s in order)
    logger.info(
        "prepare_gencde_specgen: %d coded tail edges -> %d unique (gencde, source-encoding) prompts (%d collapsed)",
        n_edges,
        len(out),
        n_edges - len(out),
    )
    return out


def assemble_gencde_specgen(
    specgen_records: list[PromptRecord],
    responses: dict[str, object],
    records: list[LeanBRecord],
) -> list[LeanBRecord]:
    """Attach each GenCDE recode as a :class:`TransformSpec` to its novel record's member edges.

    Identical to :func:`assemble_specgen` except the edge's ``target_cde_id`` is the GenCDE id (the tail's
    synthesized target), so the novel path carries member->GenCDE recodes just like adopt/refine carries
    member->CDE recodes. ``needs_review`` never changes the ``novel`` verdict.
    """
    by_key = {(r.group_id or r.cluster_id): r for r in records}
    for sgr in specgen_records:
        ctx = sgr.context
        r = _compute_recode(ctx, responses.get(sgr.id))
        gencde_id = str(ctx.get("gencde_id", ""))
        for record_key, sv in ctx.get("edges", []):
            rec = by_key.get(record_key)
            if rec is None:
                continue
            rec.transforms.append(
                TransformSpec(
                    source_variable=sv,
                    target_cde_id=(rec.gencde.gencde_id if rec.gencde else gencde_id),
                    kind=r["kind"],
                    code_map=dict(r["code_map"]),
                    unmapped_source_codes=list(r["unmapped"]),
                    coverage=r["coverage"],
                    confidence=r["confidence"],
                    needs_review=r["needs_review"],
                    rationale=r["notes"],
                    generated_by="llm",
                )
            )
    return records


# ── Structural: wide->long repeating-measure detection ($0, deterministic) ────


def _field_label(fld: Field | None) -> str:
    """Human label for repeating-measure detection (question > short_label > variable_name)."""
    if fld is None:
        return ""
    for cand in (fld.question_text, fld.short_label, fld.variable_name):
        if cand and cand.strip():
            return cand.strip()
    return ""


# A derived aggregate over a repeating measure (…_AVG, …_MEAN) — NOT an occurrence column, so it is
# excluded from a numbered family (M6). Guarded by a delimiter so a legitimate name that merely contains
# "mean"/"avg" as a substring is not misclassified.
_AGG_NAME_RE = re.compile(r"(?:^|[_\-.])(?:avg|mean)(?:$|[_\-.])", re.IGNORECASE)


def _is_aggregate_name(var: str) -> bool:
    """True for a derived-aggregate variable name (``bp_avg``, ``hdl_mean``) — dropped from a wide family."""
    return bool(_AGG_NAME_RE.search(var or ""))


def generate_wide_to_long_specs(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    cde_fields: dict[str, Field],
) -> list[LeanBRecord]:
    """Collapse a repeating-measure record to ONE ``WIDE_TO_LONG`` spec (deterministic pre-pass, no LLM).

    When an adopt/refine record's non-CDE members are a positional enumeration — numbered occurrence
    columns like ``Medication 1..40`` (:func:`~ddharmon.harmonization.positional.detect_positional_enumeration`)
    — they are ONE repeating concept, not N distinct source->CDE edges. We attach a single WIDE_TO_LONG spec
    describing the structural reshape (occurrence index -> array position). Because the categorical (C1) and
    unit (N1) generators skip a record that already carries a WIDE_TO_LONG spec, the record produces ZERO
    spec-gen LLM prompts and ONE review row instead of N — a cost + review-noise + correctness win.

    Runs FIRST in stage 4 (before N1/C1). Metadata-level: emit the reshape recipe, never execute it.
    Idempotent per record.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    n = 0
    for rec in records:
        if rec.verdict not in ("adopt", "refine") or not rec.cde_id:
            continue
        if any(t.kind == TransformKind.WIDE_TO_LONG for t in rec.transforms):
            continue  # idempotent
        members = list(rec.member_variable_names)
        q_labels: list[str] = []
        name_of: dict[str, str] = {}
        for sv in members:
            cohort, _, var = sv.partition(":")
            q_labels.append(_field_label(field_lookup.get((cohort, var))))
            name_of[sv] = var
        pe = detect_positional_enumeration(q_labels)
        if pe is None:
            # M6: the occurrence index frequently lives ONLY in the variable NAME (``hdl1``..``hdl31``, BP
            # occurrences, dated columns), not the question_text label, which is often identical across the
            # numbered columns. Retry detection on the name signature — but ONLY when the human label does
            # NOT itself distinguish the members (a single question_text signature); a genuine qualifier
            # matrix of distinct questions must never be collapsed.
            q_sigs = {signature(lbl) for lbl in q_labels if lbl}
            if len(q_sigs) <= 1:
                non_agg = [sv for sv in members if not _is_aggregate_name(name_of[sv])]
                pe = detect_positional_enumeration([name_of[sv] for sv in non_agg])
        if pe is None:
            continue
        # Derived aggregates (``_AVG``/``_MEAN``) are summaries over the repeating measure, NOT occurrences —
        # drop them from the emitted family so the wide->long spec describes only the true occurrence columns
        # (M6). ``detect_positional_enumeration`` already ignored them (minority / off-signature), so this only
        # trims the ``inputs`` set; the fallback keeps the family non-empty in the degenerate all-aggregate case.
        members = [sv for sv in members if not _is_aggregate_name(name_of[sv])] or members
        lo, hi = pe.int_range
        rec.transforms.append(
            TransformSpec(
                source_variable=members[0] if members else "",  # representative; full set in `inputs`
                target_cde_id=rec.cde_id or "",
                kind=TransformKind.WIDE_TO_LONG,
                method="wide_to_long",
                params={
                    "signature": pe.signature,
                    "n_occurrences": pe.n_occurrences,
                    "int_range": [lo, hi],
                    "density": pe.density,
                    "dominant_share": pe.dominant_share,
                },
                inputs=sorted(members),
                confidence=0.6,  # deterministic + high-precision, but a structural reshape -> always review
                needs_review=True,
                generated_by="rule",
                rationale=(
                    f"{len(members)} numbered columns ('{pe.signature}', occurrences {lo}..{hi}) are ONE "
                    "repeating measure; reshape wide->long (occurrence index -> array position)."
                ),
            )
        )
        n += 1
    logger.info("generate_wide_to_long_specs: %d repeating-measure records collapsed to wide->long", n)
    return records


# ── N1: deterministic unit/scale specs (C2) ──────────────────────────────────


def _unit_spec(
    source_variable: str, cde_id: str, src_unit: str, cde_unit: str, canon: UnitCanonicalizer
) -> TransformSpec:
    """Author one numeric-edge spec from source + CDE unit strings (no LLM, no row data).

    Known same-family conversion -> UNIT (factor/offset); a no-op or same-unit -> IDENTITY; otherwise a
    ``needs_units`` UNIT (the residual the N2 arithmetic path may upgrade).
    """
    conv = canon.convert(src_unit or None, cde_unit or None)
    if conv is not None:
        factor, offset = conv
        if is_identity_conversion(factor, offset):
            return TransformSpec(
                source_variable=source_variable,
                target_cde_id=cde_id,
                kind=TransformKind.IDENTITY,
                source_unit=src_unit or None,
                target_unit=cde_unit or None,
                generated_by="rule",
                rationale=f"units already aligned ({src_unit})" if src_unit else "no unit conversion needed",
            )
        return TransformSpec(
            source_variable=source_variable,
            target_cde_id=cde_id,
            kind=TransformKind.UNIT,
            factor=factor,
            offset=offset,
            source_unit=src_unit or None,
            target_unit=cde_unit or None,
            generated_by="rule",
            rationale=f"linear unit conversion {src_unit} -> {cde_unit} (target = source * {factor:g} + {offset:g})",
        )
    # not deterministically convertible
    if src_unit and cde_unit and src_unit.lower() == cde_unit.lower():
        return TransformSpec(  # same (unrecognized) unit string on both sides -> no conversion needed
            source_variable=source_variable,
            target_cde_id=cde_id,
            kind=TransformKind.IDENTITY,
            source_unit=src_unit,
            target_unit=cde_unit,
            generated_by="rule",
            rationale=f"same unit ({src_unit}) — no conversion",
        )
    reason = (
        f"units not reconcilable ({src_unit or '?'} vs {cde_unit or '?'})"
        if (src_unit or cde_unit)
        else "numeric variable with no declared units — cannot author a unit conversion"
    )
    return TransformSpec(
        source_variable=source_variable,
        target_cde_id=cde_id,
        kind=TransformKind.UNIT,
        needs_units=True,
        needs_review=True,
        source_unit=src_unit or None,
        target_unit=cde_unit or None,
        generated_by="rule",
        rationale=reason,
    )


def generate_unit_specs(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    cde_fields: dict[str, Field],
    *,
    config: TransformConfidenceConfig | None = None,
) -> list[LeanBRecord]:
    """N1: attach deterministic unit/scale specs for NUMERIC source edges of adopt/refine records.

    A member edge is numeric here when the SOURCE variable has no coded value set — categorical edges are
    handled by the C1 LLM path (:func:`prepare_specgen` / :func:`assemble_specgen`), and the two member
    sets are disjoint. For each numeric edge :func:`_unit_spec` emits a UNIT (known conversion), IDENTITY
    (no-op / same unit), or ``needs_units`` UNIT (units missing / unrecognized / cross-family) spec, scored
    via :func:`~ddharmon.matching.confidence.score_transform_spec`.

    Deterministic and data-free, so it runs unconditionally (no LLM, no Batch round-trip). Idempotent per
    (record, source_var): an edge already carrying a transform is left untouched.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    canon = UnitCanonicalizer()
    n_unit = n_identity = n_needs = 0
    for rec in records:
        if rec.verdict not in ("adopt", "refine") or not rec.cde_id:
            continue
        if any(t.kind == TransformKind.WIDE_TO_LONG for t in rec.transforms):
            continue  # repeating measure -> handled by the single wide->long spec, not per-member units
        cde_fld = cde_fields.get(rec.cde_id)
        cde_unit = (cde_fld.units or "").strip() if cde_fld and cde_fld.units else ""
        existing = {t.source_variable for t in rec.transforms}
        for sv in rec.member_variable_names:
            if sv in existing:
                continue
            cohort, _, var = sv.partition(":")
            src_fld = field_lookup.get((cohort, var))
            if src_fld is None or _value_set_text(src_fld):
                continue  # missing field, or a categorical edge (coded source -> C1 handles it)
            src_unit = (src_fld.units or "").strip() if src_fld.units else ""
            spec = _unit_spec(sv, rec.cde_id, src_unit, cde_unit, canon)
            spec.confidence = round(score_transform_spec(spec, config), 3)
            rec.transforms.append(spec)
            n_identity += spec.kind == TransformKind.IDENTITY
            n_unit += spec.kind == TransformKind.UNIT and not spec.needs_units
            n_needs += spec.needs_units
    logger.info(
        "generate_unit_specs: %d unit conversions, %d identity, %d needs-units residual", n_unit, n_identity, n_needs
    )
    return records


# ── N2: arithmetic formula verify harness (deterministic, $0) ─────────────────

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
# whitelisted math funcs — no builtins, no attribute access, no data-distribution functions.
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
}


class FormulaError(ValueError):
    """A formula that can't be safely parsed/evaluated (disallowed syntax, unknown name, bad call)."""


def _eval_node(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaError(f"non-numeric constant: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise FormulaError(f"unknown variable: {node.id}")
        return float(env[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return float(_BIN_OPS[type(node.op)](_eval_node(node.left, env), _eval_node(node.right, env)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand, env)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise FormulaError("disallowed function call")
        return float(_FUNCS[node.func.id](*[_eval_node(a, env) for a in node.args]))
    raise FormulaError(f"disallowed syntax: {type(node).__name__}")


def eval_formula(formula: str, env: dict[str, float]) -> float:
    """Safely evaluate an arithmetic ``formula`` over named inputs ``env`` (no builtins, whitelisted math).

    Allowed: ``+ - * / // % **``, unary ``+``/``-``, numeric literals, the input names present in ``env``,
    and a small whitelist of math functions (sqrt/log/exp/abs/round/min/max/floor/ceil). Anything else —
    attribute access, calls to other names, comprehensions, etc. — raises :class:`FormulaError`.
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"unparseable formula: {formula!r}") from e
    return _eval_node(tree, env)


def formula_names(formula: str) -> set[str]:
    """The set of variable names a formula references (empty set if it doesn't parse)."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in _FUNCS}


def is_safe_formula(formula: str, input_names: list[str]) -> bool:
    """True iff ``formula`` parses + evaluates under the safe evaluator (arithmetic edge cases aside)."""
    if not formula or not formula.strip():
        return False
    try:
        eval_formula(formula, dict.fromkeys(input_names, 1.0))
    except FormulaError:
        return False
    except ArithmeticError:
        pass  # structurally valid; a dummy-input divide-by-zero / domain error is not "unsafe"
    return True


def is_identity_formula(formula: str) -> bool:
    """True iff ``formula`` is a numeric no-op in ``source`` alone (target == source, unchanged) — M8.

    ``"source"``, ``"source * 1"``, ``"source + 0"``, ``"(source)"`` all qualify; a formula referencing any
    input other than ``source`` cannot be a single-edge identity and is rejected. Probed at several distinct
    points so a formula that merely coincides with the identity at one value is not misclassified. Used to
    downgrade an LLM-proposed no-op ARITHMETIC spec to a deterministic IDENTITY (no review row).
    """
    if not is_safe_formula(formula, ["source"]):
        return False
    if formula_names(formula) - {"source"}:
        return False
    try:
        return all(abs(eval_formula(formula, {"source": x}) - x) <= 1e-9 for x in (2.0, 3.0, 5.0))
    except (FormulaError, ArithmeticError):
        return False


def verify_formula(
    formula: str,
    cases: list[dict],
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> dict:
    """Deterministically grade ``formula`` against test ``cases`` — the N2 verification core.

    Each case is a dict of ``input_name -> value`` plus an ``"expected"`` key. The formula is evaluated
    per case and compared to ``expected`` with :func:`math.isclose`. Cases that raise (unknown name,
    arithmetic error) count as wrong. Returns ``{n, correct, accuracy, errors}``.
    """
    n = correct = errors = 0
    for case in cases:
        n += 1
        env = {k: float(v) for k, v in case.items() if k != "expected"}
        try:
            got = eval_formula(formula, env)
        except (FormulaError, ArithmeticError, ValueError, OverflowError):
            errors += 1
            continue
        if math.isclose(got, float(case["expected"]), rel_tol=rel_tol, abs_tol=abs_tol):
            correct += 1
    return {"n": n, "correct": correct, "accuracy": round(correct / n, 4) if n else 0.0, "errors": errors}


# ── N2: arithmetic-formula spec generation (LLM proposal -> upgrade N1 residual) ──

SYS_ARITH = (
    "You decide whether a TARGET numeric variable can be DETERMINISTICALLY derived from a SOURCE numeric "
    "variable by a FIXED arithmetic formula — one that does NOT depend on the data distribution (no "
    "quantiles, percentiles, z-scores, ranks, or dataset statistics). You are given the source variable "
    "(name, units, data type) and the target CDE (name, units, data type). If a fixed formula expresses "
    "the target as a function of the variable named `source` (e.g. 'source / 12', 'source * 0.01', "
    "'source * 2.20462'), return it using only + - * / ** and the name `source`. If the relationship needs "
    "the data distribution, or the two are not numerically derivable from one another, return formula null. "
    "Return a confidence in [0,1]."
)
ARITH_SCHEMA = json.dumps(
    {"formula": "<expression in terms of source, or null>", "confidence": "<0.0-1.0>", "notes": "<short rationale>"}
)


def _field_meta(fld: Field | None) -> str:
    bits: list[str] = []
    if fld and fld.data_type and fld.data_type.strip():
        bits.append(f"type {fld.data_type.strip()}")
    if fld and fld.units and fld.units.strip():
        bits.append(f"units {fld.units.strip()}")
    return "; ".join(bits) or "no units/type declared"


def build_arith_user_prompt(src_fld: Field | None, src_var: str, cde_id: str, cde_fld: Field | None) -> str:
    src_name = (src_fld.question_text or src_fld.short_label or src_var) if src_fld else src_var
    cde_name = (cde_fld.question_text or cde_fld.variable_name or cde_id) if cde_fld else cde_id
    return (
        f"SOURCE variable `source` = {src_name} [{_field_meta(src_fld)}]\n"
        f"TARGET CDE {cde_id} = {cde_name} [{_field_meta(cde_fld)}]\n\n"
        "Return a fixed arithmetic formula for the target as a function of `source`, or null."
    )


def prepare_arith_specgen(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    cde_fields: dict[str, Field],
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
) -> list[PromptRecord]:
    """N2: one arithmetic-formula prompt per numeric RESIDUAL edge (a ``needs_units`` UNIT spec from N1).

    Only edges N1 could not convert deterministically are sent to the LLM — a known unit conversion needs
    no formula. Must run AFTER :func:`generate_unit_specs` (it reads the residuals it leaves).
    """
    field_lookup = build_field_lookup(embedded_dicts)
    out: list[PromptRecord] = []
    for rec in records:
        if rec.verdict not in ("adopt", "refine") or not rec.cde_id:
            continue
        cde_fld = cde_fields.get(rec.cde_id)
        key = rec.group_id or rec.cluster_id
        for i, t in enumerate(rec.transforms):
            if t.kind != TransformKind.UNIT or not t.needs_units:
                continue
            cohort, _, var = t.source_variable.partition(":")
            src_fld = field_lookup.get((cohort, var))
            out.append(
                PromptRecord(
                    id=f"leanb:arith:{key}:{i}",
                    system_prompt=SYS_ARITH,
                    user_prompt=build_arith_user_prompt(src_fld, var, rec.cde_id, cde_fld),
                    schema=ARITH_SCHEMA,
                    model_tag=model_tag,
                    context={"record_key": key, "source_variable": t.source_variable},
                )
            )
    logger.info("prepare_arith_specgen: %d records -> %d arithmetic prompts", len(records), len(out))
    return out


def _parse_obj(resp: object) -> dict:
    if resp is None:
        return {}
    try:
        payload = extract_json(resp if isinstance(resp, str) else json.dumps(resp))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def assemble_arith_specgen(
    arith_records: list[PromptRecord],
    responses: dict[str, object],
    records: list[LeanBRecord],
    *,
    config: TransformConfidenceConfig | None = None,
) -> list[LeanBRecord]:
    """Upgrade N1 ``needs_units`` UNIT residuals to ARITHMETIC specs from the LLM formula responses.

    A null / empty / unsafe formula leaves the residual UNIT spec untouched. A safe single-input formula
    (in terms of ``source``) becomes an ARITHMETIC spec replacing the residual. A formula referencing
    inputs other than ``source`` cannot be bound from a single edge at the metadata layer, so it is kept
    ARITHMETIC but flagged ``needs_data`` + ``needs_review`` (the N3-adjacent case). The proposed formula
    is unverified here (no row data), so it always routes to review — never auto-approve.
    """
    by_key = {(r.group_id or r.cluster_id): r for r in records}
    for ar in arith_records:
        ctx = ar.context
        record_key = ctx.get("record_key")
        rec = by_key.get(record_key) if isinstance(record_key, str) else None
        sv = ctx.get("source_variable")
        if rec is None or not isinstance(sv, str):
            continue
        payload = _parse_obj(responses.get(ar.id))
        formula = str(payload.get("formula") or "").strip()
        if not formula or formula.lower() == "null":
            continue  # no fixed formula -> keep the needs_units residual
        names = formula_names(formula)
        if not is_safe_formula(formula, sorted(names) or ["source"]):
            continue  # unparseable / disallowed -> keep the residual, don't author a bad spec
        if is_identity_formula(formula):
            # M8: the LLM returned a no-op formula (target == source, e.g. "source"). That is not an
            # arithmetic conversion — emit a deterministic IDENTITY spec (no review) and drop the
            # needs_units residual, instead of a needs_review ARITHMETIC row.
            identity = TransformSpec(
                source_variable=sv,
                target_cde_id=rec.cde_id or "",
                kind=TransformKind.IDENTITY,
                inputs=["source"],
                generated_by="rule",
                needs_review=False,
                rationale="source maps to the target unchanged (LLM formula was a no-op) -> identity",
            )
            identity.confidence = round(score_transform_spec(identity, config), 3)
            rec.transforms = [
                t
                for t in rec.transforms
                if not (t.source_variable == sv and t.kind == TransformKind.UNIT and t.needs_units)
            ]
            rec.transforms.append(identity)
            continue
        extra = sorted(n for n in names if n != "source")
        llm_conf = _as_float(payload.get("confidence"))
        note = str(payload.get("notes", "") or "")
        if extra:
            note = (note + " " if note else "") + f"references inputs beyond `source` ({', '.join(extra)})"
        spec = TransformSpec(
            source_variable=sv,
            target_cde_id=rec.cde_id or "",
            kind=TransformKind.ARITHMETIC,
            formula=formula,
            inputs=sorted(names) or ["source"],
            generated_by="llm",
            needs_data=bool(extra),
            needs_review=True,  # LLM-proposed, unverified at the metadata layer
            rationale=(note + f" (llm_confidence {llm_conf:.2f})").strip(),
        )
        spec.confidence = round(score_transform_spec(spec, config), 3)
        # replace the residual needs_units UNIT spec for this edge
        rec.transforms = [
            t
            for t in rec.transforms
            if not (t.source_variable == sv and t.kind == TransformKind.UNIT and t.needs_units)
        ]
        rec.transforms.append(spec)
    return records


# ── GenCDE numeric specs: N1/N2 for the tail (member -> NUMERIC GenCDE) ───────


def generate_gencde_unit_specs(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    *,
    config: TransformConfidenceConfig | None = None,
) -> list[LeanBRecord]:
    """N1 for the tail: deterministic unit/scale specs for NUMERIC edges of NUMERIC-GenCDE novel records.

    Mirrors :func:`generate_unit_specs` but the target unit comes from the record's
    :class:`~.models.GenCDE` (``units``) instead of a catalog CDE, and each spec's ``target_cde_id`` is the
    GenCDE id — so the numeric novel path carries member->GenCDE unit conversions just like adopt/refine
    carries member->CDE ones. A GenCDE is "numeric" here when it has no ``permissible_values`` (a categorical
    GenCDE is the C1 recode path, :func:`prepare_gencde_specgen`); coded source edges and wide->long records
    are skipped. Deterministic, data-free, idempotent per (record, source_var).
    """
    field_lookup = build_field_lookup(embedded_dicts)
    canon = UnitCanonicalizer()
    n_unit = n_identity = n_needs = 0
    for rec in records:
        g = rec.gencde
        if g is None or _gencde_value_set_text(g):
            continue  # no GenCDE, or a categorical GenCDE -> C1 recode path, not a unit conversion
        if any(t.kind == TransformKind.WIDE_TO_LONG for t in rec.transforms):
            continue
        gencde_unit = (g.units or "").strip() if g.units else ""
        existing = {t.source_variable for t in rec.transforms}
        for sv in rec.member_variable_names:
            if sv in existing:
                continue
            cohort, _, var = sv.partition(":")
            src_fld = field_lookup.get((cohort, var))
            if src_fld is None or _value_set_text(src_fld):
                continue  # missing field, or a coded source edge (no numeric conversion into a numeric target)
            src_unit = (src_fld.units or "").strip() if src_fld.units else ""
            spec = _unit_spec(sv, g.gencde_id, src_unit, gencde_unit, canon)
            spec.confidence = round(score_transform_spec(spec, config), 3)
            rec.transforms.append(spec)
            n_identity += spec.kind == TransformKind.IDENTITY
            n_unit += spec.kind == TransformKind.UNIT and not spec.needs_units
            n_needs += spec.needs_units
    logger.info(
        "generate_gencde_unit_specs: %d unit conversions, %d identity, %d needs-units residual",
        n_unit,
        n_identity,
        n_needs,
    )
    return records


def build_gencde_arith_user_prompt(src_fld: Field | None, src_var: str, gencde: object) -> str:
    """Arith prompt with the TARGET drawn from a numeric GenCDE (mirrors :func:`build_arith_user_prompt`)."""
    src_name = (src_fld.question_text or src_fld.short_label or src_var) if src_fld else src_var
    tgt_name = (
        getattr(gencde, "question_text", "")
        or getattr(gencde, "title", "")
        or getattr(gencde, "preferred_name", "")
        or getattr(gencde, "gencde_id", "")
    )
    bits: list[str] = []
    dtype = (getattr(gencde, "data_type", "") or "").strip()
    units = (getattr(gencde, "units", "") or "").strip()
    if dtype:
        bits.append(f"type {dtype}")
    if units:
        bits.append(f"units {units}")
    tgt_meta = "; ".join(bits) or "no units/type declared"
    gencde_id = getattr(gencde, "gencde_id", "")
    return (
        f"SOURCE variable `source` = {src_name} [{_field_meta(src_fld)}]\n"
        f"TARGET CDE {gencde_id} = {tgt_name} [{tgt_meta}]\n\n"
        "Return a fixed arithmetic formula for the target as a function of `source`, or null."
    )


def prepare_gencde_arith_specgen(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
) -> list[PromptRecord]:
    """N2 for the tail: one arithmetic-formula prompt per numeric-GenCDE ``needs_units`` residual edge.

    Mirrors :func:`prepare_arith_specgen` but only upgrades the residuals left by
    :func:`generate_gencde_unit_specs` (``UNIT`` + ``needs_units`` whose ``target_cde_id`` is the record's
    GenCDE id). Must run AFTER the numeric-GenCDE N1 pass.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    out: list[PromptRecord] = []
    for rec in records:
        g = rec.gencde
        if g is None or _gencde_value_set_text(g):
            continue
        key = rec.group_id or rec.cluster_id
        for i, t in enumerate(rec.transforms):
            if t.kind != TransformKind.UNIT or not t.needs_units or t.target_cde_id != g.gencde_id:
                continue
            cohort, _, var = t.source_variable.partition(":")
            src_fld = field_lookup.get((cohort, var))
            out.append(
                PromptRecord(
                    id=f"leanb:gencde_arith:{key}:{i}",
                    system_prompt=SYS_ARITH,
                    user_prompt=build_gencde_arith_user_prompt(src_fld, var, g),
                    schema=ARITH_SCHEMA,
                    model_tag=model_tag,
                    context={"record_key": key, "source_variable": t.source_variable},
                )
            )
    logger.info("prepare_gencde_arith_specgen: %d records -> %d arithmetic prompts", len(records), len(out))
    return out


def assemble_gencde_arith_specgen(
    arith_records: list[PromptRecord],
    responses: dict[str, object],
    records: list[LeanBRecord],
    *,
    config: TransformConfidenceConfig | None = None,
) -> list[LeanBRecord]:
    """Upgrade numeric-GenCDE ``needs_units`` residuals to ARITHMETIC specs from the LLM formula responses.

    Identical to :func:`assemble_arith_specgen` except the spec's ``target_cde_id`` is the record's GenCDE id
    (a ``novel`` record has ``cde_id`` None). Every LLM-proposed formula is unverified at the metadata layer,
    so an ARITHMETIC spec always routes to review — never auto-approve.
    """
    by_key = {(r.group_id or r.cluster_id): r for r in records}
    for ar in arith_records:
        ctx = ar.context
        record_key = ctx.get("record_key")
        rec = by_key.get(record_key) if isinstance(record_key, str) else None
        sv = ctx.get("source_variable")
        if rec is None or rec.gencde is None or not isinstance(sv, str):
            continue
        gencde_id = rec.gencde.gencde_id
        payload = _parse_obj(responses.get(ar.id))
        formula = str(payload.get("formula") or "").strip()
        if not formula or formula.lower() == "null":
            continue  # no fixed formula -> keep the needs_units residual
        names = formula_names(formula)
        if not is_safe_formula(formula, sorted(names) or ["source"]):
            continue  # unparseable / disallowed -> keep the residual, don't author a bad spec
        if is_identity_formula(formula):
            # M8: a no-op formula (target == source) is not an arithmetic conversion -> deterministic IDENTITY.
            identity = TransformSpec(
                source_variable=sv,
                target_cde_id=gencde_id,
                kind=TransformKind.IDENTITY,
                inputs=["source"],
                generated_by="rule",
                needs_review=False,
                rationale="source maps to the GenCDE unchanged (LLM formula was a no-op) -> identity",
            )
            identity.confidence = round(score_transform_spec(identity, config), 3)
            rec.transforms = [
                t
                for t in rec.transforms
                if not (
                    t.source_variable == sv
                    and t.kind == TransformKind.UNIT
                    and t.needs_units
                    and t.target_cde_id == gencde_id
                )
            ]
            rec.transforms.append(identity)
            continue
        extra = sorted(n for n in names if n != "source")
        llm_conf = _as_float(payload.get("confidence"))
        note = str(payload.get("notes", "") or "")
        if extra:
            note = (note + " " if note else "") + f"references inputs beyond `source` ({', '.join(extra)})"
        spec = TransformSpec(
            source_variable=sv,
            target_cde_id=gencde_id,
            kind=TransformKind.ARITHMETIC,
            formula=formula,
            inputs=sorted(names) or ["source"],
            generated_by="llm",
            needs_data=bool(extra),
            needs_review=True,  # LLM-proposed, unverified at the metadata layer
            rationale=(note + f" (llm_confidence {llm_conf:.2f})").strip(),
        )
        spec.confidence = round(score_transform_spec(spec, config), 3)
        rec.transforms = [
            t
            for t in rec.transforms
            if not (
                t.source_variable == sv
                and t.kind == TransformKind.UNIT
                and t.needs_units
                and t.target_cde_id == gencde_id
            )
        ]
        rec.transforms.append(spec)
    return records


# ── M3: NONE-fraction coherence gate (deterministic, $0) ─────────────────────


def apply_coherence_gate(
    records: list[LeanBRecord],
    *,
    none_fraction_tau: float = 0.5,
    min_coded_edges: int = 3,
    demote: bool = True,
) -> list[LeanBRecord]:
    """M3 — flag/demote an adopt/refine record whose CODED edges are mostly unmappable (deterministic).

    A record whose categorical edges are MOSTLY ``NONE`` (no recode could be authored) is an over-broad
    match: the CDE was force-fit onto a heterogeneous group whose values don't recode to it — the giant-blob
    symptom (671 NONE edges across 47 records in the full-5 audit). This gate sets ``coherence_gap=True`` and,
    when ``demote`` is set, demotes an ``adopt`` to ``refine`` (never to novel — that would discard the
    candidate; the structural re-split is M2). ``route`` is unchanged.

    Only fires with ``>= min_coded_edges`` categorical edges, so it reflects RECORD-level incoherence (a
    blob), not a single hard edge already flagged by its spec's ``needs_review``. Reads only the C1 specs
    already attached; a no-op for numeric/structural records or when spec-gen did not run. The thresholds are
    tunable diagnostics, not hard rejects — the verdict is only ever softened, never hardened to novel.
    """
    coded_kinds = {TransformKind.CATEGORICAL, TransformKind.IDENTITY, TransformKind.NONE}
    n_gated = 0
    for rec in records:
        if rec.verdict not in ("adopt", "refine"):
            continue
        coded = [t for t in rec.transforms if t.kind in coded_kinds]
        if len(coded) < min_coded_edges:
            continue
        none_frac = sum(t.kind == TransformKind.NONE for t in coded) / len(coded)
        if none_frac < none_fraction_tau:
            continue
        rec.coherence_gap = True
        note = (
            f"coherence gate: {round(none_frac * 100)}% of {len(coded)} coded edges unmappable (NONE) — "
            "likely an over-broad match; review / re-split"
        )
        rec.rationale = (rec.rationale + " " if rec.rationale else "") + f"[{note}]"
        if demote and rec.verdict == "adopt":
            rec.verdict = "refine"  # don't auto-adopt an incoherent match; route stays 'assigned'
        n_gated += 1
    logger.info(
        "apply_coherence_gate: %d records flagged (tau=%.2f, min_edges=%d)", n_gated, none_fraction_tau, min_coded_edges
    )
    return records


# ── M7: concept-match gate (opt-in LLM stage) ─────────────────────────────────────────────────────
# The safety net for the failure M3 (coherence gate) is BLIND to: a recode with FULL coverage onto the WRONG
# concept. Two variables that share an answer FORMAT (both 1-5 Likert) recode 1:1 -> coverage 1.0 -> looks
# high-confidence even when the QUESTIONS differ ("confident filling out medical forms" vs "I felt happy",
# audit-observed @100%). M3 keys on the NONE fraction, so a 0-NONE wrong recode sails through. This gate asks
# the LLM whether the assigned CDE measures the SAME concept as the source (judging the QUESTION, not the
# answer format) and FLAGS mismatches (concept_mismatch=True + needs_review on the record's recodes). It never
# flips the verdict — the human decides — the same flag-not-gate discipline as M3.
SYS_CONCEPT_GATE = (
    "You verify that a source survey/clinical variable was assigned to the RIGHT Common Data Element (CDE). "
    "You are given (1) the SOURCE concept: a label, sample source fields (variable name + question), and an "
    "independently-written ideal CDE description; and (2) the ASSIGNED CDE: its designation + "
    "question/definition. Decide whether the CDE measures the SAME underlying concept as the source. Judge the "
    "QUESTION / measurement, NOT the answer format — a shared response scale (e.g. both are 1-5 agree/disagree "
    "Likerts) is NOT a match. Answer match=false when the CDE asks about a different thing than the source "
    "(e.g. source 'How confident are you filling out medical forms?' vs CDE 'I felt happy' — both agree-scales "
    "but different concepts). Be strict about the object/topic; tolerate wording or granularity differences "
    "that keep the same concept (a refine is expected to specialize). Return JSON only."
)
CONCEPT_GATE_SCHEMA = json.dumps({"match": "true|false", "reason": "<one sentence>"})


def _concept_gate_source_lines(rec: LeanBRecord, field_lookup: dict, cap: int = 4) -> list[str]:
    """Sample source fields as ``name — question`` lines (M7 needs the source NAME + question, not just codes)."""
    lines: list[str] = []
    for sv in rec.member_variable_names[:cap]:
        cohort, _, var = sv.partition(":")
        fld = field_lookup.get((cohort, var))
        q = (fld.question_text or fld.description or fld.short_label or "").strip() if fld is not None else ""
        lines.append(f"{var} — {q[:120]}" if q else var)
    return lines


def build_concept_gate_user_prompt(
    concept: str, ideal_cde: str, source_lines: list[str], cde_id: str, cde_concept: str
) -> str:
    """Render the M7 concept-match prompt for one assigned record."""
    return "\n".join(
        p
        for p in [
            f"SOURCE concept: {concept[:160]}",
            f"SOURCE ideal CDE: {ideal_cde[:200]}" if ideal_cde else "",
            f"SOURCE fields: {('; '.join(source_lines))[:400]}" if source_lines else "",
            "",
            f"ASSIGNED CDE: {cde_id}",
            f"CDE concept: {cde_concept[:300]}",
            "",
            "Do they measure the same concept? Judge the question, not the answer format. Return JSON.",
        ]
        if p or p == ""
    )


def prepare_concept_gate(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    cde_fields: dict[str, Field],
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
) -> list[PromptRecord]:
    """M7 — one concept-match prompt per adopt/refine record that has an assigned CDE.

    Content-addressed by ``(record key, cde_id)`` so it's stable across frozen-substrate runs. The prompt
    shows the source concept (label + ideal + member name/question samples) and the assigned CDE's
    question/definition; :func:`assemble_concept_gate` flags the records the LLM judges concept-mismatched.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    out: list[PromptRecord] = []
    for rec in records:
        if rec.verdict not in ("adopt", "refine") or not rec.cde_id:
            continue
        cde_fld = cde_fields.get(rec.cde_id)
        cde_concept = ""
        if cde_fld is not None:
            cde_concept = " ".join(p for p in (cde_fld.question_text, cde_fld.description) if p and p.strip())
        key = rec.group_id or rec.cluster_id
        out.append(
            PromptRecord(
                id=f"leanb:conceptgate:{content_token(key, rec.cde_id)}",
                system_prompt=SYS_CONCEPT_GATE,
                user_prompt=build_concept_gate_user_prompt(
                    rec.concept or rec.ideal_cde,
                    rec.ideal_cde,
                    _concept_gate_source_lines(rec, field_lookup),
                    rec.cde_id,
                    cde_concept or rec.cde_id,
                ),
                schema=CONCEPT_GATE_SCHEMA,
                model_tag=model_tag,
                context={"record_key": key, "cde_id": rec.cde_id},
            )
        )
    logger.info("prepare_concept_gate: %d adopt/refine records -> %d concept-match prompts", len(records), len(out))
    return out


def _parse_concept_gate(resp: object) -> dict | None:
    if resp is None:
        return None
    try:
        payload = extract_json(resp if isinstance(resp, str) else json.dumps(resp))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def assemble_concept_gate(
    gate_records: list[PromptRecord], responses: dict[str, object], records: list[LeanBRecord]
) -> list[LeanBRecord]:
    """M7 — flag records whose assigned CDE fails the concept-match check. Never flips the verdict.

    Sets ``concept_mismatch=True`` and ``needs_review`` on the record's transforms so the recodes route to
    EITL. An unparseable/absent response leaves the record untouched (no spurious flag).
    """
    by_key = {(r.group_id or r.cluster_id): r for r in records}
    n_flagged = 0
    for gr in gate_records:
        rec = by_key.get(gr.context["record_key"])
        if rec is None:
            continue
        payload = _parse_concept_gate(responses.get(gr.id))
        if payload is None:
            continue
        match = payload.get("match")
        is_match = match if isinstance(match, bool) else str(match).strip().lower() in ("true", "yes", "1")
        if is_match:
            continue
        rec.concept_mismatch = True
        reason = str(payload.get("reason", "")).strip()
        note = "concept-gate: assigned CDE may not match the source concept" + (f" — {reason}" if reason else "")
        rec.rationale = (rec.rationale + " " if rec.rationale else "") + f"[{note}]"
        for t in rec.transforms:
            t.needs_review = True
        n_flagged += 1
    logger.info("assemble_concept_gate: %d records flagged concept_mismatch", n_flagged)
    return records


_MAP_LINE_CAP = 16  # cap recode pairs shown in one cell (long code sets)
_VALUE_OPTS_CAP = 14  # cap response options shown per side


def _options(fld: Field | None) -> list[ResponseOption]:
    """Response options for a field — the parsed list, else parsed from the raw encoding string."""
    if fld is None:
        return []
    if fld.response_options:
        return fld.response_options
    return parse_value_encoding(fld.value_encoding_raw or "")


def _label_map(fld: Field | None) -> dict[str, str]:
    return {ro.code: ro.label for ro in _options(fld)}


def _options_text(fld: Field | None) -> str:
    """'code=label; code=label' for a field's response options (single line, capped)."""
    return "; ".join(f"{ro.code}={ro.label}" for ro in _options(fld)[:_VALUE_OPTS_CAP])


def _num_source_meta(t: TransformSpec, src_fld: Field | None) -> str:
    """Units / data-type summary for a numeric source edge (shown in source_text instead of value codes)."""
    unit = t.source_unit or (src_fld.units if src_fld and src_fld.units else "") or ""
    dtype = (src_fld.data_type if src_fld and src_fld.data_type else "") or ""
    parts = [p for p in (unit, f"({dtype})" if dtype else "") if p]
    return " ".join(parts) or "(no units declared)"


def _conv_line(factor: float, offset: float) -> str:
    """Human-readable linear unit conversion: 'target = source * factor [+/- offset]'."""
    s = f"target = source * {factor:g}"
    if offset:
        s += f" + {offset:g}" if offset > 0 else f" - {abs(offset):g}"
    return s


def export_transform_review(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    cde_fields: dict[str, Field],
    path: str | Path,
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
) -> int:
    """Export the transform specs as an EITL ``transform_review`` campaign (importable pair contract).

    Emits the A->B fields the EITL importer enforces — ``source_text``/``source_id``/``source_dataset`` /
    ``target_text``/``target_id``/``target_dataset``/``pair_type`` (+ ``llm_reasoning``/``llm_model``/
    ``resolution_layer``) — one row per non-identity transform edge (source var -> CDE). Rendering is
    kind-aware:

    - **categorical (C1)**: the proposed ``source-code -> canonical-code`` mapping renders inline in
      ``target_text`` (labels from both value sets); ``pair_type = value_map/{verdict}``.
    - **unit (N1)**: the linear conversion ``target = source * factor + offset`` + units;
      ``pair_type = unit_convert/{verdict}``. A ``needs_units`` residual renders "none authored".
    - **arithmetic (N2)**: the proposed formula; ``pair_type = arithmetic/{verdict}``.
    - **wide→long (structural)**: N numbered occurrence columns as ONE repeated field; one row for the whole
      group (anchored on a representative member); ``pair_type = wide_to_long/{verdict}``.

    Binary review is "is this transform correct?"; the EITL showAlternatives box turns a rejection into a
    corrected mapping/conversion/formula. Identity (no-op) specs are skipped. Uses the eitl encoding
    contract (labeled/pack/write_csv: U+2028 breaks, QUOTE_ALL, no raw CR/LF). Returns the row count.

    Per-source-encoding dedup (family collapse) is a later refinement.
    """
    field_lookup = build_field_lookup(embedded_dicts)
    rows: list[dict] = []
    for rec in records:
        tiny = rec.cde_external_id or ""
        cde_fld = cde_fields.get(rec.cde_id or "")
        cde_opts = _options_text(cde_fld)
        cde_lbl = _label_map(cde_fld)
        for t in rec.transforms:
            if t.kind == TransformKind.IDENTITY:
                continue  # source already aligned — nothing to review
            cohort, _, var = t.source_variable.partition(":")
            src_fld = field_lookup.get((cohort, var))
            base = f"The variable->CDE match ({rec.verdict}) is assumed approved."
            cov_str = ""

            if t.kind == TransformKind.WIDE_TO_LONG:
                rng = t.params.get("int_range") or []
                span = f"occurrences {rng[0]}..{rng[1]}" if len(rng) == 2 else f"{len(t.inputs)} occurrences"
                source_extra = ("Current structure", f"WIDE — {len(t.inputs)} numbered columns ({span})")
                proposal = (
                    "Proposed reshape",
                    "wide → long: collapse the numbered columns into ONE repeated field with an occurrence index",
                )
                canonical_ref = ("Matched CDE", rec.cde_id or "")
                reasoning = [
                    base,
                    f"These {len(t.inputs)} numbered columns are ONE repeating measure, not "
                    f"{len(t.inputs)} distinct variables.",
                    "Confirm the wide→long reshape (occurrence index → array position).",
                ]
                pair_kind = "wide_to_long"
            elif t.kind == TransformKind.UNIT:
                source_extra = ("Source units / type", _num_source_meta(t, src_fld))
                canonical_ref = (
                    "Canonical units",
                    t.target_unit or (cde_fld.units if cde_fld else None) or "(unspecified)",
                )
                if t.factor is not None:
                    proposal = (
                        "Proposed unit conversion",
                        f"{_conv_line(t.factor, t.offset or 0.0)}  [{t.source_unit or '?'} -> {t.target_unit or '?'}]",
                    )
                    reasoning = [
                        base,
                        "Confirm the UNIT conversion: does source * factor + offset yield the canonical units?",
                    ]
                else:  # needs_units residual (no deterministic conversion authored)
                    proposal = (
                        "Proposed unit conversion",
                        "(none authored — source/target units missing or unreconcilable)",
                    )
                    reasoning = [
                        base,
                        "Source/target units could not be reconciled: supply the conversion or confirm none is needed.",
                    ]
                pair_kind = "unit_convert"
            elif t.kind == TransformKind.ARITHMETIC:
                source_extra = ("Source units / type", _num_source_meta(t, src_fld))
                canonical_ref = (
                    "Canonical units",
                    t.target_unit or (cde_fld.units if cde_fld else None) or "(unspecified)",
                )
                proposal = ("Proposed derivation formula", f"target = {t.formula}")
                reasoning = [
                    base,
                    "Confirm the DERIVATION: does the formula compute the canonical value from the source?",
                ]
                if t.needs_data:
                    reasoning.append(
                        "formula references inputs beyond the source variable — needs multi-field binding / row data."
                    )
                pair_kind = "arithmetic"
            else:  # categorical (or a NONE residual whose mapping could not be authored)
                src_lbl = _label_map(src_fld)
                map_line = "; ".join(
                    f"{sc}={src_lbl.get(sc, '?')}→{tc}={cde_lbl.get(tc, '?')}"
                    for sc, tc in sorted(t.code_map.items())[:_MAP_LINE_CAP]
                )
                source_extra = ("Source response values", _options_text(src_fld))
                proposal = ("Proposed value mapping (source→canonical)", map_line)
                canonical_ref = ("Canonical value set", cde_opts)
                reasoning = [
                    base,
                    "Confirm the VALUE recode: does each source response code map to the correct canonical code?",
                    f"coverage {t.coverage:.0%}",
                ]
                if t.unmapped_source_codes:
                    reasoning.append("unmapped source codes: " + ", ".join(t.unmapped_source_codes))
                if not t.code_map:
                    reasoning.append("the generator could not produce a mapping -- please supply it.")
                pair_kind = "value_map"
                cov_str = f"{t.coverage:.2f}"

            if t.needs_review:
                reasoning.append("flagged NEEDS REVIEW")
            if t.rationale:
                reasoning.append(t.rationale)
            rows.append(
                {
                    "source_text": labeled([("Variable name", var), ("Question text", qtext(src_fld)), source_extra]),
                    "source_id": t.source_variable,
                    "source_dataset": cohort,
                    "target_text": labeled([("Approved match", rec.cde_id or ""), proposal, canonical_ref]),
                    "target_id": tiny,
                    "target_dataset": "NIH_CDE",
                    "target_url": cde_url(tiny),
                    "pair_type": f"{pair_kind}/{rec.verdict}",
                    "llm_model": model_tag,
                    "llm_reasoning": pack(reasoning),
                    "resolution_layer": "ai_assisted",
                    "transform_kind": str(t.kind),
                    "coverage": cov_str,
                }
            )
    write_csv(Path(path), rows)
    return len(rows)
