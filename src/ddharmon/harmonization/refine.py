"""Refinement authoring for the ``refine`` verdict — a GenCDE *derived from* a real CDE.

A ``refine`` says "this existing CDE is close, but not right for this concept group". Until now that
was the end of it: a refine took the identical code path to an ``adopt`` — routed ``assigned``, recodes
generated against the *unmodified* parent CDE, same review row — so the verdict's entire semantic
content (*what* about the CDE is wrong) was discarded into a free-text rationale. This module closes
that gap, the way :mod:`~ddharmon.harmonization.gencde` closed it for ``novel``:

    novel   ->  GenCDE synthesized FROM SCRATCH from the group's pooled empirics
    refine  ->  GenCDE DERIVED FROM the matched CDE: parent + a typed, minimal delta

Both produce the same artifact (:class:`~ddharmon.harmonization.models.GenCDE`), so they share one
review panel and one NIH-conformant serializer (:mod:`ddharmon.export.cde_json`); ``parent_cde_id``
distinguishes them. The refinement RELATION is an SSSOM/SKOS predicate because the NIH CDE model has
no "refines" pointer of its own (see the :class:`GenCDE` docstring for the measurement behind that).

Two things are deliberate:

**The axis is classified deterministically, the delta is authored by the LLM.** Everything needed to
say *which way* a parent CDE fails is already on the record — unmapped source codes, ``needs_units``
residuals, a wide->long spec, the M7 concept-match flag. So :func:`classify_refinement_axis` is a
$0, no-LLM, replayable function of the record alone (it does not even need the dictionaries), and the
LLM is only ever asked to author the delta it cannot derive. Stage 2 (R2) adds that authoring; this
module ships the triage plus the deltas that are genuinely computable without a model.

**A mis-assigned match is gated OUT, never refined.** On the held-out full-5 run, 304 of 667 refines
(46%) are already flagged ``concept_mismatch`` or carry no concept label at all. Authoring a
refinement for those would launder a bad match into a plausible-looking new element — the failure
mode is worse than leaving it alone, because the output *looks* curated. They route to
re-adjudication instead. (Same discipline as gating GenCDE spec-gen on group coherence: a recode into
an incoherent target is meaningless.)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ddharmon.embedding.service import EmbeddedDictionary
from ddharmon.harmonization.anchor import build_field_lookup

# Deliberate reuse of gencde's helpers: a refined element and a synthesized one are the same artifact,
# so they must reconcile answer concepts and score value coverage identically. (Cross-module private
# reuse follows the existing precedent — transform.py imports leanb._value_set_text the same way.)
from ddharmon.harmonization.gencde import (
    REVIEW_CONFIDENCE,
    REVIEW_COVERAGE,
    _label_coverage,
    _member_line,
    observed_answer_labels,
)
from ddharmon.harmonization.leanb import DEFAULT_MODEL_TAG, _value_set_text
from ddharmon.harmonization.models import (
    REFINEMENT_RELATIONS,
    RELATION_CLOSE,
    GenCDE,
    LeanBRecord,
    RefinementAxis,
    TransformKind,
)
from ddharmon.harmonization.parse import extract_json
from ddharmon.harmonization.pipeline import PromptRecord
from ddharmon.models.data_dictionary import Field, ResponseOption
from ddharmon.values.response_parser import parse_value_encoding
from ddharmon.values.units import UnitCanonicalizer, is_identity_conversion

logger = logging.getLogger(__name__)

# Why a refine record was not refined. Distinct from "no axis": these are records we deliberately
# decline to author for, and the reason is carried through to review.
GATE_MIS_ASSIGNED = "mis_assigned"  # M7 concept-mismatch / no concept / judged incoherent -> re-adjudicate
GATE_NO_EVIDENCE = "no_specs"  # no transform specs on the record -> nothing to reason from
GATE_NOT_REFINE = "not_refine"  # adopt / novel — the classifier is a no-op

# The parent-CDE slots a refinement can change. `delta_size` is the fraction of these the delta
# touches — the minimality signal. Kept small and explicit so the number means something concrete.
DELTA_SLOTS = ("preferred_name", "definition", "question_text", "data_type", "units", "permissible_values")
# Above this fraction the "delta" is a rewrite, not a refinement -> `over_refined` (the honest verdict
# would have been `novel`). 0.5 = more than half the element replaced. FLAG, never a gate.
OVER_REFINED_DELTA = 0.5


def _norm(s: str) -> str:
    """Normalize a label for comparison: collapse whitespace, lowercase, strip. (Mirrors gencde.)"""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


@dataclass
class RefineTriage:
    """The deterministic read on one ``refine`` record: which axis, or why we decline to author.

    ``refinable`` is the gate. When it is False, ``axis`` is None and ``reason`` is one of the
    ``GATE_*`` constants; when True, ``axis`` is the classified :class:`RefinementAxis` and
    ``evidence`` says what on the record implied it (carried into the prompt + the review row).
    """

    axis: RefinementAxis | None
    refinable: bool
    reason: str = ""
    evidence: str = ""


def classify_refinement_axis(rec: LeanBRecord) -> RefineTriage:
    """Classify WHY a ``refine`` record's parent CDE doesn't fit — deterministically, from the record.

    Reads only signals already computed upstream, so this is $0, needs no LLM, no dictionaries and no
    catalog, and can be replayed over a saved ``records.json`` to audit a run's triage.

    Order matters only for readability — the buckets are disjoint by construction: a wide->long record
    is claimed by :func:`~ddharmon.harmonization.transform.generate_wide_to_long_specs` before the C1
    and N1 generators run, so it carries no categorical or unit specs to be classified by.

    ``SCOPE`` is never returned here. Telling "the parent needs a qualifier" (QUALIFIER) from "the
    parent is narrower than the concept" (SCOPE) is a semantic judgment with no deterministic signal
    behind it, so the conceptual residual is labelled QUALIFIER — the dominant case in practice — and
    the LLM authoring stage may resolve it to SCOPE.
    """
    if rec.verdict != "refine":
        return RefineTriage(axis=None, refinable=False, reason=GATE_NOT_REFINE)

    # ── the gate: a match that is probably WRONG must not be dressed up as a refinement ──
    if rec.concept_mismatch:
        return RefineTriage(None, False, GATE_MIS_ASSIGNED, "M7 concept-match gate failed")
    if rec.incoherent:
        return RefineTriage(None, False, GATE_MIS_ASSIGNED, "coherence judge: group is over-merged (split)")
    if not (rec.concept or "").strip():
        return RefineTriage(None, False, GATE_MIS_ASSIGNED, "no concept label — the group never resolved")

    specs = rec.transforms
    if not specs:
        return RefineTriage(None, False, GATE_NO_EVIDENCE, "no transform specs to reason from")

    n_wide = sum(1 for t in specs if t.kind == TransformKind.WIDE_TO_LONG)
    if n_wide:
        occ = next((int(t.params.get("n_occurrences", 0) or 0) for t in specs if t.params), 0)
        return RefineTriage(
            RefinementAxis.STRUCTURAL, True, evidence=f"repeating measure ({occ or 'n'} numbered occurrences)"
        )

    unmapped = [c for t in specs for c in t.unmapped_source_codes]
    if unmapped:
        shown = ", ".join(unmapped[:6]) + ("…" if len(unmapped) > 6 else "")
        return RefineTriage(
            RefinementAxis.VALUE_DOMAIN, True, evidence=f"{len(unmapped)} source code(s) unmappable: {shown}"
        )

    if any(t.needs_units or (t.kind == TransformKind.UNIT and t.needs_review) for t in specs):
        pairs = {(t.source_unit or "", t.target_unit or "") for t in specs if t.needs_units or t.needs_review}
        declared = [(s, t) for s, t in pairs if s or t]
        if not declared:
            # The common case (66 of 82 on the full-5 run): numeric edges where NEITHER side declares a
            # unit. Say so — "? -> ?" reads like a failed lookup when it is really missing metadata.
            evidence = "numeric edges with no unit of measure declared on either side"
        else:
            rendered = "; ".join(f"{s or '(none)'} -> {t or '(none)'}" for s, t in sorted(declared)[:4])
            evidence = f"unit/scale unreconciled: {rendered}"
        return RefineTriage(RefinementAxis.REPRESENTATION, True, evidence=evidence)

    return RefineTriage(
        RefinementAxis.QUALIFIER, True, evidence="values reconcile — the gap is conceptual (qualifier or scope)"
    )


def triage_summary(records: list[LeanBRecord]) -> dict[str, int]:
    """Bucket counts over a run's records: one key per axis plus each ``GATE_*`` reason.

    The auditable view of what a run's ``refine`` bucket is actually made of. Only ``refine`` records
    are counted (``GATE_NOT_REFINE`` is omitted — adopts and novels are not this stage's business).
    """
    out: dict[str, int] = {}
    for rec in records:
        tri = classify_refinement_axis(rec)
        if tri.reason == GATE_NOT_REFINE:
            continue
        key = tri.axis.value if tri.axis is not None else tri.reason
        out[key] = out.get(key, 0) + 1
    return out


def refined_cde_id(rec: LeanBRecord) -> str:
    """Deterministic id for a refine-derived element (``REFCDE:`` — ``GENCDE:`` stays for from-scratch)."""
    return f"REFCDE:{rec.group_id or rec.cluster_id}"


def parent_values(parent: Field) -> list[ResponseOption]:
    """The parent CDE's permissible values, from its parsed options or its raw value encoding."""
    if parent.response_options:
        return list(parent.response_options)
    raw = (parent.value_encoding_raw or "").strip()
    return parse_value_encoding(raw) if raw else []


def compute_delta(refined: GenCDE, parent: Field) -> tuple[list[str], list[str], float, bool]:
    """Compare a refined element against its parent.

    Returns ``(changed_fields, completed_fields, delta_size, over_refined)``.

    The minimality check, and refine's analog of the GenCDE coverage check: a refinement is supposed to
    be a small, targeted edit. ``over_refined`` fires when the delta exceeds :data:`OVER_REFINED_DELTA`,
    or when the parent's value domain is replaced wholesale (no parent label survives) — either way the
    element is not really a refinement of that CDE and the honest verdict would have been ``novel``.

    **Filling an empty slot is not a change.** The public CDE catalog is sparsely populated: measured on
    the subset run, 90% of matched parents carry NO ``question_text`` and 37% no ``definition`` at all.
    An element that supplies one is COMPLETING absent metadata, not altering the concept — counting it
    as a delta inflated ``delta_size`` and made ``over_refined`` fire on elements that had changed
    nothing the parent actually asserted. Completions are reported separately (they are still worth a
    reviewer's eye) and excluded from ``delta_size``, so the minimality number means what it claims:
    the fraction of what the parent DID say that this element contradicts.

    A FLAG, never a gate: the caller records it and routes to review; the assign verdict stands
    (``C-TRANSFORM-SPEC-SCOPE.md`` §0a.4 — spec-gen verifies, it does not override).
    """
    changed: list[str] = []
    completed: list[str] = []

    def compare(slot: str, new: str | None, old: str | None) -> None:
        """Classify one text slot: absent-then-supplied = completion; asserted-then-differing = change."""
        new_t, old_t = _norm(new or ""), _norm(old or "")
        if not new_t or new_t == old_t:
            return
        (completed if not old_t else changed).append(slot)

    compare("preferred_name", refined.preferred_name, parent.variable_name)
    compare("definition", refined.definition, parent.description)
    compare("question_text", refined.question_text, parent.question_text)
    compare("data_type", refined.data_type, parent.data_type)
    compare("units", refined.units, parent.units)

    p_values = parent_values(parent)
    p_labels = {_norm(ro.label) for ro in p_values}
    r_labels = {_norm(ro.label) for ro in refined.permissible_values}
    values_touched = bool(refined.added_permissible_values or refined.relabeled_values or refined.deprecated_values)
    if values_touched or (p_labels and r_labels and p_labels != r_labels):
        # A parent with no value domain at all gets the same treatment: supplying one completes it.
        (changed if p_labels else completed).append("permissible_values")

    delta_size = round(len(changed) / len(DELTA_SLOTS), 3)
    # A wholly-replaced value domain is a rewrite even if it is the only slot touched: nothing the
    # parent could express survives, so the element no longer refines that CDE.
    domain_replaced = bool(p_labels) and bool(r_labels) and not (p_labels & r_labels)
    return changed, completed, delta_size, delta_size > OVER_REFINED_DELTA or domain_replaced


def _derived_from_parent(rec: LeanBRecord, parent: Field, axis: RefinementAxis) -> GenCDE:
    """A derived element pre-filled with the parent's metadata — the base every delta is applied to.

    Everything not changed by the delta is inherited verbatim, which is what makes the result a
    refinement rather than a re-authoring, and what lets :func:`compute_delta` measure the difference.
    """
    return GenCDE(
        gencde_id=refined_cde_id(rec),
        preferred_name=parent.variable_name,
        title=(parent.short_label or parent.variable_name),
        definition=parent.description,
        question_text=parent.question_text or "",
        data_type=parent.data_type or "",
        permissible_values=parent_values(parent),
        units=parent.units,
        source_variables=list(rec.member_variable_names),
        source_cohorts=list(rec.cohorts),
        ideal_seed=rec.ideal_cde,
        parent_cde_id=rec.cde_id,
        parent_cde_external_id=rec.cde_external_id,
        refinement_axis=axis.value,
        generated_by="rule",
    )


def build_deterministic_refinement(
    rec: LeanBRecord, parent: Field, tri: RefineTriage, *, canon: UnitCanonicalizer | None = None
) -> GenCDE | None:
    """Author a derived element for the cases needing NO model call; ``None`` when the LLM is required.

    Only two axes are computable without judgment, and only in their unambiguous sub-cases:

    - **REPRESENTATION** — the parent declares no unit and the source edges agree on exactly one
      *recognized* unit. Then the refinement is simply "declare that unit", which the curated table
      (:mod:`ddharmon.values.units`) can verify. Measured on the full-5 run this is 8 of 82
      representation records: most of the rest declare no units *anywhere* (nothing to compute) or are
      cross-family conversions the table deliberately excludes as analyte-specific.
    - **STRUCTURAL** — a wide->long record. The parent element is right; what is wrong is that the
      cohorts collect it as N numbered occurrences. That is a statement about the element's cardinality
      and it comes straight off the detector, so no model is needed to make it.

    VALUE_DOMAIN and QUALIFIER always need the LLM: reconciling answer concepts across cohorts, and
    naming a missing qualifier, are exactly the judgments a rule cannot make.
    """
    if not tri.refinable or tri.axis is None:
        return None
    canon = canon or UnitCanonicalizer()

    if tri.axis is RefinementAxis.REPRESENTATION:
        if (parent.units or "").strip():
            return None  # parent HAS a unit and N1 still couldn't reconcile -> cross-family, needs judgment
        src_units = {(t.source_unit or "").strip() for t in rec.transforms if (t.source_unit or "").strip()}
        known = {u for u in src_units if canon.canonical(u)}
        if len(known) != 1:
            return None  # no recognized unit, or the sources disagree -> not deterministic
        unit = known.pop()
        refined = _derived_from_parent(rec, parent, tri.axis)
        refined.units = unit
        refined.relation = RELATION_CLOSE  # same concept, its representation is what changes
        refined.rationale = (
            f"The parent declares no unit of measure; every numeric source edge is in '{unit}'. "
            "Declaring it makes the element's value domain checkable and the recode verifiable."
        )
        refined.confidence = 0.9  # deterministic + unit recognized by the curated table

    elif tri.axis is RefinementAxis.STRUCTURAL:
        spec = next((t for t in rec.transforms if t.kind == TransformKind.WIDE_TO_LONG), None)
        if spec is None:
            return None
        occ = int(spec.params.get("n_occurrences", 0) or 0) or len(spec.inputs)
        refined = _derived_from_parent(rec, parent, tri.axis)
        refined.qualifier_added = f"repeating measure ({occ} occurrences)"
        refined.definition = (
            f"{parent.description} Collected as a repeating measure across {occ} occurrences "
            "(the cohorts record it as numbered columns; see the wide-to-long transform spec)."
        ).strip()
        refined.relation = RELATION_CLOSE  # the concept is unchanged; its cardinality is not
        refined.rationale = spec.rationale
        refined.confidence = 0.6  # deterministic detection, but a structural claim -> always reviewed
        refined.needs_review = True

    else:
        return None  # VALUE_DOMAIN / QUALIFIER / SCOPE -> LLM authoring

    (
        refined.changed_fields,
        refined.completed_fields,
        refined.delta_size,
        refined.over_refined,
    ) = compute_delta(refined, parent)
    if refined.over_refined:
        refined.needs_review = True
    return refined


def apply_deterministic_refinements(
    records: list[LeanBRecord], cde_fields: dict[str, Field], *, canon: UnitCanonicalizer | None = None
) -> list[LeanBRecord]:
    """Attach every deterministically-authorable derived element to its record (``rec.gencde``).

    Runs with no LLM and no Batch round-trip, so it is unconditional and free. Idempotent: a record
    that already carries an element is left alone. Records needing model authoring are untouched here
    and picked up by the LLM stage; gated records are never authored at all.
    """
    canon = canon or UnitCanonicalizer()
    n_authored = 0
    gated = 0
    for rec in records:
        if rec.gencde is not None:
            continue  # idempotent — also protects a novel record's from-scratch GenCDE
        tri = classify_refinement_axis(rec)
        if tri.reason == GATE_NOT_REFINE:
            continue
        if not tri.refinable:
            gated += 1
            continue
        parent = cde_fields.get(rec.cde_id or "")
        if parent is None:
            continue
        refined = build_deterministic_refinement(rec, parent, tri, canon=canon)
        if refined is not None:
            rec.gencde = refined
            n_authored += 1
    logger.info(
        "apply_deterministic_refinements: %d derived elements authored ($0), %d refines gated as mis-assigned",
        n_authored,
        gated,
    )
    return records


# ── LLM authoring (Batch API) ─────────────────────────────────────────────────

_MEMBER_CAP = 12  # members shown in the prompt (cost bound; provenance keeps the full list) — as gencde
_PARENT_VALUES_TRUNC = 900  # the parent's value set can be long (country lists); cap what the prompt shows

SYS_REFINE = (
    "You are a biomedical Common Data Element (CDE) expert. You are given ONE existing CDE and a set of "
    "harmonized data-dictionary fields (from one or more cohorts) that all measure a single concept. The CDE "
    "was judged a CLOSE BUT IMPERFECT match for that concept.\n\n"
    "Author the SMALLEST change to the existing CDE that would make it correctly cover the concept — a "
    "refinement, not a replacement. Keep everything that already works: leave a field null unless that "
    "specific field is what is wrong. When extending the value set, PRESERVE the CDE's existing permissible "
    "values and their codes and add new ones alongside; never renumber or restate the existing values. "
    "Do not invent answer concepts that no source field expresses.\n\n"
    "State how your refined element relates to the original:\n"
    "  skos:narrowMatch  — you specialized it (added a qualifier: body site, method, person, laterality, "
    "time window)\n"
    "  skos:broadMatch   — you generalized it (the original was too narrow for the concept)\n"
    "  skos:closeMatch   — same concept; you changed its representation (unit, data type) or extended its "
    "value domain\n"
    "  skos:relatedMatch — related, but neither strictly narrower nor broader\n\n"
    "IMPORTANT: if making this CDE fit would require rewriting most of it — a different definition AND a "
    "different question AND a different value set — then it is NOT a refinement of this CDE, and the honest "
    'answer is that a new element is needed. Say so by setting "is_refinement": false with a brief reason, '
    "and do not force a rewrite into the delta. Returning false is a correct, expected answer.\n\n"
    "Return JSON only."
)

REFINE_SCHEMA = (
    '{"is_refinement": "true|false — false if this needs a new element, not a refinement", '
    '"relation": "skos:narrowMatch | skos:broadMatch | skos:closeMatch | skos:relatedMatch", '
    '"axis": "value_domain | qualifier | representation | structural | scope", '
    '"qualifier_added": "<the qualifier the CDE lacked, or null>", '
    '"definition_revised": "<revised definition, or null to keep the CDE\'s>", '
    '"question_text_revised": "<revised question text, or null to keep the CDE\'s>", '
    '"preferred_name_revised": "<revised variable name, or null to keep the CDE\'s>", '
    '"data_type_revised": "<numeric|categorical|binary|date|text, or null to keep the CDE\'s>", '
    '"units_revised": "<UCUM unit, or null to keep the CDE\'s>", '
    '"added_permissible_values": [{"code": "<new code>", "label": "<label>"}], '
    '"relabeled_values": {"<existing code>": "<clearer label>"}, '
    '"deprecated_values": ["<existing code the concept does not use>"], '
    '"confidence": "<0.0-1.0>", "notes": "<one-sentence rationale>"}'
)


def _parent_card(parent: Field) -> str:
    """Render the parent CDE for the prompt: what it says now, so the model can change only what's wrong."""
    lines = [f"Designation: {parent.variable_name}"]
    if parent.question_text:
        lines.append(f"Question text: {parent.question_text}")
    if parent.description:
        lines.append(f"Definition: {parent.description}")
    if parent.data_type:
        lines.append(f"Data type: {parent.data_type}")
    if parent.units:
        lines.append(f"Unit of measure: {parent.units}")
    values = _value_set_text(parent)
    lines.append(f"Permissible values: {values[:_PARENT_VALUES_TRUNC] if values else '(none — not a coded element)'}")
    return "\n".join(lines)


def build_refine_user_prompt(
    concept: str, parent: Field, tri: RefineTriage, rationale: str, member_lines: list[str], observed: list[str]
) -> str:
    """Assemble the authoring prompt: the concept, the CDE as it stands, why it fails, and the evidence."""
    parts = [
        f"CONCEPT the source fields measure: {concept}",
        "EXISTING CDE (change only what is wrong with it):\n" + _parent_card(parent),
    ]
    why = [f"Detected mismatch ({tri.axis.value if tri.axis else 'unknown'}): {tri.evidence}"]
    if rationale:
        why.append(f"The matching step's reasoning: {rationale.strip()[:600]}")
    parts.append("WHY THE CDE DOES NOT FIT AS-IS:\n" + "\n".join(why))
    parts.append("SOURCE FIELDS (all harmonized to this concept across cohorts):\n" + "\n".join(member_lines))
    if observed:
        parts.append(
            "ANSWER CONCEPTS observed across the source fields — the value domain must be able to express "
            "these; add only what is genuinely missing:\n" + ", ".join(observed[:60])
        )
    parts.append("Author the minimal refinement as JSON per the schema.")
    return "\n\n".join(parts)


def prepare_refine(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    cde_fields: dict[str, Field],
    model_tag: str = DEFAULT_MODEL_TAG,
) -> list[PromptRecord]:
    """One authoring prompt per ``refine`` record that needs a model — the LLM axes only.

    Skipped: records the gate declined (mis-assigned), records with no evidence, and records a
    deterministic delta already covered (run :func:`apply_deterministic_refinements` first — its output
    is free, so paying a model for the same answer would be waste). The deterministic reconciliation
    runs here too, so the prompt shows the pooled answer concepts and :func:`assemble_refine` can
    verify the authored domain against them.
    """
    lookup = build_field_lookup(embedded_dicts)
    prompts: list[PromptRecord] = []
    n_refine = n_gated = n_prefilled = 0
    for rec in records:
        tri = classify_refinement_axis(rec)
        if tri.reason == GATE_NOT_REFINE:
            continue
        n_refine += 1
        if not tri.refinable:
            n_gated += 1
            continue
        if rec.gencde is not None:
            n_prefilled += 1
            continue  # already authored deterministically ($0) — do not pay for it twice
        parent = cde_fields.get(rec.cde_id or "")
        if parent is None:
            continue
        fields = [
            fld
            for sv in rec.member_variable_names
            if (fld := lookup.get((sv.partition(":")[0], sv.partition(":")[2]))) is not None
        ]
        observed = observed_answer_labels(fields)
        member_lines = [
            _member_line(sv, lookup.get((sv.partition(":")[0], sv.partition(":")[2])))
            for sv in rec.member_variable_names[:_MEMBER_CAP]
        ]
        key = rec.group_id or rec.cluster_id
        prompts.append(
            PromptRecord(
                id=f"refine:{key}",
                system_prompt=SYS_REFINE,
                user_prompt=build_refine_user_prompt(rec.concept, parent, tri, rec.rationale, member_lines, observed),
                schema=REFINE_SCHEMA,
                model_tag=model_tag,
                context={
                    "record_key": key,
                    "axis": tri.axis.value if tri.axis else "",
                    "observed_labels": observed,
                    "parent_cde_id": rec.cde_id,
                },
            )
        )
    logger.info(
        "prepare_refine: %d refine records -> %d prompts (%d gated as mis-assigned, %d already authored $0)",
        n_refine,
        len(prompts),
        n_gated,
        n_prefilled,
    )
    return prompts


def _parse_refine(resp: object) -> dict:
    """Tolerant parse of an authoring response (the Batch schema is soft/appended-as-text)."""
    if resp is None:
        return {}
    if isinstance(resp, dict):
        return resp
    try:
        obj = extract_json(str(resp))
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001 - any parse failure -> empty, handled as needs_review
        return {}


def _revised(payload: dict, key: str) -> str | None:
    """A ``*_revised`` field, or None when the model left it null/blank (meaning "keep the parent's")."""
    val = payload.get(key)
    if val is None:
        return None
    text = str(val).strip()
    return None if text.lower() in ("", "null", "none", "n/a", "unchanged") else text


def _as_float(v: object) -> float:
    try:
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def assemble_refine(
    refine_prompts: list[PromptRecord],
    responses: dict[str, object],
    records: list[LeanBRecord],
    cde_fields: dict[str, Field],
) -> list[LeanBRecord]:
    """Apply each authored delta to its parent and attach the resulting element to ``rec.gencde``.

    The element is built by starting from the parent and applying ONLY the fields the model revised, so
    the result is a genuine refinement by construction rather than by the model's promise. Three
    verification signals are then computed and recorded — none of them changes the assign verdict
    (flag-not-gate, ``C-TRANSFORM-SPEC-SCOPE.md`` §0a.4):

    - ``value_coverage`` — does the refined value domain express the answer concepts the cohorts
      actually use? (N/A for a numeric concept, exactly as for a synthesized GenCDE.)
    - ``delta_size`` / ``over_refined`` — is this still a refinement, or a rewrite wearing one's clothes?
    - the model's own ``is_refinement: false`` — the element is still attached (the reviewer needs to see
      the parent and the reason), marked ``over_refined`` so it reads as "this should probably be novel".
    """
    by_key = {(r.group_id or r.cluster_id): r for r in records}
    n_attached = n_rejected = 0
    for pr in refine_prompts:
        ctx = pr.context
        rec = by_key.get(str(ctx.get("record_key", "")))
        if rec is None:
            continue
        parent = cde_fields.get(str(ctx.get("parent_cde_id") or ""))
        if parent is None:
            continue
        payload = _parse_refine(responses.get(pr.id))
        axis_name = str(payload.get("axis", "") or ctx.get("axis", ""))
        try:
            axis = RefinementAxis(axis_name)
        except ValueError:
            axis = RefinementAxis(str(ctx.get("axis")) or RefinementAxis.QUALIFIER)

        refined = _derived_from_parent(rec, parent, axis)
        refined.generated_by = "llm"
        relation = str(payload.get("relation", "")).strip()
        refined.relation = relation if relation in REFINEMENT_RELATIONS else RELATION_CLOSE

        # Apply only what the model revised; everything else stays the parent's.
        for payload_key, attr in (
            ("preferred_name_revised", "preferred_name"),
            ("definition_revised", "definition"),
            ("question_text_revised", "question_text"),
            ("data_type_revised", "data_type"),
            ("units_revised", "units"),
        ):
            revised = _revised(payload, payload_key)
            if revised is not None:
                setattr(refined, attr, revised)
        refined.qualifier_added = _revised(payload, "qualifier_added") or ""

        added = [
            ResponseOption(
                code=str(it.get("code", "")).strip() or str(it.get("label", "")).strip(),
                label=str(it.get("label", "")).strip(),
            )
            for it in (payload.get("added_permissible_values") or [])
            if isinstance(it, dict) and str(it.get("label", "")).strip()
        ]
        relabeled = (
            {
                str(k).strip(): str(v).strip()
                for k, v in (payload.get("relabeled_values") or {}).items()
                if str(v).strip()
            }
            if isinstance(payload.get("relabeled_values"), dict)
            else {}
        )
        deprecated = [str(c).strip() for c in (payload.get("deprecated_values") or []) if str(c).strip()]
        refined.added_permissible_values = added
        refined.relabeled_values = relabeled
        refined.deprecated_values = deprecated
        # Materialize the effective value domain: the parent's values (relabeled / minus deprecated) plus
        # the additions. Consumers get a complete element, never a patch they must apply themselves.
        effective = [
            ResponseOption(code=ro.code, label=relabeled.get(ro.code, ro.label), order=ro.order)
            for ro in parent_values(parent)
            if ro.code not in deprecated
        ]
        refined.permissible_values = effective + added

        observed = list(ctx.get("observed_labels", []))
        llm_conf = _as_float(payload.get("confidence"))
        if observed:
            coverage, uncovered = _label_coverage(observed, refined.permissible_values)
            refined.value_coverage, refined.uncovered_labels = coverage, uncovered
            refined.confidence = round(0.5 * coverage + 0.5 * llm_conf, 3)
        else:
            refined.value_coverage, refined.uncovered_labels = None, []
            refined.confidence = round(llm_conf, 3)
        refined.rationale = str(payload.get("notes", "") or "")

        (
            refined.changed_fields,
            refined.completed_fields,
            refined.delta_size,
            refined.over_refined,
        ) = compute_delta(refined, parent)
        # The model's own verdict overrides an optimistic delta measurement: if it says this is not a
        # refinement of this CDE, that is the finding, and it is the more informed of the two.
        if str(payload.get("is_refinement", "true")).strip().lower() in ("false", "no", "0"):
            refined.over_refined = True
            n_rejected += 1
        refined.needs_review = bool(
            refined.over_refined
            or refined.confidence < REVIEW_CONFIDENCE
            or (refined.value_coverage is not None and refined.value_coverage < REVIEW_COVERAGE)
            or not payload  # unparseable response -> the reviewer decides, we do not guess
        )
        rec.gencde = refined
        n_attached += 1
    logger.info(
        "assemble_refine: attached %d refined elements (%d the model judged NOT refinements -> flagged)",
        n_attached,
        n_rejected,
    )
    return records


# ── re-targeting the transform specs at the refined element ───────────────────


def _source_labels(fld: Field | None) -> dict[str, str]:
    """``code -> label`` for a source field's value set (the bridge from an unmapped CODE to a concept)."""
    if fld is None:
        return {}
    opts = fld.response_options or (
        parse_value_encoding(fld.value_encoding_raw or "") if fld.value_encoding_raw else []
    )
    return {ro.code: ro.label for ro in opts if ro.code}


def retarget_refined_specs(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    *,
    canon: UnitCanonicalizer | None = None,
) -> list[LeanBRecord]:
    """Point a refined record's transform specs at the refined element, and close the gaps it filled.

    This is the payoff, and it is deterministic. Specs were generated against the *parent* CDE, so on a
    refine they routinely fail: measured on the held-out full-5 run, 24% of refine edges came back
    ``kind=none`` (vs 4% on adopts) and 2,662 source codes had nowhere to map — precisely because the
    parent's value domain could not express them. Once the refinement ADDS those values, the mapping
    that was impossible becomes mechanical:

    - **Categorical.** Each previously-unmapped source code is looked up by its own label and matched,
      by normalized label, against the refined element's value domain. A hit becomes a real ``code_map``
      entry and leaves ``unmapped_source_codes``; ``coverage`` is recomputed. No model call — the
      refinement already stated which concepts it added, and the source already stated what its codes
      mean, so the join is exact.
    - **Unit.** When the refinement declares a unit the parent lacked, a previously ``needs_units``
      residual can become a real conversion (or an identity) via the curated table.

    Every spec's ``target_cde_id`` is repointed so the Sankey edge names the element the recode actually
    targets. Records with no refined element are untouched. Idempotent.
    """
    from ddharmon.matching.confidence import score_transform_spec

    canon = canon or UnitCanonicalizer()
    lookup = build_field_lookup(embedded_dicts)
    n_recs = n_codes_closed = n_units_closed = 0
    for rec in records:
        el = rec.gencde
        if el is None or not el.parent_cde_id:
            continue  # not a refined element (a from-scratch GenCDE keeps the M12 novel path)
        n_recs += 1
        refined_labels = {_norm(ro.label): ro.code for ro in el.permissible_values if (ro.label or "").strip()}
        for spec in rec.transforms:
            spec.target_cde_id = el.gencde_id

            if spec.kind == TransformKind.CATEGORICAL and spec.unmapped_source_codes:
                cohort, _, var = spec.source_variable.partition(":")
                src_labels = _source_labels(lookup.get((cohort, var)))
                still_unmapped: list[str] = []
                for code in spec.unmapped_source_codes:
                    target = refined_labels.get(_norm(src_labels.get(code, "")))
                    if target is not None:
                        spec.code_map[code] = target
                        n_codes_closed += 1
                    else:
                        still_unmapped.append(code)
                if len(still_unmapped) != len(spec.unmapped_source_codes):
                    spec.unmapped_source_codes = still_unmapped
                    total = len(spec.code_map) + len(still_unmapped)
                    spec.coverage = round(len(spec.code_map) / total, 3) if total else spec.coverage
                    spec.needs_review = bool(still_unmapped)
                    spec.rationale = (
                        f"{spec.rationale} Extended against the refined element "
                        f"({el.gencde_id}): {len(spec.code_map)} of {total} source codes now map."
                    ).strip()

            elif spec.kind == TransformKind.UNIT and spec.needs_units and el.units:
                conv = canon.convert(spec.source_unit, el.units)
                if conv is not None:
                    spec.factor, spec.offset = conv
                    spec.target_unit = el.units
                    spec.needs_units = False
                    spec.needs_review = False
                    spec.kind = TransformKind.IDENTITY if is_identity_conversion(*conv) else TransformKind.UNIT
                    spec.rationale = (
                        f"Unit declared by the refined element ({el.units}); conversion resolved from the "
                        "curated unit table."
                    )
                    spec.confidence = round(score_transform_spec(spec), 3)
                    n_units_closed += 1
    logger.info(
        "retarget_refined_specs: %d records re-targeted; %d source codes newly mapped, %d unit residuals resolved",
        n_recs,
        n_codes_closed,
        n_units_closed,
    )
    return records
