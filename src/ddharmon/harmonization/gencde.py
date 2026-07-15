"""GenCDE synthesis — author a Common Data Element for a ``novel`` concept group (the tail's target).

Every ``novel`` :class:`~ddharmon.harmonization.models.LeanBRecord` (route ``gencde_residual``) reached no
existing CDE. Without a target, "novel" is a dead end. This stage synthesizes a structured
:class:`~ddharmon.harmonization.models.GenCDE` so the tail HAS a harmonization target.

"GenCDE" is DataTecnica/FAIRkit's term (Long et al., npj Digit Med 2026). FAIRkit generates one CDE from
ONE sparse dictionary entry (generate-from-template). This inverts that: a novel record is already a
cluster of harmonized fields across cohorts measuring the same concept, so we synthesize from the POOLED
empirical evidence — the members' reconciled answer options, units, and question texts
(generate-from-cluster-empirics). The reconciliation (:func:`observed_answer_labels`) is deterministic and
$0-testable; only the final authoring is an LLM call, run via the Batch API like the other leanb stages
(``prepare_gencde`` -> ``gencde(prompts)`` -> ``assemble_gencde``).

Metadata-level (like transform specs): the GenCDE is EMITTED and routed to review, never executed on row
data. ``value_coverage`` is a verification signal (did the synthesized value domain represent the answer
concepts actually observed across cohorts?); low coverage flags ``needs_review`` but never changes the
``novel`` verdict.
"""

from __future__ import annotations

import logging
import re

from ddharmon.embedding.service import EmbeddedDictionary
from ddharmon.harmonization.anchor import build_field_lookup
from ddharmon.harmonization.leanb import DEFAULT_MODEL_TAG
from ddharmon.harmonization.models import ROUTE_RESIDUAL, GenCDE, LeanBRecord
from ddharmon.harmonization.parse import extract_json
from ddharmon.harmonization.pipeline import PromptRecord
from ddharmon.models.data_dictionary import Field, ResponseOption
from ddharmon.values.response_parser import parse_value_encoding

logger = logging.getLogger(__name__)

REVIEW_COVERAGE = 0.8  # observed answer-concepts represented below this fraction -> needs_review
REVIEW_CONFIDENCE = 0.6  # LLM confidence below this -> needs_review
NUMERIC_NO_DOMAIN_PENALTY = 0.7  # a numeric GenCDE with no units/bounds -> confidence scaled by this
_MEMBER_CAP = 12  # members shown in the synthesis prompt (cost bound; provenance keeps the full list)

SYS_GENCDE = (
    "You are a biomedical Common Data Element (CDE) expert. You are given a CONCEPT that a set of harmonized "
    "data-dictionary fields (from one or more cohorts) all measure, but which matched no existing CDE. "
    "Author a single, catalog-quality GenCDE (generated CDE) for it, mirroring standard NIH-CDE metadata. "
    "Base the value domain on the POOLED answer options actually observed across the source fields: "
    "reconcile them into ONE canonical permissible-value set (merge synonyms, assign clean sequential codes, "
    "drop missing-data sentinels). For a numeric concept, give the data type, unit of measure, and "
    "minimum/maximum instead of permissible values. Do NOT invent answer concepts no source field expresses. "
    "Preserve the source qualifier (body site, person, laterality, time window). Add aliases (synonyms found "
    "in medical ontologies/vocabularies) when confident. Return a confidence in [0,1]. Return JSON only."
)

GENCDE_SCHEMA = (
    '{"preferred_name": "<canonical snake_case variable name>", '
    '"title": "<expanded descriptive label>", '
    '"definition": "<one-sentence clinical definition>", '
    '"question_text": "<the acquisition question this element answers>", '
    '"data_type": "categorical | numeric | binary | date | text", '
    '"permissible_values": [{"code": "<code>", "label": "<label>"}], '
    '"units": "<UCUM unit or null>", "minimum_value": "<number or null>", "maximum_value": "<number or null>", '
    '"aliases": ["<synonym>"], "confidence": "<0.0-1.0>", "notes": "<short rationale>"}'
)

# answer labels that carry no concept — dropped before reconciliation so they don't pad the value domain
_SENTINELS = {
    "",
    "missing",
    "unknown",
    "not applicable",
    "n/a",
    "na",
    "refused",
    "prefer not to answer",
    "don't know",
    "dont know",
    "do not know",
    "no answer",
    "not answered",
    "skipped",
    "no response",
    "declined",
}


def _norm(s: str) -> str:
    """Normalize a label for cross-cohort comparison: collapse whitespace, lowercase, strip."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _as_float(v: object) -> float:
    """Clamp an LLM-supplied confidence into [0, 1]; unparseable -> 0.0."""
    try:
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _num(v: object) -> float | None:
    """Parse an optional numeric bound; None/null/unparseable -> None."""
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "na")):
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _member_fields(rec: LeanBRecord, lookup: dict) -> list[Field]:
    """The group's member :class:`Field`s, resolved from ``member_variable_names`` ('cohort:var')."""
    out: list[Field] = []
    for sv in rec.member_variable_names:
        cohort, _, var = sv.partition(":")
        fld = lookup.get((cohort, var))
        if fld is not None:
            out.append(fld)
    return out


def _member_line(sv: str, fld: Field | None) -> str:
    """A compact one-line rendering of a member field for the synthesis prompt (symbolic value metadata)."""
    if fld is None:
        return sv
    parts = [sv]
    q = (fld.question_text or fld.description or "").strip()
    if q:
        parts.append(q)
    if fld.data_type and fld.data_type.strip():
        parts.append(f"[type {fld.data_type.strip()}]")
    if fld.units and fld.units.strip():
        parts.append(f"[units {fld.units.strip()}]")
    vs = (fld.value_encoding_raw or "").strip()
    if not vs and fld.response_options:
        vs = "|".join(f"{ro.code}={ro.label}" for ro in fld.response_options)
    if vs:
        parts.append(f"values: {vs}")
    return " — ".join(parts)


def observed_answer_labels(fields: list[Field]) -> list[str]:
    """Distinct answer-concept labels pooled across a group's member fields (the empirical value evidence).

    Reconciles ACROSS cohorts by normalized label — codes are cohort-specific and not comparable (AoU's
    ``1=Yes`` and CLSA's ``0=Yes`` are the same concept under different codes). Missing-data sentinels are
    dropped. Returns representative original-cased labels in first-seen order. Empty for a numeric concept
    (no coded options), which is correct: a numeric GenCDE carries units/bounds, not permissible values.
    """
    seen: dict[str, str] = {}
    for fld in fields:
        raw = (fld.value_encoding_raw or "").strip()
        opts = parse_value_encoding(raw) if raw else list(fld.response_options or [])
        for ro in opts:
            n = _norm(ro.label)
            if not n or n in _SENTINELS:
                continue
            seen.setdefault(n, ro.label.strip())
    return list(seen.values())


def _label_coverage(observed: list[str], synthesized: list[ResponseOption]) -> tuple[float, list[str]]:
    """Fraction of observed answer-concepts represented (by normalized label) in the synthesized value set.

    Returns ``(coverage, uncovered_labels)``. No observed labels (numeric concept) -> coverage 1.0, [].
    """
    if not observed:
        return 1.0, []
    syn = {_norm(ro.label) for ro in synthesized}
    uncovered = [lab for lab in observed if _norm(lab) not in syn]
    covered = len(observed) - len(uncovered)
    return round(covered / len(observed), 3), uncovered


def _candidate_names(rec: LeanBRecord, limit: int = 3) -> list[str]:
    """Near-miss candidate designations the assign stage saw (for aliases/relation only)."""
    out: list[str] = []
    for c in (rec.candidates or [])[:limit]:
        name = (
            getattr(c, "designation", "")
            or getattr(c, "title", "")
            or getattr(c, "name", "")
            or getattr(c, "cde_id", "")
            or ""
        )
        if name:
            out.append(str(name).strip())
    return out


def build_gencde_user_prompt(
    ideal_seed: str,
    concept: str,
    member_lines: list[str],
    observed_labels: list[str],
    related: list[str],
) -> str:
    """Assemble the synthesis user prompt from the concept's pooled cross-cohort evidence."""
    parts: list[str] = []
    if concept:
        parts.append(f"CONCEPT: {concept}")
    if ideal_seed:
        parts.append(f"IDEAL CDE (independently drafted anchor for this concept):\n{ideal_seed[:600]}")
    parts.append("SOURCE FIELDS (all harmonized to this concept across cohorts):\n" + "\n".join(member_lines))
    if observed_labels:
        parts.append(
            "ANSWER CONCEPTS observed across the source fields — reconcile these into the canonical "
            "permissible-value set; do NOT add concepts absent here:\n" + ", ".join(observed_labels[:60])
        )
    if related:
        parts.append(
            "NEAR-MISS existing CDEs (the concept did NOT match these — use only for aliases/relations):\n"
            + "; ".join(related[:5])
        )
    parts.append("Author the GenCDE as JSON per the schema.")
    return "\n\n".join(parts)


def prepare_gencde(
    records: list[LeanBRecord],
    embedded_dicts: list[EmbeddedDictionary],
    model_tag: str = DEFAULT_MODEL_TAG,
) -> list[PromptRecord]:
    """Build one GenCDE-synthesis prompt per ``novel`` record (route ``gencde_residual``).

    Non-novel records are skipped (they already have a CDE). The deterministic value reconciliation runs
    here so the prompt shows the LLM the pooled answer concepts, and ``assemble_gencde`` can verify the
    synthesized domain against them. Each prompt carries the provenance in ``context`` for assembly.
    """
    lookup = build_field_lookup(embedded_dicts)
    prompts: list[PromptRecord] = []
    n_novel = 0
    for rec in records:
        if rec.route != ROUTE_RESIDUAL:
            continue
        n_novel += 1
        fields = _member_fields(rec, lookup)
        observed = observed_answer_labels(fields)
        related = _candidate_names(rec)
        member_lines = [
            _member_line(sv, lookup.get((sv.partition(":")[0], sv.partition(":")[2])))
            for sv in rec.member_variable_names[:_MEMBER_CAP]
        ]
        key = rec.group_id or rec.cluster_id
        prompts.append(
            PromptRecord(
                id=f"gencde:{key}",
                system_prompt=SYS_GENCDE,
                user_prompt=build_gencde_user_prompt(rec.ideal_cde, rec.concept, member_lines, observed, related),
                schema=GENCDE_SCHEMA,
                model_tag=model_tag,
                context={
                    "record_key": key,
                    "observed_labels": observed,
                    "source_variables": list(rec.member_variable_names),
                    "source_cohorts": list(rec.cohorts),
                    "ideal_seed": rec.ideal_cde,
                    "related_cdes": related,
                },
            )
        )
    logger.info("prepare_gencde: %d records (%d novel) -> %d GenCDE prompts", len(records), n_novel, len(prompts))
    return prompts


def _parse_gencde(resp: object) -> dict:
    """Tolerant parse of a GenCDE synthesis response into a dict (Batch schema is soft/appended-as-text)."""
    if resp is None:
        return {}
    if isinstance(resp, dict):
        return resp
    try:
        obj = extract_json(str(resp))
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001 - any parse failure -> empty, handled as needs_review
        return {}


def _permissible_values(payload: dict) -> list[ResponseOption]:
    """Build ResponseOptions from the payload; fall back to the label as code when no code is given."""
    out: list[ResponseOption] = []
    for item in payload.get("permissible_values") or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip()
        label = str(item.get("label", "")).strip()
        if label:
            out.append(ResponseOption(code=code or label, label=label))
    return out


def assemble_gencde(
    gencde_records: list[PromptRecord],
    responses: dict[str, object],
    records: list[LeanBRecord],
) -> list[LeanBRecord]:
    """Parse each synthesis response into a :class:`GenCDE` and attach it to its ``novel`` record.

    For a CATEGORICAL concept ``value_coverage`` = fraction of the observed answer concepts
    (``context['observed_labels']``) the synthesized ``permissible_values`` represents, and confidence blends
    it with the LLM confidence. For a NUMERIC concept there are no observed answer-labels, so coverage is N/A
    (``value_coverage = None`` — never a vacuous 1.0), and confidence rests on the LLM confidence penalized by
    numeric-domain completeness (units / bounds). ``needs_review`` fires on low categorical coverage, a
    missing numeric domain, low LLM confidence, or an empty spec — never overriding the ``novel`` verdict.
    Provenance (source vars/cohorts, ideal seed) comes from the deterministic ``context``, not the LLM.
    """
    by_key = {(r.group_id or r.cluster_id): r for r in records}
    n_attached = 0
    for gr in gencde_records:
        ctx = gr.context
        rec = by_key.get(str(ctx.get("record_key", "")))
        if rec is None:
            continue
        payload = _parse_gencde(responses.get(gr.id))
        observed = list(ctx.get("observed_labels", []))
        pv = _permissible_values(payload)
        units = str(payload.get("units")).strip() if _num_unit(payload.get("units")) else None
        data_type = str(payload.get("data_type", "")).strip()
        min_v, max_v = _num(payload.get("minimum_value")), _num(payload.get("maximum_value"))
        llm_conf = _as_float(payload.get("confidence"))
        has_domain = bool(pv) or bool(units) or data_type in ("numeric", "date", "text", "boolean")
        if observed:
            # Categorical concept: value_coverage = fraction of observed answer-labels the synthesized domain
            # represents; confidence blends coverage with the LLM confidence; the coverage gate can fire.
            value_coverage, uncovered = _label_coverage(observed, pv)
            confidence = round(0.5 * value_coverage + 0.5 * llm_conf, 3)
            needs_review = value_coverage < REVIEW_COVERAGE or llm_conf < REVIEW_CONFIDENCE or not has_domain
        else:
            # Numeric concept: NO observed answer-labels, so value_coverage is N/A (None) — never a vacuous
            # 1.0 that would inflate confidence and mask an incoherent group. Confidence rests on the LLM
            # confidence, penalized when the model gave no usable numeric value domain (units or bounds);
            # review fires on a missing numeric domain in place of the (inapplicable) coverage gate.
            has_num_domain = bool(units) or min_v is not None or max_v is not None
            value_coverage, uncovered = None, []
            confidence = round(llm_conf * (1.0 if has_num_domain else NUMERIC_NO_DOMAIN_PENALTY), 3)
            needs_review = not has_num_domain or llm_conf < REVIEW_CONFIDENCE or not has_domain
        aliases = [str(a).strip() for a in (payload.get("aliases") or []) if str(a).strip()]
        rec.gencde = GenCDE(
            gencde_id=f"GENCDE:{rec.group_id or rec.cluster_id}",
            preferred_name=str(payload.get("preferred_name", "")).strip(),
            title=str(payload.get("title", "")).strip(),
            definition=str(payload.get("definition", "")).strip(),
            question_text=str(payload.get("question_text", "")).strip(),
            data_type=data_type,
            permissible_values=pv,
            units=units,
            minimum_value=min_v,
            maximum_value=max_v,
            aliases=aliases,
            source_variables=list(ctx.get("source_variables", [])),
            source_cohorts=list(ctx.get("source_cohorts", [])),
            ideal_seed=str(ctx.get("ideal_seed", "")),
            related_cdes=list(ctx.get("related_cdes", [])),
            value_coverage=value_coverage,
            uncovered_labels=uncovered,
            confidence=confidence,
            needs_review=needs_review,
            rationale=str(payload.get("notes", "") or ""),
        )
        n_attached += 1
    logger.info("assemble_gencde: attached %d GenCDEs to %d novel records", n_attached, len(gencde_records))
    return records


def _num_unit(v: object) -> bool:
    """True when ``v`` is a non-empty, non-null unit string."""
    return isinstance(v, str) and v.strip().lower() not in ("", "null", "none", "na")
