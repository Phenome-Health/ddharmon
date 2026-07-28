"""Serialize a synthesized or refined element into the NIH CDE Repository's own JSON shape.

A :class:`~ddharmon.harmonization.models.GenCDE` — whether synthesized from scratch for a ``novel``
group or derived from a real CDE for a ``refine`` group — is an internal dataclass. Nothing downstream
of ddharmon speaks it. The NIH CDE Repository's ``CdeDocument`` JSON, on the other hand, is what the
submission workflow and its validators consume, so emitting that shape is what makes our output
*submittable* rather than merely readable: the files this module writes open directly in a CDE
curation/validation tool and can be checked before they ever reach NIH endorsement review.

**Where the provenance goes, and why it goes there.** The obvious place to record "this element refines
CDE X" would be a derivation pointer on the element. The NIH model does not have one. Measured against
the public dump of 22,743 CDEs (``data/examples/cde/All-CDEs.json``):

- ``derivationRules`` is the only self-referential field, and it is a score-AGGREGATION slot
  (``ruleType: "score"``, ``formula: "sumAll"``, ``inputs: [tinyId, …]``) — "this total is the sum of
  those items". It is populated on **2 of 22,743** elements. It is emphatically NOT a "refines"
  relation, and this module leaves it empty on purpose. Do not repurpose it.
- ``history`` (version lineage) is empty on **all 22,743** records in the public export.
- The ISO 11179 slots a refinement would naturally ride in — ``objectClass``, ``property``,
  ``dataElementConcept`` concepts — are populated on ~4%, so they cannot be relied on either.

So the relation is carried two ways: as an SSSOM/SKOS predicate in ``properties[]`` (machine-readable,
and the same vocabulary the pairwise mapping export uses), and as a ``referenceDocuments[]`` link back
to the parent's repository entry (human-followable). Both are ordinary, spec-legal fields — a
consumer that has never heard of ddharmon still sees a well-formed CDE.

Emitted elements are always ``registrationStatus: "Candidate"`` / ``nihEndorsed: false`` with an empty
``tinyId``: they are proposals awaiting human review, and NIH assigns the identifier on acceptance.
Claiming anything stronger would misrepresent a machine-generated element as an endorsed standard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ddharmon.export.eitl import cde_url
from ddharmon.harmonization.models import GenCDE

logger = logging.getLogger(__name__)

STEWARD_ORG = "ddharmon"  # the steward of a machine-authored candidate, until a human org adopts it
SOURCE_NAME = "ddharmon"  # `sources[].sourceName` — provenance for every designation/definition we emit
PROPERTY_PREFIX = "ddharmon"  # namespace for our `properties[]` keys, so they never collide with NIH's

# Our data_type vocabulary -> the NIH valueDomain.datatype spellings seen in the public dump.
_DATATYPE = {
    "numeric": "Number",
    "number": "Number",
    "integer": "Number",
    "continuous": "Number",
    "categorical": "Value List",
    "binary": "Value List",
    "text": "Text",
    "string": "Text",
    "date": "Date",
    "time": "Time",
}


def _datatype(data_type: str, has_values: bool) -> str:
    """Map our data_type to an NIH datatype, falling back to the shape of the value domain."""
    mapped = _DATATYPE.get((data_type or "").strip().lower())
    if mapped:
        return mapped
    return "Value List" if has_values else "Text"


def _designations(el: GenCDE) -> list[dict]:
    """Name + title + question text as NIH designations (the question tagged, as the repo does)."""
    out: list[dict] = []
    seen: set[str] = set()
    for text, tags in ((el.preferred_name, []), (el.title, []), (el.question_text, ["Preferred Question Text"])):
        text = (text or "").strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append({"designation": text, "tags": tags, "sources": [SOURCE_NAME]})
    for alias in el.aliases:
        alias = (alias or "").strip()
        if alias and alias.lower() not in seen:
            seen.add(alias.lower())
            out.append({"designation": alias, "tags": ["Alternate Name"], "sources": [SOURCE_NAME]})
    return out


def _value_domain(el: GenCDE) -> dict:
    """The NIH valueDomain: datatype + unit of measure + permissible values (code/meaning pairs)."""
    pvs = [
        {"permissibleValue": ro.code or ro.label, "valueMeaning": ro.label}
        for ro in el.permissible_values
        if (ro.label or "").strip()
    ]
    vd: dict = {
        "datatype": _datatype(el.data_type, bool(pvs)),
        "uom": (el.units or "").strip(),
        "permissibleValues": pvs,
        "identifiers": [],
        "ids": [],
    }
    if el.minimum_value is not None:
        vd["minValue"] = el.minimum_value
    if el.maximum_value is not None:
        vd["maxValue"] = el.maximum_value
    return vd


def _properties(el: GenCDE) -> list[dict]:
    """Provenance + derivation as namespaced key/value properties (see the module docstring)."""
    props: list[dict] = []

    def add(key: str, value: object) -> None:
        text = ", ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value or "")
        if text.strip():
            props.append({"key": f"{PROPERTY_PREFIX}:{key}", "value": text.strip(), "source": SOURCE_NAME})

    if el.parent_cde_id:
        # The refinement relation. `refines` names the parent; `relation` is the SSSOM/SKOS predicate.
        add("refines", el.parent_cde_external_id or el.parent_cde_id)
        add("refines_designation", el.parent_cde_id)
        add("relation", el.relation)
        add("refinement_axis", el.refinement_axis)
        add("changed_fields", el.changed_fields)
        add("delta_size", f"{el.delta_size:.3f}")
        if el.qualifier_added:
            add("qualifier_added", el.qualifier_added)
        if el.deprecated_values:
            add("deprecated_values", el.deprecated_values)
        if el.over_refined:
            # Surfaced deliberately: a reviewer should see that the tool itself doubts this is a
            # refinement of that parent at all.
            add("over_refined", "true — the delta rewrites rather than refines; consider a new element")
    add("source_variables", el.source_variables)
    add("source_cohorts", el.source_cohorts)
    add("generated_by", el.generated_by)
    add("confidence", f"{el.confidence:.3f}")
    if el.value_coverage is not None:
        add("value_coverage", f"{el.value_coverage:.3f}")
    if el.uncovered_labels:
        add("uncovered_labels", el.uncovered_labels)
    if el.needs_review:
        add("needs_review", "true")
    return props


def to_cde_document(el: GenCDE) -> dict:
    """Render one element as an NIH ``CdeDocument`` dict — submission-shaped, not submitted.

    Works for both provenances: a from-scratch ``novel`` GenCDE and a ``refine``-derived element. The
    derived case inherits the parent's unchanged metadata (that inheritance happens upstream, in
    :mod:`ddharmon.harmonization.refine`), so what is serialized here is always the complete effective
    element — never a bare patch a consumer would have to apply itself.
    """
    doc: dict = {
        "tinyId": "",  # NIH assigns on acceptance; an invented id would collide with a real one
        "elementType": "cde",
        "stewardOrg": {"name": STEWARD_ORG},
        "createdBy": {"username": SOURCE_NAME},
        "nihEndorsed": False,
        "archived": False,
        "registrationState": {"registrationStatus": "Candidate", "administrativeStatus": "Not Endorsed"},
        "designations": _designations(el),
        "definitions": (
            [{"definition": el.definition.strip(), "tags": [], "sources": [SOURCE_NAME]}]
            if (el.definition or "").strip()
            else []
        ),
        "valueDomain": _value_domain(el),
        "dataElementConcept": {"concepts": []},
        "objectClass": {"concepts": []},
        "property": {"concepts": []},
        "sources": [{"sourceName": _source_name(el), "registrationStatus": "Candidate"}],
        "referenceDocuments": _reference_documents(el),
        "properties": _properties(el),
        "classification": [],
        "ids": [],
        "partOfBundles": [],
        # Intentionally empty: `derivationRules` is the NIH model's score-AGGREGATION slot (2/22,743
        # public CDEs, all `ruleType: "score"`), NOT a "refines" relation. The derivation of a refined
        # element is carried in `properties[]` + `referenceDocuments[]` instead — see the module
        # docstring. Do not repurpose this field.
        "derivationRules": [],
    }
    return doc


def _source_name(el: GenCDE) -> str:
    """A human-readable provenance sentence for ``sources[]`` — the one free-text slot NIH gives us."""
    cohorts = ", ".join(el.source_cohorts) if el.source_cohorts else "the source cohorts"
    n = len(el.source_variables)
    if el.parent_cde_id:
        return (
            f"ddharmon — refinement of NIH CDE '{el.parent_cde_id}' "
            f"({el.relation or 'related'}; axis: {el.refinement_axis or 'unspecified'}), "
            f"derived from {n} harmonized variable(s) across {cohorts}."
        )
    return f"ddharmon — generated from {n} harmonized variable(s) across {cohorts} that matched no existing CDE."


def _reference_documents(el: GenCDE) -> list[dict]:
    """A followable link back to the parent's repository entry, for a derived element."""
    uri = cde_url(el.parent_cde_external_id or "")
    if not uri:
        return []
    return [{"document": f"Parent CDE this element refines: {el.parent_cde_id}", "uri": uri, "source": SOURCE_NAME}]


def write_cde_documents(elements: list[GenCDE], out_dir: Path) -> list[Path]:
    """Write one ``<element_id>.json`` per element into ``out_dir``; returns the paths written.

    One file per element (rather than one array) because that is the unit a CDE curation tool opens
    and a reviewer edits.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for el in elements:
        stem = (el.gencde_id or "element").replace(":", "_").replace("/", "_").replace("#", "_")
        path = out_dir / f"{stem}.json"
        path.write_text(json.dumps(to_cde_document(el), indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(path)
    logger.info("write_cde_documents: wrote %d CdeDocument files to %s", len(written), out_dir)
    return written


def cde_revision_proposals(elements: list[GenCDE]) -> list[dict]:
    """Aggregate refined elements BY PARENT CDE — the steward-facing view of the same deltas.

    One row per parent CDE that our runs say needs changing, with the evidence pooled across every
    concept group that hit it: "N groups spanning M cohorts show CDE X needs these permissible values
    / this qualifier / this unit". A single group's refinement is an opinion; the same gap found by
    several independent groups across several cohorts is evidence worth sending upstream.

    From-scratch elements are skipped (no parent to revise).
    """
    by_parent: dict[str, dict] = {}
    for el in elements:
        if not el.parent_cde_id:
            continue
        row = by_parent.setdefault(
            el.parent_cde_id,
            {
                "parent_cde_id": el.parent_cde_id,
                "parent_cde_external_id": el.parent_cde_external_id or "",
                "parent_cde_url": cde_url(el.parent_cde_external_id or ""),
                "n_groups": 0,
                "axes": [],
                "cohorts": [],
                "source_variables": [],
                "added_permissible_values": [],
                "qualifiers_added": [],
                "units_proposed": [],
                "element_ids": [],
                "over_refined_groups": 0,
            },
        )
        row["n_groups"] += 1
        row["element_ids"].append(el.gencde_id)
        row["over_refined_groups"] += int(el.over_refined)
        for axis in ([el.refinement_axis] if el.refinement_axis else []):
            if axis not in row["axes"]:
                row["axes"].append(axis)
        for cohort in el.source_cohorts:
            if cohort not in row["cohorts"]:
                row["cohorts"].append(cohort)
        row["source_variables"].extend(el.source_variables)
        for ro in el.added_permissible_values:
            label = (ro.label or "").strip()
            if label and label not in row["added_permissible_values"]:
                row["added_permissible_values"].append(label)
        if el.qualifier_added and el.qualifier_added not in row["qualifiers_added"]:
            row["qualifiers_added"].append(el.qualifier_added)
        if el.units and el.units not in row["units_proposed"]:
            row["units_proposed"].append(el.units)
    rows = sorted(by_parent.values(), key=lambda r: (-r["n_groups"], r["parent_cde_id"]))
    logger.info("cde_revision_proposals: %d parent CDEs with proposed revisions", len(rows))
    return rows
