"""Tests for the NIH ``CdeDocument`` serializer (deterministic, no LLM).

The contract that matters here is conformance to somebody else's schema: what we emit has to be
loadable and checkable by a CDE curation/validation tool, and it must not overclaim (an
LLM-authored element is a *candidate*, never an endorsed standard).
"""

from __future__ import annotations

import json

from ddharmon.export.cde_json import (
    cde_revision_proposals,
    to_cde_document,
    write_cde_documents,
)
from ddharmon.harmonization.models import RELATION_NARROWER, GenCDE, RefinementAxis
from ddharmon.models.data_dictionary import ResponseOption


def _derived(**kw) -> GenCDE:
    base = {
        "gencde_id": "REFCDE:c1#g0",
        "preferred_name": "carotid_bulb_plaque_surface_right",
        "title": "Right carotid bulb plaque surface morphology",
        "definition": "Surface morphology of plaque at the right carotid bulb.",
        "question_text": "Is the plaque surface at the right carotid bulb irregular or regular?",
        "data_type": "categorical",
        "permissible_values": [ResponseOption(code="1", label="Regular"), ResponseOption(code="2", label="Irregular")],
        "source_variables": ["MESA:cplq1", "CLSA:car_surf"],
        "source_cohorts": ["MESA", "CLSA"],
        "confidence": 0.82,
        "parent_cde_id": "Imaging plaque surface type",
        "parent_cde_external_id": "tiny999",
        "relation": RELATION_NARROWER,
        "refinement_axis": RefinementAxis.QUALIFIER.value,
        "qualifier_added": "right carotid bulb",
        "changed_fields": ["question_text"],
        "delta_size": 0.167,
    }
    base.update(kw)
    return GenCDE(**base)


def _from_scratch(**kw) -> GenCDE:
    base = {
        "gencde_id": "GENCDE:c9#g0",
        "preferred_name": "ever_used_needle_age",
        "definition": "Age when a needle was first used to inject a non-prescribed drug.",
        "question_text": "How old were you when you first used a needle?",
        "data_type": "numeric",
        "units": "years",
        "minimum_value": 0.0,
        "maximum_value": 120.0,
        "source_variables": ["AoU:needle_age"],
        "source_cohorts": ["AoU"],
    }
    base.update(kw)
    return GenCDE(**base)


def _props(doc: dict) -> dict[str, str]:
    return {p["key"]: p["value"] for p in doc["properties"]}


# ── the shape somebody else's validator expects ───────────────────────────────


def test_emitted_element_is_a_candidate_not_an_endorsed_standard() -> None:
    """An LLM-authored element must never present itself as endorsed, or carry an invented tinyId."""
    doc = to_cde_document(_derived())
    assert doc["registrationState"] == {"registrationStatus": "Candidate", "administrativeStatus": "Not Endorsed"}
    assert doc["nihEndorsed"] is False
    assert doc["tinyId"] == ""  # NIH assigns on acceptance; inventing one could collide with a real CDE
    assert doc["elementType"] == "cde"


def test_derivation_rules_stays_empty() -> None:
    """`derivationRules` is the model's score-AGGREGATION slot (2/22,743 CDEs), not a 'refines' relation.

    Regression guard: the field is tempting and wrong. The derivation lives in properties/refs instead.
    """
    assert to_cde_document(_derived())["derivationRules"] == []
    assert to_cde_document(_from_scratch())["derivationRules"] == []


def test_question_text_is_tagged_as_the_preferred_question() -> None:
    """The repo distinguishes a name from the question asked; so must we."""
    doc = to_cde_document(_derived())
    tagged = [d for d in doc["designations"] if "Preferred Question Text" in d["tags"]]
    assert len(tagged) == 1
    assert tagged[0]["designation"].startswith("Is the plaque surface")


def test_value_domain_renders_permissible_values_and_bounds() -> None:
    cat = to_cde_document(_derived())["valueDomain"]
    assert cat["datatype"] == "Value List"
    assert cat["permissibleValues"] == [
        {"permissibleValue": "1", "valueMeaning": "Regular"},
        {"permissibleValue": "2", "valueMeaning": "Irregular"},
    ]

    num = to_cde_document(_from_scratch())["valueDomain"]
    assert num["datatype"] == "Number"
    assert num["uom"] == "years"
    assert num["minValue"] == 0.0 and num["maxValue"] == 120.0
    assert num["permissibleValues"] == []


# ── derivation provenance ─────────────────────────────────────────────────────


def test_derived_element_carries_the_relation_and_a_link_to_its_parent() -> None:
    doc = to_cde_document(_derived())
    props = _props(doc)
    assert props["ddharmon:refines"] == "tiny999"
    assert props["ddharmon:refines_designation"] == "Imaging plaque surface type"
    assert props["ddharmon:relation"] == RELATION_NARROWER
    assert props["ddharmon:refinement_axis"] == "qualifier"
    assert props["ddharmon:qualifier_added"] == "right carotid bulb"
    assert doc["referenceDocuments"][0]["uri"] == "https://cde.nlm.nih.gov/deView?tinyId=tiny999"
    assert "refinement of NIH CDE 'Imaging plaque surface type'" in doc["sources"][0]["sourceName"]


def test_from_scratch_element_claims_no_parent() -> None:
    doc = to_cde_document(_from_scratch())
    props = _props(doc)
    assert "ddharmon:refines" not in props
    assert "ddharmon:relation" not in props
    assert doc["referenceDocuments"] == []
    assert "matched no existing CDE" in doc["sources"][0]["sourceName"]


def test_over_refined_is_surfaced_to_the_reviewer() -> None:
    """If the tool doubts its own refinement, the reviewer must be able to see that."""
    props = _props(to_cde_document(_derived(over_refined=True)))
    assert "over_refined" in " ".join(props)
    assert "consider a new element" in props["ddharmon:over_refined"]


def test_document_is_json_serializable(tmp_path) -> None:
    """It has to survive a round-trip to disk — that is the whole point of emitting it."""
    doc = to_cde_document(_derived())
    assert json.loads(json.dumps(doc)) == doc


def test_write_cde_documents_writes_one_file_per_element(tmp_path) -> None:
    paths = write_cde_documents([_derived(), _from_scratch()], tmp_path)
    assert len(paths) == 2
    assert {p.name for p in paths} == {"REFCDE_c1_g0.json", "GENCDE_c9_g0.json"}  # ':'/'#' are not path-safe
    reloaded = json.loads(paths[0].read_text(encoding="utf-8"))
    assert reloaded["registrationState"]["registrationStatus"] == "Candidate"


# ── steward-facing aggregation ────────────────────────────────────────────────


def test_revision_proposals_pool_evidence_across_groups() -> None:
    """One group's refinement is an opinion; the same gap from several groups is evidence."""
    a = _derived(gencde_id="REFCDE:1", source_cohorts=["MESA"], qualifier_added="right carotid bulb")
    b = _derived(
        gencde_id="REFCDE:2",
        source_cohorts=["CLSA"],
        qualifier_added="left carotid bulb",
        added_permissible_values=[ResponseOption(code="3", label="Ulcerated")],
    )
    other = _derived(gencde_id="REFCDE:3", parent_cde_id="A different CDE", parent_cde_external_id="tiny111")

    rows = cde_revision_proposals([a, b, other, _from_scratch()])
    assert len(rows) == 2  # from-scratch has no parent to revise
    top = rows[0]  # ordered by evidence weight
    assert top["parent_cde_id"] == "Imaging plaque surface type"
    assert top["n_groups"] == 2
    assert sorted(top["cohorts"]) == ["CLSA", "MESA"]
    assert top["qualifiers_added"] == ["right carotid bulb", "left carotid bulb"]
    assert top["added_permissible_values"] == ["Ulcerated"]
    assert top["parent_cde_url"] == "https://cde.nlm.nih.gov/deView?tinyId=tiny999"


def test_revision_proposals_count_doubted_refinements() -> None:
    rows = cde_revision_proposals([_derived(gencde_id="REFCDE:1", over_refined=True), _derived(gencde_id="REFCDE:2")])
    assert rows[0]["n_groups"] == 2 and rows[0]["over_refined_groups"] == 1
