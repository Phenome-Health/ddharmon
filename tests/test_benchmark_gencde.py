"""Deterministic tests for the GenCDE benchmark (Benchmark E) — no network, no embedder.

Covers the RoP-anchor -> GenCDE mapping, the component-wise equivalence scorer (lexical fallback), and the
FAIRkit-style reproducibility report. The real run injects the BioLORD embedder and fetches the anchors.
"""

from __future__ import annotations

from benchmarks.gencde import (
    PUBLISHED_FAIRKIT,
    component_scores,
    reproducibility_report,
    rop_anchor_to_gencde,
)
from ddharmon.harmonization.models import GenCDE
from ddharmon.models.data_dictionary import ResponseOption


def _g(gid: str, name: str, title: str, definition: str, labels: list[str]) -> GenCDE:
    return GenCDE(
        gencde_id=gid,
        preferred_name=name,
        title=title,
        definition=definition,
        data_type="categorical",
        permissible_values=[ResponseOption(code=x, label=x) for x in labels],
    )


def test_rop_anchor_to_gencde_enum() -> None:
    anchor = {
        "rop_accession": "RoP:0000001",
        "item": "SexAtBirth",
        "description": "Biological sex assigned at birth.",
        "item_type": "enum",
        "values": "Male|Female|Intersex|Unknown",
        "alternate_names": "Sex|BirthSex",
        "source_authority": "LOINC",
    }
    g = rop_anchor_to_gencde(anchor)
    assert g.gencde_id == "RoP:0000001"
    assert g.preferred_name == "SexAtBirth"
    assert g.data_type == "categorical"
    assert [ro.label for ro in g.permissible_values] == ["Male", "Female", "Intersex", "Unknown"]
    assert g.aliases == ["Sex", "BirthSex"]


def test_rop_anchor_to_gencde_numeric_range() -> None:
    anchor = {
        "item": "BMI",
        "description": "Body mass index.",
        "item_type": "numeric",
        "values": "0-200",
        "unit_of_measure": "kg/m2",
    }
    g = rop_anchor_to_gencde(anchor)
    assert g.data_type == "numeric"
    assert g.permissible_values == []
    assert g.minimum_value == 0.0 and g.maximum_value == 200.0
    assert g.units == "kg/m2"


def test_component_scores_identical_are_one() -> None:
    a = _g("x", "ever_smoked", "Ever Smoked", "Whether the participant ever smoked.", ["Yes", "No"])
    scores = component_scores(a, a)  # no embedder -> lexical; identical strings -> 1.0
    assert scores == {"variable_name": 1.0, "title": 1.0, "definition": 1.0, "permissible_values": 1.0}


def test_component_scores_divergent_below_one() -> None:
    a = _g("x", "ever_smoked", "Ever Smoked", "Whether the participant ever smoked.", ["Yes", "No"])
    b = _g("x", "smoking_status", "Cigarette Use", "Current cigarette smoking category.", ["Yes", "No", "Former"])
    scores = component_scores(a, b)
    assert scores["variable_name"] < 1.0
    assert scores["permissible_values"] == 2 / 3  # {yes,no} ∩ {yes,no,former} / union


def test_reproducibility_report_shape_and_values() -> None:
    stable = _g("c1", "ever_smoked", "Ever Smoked", "Whether ever smoked.", ["Yes", "No"])
    run1 = [stable, _g("c2", "pain_level", "Pain Level", "Reported pain severity.", ["Mild", "Severe"])]
    run2 = [stable, _g("c2", "pain_score", "Pain Score", "Severity of reported pain.", ["Mild", "Moderate"])]
    rep = reproducibility_report([run1, run2])
    assert rep["n_runs"] == 2
    assert rep["n_concepts"] == 2  # c1, c2 shared across both runs
    assert rep["published_fairkit"] == PUBLISHED_FAIRKIT
    # c1 is byte-identical across runs, c2 diverges -> mean equivalence strictly between 0 and 1
    for comp in ("variable_name", "title", "definition", "permissible_values"):
        assert 0.0 < rep[comp]["mean"] <= 1.0
