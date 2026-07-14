"""Deterministic tests for the GenCDE synthesis stage (no LLM).

Covers the $0 core: cross-cohort answer-concept reconciliation, novel-only prompt preparation, and
response assembly with the value-coverage verification signal. The LLM authoring itself is exercised on
a metered Batch run, not here.
"""

from __future__ import annotations

import numpy as np

from ddharmon.harmonization.gencde import (
    assemble_gencde,
    observed_answer_labels,
    prepare_gencde,
)
from ddharmon.harmonization.leanb import LeanBResult, export_leanb_eitl_queue
from ddharmon.harmonization.models import ROUTE_ASSIGNED, ROUTE_RESIDUAL, GenCDE, LeanBRecord


def _dicts(hf):
    """Two cohorts: AoU + CLSA, each with one coded 'ever smoked' field using different codes."""
    aou = hf.field("smk", "Ever smoked", encoding="1=Yes|2=No|99=Missing", question_text="Have you ever smoked?")
    clsa = hf.field("smoke", "Smoking status", encoding="0=No|1=Yes|8=Refused", question_text="Ever smoked cigarettes?")
    v = np.ones((1, 8), dtype=np.float32)
    return [
        hf.embedded_dict("AoU", [aou], sem_vecs=v),
        hf.embedded_dict("CLSA", [clsa], sem_vecs=v),
    ]


def _novel_record() -> LeanBRecord:
    return LeanBRecord(
        cluster_id="c1",
        verdict="novel",
        route=ROUTE_RESIDUAL,
        group_id="c1#g0",
        concept="Ever smoked cigarettes",
        member_variable_names=["AoU:smk", "CLSA:smoke"],
        cohorts=["AoU", "CLSA"],
        ideal_cde="Whether the participant has ever smoked cigarettes (yes/no).",
    )


def test_observed_answer_labels_reconciles_across_cohorts(hf) -> None:
    """Pooled distinct answer concepts, deduped by label across cohorts, sentinels dropped."""
    dicts = _dicts(hf)
    fields = [dicts[0].dictionary.fields["smk"], dicts[1].dictionary.fields["smoke"]]
    labels = observed_answer_labels(fields)
    assert {label.lower() for label in labels} == {
        "yes",
        "no",
    }  # Missing/Refused dropped; Yes/No deduped across cohorts


def test_prepare_gencde_only_novel(hf) -> None:
    """Only route=gencde_residual records produce prompts; the prompt carries members + seed + provenance."""
    dicts = _dicts(hf)
    novel = _novel_record()
    assigned = LeanBRecord(
        cluster_id="c2",
        verdict="adopt",
        route=ROUTE_ASSIGNED,
        group_id="c2#g0",
        member_variable_names=["AoU:smk"],
        cohorts=["AoU"],
    )
    prompts = prepare_gencde([novel, assigned], dicts)
    assert len(prompts) == 1
    p = prompts[0]
    assert p.id == "gencde:c1#g0"
    assert "Have you ever smoked?" in p.user_prompt  # member question text surfaced
    assert "ever smoked cigarettes" in p.user_prompt.lower()  # ideal seed surfaced
    assert p.context["record_key"] == "c1#g0"
    assert set(p.context["source_cohorts"]) == {"AoU", "CLSA"}
    assert {label.lower() for label in p.context["observed_labels"]} == {"yes", "no"}


def test_assemble_gencde_full_coverage_no_review(hf) -> None:
    """A response covering every observed answer concept -> coverage 1.0, no review, provenance from context."""
    dicts = _dicts(hf)
    novel = _novel_record()
    prompts = prepare_gencde([novel], dicts)
    responses = {
        prompts[0].id: {
            "preferred_name": "ever_smoked",
            "title": "Ever Smoked Cigarettes",
            "definition": "Whether the participant has ever smoked cigarettes.",
            "question_text": "Have you ever smoked cigarettes?",
            "data_type": "binary",
            "permissible_values": [{"code": "1", "label": "Yes"}, {"code": "0", "label": "No"}],
            "aliases": ["smoking_history"],
            "confidence": 0.9,
            "notes": "reconciled from AoU + CLSA",
        }
    }
    assemble_gencde(prompts, responses, [novel])
    g = novel.gencde
    assert isinstance(g, GenCDE)
    assert g.gencde_id == "GENCDE:c1#g0"
    assert g.value_coverage == 1.0
    assert g.needs_review is False
    assert len(g.permissible_values) == 2
    assert g.aliases == ["smoking_history"]
    # provenance is deterministic (from context), not the LLM
    assert set(g.source_variables) == {"AoU:smk", "CLSA:smoke"}
    assert g.ideal_seed.startswith("Whether the participant")


def test_assemble_gencde_low_coverage_flags_review(hf) -> None:
    """A response that misses an observed answer concept -> coverage < 0.8 -> needs_review, uncovered listed."""
    dicts = _dicts(hf)
    novel = _novel_record()
    prompts = prepare_gencde([novel], dicts)
    responses = {
        prompts[0].id: {
            "data_type": "binary",
            "permissible_values": [{"code": "1", "label": "Yes"}],  # dropped "No"
            "confidence": 0.9,
        }
    }
    assemble_gencde(prompts, responses, [novel])
    g = novel.gencde
    assert g.value_coverage == 0.5
    assert g.needs_review is True
    assert [label.lower() for label in g.uncovered_labels] == ["no"]


def test_assemble_gencde_numeric_concept(hf) -> None:
    """A numeric concept (no coded options) -> units/bounds, coverage 1.0, not flagged for empty values."""
    aou = hf.field("bmi_a", "Body mass index", data_type="numeric", units="kg/m2", question_text="BMI")
    clsa = hf.field("bmi_c", "BMI", data_type="numeric", units="kg/m2", question_text="Body mass index")
    v = np.ones((1, 8), dtype=np.float32)
    dicts = [hf.embedded_dict("AoU", [aou], sem_vecs=v), hf.embedded_dict("CLSA", [clsa], sem_vecs=v)]
    novel = LeanBRecord(
        cluster_id="c3",
        verdict="novel",
        route=ROUTE_RESIDUAL,
        group_id="c3#g0",
        concept="Body mass index",
        member_variable_names=["AoU:bmi_a", "CLSA:bmi_c"],
        cohorts=["AoU", "CLSA"],
        ideal_cde="BMI in kg/m2.",
    )
    prompts = prepare_gencde([novel], dicts)
    responses = {
        prompts[0].id: {
            "preferred_name": "bmi",
            "data_type": "numeric",
            "permissible_values": [],
            "units": "kg/m2",
            "minimum_value": 0,
            "maximum_value": 200,
            "confidence": 0.95,
        }
    }
    assemble_gencde(prompts, responses, [novel])
    g = novel.gencde
    assert g.value_coverage == 1.0  # no observed answer concepts -> trivially covered
    assert g.needs_review is False
    assert g.units == "kg/m2"
    assert g.minimum_value == 0.0
    assert g.maximum_value == 200.0


def test_export_leanb_eitl_queue_surfaces_gencde(hf, tmp_path) -> None:
    """The EITL review queue carries the synthesized GenCDE for a novel; records without one get empty cells."""
    dicts = _dicts(hf)
    novel = _novel_record()
    prompts = prepare_gencde([novel], dicts)
    responses = {
        prompts[0].id: {
            "preferred_name": "ever_smoked",
            "definition": "Whether the participant has ever smoked cigarettes.",
            "data_type": "binary",
            "permissible_values": [{"code": "1", "label": "Yes"}, {"code": "0", "label": "No"}],
            "confidence": 0.9,
        }
    }
    assemble_gencde(prompts, responses, [novel])
    adopt = LeanBRecord(
        cluster_id="c2",
        verdict="adopt",
        route=ROUTE_ASSIGNED,
        group_id="c2#g0",
        concept="Age at enrollment",
        member_variable_names=["AoU:age"],
        cohorts=["AoU"],
    )

    tsv = tmp_path / "eitl.tsv"
    export_leanb_eitl_queue(LeanBResult(records=[novel, adopt]), tsv)
    lines = tsv.read_text().splitlines()
    header = lines[0].split("\t")
    for col in (
        "gencde_name",
        "gencde_definition",
        "gencde_data_type",
        "gencde_permissible_values",
        "gencde_units",
        "gencde_value_coverage",
        "gencde_needs_review",
    ):
        assert col in header

    rows = {ln.split("\t")[header.index("group_id")]: ln.split("\t") for ln in lines[1:]}
    nrow = rows["c1#g0"]
    assert nrow[header.index("gencde_name")] == "ever_smoked"
    assert nrow[header.index("gencde_data_type")] == "binary"
    assert nrow[header.index("gencde_permissible_values")] == "1=Yes;0=No"
    assert nrow[header.index("gencde_value_coverage")] == "1.00"
    assert nrow[header.index("gencde_needs_review")] == "False"
    # a record without a synthesized GenCDE (adopt/refine, or novel with the stage off) -> empty cells
    arow = rows["c2#g0"]
    assert arow[header.index("gencde_name")] == ""
    assert arow[header.index("gencde_permissible_values")] == ""
