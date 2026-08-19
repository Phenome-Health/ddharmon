"""Deterministic tests for the refine stage (no LLM).

Covers the $0 core: axis triage from evidence already on the record, the mis-assigned gate, the
minimality (``delta_size`` / ``over_refined``) check, and the two deltas that are computable without a
model. The LLM authoring itself is exercised on a metered Batch run, not here.
"""

from __future__ import annotations

from ddharmon.harmonization.models import (
    RELATION_CLOSE,
    ROUTE_ASSIGNED,
    GenCDE,
    LeanBRecord,
    RefinementAxis,
    TransformKind,
    TransformSpec,
)
from ddharmon.harmonization.refine import (
    GATE_MIS_ASSIGNED,
    GATE_NO_EVIDENCE,
    GATE_NOT_REFINE,
    apply_deterministic_refinements,
    build_deterministic_refinement,
    classify_refinement_axis,
    compute_delta,
    refined_cde_id,
    triage_summary,
)
from ddharmon.models.data_dictionary import Field, ResponseOption


def _rec(**kw) -> LeanBRecord:
    """A refine record that passes the gate, with no specs unless the test adds them."""
    base = {
        "cluster_id": "c1",
        "verdict": "refine",
        "route": ROUTE_ASSIGNED,
        "group_id": "c1#g0",
        "concept": "Systolic blood pressure",
        "cde_id": "Blood pressure systolic measurement",
        "cde_external_id": "tiny123",
        "member_variable_names": ["AoU:sbp", "CLSA:bp_sys"],
        "cohorts": ["AoU", "CLSA"],
    }
    base.update(kw)
    return LeanBRecord(**base)


def _parent(**kw) -> Field:
    base = {
        "variable_name": "Blood pressure systolic measurement",
        "description": "The systolic blood pressure of the participant.",
        "question_text": "What is the systolic blood pressure?",
        "data_type": "Number",
    }
    base.update(kw)
    return Field(**base)


# ── triage ────────────────────────────────────────────────────────────────────


def test_triage_ignores_non_refine_verdicts() -> None:
    """adopt/novel are not this stage's business — the classifier is a no-op on them."""
    for verdict in ("adopt", "novel", ""):
        tri = classify_refinement_axis(_rec(verdict=verdict))
        assert not tri.refinable and tri.reason == GATE_NOT_REFINE and tri.axis is None


def test_triage_gates_concept_mismatch() -> None:
    """A match the M7 gate already doubts must never be dressed up as a refinement."""
    tri = classify_refinement_axis(_rec(concept_mismatch=True))
    assert not tri.refinable and tri.reason == GATE_MIS_ASSIGNED and tri.axis is None


def test_triage_gates_incoherent_group() -> None:
    """An over-merged group has no single concept to refine toward."""
    tri = classify_refinement_axis(_rec(incoherent=True))
    assert not tri.refinable and tri.reason == GATE_MIS_ASSIGNED


def test_triage_gates_missing_concept_label() -> None:
    """No concept label means the group never resolved — refining it would invent the target."""
    tri = classify_refinement_axis(_rec(concept="   "))
    assert not tri.refinable and tri.reason == GATE_MIS_ASSIGNED


def test_triage_no_specs_is_not_refinable() -> None:
    """With no transform specs there is no evidence to classify from."""
    tri = classify_refinement_axis(_rec())
    assert not tri.refinable and tri.reason == GATE_NO_EVIDENCE


def test_triage_value_domain_from_unmapped_codes() -> None:
    spec = TransformSpec(
        source_variable="AoU:sbp",
        target_cde_id="cde",
        kind=TransformKind.CATEGORICAL,
        unmapped_source_codes=["7", "8", "9"],
    )
    tri = classify_refinement_axis(_rec(transforms=[spec]))
    assert tri.refinable and tri.axis is RefinementAxis.VALUE_DOMAIN
    assert "3 source code(s) unmappable" in tri.evidence


def test_triage_representation_from_needs_units() -> None:
    spec = TransformSpec(
        source_variable="AoU:sbp",
        target_cde_id="cde",
        kind=TransformKind.UNIT,
        needs_units=True,
        source_unit="mmHg",
        target_unit="kPa",
    )
    tri = classify_refinement_axis(_rec(transforms=[spec]))
    assert tri.refinable and tri.axis is RefinementAxis.REPRESENTATION
    assert "mmHg -> kPa" in tri.evidence


def test_triage_representation_names_the_undeclared_units_case() -> None:
    """The dominant sub-case (neither side declares a unit) reads as missing metadata, not a failed lookup."""
    spec = TransformSpec(source_variable="AoU:sbp", target_cde_id="cde", kind=TransformKind.UNIT, needs_units=True)
    tri = classify_refinement_axis(_rec(transforms=[spec]))
    assert tri.axis is RefinementAxis.REPRESENTATION
    assert "no unit of measure declared on either side" in tri.evidence


def test_triage_structural_from_wide_to_long() -> None:
    spec = TransformSpec(
        source_variable="AoU:bp1",
        target_cde_id="cde",
        kind=TransformKind.WIDE_TO_LONG,
        params={"n_occurrences": 5},
    )
    tri = classify_refinement_axis(_rec(transforms=[spec]))
    assert tri.refinable and tri.axis is RefinementAxis.STRUCTURAL
    assert "5 numbered occurrences" in tri.evidence


def test_triage_qualifier_is_the_conceptual_residual() -> None:
    """Values reconcile cleanly and units are fine -> whatever is wrong is conceptual."""
    spec = TransformSpec(source_variable="AoU:sbp", target_cde_id="cde", kind=TransformKind.CATEGORICAL, coverage=1.0)
    tri = classify_refinement_axis(_rec(transforms=[spec]))
    assert tri.refinable and tri.axis is RefinementAxis.QUALIFIER


def test_triage_summary_counts_only_refines() -> None:
    unmapped = TransformSpec(
        source_variable="s", target_cde_id="c", kind=TransformKind.CATEGORICAL, unmapped_source_codes=["9"]
    )
    records = [
        _rec(group_id="a", transforms=[unmapped]),
        _rec(group_id="b", concept_mismatch=True),
        _rec(group_id="c"),  # no specs
        _rec(group_id="d", verdict="adopt"),  # not counted
        _rec(group_id="e", verdict="novel"),  # not counted
    ]
    assert triage_summary(records) == {"value_domain": 1, GATE_MIS_ASSIGNED: 1, GATE_NO_EVIDENCE: 1}


# ── minimality (compute_delta) ────────────────────────────────────────────────


def test_compute_delta_minimal_change() -> None:
    """One slot the parent DID assert, contradicted — a refinement, not a rewrite."""
    parent = _parent(units="kPa")
    refined = GenCDE(
        gencde_id="REFCDE:x",
        preferred_name=parent.variable_name,
        definition=parent.description,
        question_text=parent.question_text or "",
        data_type=parent.data_type or "",
        units="mmHg",
    )
    changed, completed, size, over = compute_delta(refined, parent)
    assert changed == ["units"] and completed == []
    assert size == round(1 / 6, 3)
    assert over is False


def test_compute_delta_treats_filling_an_empty_slot_as_completion() -> None:
    """You cannot change what the parent never said. 90% of matched CDEs carry no question_text.

    Counting a supplied value as a delta inflated delta_size and made over_refined fire on elements
    that contradicted nothing — so completions are tracked separately and excluded from the metric.
    """
    parent = _parent(units=None, question_text=None, data_type=None)
    refined = GenCDE(
        gencde_id="REFCDE:x",
        preferred_name=parent.variable_name,
        definition=parent.description,
        question_text="What is the systolic blood pressure?",
        data_type="numeric",
        units="mmHg",
    )
    changed, completed, size, over = compute_delta(refined, parent)
    assert changed == []
    assert sorted(completed) == ["data_type", "question_text", "units"]
    assert size == 0.0  # three slots supplied, nothing contradicted
    assert over is False


def test_compute_delta_flags_a_rewrite_as_over_refined() -> None:
    """Replace name + definition + question + type -> not a refinement of that parent."""
    parent = _parent(units="mmHg")
    refined = GenCDE(
        gencde_id="REFCDE:x",
        preferred_name="Something else entirely",
        definition="A different concept.",
        question_text="A different question?",
        data_type="Value List",
        units="kPa",
    )
    changed, completed, size, over = compute_delta(refined, parent)
    assert set(changed) >= {"preferred_name", "definition", "question_text", "data_type", "units"}
    assert size > 0.5
    assert over is True


def test_compute_delta_wholly_replaced_value_domain_is_over_refined() -> None:
    """Even as the only changed slot, a value domain sharing nothing with the parent is a rewrite."""
    parent = _parent(value_encoding_raw="1=Yes|2=No")
    refined = GenCDE(
        gencde_id="REFCDE:x",
        preferred_name=parent.variable_name,
        definition=parent.description,
        question_text=parent.question_text or "",
        data_type=parent.data_type or "",
        permissible_values=[ResponseOption(code="1", label="Completed"), ResponseOption(code="2", label="Skipped")],
    )
    changed, completed, _size, over = compute_delta(refined, parent)
    assert changed == ["permissible_values"]
    assert over is True


def test_compute_delta_extending_a_value_domain_is_not_over_refined() -> None:
    """Adding an option while keeping the parent's own values is exactly what a refinement looks like."""
    parent = _parent(value_encoding_raw="1=Yes|2=No")
    refined = GenCDE(
        gencde_id="REFCDE:x",
        preferred_name=parent.variable_name,
        definition=parent.description,
        question_text=parent.question_text or "",
        data_type=parent.data_type or "",
        permissible_values=[
            ResponseOption(code="1", label="Yes"),
            ResponseOption(code="2", label="No"),
            ResponseOption(code="3", label="Prefer not to answer"),
        ],
        added_permissible_values=[ResponseOption(code="3", label="Prefer not to answer")],
    )
    changed, completed, _size, over = compute_delta(refined, parent)
    assert changed == ["permissible_values"]
    assert over is False


# ── deterministic authoring ───────────────────────────────────────────────────


def _unit_rec(source_unit: str, **kw) -> LeanBRecord:
    spec = TransformSpec(
        source_variable="AoU:sbp",
        target_cde_id="cde",
        kind=TransformKind.UNIT,
        needs_units=True,
        source_unit=source_unit,
    )
    return _rec(transforms=[spec], **kw)


def test_deterministic_representation_declares_the_agreed_source_unit() -> None:
    rec = _unit_rec("mmHg")
    parent = _parent(units=None)
    refined = build_deterministic_refinement(rec, parent, classify_refinement_axis(rec))
    assert refined is not None
    assert refined.gencde_id == refined_cde_id(rec) == "REFCDE:c1#g0"
    assert refined.units == "mmHg"
    assert refined.parent_cde_id == rec.cde_id and refined.parent_cde_external_id == "tiny123"
    assert refined.relation == RELATION_CLOSE
    assert refined.refinement_axis == RefinementAxis.REPRESENTATION.value
    # the parent declared no unit, so this COMPLETES it rather than contradicting it
    assert refined.changed_fields == [] and refined.completed_fields == ["units"]
    assert refined.delta_size == 0.0 and refined.over_refined is False
    assert refined.generated_by == "rule"
    # the parent's unchanged metadata is inherited, so the element is complete, not a bare patch
    assert refined.definition == parent.description and refined.question_text == parent.question_text


def test_deterministic_representation_declines_when_parent_has_units() -> None:
    """Parent HAS a unit and N1 still failed -> cross-family/analyte-specific, needs judgment."""
    rec = _unit_rec("mg/dL")
    assert build_deterministic_refinement(rec, _parent(units="mmol/L"), classify_refinement_axis(rec)) is None


def test_deterministic_representation_declines_on_disagreeing_sources() -> None:
    specs = [
        TransformSpec(
            source_variable="AoU:a", target_cde_id="c", kind=TransformKind.UNIT, needs_units=True, source_unit="kg"
        ),
        TransformSpec(
            source_variable="CLSA:b", target_cde_id="c", kind=TransformKind.UNIT, needs_units=True, source_unit="lb"
        ),
    ]
    rec = _rec(transforms=specs)
    assert build_deterministic_refinement(rec, _parent(units=None), classify_refinement_axis(rec)) is None


def test_deterministic_representation_declines_on_unrecognized_unit() -> None:
    """A unit the curated table cannot verify is not a deterministic delta."""
    rec = _unit_rec("widgets per fortnight")
    assert build_deterministic_refinement(rec, _parent(units=None), classify_refinement_axis(rec)) is None


def test_deterministic_structural_records_the_repeating_measure() -> None:
    spec = TransformSpec(
        source_variable="AoU:bp1",
        target_cde_id="cde",
        kind=TransformKind.WIDE_TO_LONG,
        params={"n_occurrences": 5},
        rationale="5 numbered columns are ONE repeating measure",
    )
    rec = _rec(transforms=[spec])
    refined = build_deterministic_refinement(rec, _parent(), classify_refinement_axis(rec))
    assert refined is not None
    assert refined.refinement_axis == RefinementAxis.STRUCTURAL.value
    assert "5 occurrences" in refined.qualifier_added
    assert "5 occurrences" in refined.definition
    assert refined.needs_review is True  # a structural claim is always reviewed
    assert refined.changed_fields == ["definition"]


def test_deterministic_authoring_declines_the_llm_axes() -> None:
    """VALUE_DOMAIN and QUALIFIER are judgments a rule cannot make."""
    unmapped = TransformSpec(
        source_variable="s", target_cde_id="c", kind=TransformKind.CATEGORICAL, unmapped_source_codes=["9"]
    )
    rec = _rec(transforms=[unmapped])
    assert build_deterministic_refinement(rec, _parent(), classify_refinement_axis(rec)) is None

    clean = TransformSpec(source_variable="s", target_cde_id="c", kind=TransformKind.CATEGORICAL, coverage=1.0)
    rec2 = _rec(transforms=[clean])
    assert build_deterministic_refinement(rec2, _parent(), classify_refinement_axis(rec2)) is None


# ── orchestration ─────────────────────────────────────────────────────────────


def test_apply_attaches_authored_elements_and_skips_the_rest() -> None:
    authorable = _unit_rec("mmHg", group_id="authorable")
    gated = _unit_rec("mmHg", group_id="gated", concept_mismatch=True)
    needs_llm = _rec(
        group_id="needs_llm",
        transforms=[
            TransformSpec(
                source_variable="s", target_cde_id="c", kind=TransformKind.CATEGORICAL, unmapped_source_codes=["9"]
            )
        ],
    )
    records = [authorable, gated, needs_llm]
    apply_deterministic_refinements(records, {"Blood pressure systolic measurement": _parent(units=None)})
    assert authorable.gencde is not None and authorable.gencde.units == "mmHg"
    assert gated.gencde is None
    assert needs_llm.gencde is None


def test_apply_is_idempotent_and_preserves_a_novel_gencde() -> None:
    """Re-running must not re-author, and a novel record's from-scratch element must survive untouched."""
    rec = _unit_rec("mmHg")
    novel = _rec(verdict="novel", group_id="n1", gencde=GenCDE(gencde_id="GENCDE:n1", preferred_name="kept"))
    cde_fields = {"Blood pressure systolic measurement": _parent(units=None)}

    apply_deterministic_refinements([rec, novel], cde_fields)
    first = rec.gencde
    apply_deterministic_refinements([rec, novel], cde_fields)
    assert rec.gencde is first  # same object — not re-authored
    assert novel.gencde is not None and novel.gencde.preferred_name == "kept"


def test_apply_skips_records_whose_parent_is_not_in_the_catalog() -> None:
    rec = _unit_rec("mmHg", cde_id="A CDE not in the catalog")
    apply_deterministic_refinements([rec], {})
    assert rec.gencde is None


# ── LLM authoring (mock responses) ────────────────────────────────────────────


def _dicts(hf):
    """One cohort whose 'nervous condition' field uses codes the parent CDE cannot express."""
    fld = hf.field(
        "nerv",
        "Nervous condition diagnosed",
        encoding="1=Epilepsy|2=Chronic fatigue|3=Insomnia",
        question_text="Have you been diagnosed with a nervous system condition?",
    )
    import numpy as np

    return [hf.embedded_dict("AoU", [fld], sem_vecs=np.ones((1, 8), dtype=np.float32))]


def _value_domain_rec() -> LeanBRecord:
    spec = TransformSpec(
        source_variable="AoU:nerv",
        target_cde_id="Past current neurological illness type",
        kind=TransformKind.CATEGORICAL,
        code_map={"1": "1"},
        unmapped_source_codes=["2", "3"],
        coverage=0.333,
        needs_review=True,
        rationale="Partial recode.",
    )
    return _rec(
        cde_id="Past current neurological illness type",
        concept="Diagnosed nervous system condition",
        member_variable_names=["AoU:nerv"],
        cohorts=["AoU"],
        transforms=[spec],
        rationale="Candidate covers neurological illness but lacks these specific condition types.",
    )


def _neuro_parent() -> Field:
    return Field(
        variable_name="Past current neurological illness type",
        description="Type of past or current neurological illness.",
        question_text="What type of neurological illness?",
        data_type="categorical",
        value_encoding_raw="1=Epilepsy|4=Stroke",
    )


def test_prepare_refine_skips_gated_and_already_authored(hf) -> None:
    """Only records that need a model get a prompt: not the gated, not the $0-authored ones."""
    from ddharmon.harmonization.refine import prepare_refine

    needs_llm = _value_domain_rec()
    gated = _rec(group_id="g2", concept_mismatch=True, transforms=list(needs_llm.transforms))
    already = _rec(group_id="g3", transforms=list(needs_llm.transforms))
    already.gencde = GenCDE(gencde_id="REFCDE:g3", parent_cde_id="p")
    adopt = _rec(group_id="g4", verdict="adopt", transforms=list(needs_llm.transforms))

    prompts = prepare_refine(
        [needs_llm, gated, already, adopt], _dicts(hf), {"Past current neurological illness type": _neuro_parent()}
    )
    assert [p.id for p in prompts] == ["refine:c1#g0"]
    body = prompts[0].user_prompt
    assert "Past current neurological illness type" in body  # the parent card
    assert "unmappable" in body  # the deterministic evidence
    assert "Candidate covers neurological illness" in body  # the assign rationale
    assert prompts[0].context["axis"] == "value_domain"


def _assemble(hf, payload: dict, rec: LeanBRecord | None = None):
    import json as _json

    from ddharmon.harmonization.refine import assemble_refine, prepare_refine

    rec = rec or _value_domain_rec()
    cde_fields = {"Past current neurological illness type": _neuro_parent()}
    prompts = prepare_refine([rec], _dicts(hf), cde_fields)
    assemble_refine(prompts, {prompts[0].id: _json.dumps(payload)}, [rec], cde_fields)
    return rec


def test_assemble_refine_extends_the_value_domain_and_keeps_the_parent(hf) -> None:
    """The canonical refinement: add the missing values, change nothing else."""
    rec = _assemble(
        hf,
        {
            "is_refinement": True,
            "relation": "skos:closeMatch",
            "axis": "value_domain",
            "added_permissible_values": [
                {"code": "2", "label": "Chronic fatigue"},
                {"code": "3", "label": "Insomnia"},
            ],
            "confidence": 0.9,
            "notes": "Extended with the condition types the cohorts record.",
        },
    )
    el = rec.gencde
    assert el is not None and el.parent_cde_id == "Past current neurological illness type"
    assert el.gencde_id == "REFCDE:c1#g0" and el.generated_by == "llm"
    assert el.relation == "skos:closeMatch" and el.refinement_axis == "value_domain"
    # the parent's own value survives, the additions sit alongside it
    assert [ro.label for ro in el.permissible_values] == ["Epilepsy", "Stroke", "Chronic fatigue", "Insomnia"]
    assert el.definition == "Type of past or current neurological illness."  # untouched
    assert el.changed_fields == ["permissible_values"] and el.over_refined is False
    assert el.value_coverage == 1.0  # all observed answer concepts now expressible


def test_assemble_refine_honours_the_models_own_refusal(hf) -> None:
    """`is_refinement: false` is a valid, expected answer — flagged, never silently forced through."""
    rec = _assemble(
        hf,
        {
            "is_refinement": False,
            "relation": "skos:relatedMatch",
            "axis": "scope",
            "confidence": 0.8,
            "notes": "This measures a different construct; a new element is needed.",
        },
    )
    el = rec.gencde
    assert el is not None and el.over_refined is True and el.needs_review is True
    assert "new element is needed" in el.rationale
    assert rec.verdict == "refine"  # flag-not-gate: the assign verdict is untouched


def test_assemble_refine_handles_an_unparseable_response(hf) -> None:
    """A garbled response must not silently produce a confident element."""
    from ddharmon.harmonization.refine import assemble_refine, prepare_refine

    rec = _value_domain_rec()
    cde_fields = {"Past current neurological illness type": _neuro_parent()}
    prompts = prepare_refine([rec], _dicts(hf), cde_fields)
    assemble_refine(prompts, {prompts[0].id: "not json at all"}, [rec], cde_fields)
    el = rec.gencde
    assert el is not None and el.needs_review is True
    # No delta is invented from a garbled response: the element is the bare parent, and its (genuinely
    # poor) coverage of the observed answer concepts is what the reviewer sees.
    assert el.changed_fields == [] and el.delta_size == 0.0
    assert el.value_coverage is not None and el.value_coverage < 1.0


def test_assemble_refine_deprecates_and_relabels(hf) -> None:
    rec = _assemble(
        hf,
        {
            "is_refinement": True,
            "relation": "skos:narrowMatch",
            "axis": "qualifier",
            "relabeled_values": {"1": "Epilepsy or seizure disorder"},
            "qualifier_added": "self-reported",
            "confidence": 0.7,
        },
    )
    el = rec.gencde
    assert el is not None
    assert [ro.label for ro in el.permissible_values] == ["Epilepsy or seizure disorder", "Stroke"]
    assert el.qualifier_added == "self-reported"
    assert el.relation == "skos:narrowMatch"


# ── re-targeting ──────────────────────────────────────────────────────────────


def test_retarget_closes_the_codes_the_refinement_added(hf) -> None:
    """The measurable win: codes the parent could not express become real mappings, with no LLM."""
    from ddharmon.harmonization.refine import retarget_refined_specs

    rec = _assemble(
        hf,
        {
            "is_refinement": True,
            "relation": "skos:closeMatch",
            "axis": "value_domain",
            "added_permissible_values": [
                {"code": "2", "label": "Chronic fatigue"},
                {"code": "3", "label": "Insomnia"},
            ],
            "confidence": 0.9,
        },
    )
    spec = rec.transforms[0]
    assert spec.unmapped_source_codes == ["2", "3"] and spec.coverage == 0.333  # before

    retarget_refined_specs([rec], _dicts(hf))

    assert spec.target_cde_id == "REFCDE:c1#g0"  # the edge names the element it actually recodes into
    assert spec.unmapped_source_codes == []  # both closed by label match
    assert spec.code_map == {"1": "1", "2": "2", "3": "3"}
    assert spec.coverage == 1.0
    assert spec.needs_review is False


def test_retarget_resolves_a_unit_residual_the_refinement_declared(hf) -> None:
    from ddharmon.harmonization.refine import retarget_refined_specs

    spec = TransformSpec(
        source_variable="AoU:w",
        target_cde_id="Body weight",
        kind=TransformKind.UNIT,
        needs_units=True,
        needs_review=True,
        source_unit="kg",
    )
    rec = _rec(cde_id="Body weight", transforms=[spec])
    rec.gencde = GenCDE(gencde_id="REFCDE:c1#g0", parent_cde_id="Body weight", units="g")

    retarget_refined_specs([rec], _dicts(hf))

    assert spec.needs_units is False and spec.needs_review is False
    assert spec.target_unit == "g" and spec.factor == 1000.0
    assert spec.kind == TransformKind.UNIT


def test_retarget_leaves_from_scratch_gencdes_alone(hf) -> None:
    """A novel record's GenCDE is the M12 path's business, not this one's."""
    from ddharmon.harmonization.refine import retarget_refined_specs

    spec = TransformSpec(source_variable="AoU:x", target_cde_id="orig", kind=TransformKind.CATEGORICAL)
    rec = _rec(verdict="novel", transforms=[spec])
    rec.gencde = GenCDE(gencde_id="GENCDE:c1#g0")  # no parent_cde_id
    retarget_refined_specs([rec], _dicts(hf))
    assert spec.target_cde_id == "orig"


def test_retarget_is_idempotent(hf) -> None:
    from ddharmon.harmonization.refine import retarget_refined_specs

    rec = _assemble(
        hf,
        {
            "is_refinement": True,
            "relation": "skos:closeMatch",
            "axis": "value_domain",
            "added_permissible_values": [{"code": "2", "label": "Chronic fatigue"}],
            "confidence": 0.9,
        },
    )
    retarget_refined_specs([rec], _dicts(hf))
    first = dict(rec.transforms[0].code_map)
    retarget_refined_specs([rec], _dicts(hf))
    assert rec.transforms[0].code_map == first
