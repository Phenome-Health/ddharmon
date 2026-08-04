"""Composite / derived-variable builder — the core library capability.

Covers the two published score shapes we must both support (Fried criteria-count, FI-Lab
deficit-accumulation), transcription discipline (an unstated cutoff stays unstated), the structural
grounding guard (a concept id the judge was not offered is dropped, and the component reported missing),
per-cohort feasibility, the derivation recipe, and the free reviewer re-derive loop.

No real LLM call and no embedding model: a fake ``complete`` returns canned JSON and retrieval runs
BM25-only (``embed=None``), which is also the documented no-extras code path.
"""

from __future__ import annotations

import json

import pytest

from ddharmon.harmonization import (
    CodingKind,
    CompositeKind,
    ScoreComponent,
    ScoreDefinition,
    assess_feasibility,
    build_composite_spec,
    build_concept_index,
    derive_composite,
    extract_score_definition,
    match_components,
    records_from_payload,
    spec_to_dict,
)
from ddharmon.harmonization.composite import ComponentCoding, _cut_point, shortlist_concepts
from ddharmon.harmonization.models import GenCDE, LeanBRecord, TransformSpec
from ddharmon.harmonization.parse import salvage_objects
from ddharmon.harmonization.score_sources import ScoreSource, from_text

# --- fixtures ---------------------------------------------------------------------------------


def _rec(
    concept_id: str,
    concept: str,
    cohorts: list[str],
    *,
    verdict: str = "adopt",
    cde_id: str | None = None,
    gencde: GenCDE | None = None,
    transforms: list[TransformSpec] | None = None,
    ideal: str = "",
) -> LeanBRecord:
    return LeanBRecord(
        cluster_id=concept_id.split("#")[0],
        verdict=verdict,
        route="assigned" if cde_id else "gencde_residual",
        group_id=concept_id,
        concept=concept,
        cde_id=cde_id,
        ideal_cde=ideal,
        cohorts=cohorts,
        cross_cohort=len(cohorts) >= 2,
        n_members=len(cohorts),
        member_variable_names=[f"{c}:{concept_id}_var" for c in cohorts],
        gencde=gencde,
        transforms=transforms or [],
    )


@pytest.fixture
def run_records() -> list[LeanBRecord]:
    """A miniature run: grip strength is UKBB-only, so a Fried score is computable there and nowhere else."""
    return [
        _rec("c1#g0", "Hand grip strength maximum isometric force in kilograms", ["UKBB"], cde_id="GripStrengthMax"),
        _rec("c2#g0", "Unintentional body weight loss in the past year", ["UKBB", "AoU"], cde_id="WeightLossUnintent"),
        _rec("c3#g0", "Self-reported exhaustion or fatigue severity in the past week", ["UKBB", "AoU", "CLSA"]),
        _rec("c4#g0", "Usual walking pace / gait speed over a measured course", ["UKBB", "CLSA"], cde_id="GaitSpeed"),
        _rec("c5#g0", "Weekly physical activity level from questionnaire items", ["UKBB", "AoU", "CLSA"]),
        _rec("c6#g0", "Blood haemoglobin concentration", ["UKBB", "CLSA"], cde_id="Hemoglobin"),
        _rec("c7#g0", "Hypertension diagnosis ever told by a doctor", ["AoU", "CLSA"], cde_id="HypertensionDx"),
    ]


_FRIED_JSON = json.dumps(
    {
        "name": "Fried frailty phenotype",
        "citation": "Fried et al. 2001, J Gerontol A",
        "kind": "criteria_count",
        "combinationRule": "Count the criteria present; 3 or more of 5 indicates frailty.",
        "threshold": "frail if >=3 of 5 criteria",
        "notes": "1-2 criteria = pre-frail.",
        "components": [
            {
                "name": "Weight loss",
                "definition": "Unintentional loss of >=10 lbs in the prior year",
                "required": True,
                "coding": {"kind": "threshold", "cutoff": ">=10 lbs in the prior year", "statedInSource": True},
            },
            {
                "name": "Exhaustion",
                "definition": "Self-reported exhaustion (CES-D items)",
                "required": True,
                "coding": {"kind": "categorical", "codeMap": {"3": "1", "4": "1"}, "statedInSource": True},
            },
            {
                "name": "Low physical activity",
                "definition": "Lowest quintile of weekly kcal expenditure, sex-stratified",
                "required": True,
                "coding": {
                    "kind": "data_dependent",
                    "cutoff": "lowest quintile, sex-stratified",
                    "statedInSource": True,
                },
            },
            {
                "name": "Slow gait speed",
                "definition": "Slowest 20% on a 15-foot walk, stratified by sex and height",
                "required": True,
                "coding": {"kind": "data_dependent", "cutoff": "slowest 20%", "statedInSource": True},
            },
            {
                "name": "Weak grip strength",
                "definition": "Lowest 20% of grip strength, stratified by sex and BMI",
                "required": True,
                "coding": {"kind": "unstated", "statedInSource": False},
            },
        ],
    }
)

_FI_LAB_JSON = json.dumps(
    {
        "name": "FI-Lab",
        "citation": "Blodgett et al. 2017, GeroScience",
        "kind": "deficit_proportion",
        "combinationRule": "Each item is coded 1 when outside its sex-specific reference range; the score is "
        "the number of deficits divided by the number of items considered (range 0-1).",
        "threshold": "",
        "components": [
            {
                "name": "Haemoglobin",
                "definition": "Blood haemoglobin concentration",
                "required": True,
                "coding": {
                    "kind": "threshold",
                    "referenceRange": "130-170 g/L (men), 120-150 g/L (women)",
                    "units": "g/L",
                    "statedInSource": True,
                },
            },
            {
                "name": "Systolic blood pressure",
                "definition": "Seated systolic blood pressure",
                "required": True,
                "coding": {"kind": "threshold", "referenceRange": "90-140 mmHg", "statedInSource": True},
            },
        ],
    }
)


def _fake_complete(*responses: str):
    """A canned ``complete`` that returns each response in turn and records the prompts it saw."""
    calls: list[dict] = []

    def complete(prompt: str, *, system: str = "", max_tokens: int = 0) -> str:
        calls.append({"prompt": prompt, "system": system, "max_tokens": max_tokens})
        return responses[min(len(calls) - 1, len(responses) - 1)]

    complete.calls = calls  # type: ignore[attr-defined]
    return complete


def _match_json(pairs: dict[str, str | None], confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "matches": [
                {"component": name, "conceptId": cid, "confidence": confidence, "rationale": "measures it"}
                for name, cid in pairs.items()
            ]
        }
    )


# --- stage 1: transcription -------------------------------------------------------------------


def test_extract_fried_criteria_count_round_trips():
    """A criteria-based phenotype keeps its 5 criteria, its cut-point wording, and its per-item coding."""
    definition = extract_score_definition(from_text("Fried phenotype paper text"), _fake_complete(_FRIED_JSON))
    assert definition.name == "Fried frailty phenotype"
    assert definition.kind is CompositeKind.CRITERIA_COUNT
    assert len(definition.components) == 5 and len(definition.required_components) == 5
    assert definition.threshold == "frail if >=3 of 5 criteria"
    assert definition.source is not None and definition.provenance == "pasted text"
    codings = {c.name: c.coding.kind for c in definition.components}
    assert codings["Weight loss"] is CodingKind.THRESHOLD
    assert codings["Slow gait speed"] is CodingKind.DATA_DEPENDENT


def test_extract_fi_lab_deficit_proportion_keeps_reference_ranges_verbatim():
    """The deficit-accumulation shape is preserved, and sex-specific ranges are NOT parsed into numbers."""
    definition = extract_score_definition(from_text("FI-Lab paper text"), _fake_complete(_FI_LAB_JSON))
    assert definition.kind is CompositeKind.DEFICIT_PROPORTION
    haemoglobin = definition.components[0]
    assert haemoglobin.coding.reference_range == "130-170 g/L (men), 120-150 g/L (women)"
    assert haemoglobin.coding.units == "g/L"
    assert haemoglobin.coding.needs_review is False  # stated in the source -> auto-codable


def test_unstated_coding_is_flagged_for_review_never_invented():
    """A component the source names without a cutoff stays UNSTATED and needs_review — no invented threshold."""
    definition = extract_score_definition(from_text("Fried phenotype paper text"), _fake_complete(_FRIED_JSON))
    grip = next(c for c in definition.components if c.name == "Weak grip strength")
    assert grip.coding.kind is CodingKind.UNSTATED
    assert grip.coding.cutoff == "" and grip.coding.stated_in_source is False
    assert grip.coding.needs_review is True


def test_arithmetic_and_data_dependent_always_need_review():
    """Mirrors the transform layer's rule: a formula or a sample-relative rule is never auto-applied."""
    for kind in (CodingKind.ARITHMETIC, CodingKind.DATA_DEPENDENT, CodingKind.UNSTATED):
        assert ComponentCoding(kind=kind, stated_in_source=True).needs_review is True
    assert ComponentCoding(kind=CodingKind.THRESHOLD, cutoff="<130", stated_in_source=True).needs_review is False


def test_extract_raises_when_the_document_defines_no_score():
    """A page that does not define a composite is an honest error, not a plausible guess."""
    with pytest.raises(ValueError, match="no score components"):
        extract_score_definition(from_text("A journal landing page."), _fake_complete('{"name": "nothing"}'))


def test_extract_salvages_a_truncated_component_array():
    """A long component list truncated at the token cap keeps its complete entries."""
    truncated = (
        '{"name": "FI-Combined", "kind": "deficit_proportion", "components": ['
        '{"name": "Haemoglobin", "coding": {"kind": "threshold", "cutoff": "<130 g/L", "statedInSource": true}},'
        '{"name": "Albumin", "coding": {"kind": "threshold", "cutoff": "<35 g/L", "statedInSource": true}},'
        '{"name": "Creatinine", "coding": {"kind": "thresh'
    )
    definition = extract_score_definition(from_text("FI-Combined text"), _fake_complete(truncated))
    assert [c.name for c in definition.components] == ["Haemoglobin", "Albumin"]
    assert len(salvage_objects(truncated, "components")) == 2


def test_extract_prompt_forbids_supplying_the_score_from_memory():
    """The transcription discipline must be IN the prompt — it is the whole guard against a recalled score."""
    complete = _fake_complete(_FRIED_JSON)
    extract_score_definition(from_text("paper"), complete)
    system = complete.calls[0]["system"]
    assert "TRANSCRIBER" in system and "your own knowledge" in system
    assert '"unstated"' in system


# --- the concept index (the closed world) ------------------------------------------------------


def test_concept_index_keeps_single_cohort_concepts(run_records):
    """Unlike the analysis-ideas digest, single-cohort concepts stay: they still make a score computable THERE."""
    index = build_concept_index(run_records)
    assert len(index) == len(run_records)
    grip = next(e for e in index if e.concept_id == "c1#g0")
    assert grip.cohorts == ["UKBB"]  # single-cohort, and still indexed
    assert grip.column == "GripStrengthMax"  # the harmonized column name downstream


def test_concept_index_column_falls_back_to_gencde_then_id():
    """A novel concept's column is its GenCDE name; with neither CDE nor GenCDE, the record id."""
    gencde = GenCDE(gencde_id="GENCDE:c9#g0", preferred_name="exhaustion_frequency", data_type="categorical")
    index = build_concept_index(
        [
            _rec("c9#g0", "Exhaustion frequency", ["AoU"], verdict="novel", gencde=gencde),
            _rec("c10#g0", "Something unassigned", ["AoU"], verdict="novel"),
        ]
    )
    assert index[0].column == "exhaustion_frequency"
    assert index[1].column == "c10#g0"


def test_concept_index_skips_records_with_nothing_to_match():
    index = build_concept_index([_rec("", "", ["AoU"]), _rec("c1#g0", "", ["AoU"])])
    assert index == []


def test_concept_index_carries_units_from_transforms():
    """Units reach the judge's candidate lines, which is what lets it spot a unit-mismatch match."""
    transform = TransformSpec(
        source_variable="UKBB:grip", target_cde_id="GripStrengthMax", kind="unit", target_unit="kg"
    )
    index = build_concept_index([_rec("c1#g0", "Grip strength", ["UKBB"], cde_id="G", transforms=[transform])])
    assert index[0].units == "kg"


# --- the shared serialized-run reader ----------------------------------------------------------


def test_records_from_payload_reads_both_casings_and_both_shapes():
    """One reader for every consumer of a serialized run (CLI harness, web backend) — camelCase UI contract
    and snake_case core records.json, bare list or nested under `result`."""
    ui_shape = {
        "result": {
            "records": [
                {
                    "id": "cA#g0",
                    "clusterId": "cA",
                    "groupId": "cA#g0",
                    "concept": "Blood haemoglobin concentration",
                    "verdict": "refine",
                    "cde": {"id": "Hemoglobin"},
                    "idealCde": "haemoglobin in g/L",
                    "cohorts": ["UKBB", "CLSA"],
                    "members": ["UKBB:hb", "CLSA:hgb"],
                    "nMembers": 2,
                    "transforms": [{"sourceVariable": "UKBB:hb", "kind": "unit", "targetUnit": "g/L"}],
                }
            ]
        }
    }
    core_shape = [
        {
            "cluster_id": "cA",
            "group_id": "cA#g0",
            "concept": "Blood haemoglobin concentration",
            "verdict": "refine",
            "cde_id": "Hemoglobin",
            "cohorts": ["UKBB", "CLSA"],
            "member_variable_names": ["UKBB:hb", "CLSA:hgb"],
            "transforms": [{"source_variable": "UKBB:hb", "kind": "unit", "target_unit": "g/L"}],
        }
    ]
    for payload in (ui_shape, core_shape, {"records": core_shape}):
        index = build_concept_index(records_from_payload(payload))
        assert len(index) == 1
        assert index[0].concept_id == "cA#g0"
        assert index[0].column == "Hemoglobin"
        assert index[0].cohorts == ["CLSA", "UKBB"]
        assert index[0].members == ["UKBB:hb", "CLSA:hgb"]
        assert index[0].units == "g/L"  # recovered from the transform's target unit


def test_records_from_payload_recovers_a_gencde_column():
    payload = {
        "records": [
            {
                "groupId": "cB#g0",
                "concept": "Exhaustion frequency",
                "verdict": "novel",
                "gencde": {"gencdeId": "GENCDE:cB#g0", "preferredName": "exhaustion_freq", "dataType": "categorical"},
                "cohorts": ["AoU"],
            }
        ]
    }
    index = build_concept_index(records_from_payload(payload))
    assert index[0].column == "exhaustion_freq" and index[0].data_type == "categorical"


def test_records_from_payload_tolerates_junk_and_unknown_transform_kinds():
    payload = {
        "records": [
            "not a record",
            {"groupId": "cC#g0", "concept": "X", "cohorts": ["A"], "transforms": [{"kind": "who-knows"}]},
        ]
    }
    records = records_from_payload(payload)
    assert len(records) == 1 and records[0].transforms[0].kind == "none"


def test_records_from_payload_rejects_a_shape_with_no_records():
    with pytest.raises(ValueError, match="no `records` array found"):
        records_from_payload({"summary": {}})
    with pytest.raises(ValueError, match="cannot read records"):
        records_from_payload("a string")


# --- stage 2: matching + the grounding guard ---------------------------------------------------


def test_retrieval_shortlists_the_relevant_concepts_without_an_embedder(run_records):
    """BM25-only retrieval (no `embeddings` extra installed) still surfaces the right concept per component."""
    index = build_concept_index(run_records)
    components = [
        ScoreComponent(name="Weak grip strength", definition="Lowest 20% of grip strength by dynamometry"),
        ScoreComponent(name="Haemoglobin", definition="Blood haemoglobin concentration"),
    ]
    shortlists = shortlist_concepts(components, index, embed=None, top_k=3)
    assert "c1#g0" in [e.concept_id for e in shortlists["Weak grip strength"]]
    assert shortlists["Haemoglobin"][0].concept_id == "c6#g0"


def test_match_drops_a_concept_id_the_judge_was_never_offered(run_records):
    """The structural grounding guard: an unretrieved (or hallucinated) id becomes an honest gap."""
    index = build_concept_index(run_records)
    definition = ScoreDefinition(
        name="Fake score",
        components=[ScoreComponent(name="Weak grip strength", definition="grip strength dynamometry")],
    )
    complete = _fake_complete(_match_json({"Weak grip strength": "TOTALLY-MADE-UP-ID"}))
    matches = match_components(definition, index, complete, embed=None)
    assert matches[0].matched is False and matches[0].concept_id is None
    assert matches[0].shortlist  # retrieval DID offer candidates — the judge's answer was the problem


def test_match_rejects_an_id_borrowed_from_another_components_shortlist(run_records):
    """A real id is still invalid if it was not on THAT component's list — ids are per-component."""
    index = build_concept_index(run_records)
    definition = ScoreDefinition(
        name="Fake score",
        components=[
            ScoreComponent(name="Weak grip strength", definition="grip strength dynamometry"),
            ScoreComponent(name="Haemoglobin", definition="blood haemoglobin concentration"),
        ],
    )
    # Offer only 1 candidate each, then answer with the OTHER component's concept.
    complete = _fake_complete(_match_json({"Weak grip strength": "c6#g0", "Haemoglobin": "c6#g0"}))
    matches = match_components(definition, index, complete, embed=None, top_k=1)
    by_component = {m.component: m for m in matches}
    assert by_component["Weak grip strength"].matched is False
    assert by_component["Haemoglobin"].concept_id == "c6#g0"


def test_match_populates_the_concept_and_its_cohorts(run_records):
    index = build_concept_index(run_records)
    definition = ScoreDefinition(name="s", components=[ScoreComponent(name="Haemoglobin", definition="haemoglobin")])
    matches = match_components(definition, index, _fake_complete(_match_json({"Haemoglobin": "c6#g0"})), embed=None)
    assert matches[0].column == "Hemoglobin"
    assert matches[0].cohorts == ["CLSA", "UKBB"]
    assert matches[0].source_variables == ["UKBB:c6#g0_var", "CLSA:c6#g0_var"]
    assert matches[0].confidence == pytest.approx(0.9)


def test_match_prompt_tells_the_judge_to_prefer_a_gap_over_a_guess(run_records):
    index = build_concept_index(run_records)
    definition = ScoreDefinition(name="s", components=[ScoreComponent(name="Haemoglobin", definition="haemoglobin")])
    complete = _fake_complete(_match_json({"Haemoglobin": "c6#g0"}))
    match_components(definition, index, complete, embed=None)
    system = complete.calls[0]["system"]
    assert "Prefer null when unsure" in system
    assert "WHAT IS MEASURED, not on shared words" in system


def test_overrides_pin_and_drop_without_calling_the_judge(run_records):
    """The reviewer's accept/swap/drop loop is free: fully-pinned components make no LLM call at all."""
    index = build_concept_index(run_records)
    definition = ScoreDefinition(
        name="s",
        components=[ScoreComponent(name="Grip"), ScoreComponent(name="Gait")],
    )
    complete = _fake_complete("{}")
    matches = match_components(definition, index, complete, embed=None, overrides={"Grip": "c1#g0", "Gait": None})
    assert complete.calls == []  # nothing to ask
    by_component = {m.component: m for m in matches}
    assert by_component["Grip"].concept_id == "c1#g0" and by_component["Grip"].pinned
    assert by_component["Grip"].confidence == 1.0
    assert by_component["Gait"].matched is False and by_component["Gait"].rationale == "dropped by reviewer"


def test_a_partially_pinned_rederive_only_asks_about_the_rest(run_records):
    index = build_concept_index(run_records)
    definition = ScoreDefinition(
        name="s",
        components=[
            ScoreComponent(name="Grip", definition="grip strength"),
            ScoreComponent(name="Hb", definition="haemoglobin"),
        ],
    )
    complete = _fake_complete(_match_json({"Hb": "c6#g0"}))
    matches = match_components(definition, index, complete, embed=None, overrides={"Grip": "c1#g0"})
    assert len(complete.calls) == 1
    assert "Grip" not in complete.calls[0]["prompt"]  # the pinned component is not re-litigated
    assert {m.component: m.matched for m in matches} == {"Grip": True, "Hb": True}


# --- stage 3: feasibility ---------------------------------------------------------------------


def _fried_matches(run_records, mapping: dict[str, str | None]):
    definition = extract_score_definition(from_text("t"), _fake_complete(_FRIED_JSON))
    index = build_concept_index(run_records)
    return definition, match_components(definition, index, _fake_complete("{}"), embed=None, overrides=mapping)


def test_feasibility_full_when_every_required_component_matched(run_records):
    definition, matches = _fried_matches(
        run_records,
        {
            "Weight loss": "c2#g0",
            "Exhaustion": "c3#g0",
            "Low physical activity": "c5#g0",
            "Slow gait speed": "c4#g0",
            "Weak grip strength": "c1#g0",
        },
    )
    report = assess_feasibility(definition, matches, cohorts=["UKBB", "AoU", "CLSA"])
    assert report.verdict == "full"
    assert report.n_required == 5 and report.n_required_matched == 5
    # Grip strength exists only in UKBB, so only UKBB can actually compute the phenotype.
    assert report.computable_cohorts == ["UKBB"]
    clsa = next(c for c in report.per_cohort if c.cohort == "CLSA")
    assert clsa.computable is False and "Weak grip strength" in clsa.missing
    assert "Weight loss" in clsa.missing  # AoU/UKBB-only concept


def test_feasibility_partial_names_the_gaps_and_warns_against_comparison(run_records):
    definition, matches = _fried_matches(
        run_records,
        {
            "Weight loss": "c2#g0",
            "Exhaustion": "c3#g0",
            "Low physical activity": None,
            "Slow gait speed": None,
            "Weak grip strength": None,
        },
    )
    report = assess_feasibility(definition, matches)
    assert report.verdict == "partial"
    assert report.n_required_matched == 2
    assert set(report.missing) == {"Low physical activity", "Slow gait speed", "Weak grip strength"}
    assert any("NOT the published score" in c for c in report.caveats)


def test_feasibility_infeasible_when_nothing_matched(run_records):
    definition, matches = _fried_matches(
        run_records,
        dict.fromkeys(
            ["Weight loss", "Exhaustion", "Low physical activity", "Slow gait speed", "Weak grip strength"], None
        ),
    )
    report = assess_feasibility(definition, matches)
    assert report.verdict == "infeasible" and report.matched == []
    assert report.computable_cohorts == []


def test_feasibility_always_carries_the_metadata_only_caveat(run_records):
    definition, matches = _fried_matches(run_records, {"Weight loss": "c2#g0"})
    report = assess_feasibility(definition, matches)
    assert any("effective N and statistical power cannot be derived" in c for c in report.caveats)


def test_feasibility_reports_matched_components_whose_coding_needs_a_reviewer(run_records):
    """Matched but uncodable is its own state — the concept is there, the source's rule is not."""
    definition, matches = _fried_matches(run_records, {"Weak grip strength": "c1#g0", "Weight loss": "c2#g0"})
    report = assess_feasibility(definition, matches)
    assert "Weak grip strength" in report.needs_review  # coding kind was `unstated`
    assert "Weight loss" not in report.needs_review  # a stated cutoff
    assert any("does not invent cutoffs" in c for c in report.caveats)


def test_optional_components_do_not_block_computability(run_records):
    definition = ScoreDefinition(
        name="s",
        components=[
            ScoreComponent(
                name="Core", required=True, coding=ComponentCoding(CodingKind.IDENTITY, stated_in_source=True)
            ),
            ScoreComponent(name="Nice to have", required=False),
        ],
    )
    index = build_concept_index(run_records)
    matches = match_components(
        definition, index, _fake_complete("{}"), embed=None, overrides={"Core": "c3#g0", "Nice to have": None}
    )
    report = assess_feasibility(definition, matches)
    assert report.verdict == "full" and report.n_required == 1
    assert {c.cohort for c in report.per_cohort if c.computable} == {"UKBB", "AoU", "CLSA"}


def test_cohorts_supplying_nothing_still_appear(run_records):
    """Passing the run's cohorts surfaces a cohort with zero coverage instead of hiding it."""
    definition, matches = _fried_matches(run_records, {"Weak grip strength": "c1#g0"})
    report = assess_feasibility(definition, matches, cohorts=["UKBB", "AoU", "CLSA", "MESA"])
    mesa = next(c for c in report.per_cohort if c.cohort == "MESA")
    assert mesa.present == [] and mesa.computable is False


# --- stage 4: the derivation recipe ------------------------------------------------------------


def test_criteria_count_recipe_counts_then_thresholds(run_records):
    definition, matches = _fried_matches(
        run_records,
        {
            "Weight loss": "c2#g0",
            "Exhaustion": "c3#g0",
            "Low physical activity": "c5#g0",
            "Slow gait speed": "c4#g0",
            "Weak grip strength": "c1#g0",
        },
    )
    spec = build_composite_spec(definition, matches, assess_feasibility(definition, matches))
    kinds = [s.kind for s in spec.derivation]
    assert kinds == ["code_component"] * 5 + ["combine", "threshold"]
    assert spec.units == "count"
    combine = next(s for s in spec.derivation if s.kind == "combine")
    assert combine.expression.startswith("score = weight_loss + exhaustion")
    assert next(s for s in spec.derivation if s.kind == "threshold").expression == "positive = score >= 3"
    # The stated cutoff is applied; the unstated one is a review stub, not a guess.
    weight = next(s for s in spec.derivation if s.component == "Weight loss")
    assert "outside('WeightLossUnintent', '>=10 lbs in the prior year')" in weight.expression
    grip = next(s for s in spec.derivation if s.component == "Weak grip strength")
    assert grip.needs_review and "REVIEW" in grip.expression and "no coding rule" in grip.expression


def test_deficit_proportion_recipe_divides_by_available_items(run_records):
    """The FI denominator is what is AVAILABLE — and the spec says so, because that breaks comparability."""
    definition = extract_score_definition(from_text("t"), _fake_complete(_FI_LAB_JSON))
    index = build_concept_index(run_records)
    matches = match_components(
        definition,
        index,
        _fake_complete("{}"),
        embed=None,
        overrides={"Haemoglobin": "c6#g0", "Systolic blood pressure": None},
    )
    feasibility = assess_feasibility(definition, matches)
    spec = build_composite_spec(definition, matches, feasibility)
    combine = next(s for s in spec.derivation if s.kind == "combine")
    assert combine.expression == "score = (haemoglobin) / 1"
    assert "denominator differs whenever coverage is partial" in combine.description
    assert spec.units == "proportion (0-1)"
    assert any("[0, 1]" in r for r in spec.validation_rules)
    assert any("modified index" in r for r in spec.validation_rules)


def test_data_dependent_component_is_an_apply_time_step(run_records):
    definition, matches = _fried_matches(run_records, {"Slow gait speed": "c4#g0"})
    spec = build_composite_spec(definition, matches, assess_feasibility(definition, matches))
    step = next(s for s in spec.derivation if s.component == "Slow gait speed")
    assert step.expression.startswith("# APPLY-TIME")
    assert "not derivable from metadata" in step.description


def test_z_composite_and_custom_combinations_are_review_stubs(run_records):
    index = build_concept_index(run_records)
    for kind in (CompositeKind.Z_COMPOSITE, CompositeKind.CUSTOM):
        definition = ScoreDefinition(name="s", kind=kind, components=[ScoreComponent(name="A")])
        matches = match_components(definition, index, _fake_complete("{}"), embed=None, overrides={"A": "c3#g0"})
        spec = build_composite_spec(definition, matches, assess_feasibility(definition, matches))
        combine = next(s for s in spec.derivation if s.kind == "combine")
        assert combine.needs_review is True


def test_weighted_sum_without_full_weights_is_a_review_stub(run_records):
    index = build_concept_index(run_records)
    definition = ScoreDefinition(
        name="Charlson-like",
        kind=CompositeKind.WEIGHTED_SUM,
        components=[ScoreComponent(name="A", weight=2.0), ScoreComponent(name="B")],
    )
    matches = match_components(
        definition, index, _fake_complete("{}"), embed=None, overrides={"A": "c3#g0", "B": "c5#g0"}
    )
    spec = build_composite_spec(definition, matches, assess_feasibility(definition, matches))
    combine = next(s for s in spec.derivation if s.kind == "combine")
    assert "REVIEW" in combine.expression and "stated weight" in combine.description


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        ("frail if >=3 of 5 criteria", ">= 3"),
        ("score ≥ 0.25", ">= 0.25"),
        ("cut-point < 8", "< 8"),
        ("no number", ""),
        # Band lists are severity strata, NOT a cut-point — parsing the first number would invent a threshold
        # the source never stated (this is what the real FI-Lab paper's "categories" wording produced).
        ("Frailty categories: 0–0.1, 0.1–0.2, 0.2–0.3, 0.3–0.4, 0.4+", ""),
        ("cut-offs of 5, 10, 15 and 20", ""),
        ("mild range 0.1–0.2", ""),
    ],
)
def test_cut_point_parsing(threshold, expected):
    assert _cut_point(threshold) == expected


def test_a_document_claiming_more_items_than_it_lists_is_flagged_not_filled_in():
    """The real failure mode from the FI-Lab PMC page: the item table did not survive text extraction.

    The score's rule and item COUNT are in the prose, the 32 items are not. The builder must say the document
    was incomplete rather than reconstructing the list from the model's knowledge of the index.
    """
    partial = json.dumps(
        {
            "name": "FI-Lab",
            "kind": "deficit_proportion",
            "statedNItems": 32,
            "combinationRule": "deficits / 32 items",
            "components": [
                {
                    "name": "Haemoglobin",
                    "required": True,
                    "coding": {"kind": "threshold", "referenceRange": "130-170 g/L", "statedInSource": True},
                }
            ],
        }
    )
    definition = extract_score_definition(from_text("FI-Lab prose without the table"), _fake_complete(partial))
    assert definition.stated_n_items == 32
    assert definition.under_enumerated == 31
    report = assess_feasibility(definition, [])
    assert any(
        "only 1 item(s) could be read" in c and "NOT filled in from prior knowledge" in c for c in report.caveats
    )


def test_extract_prompt_demands_one_entry_per_item():
    """Guards the bug this fixed: the model summarized 32 lab tests into a single "32 items" component."""
    complete = _fake_complete(_FI_LAB_JSON)
    extract_score_definition(from_text("paper"), complete)
    system = complete.calls[0]["system"]
    assert "ONE component per SCORED ITEM" in system
    assert "32 laboratory tests" in system  # the anti-pattern is shown, not just described
    assert "statedNItems" in system


def test_a_transcription_with_no_required_flags_treats_every_component_as_required(run_records):
    """A score whose items are all `required: false` is a transcription artifact, not a score with no needs —
    otherwise n_required is 0 and a fully-matched score reads as "infeasible"."""
    definition = ScoreDefinition(
        name="s",
        components=[ScoreComponent(name="A", required=False), ScoreComponent(name="B", required=False)],
    )
    index = build_concept_index(run_records)
    matches = match_components(
        definition, index, _fake_complete("{}"), embed=None, overrides={"A": "c3#g0", "B": "c5#g0"}
    )
    report = assess_feasibility(definition, matches)
    assert report.n_required == 2 and report.verdict == "full"


def test_nothing_matched_yields_a_not_computable_stub_not_a_division_by_zero(run_records):
    """`score = (0) / 0` looked runnable and was a lie — the combine step must refuse instead."""
    definition = extract_score_definition(from_text("t"), _fake_complete(_FI_LAB_JSON))
    matches = match_components(
        definition,
        build_concept_index(run_records),
        _fake_complete("{}"),
        embed=None,
        overrides={"Haemoglobin": None, "Systolic blood pressure": None},
    )
    spec = build_composite_spec(definition, matches, assess_feasibility(definition, matches))
    combine = next(s for s in spec.derivation if s.kind == "combine")
    assert "NOT COMPUTABLE" in combine.expression
    assert "/ 0" not in combine.expression and spec.units == ""


# --- end to end -------------------------------------------------------------------------------


def test_derive_composite_runs_two_calls_and_reports_them(run_records):
    complete = _fake_complete(
        _FRIED_JSON,
        _match_json(
            {
                "Weight loss": "c2#g0",
                "Exhaustion": "c3#g0",
                "Low physical activity": "c5#g0",
                "Slow gait speed": "c4#g0",
                "Weak grip strength": "c1#g0",
            }
        ),
    )
    result = derive_composite(from_text("Fried paper"), run_records, complete, embed=None)
    assert result.calls_made == 2  # extract + judge; stages 3-4 are deterministic
    assert result.n_concepts_indexed == len(run_records)
    assert result.spec.feasibility.verdict == "full"
    assert result.spec.feasibility.computable_cohorts == ["UKBB"]
    assert len(result.spec.matched) == 5 and result.spec.unmatched == []


def test_rederive_from_a_definition_skips_extraction(run_records):
    """The re-derive path: reuse the transcribed definition, pin every match -> zero LLM calls."""
    definition = extract_score_definition(from_text("t"), _fake_complete(_FRIED_JSON))
    complete = _fake_complete("{}")
    result = derive_composite(
        definition,
        run_records,
        complete,
        embed=None,
        overrides={
            "Weight loss": "c2#g0",
            "Exhaustion": "c3#g0",
            "Low physical activity": "c5#g0",
            "Slow gait speed": "c4#g0",
            "Weak grip strength": None,
        },
    )
    assert result.calls_made == 0 and complete.calls == []
    assert result.spec.feasibility.verdict == "partial"
    assert result.spec.feasibility.missing == ["Weak grip strength"]


def test_derive_composite_on_a_run_with_no_concepts_is_infeasible_not_an_error():
    complete = _fake_complete(_FRIED_JSON, '{"matches": []}')
    result = derive_composite(from_text("Fried paper"), [], complete, embed=None)
    assert result.n_concepts_indexed == 0
    assert result.spec.feasibility.verdict == "infeasible"
    assert result.spec.feasibility.per_cohort == []


def test_spec_to_dict_is_json_serializable_and_camel_cased(run_records):
    complete = _fake_complete(_FI_LAB_JSON, _match_json({"Haemoglobin": "c6#g0"}))
    spec = derive_composite(from_text("FI-Lab paper"), run_records, complete, embed=None).spec
    payload = spec_to_dict(spec)
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["definition"]["kind"] == "deficit_proportion"
    assert round_tripped["definition"]["sourceSha256"]
    assert round_tripped["feasibility"]["verdict"] == "partial"
    assert round_tripped["matches"][0]["conceptId"] == "c6#g0"
    assert round_tripped["derivation"][0]["needsReview"] is False
    assert "perCohort" in round_tripped["feasibility"]


def test_embed_callable_is_used_when_supplied(run_records):
    """With an embedder, retrieval is hybrid — the dense half must actually be consulted."""
    seen: list[list[str]] = []

    def embed(texts: list[str]):
        seen.append(texts)
        import numpy as np

        vectors = np.zeros((len(texts), 4), dtype="float32")
        vectors[:, 0] = 1.0
        return vectors

    index = build_concept_index(run_records)
    definition = ScoreDefinition(name="s", components=[ScoreComponent(name="Hb", definition="haemoglobin")])
    match_components(definition, index, _fake_complete(_match_json({"Hb": "c6#g0"})), embed=embed)
    assert len(seen) == 2  # the index corpus, then the component queries
    assert len(seen[0]) == len(index) and seen[1] == ["Hb. haemoglobin"]


def test_source_provenance_survives_into_the_spec(run_records):
    source = ScoreSource(text="FI-Lab definition", kind="url", provenance="https://doi.org/10.1007/x")
    complete = _fake_complete(_FI_LAB_JSON, _match_json({"Haemoglobin": "c6#g0"}))
    spec = derive_composite(source, run_records, complete, embed=None).spec
    assert spec.definition.provenance == "https://doi.org/10.1007/x"
    assert spec_to_dict(spec)["definition"]["provenance"] == "https://doi.org/10.1007/x"
