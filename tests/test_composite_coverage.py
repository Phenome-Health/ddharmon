"""Score-builder UNION coverage + answer-option granularity.

Coverage is a per-cohort UNION over judge-accepted MEMBERS (above a confidence floor) across ALL the
groups a component reached — decoupled from the single surfaced winner group. A multi-select checklist
variable can supply several components at once, each attributed to the answer OPTION that measures it.

These are pure/deterministic: a fake ``complete`` returns canned judge JSON and retrieval runs BM25-only
(``embed=None``). The existing group-concept + variable-only tests live in ``test_harmonization_composite``;
this file adds only the coverage-model and answer-option behaviour.
"""

from __future__ import annotations

import json

import pytest

from ddharmon.harmonization import (
    CompositeKind,
    ScoreComponent,
    ScoreDefinition,
    derive_composite,
    spec_to_dict,
)
from ddharmon.harmonization.composite import (
    _COVERAGE_CONFIDENCE_FLOOR as FLOOR,
)
from ddharmon.harmonization.composite import (
    ConceptEntry,
    _answer_labels,
    _base_variable_id,
    _coverage_from_rated,
    _member_payload,
    build_variable_index,
)
from ddharmon.harmonization.models import LeanBRecord

# --- helpers ----------------------------------------------------------------------------------


def _grec(group_id, concept, members, cohorts, *, ideal=""):
    return LeanBRecord(
        cluster_id=group_id.split("#")[0],
        verdict="adopt",
        route="assigned",
        group_id=group_id,
        concept=concept,
        ideal_cde=ideal,
        cohorts=cohorts,
        cross_cohort=len(cohorts) >= 2,
        n_members=len(members),
        member_variable_names=list(members),
    )


def _fake_complete(response: str):
    def complete(prompt: str, *, system: str = "", max_tokens: int = 0) -> str:
        return response

    return complete


def _multi_match_json(by_key):
    matches = []
    for key, items in by_key.items():
        for vid, conf in items:
            matches.append({"componentKey": key, "conceptId": vid, "confidence": conf, "rationale": "measures it"})
    return json.dumps({"matches": matches})


ABOVE = round(FLOOR + 0.2, 3)  # comfortably above the floor
BELOW = round(FLOOR - 0.2, 3)  # comfortably below the floor


# --- unit: _coverage_from_rated ---------------------------------------------------------------


def _cataract_groups():
    ge_ukbb = ConceptEntry(concept_id="g_cat_ukbb", concept="Cataract", cohorts=["UKBB"], members=["UKBB:cat"])
    ge_aou = ConceptEntry(
        concept_id="g_eye_aou",
        concept="",
        ideal_cde="Eye conditions checklist",
        cohorts=["AllOfUs"],
        members=["AllOfUs:eye_cat_yes", "AllOfUs:eye_glauc_yes", "AllOfUs:eye_other_yes", "AllOfUs:eye_none"],
    )
    ge_clsa = ConceptEntry(concept_id="g_cat_clsa", concept="Cataract CLSA", cohorts=["CLSA"], members=["CLSA:cat"])
    groups_by_id = {"g_cat_ukbb": ge_ukbb, "g_eye_aou": ge_aou, "g_cat_clsa": ge_clsa}
    var_to_group = {
        "UKBB:cat": "g_cat_ukbb",
        "AllOfUs:eye_cat_yes": "g_eye_aou",
        "AllOfUs:eye_glauc_yes": "g_eye_aou",
        "AllOfUs:eye_other_yes": "g_eye_aou",
        "AllOfUs:eye_none": "g_eye_aou",
        "CLSA:cat": "g_cat_clsa",
    }
    return var_to_group, groups_by_id


def test_union_spans_every_reached_cohort_above_the_floor():
    var_to_group, groups_by_id = _cataract_groups()
    rated = [("UKBB:cat", ABOVE), ("AllOfUs:eye_cat_yes", ABOVE), ("CLSA:cat", ABOVE)]
    cohorts, cohort_members, cand_stats = _coverage_from_rated(rated, var_to_group, groups_by_id, FLOOR)
    assert cohorts == ["AllOfUs", "CLSA", "UKBB"]  # union across 3 groups / 3 cohorts, sorted
    assert set(cohort_members) == {"AllOfUs", "CLSA", "UKBB"}
    assert cohort_members["AllOfUs"] == [("AllOfUs:eye_cat_yes", ABOVE)]


def test_over_merged_group_contributes_its_one_on_topic_member():
    # g_eye_aou is over-merged (4 members); only its cataract option is on-topic. Its GROUP mean would be
    # low, but the single above-floor member must still credit AllOfUs (member-level, not group-mean).
    var_to_group, groups_by_id = _cataract_groups()
    rated = [
        ("AllOfUs:eye_cat_yes", ABOVE),  # on-topic
        ("AllOfUs:eye_glauc_yes", BELOW),  # off-topic, sub-floor
        ("AllOfUs:eye_other_yes", BELOW),
    ]
    cohorts, cohort_members, _ = _coverage_from_rated(rated, var_to_group, groups_by_id, FLOOR)
    assert cohorts == ["AllOfUs"]
    assert cohort_members["AllOfUs"] == [("AllOfUs:eye_cat_yes", ABOVE)]


def test_confidence_floor_excludes_a_sub_floor_rating():
    var_to_group, groups_by_id = _cataract_groups()
    rated = [("UKBB:cat", ABOVE), ("CLSA:cat", BELOW)]  # CLSA is sub-floor
    cohorts, cohort_members, _ = _coverage_from_rated(rated, var_to_group, groups_by_id, FLOOR)
    assert cohorts == ["UKBB"]  # CLSA excluded — coverage never claimed below the floor
    assert "CLSA" not in cohort_members


def test_group_candidates_carry_x_of_y():
    var_to_group, groups_by_id = _cataract_groups()
    rated = [("UKBB:cat", ABOVE), ("AllOfUs:eye_cat_yes", ABOVE), ("CLSA:cat", ABOVE)]
    _, _, cand_stats = _coverage_from_rated(rated, var_to_group, groups_by_id, FLOOR)
    stats = {gid: (nm, nt) for gid, _agg, nm, nt in cand_stats}
    assert stats["g_eye_aou"] == (1, 4)  # 1 of the checklist group's 4 members matched (X of Y)
    assert stats["g_cat_ukbb"] == (1, 1)
    # every entry is a 4-tuple (group_id, aggregate, n_matched, n_total)
    assert all(len(entry) == 4 for entry in cand_stats)


def test_ungrouped_outlier_never_contributes_coverage():
    var_to_group, groups_by_id = _cataract_groups()
    rated = [("UKBB:cat", ABOVE), ("AoU:loose", ABOVE)]  # loose belongs to no group
    cohorts, _, cand_stats = _coverage_from_rated(rated, var_to_group, groups_by_id, FLOOR)
    assert cohorts == ["UKBB"]
    assert all(gid != "g_gone" for gid, *_ in cand_stats)


# --- integration: derive_composite surfaces a winner but reports the union --------------------


def _def_one(component_name="Cataracts"):
    return ScoreDefinition(
        name="mini",
        kind=CompositeKind.DEFICIT_PROPORTION,
        components=[ScoreComponent(name=component_name, definition="self-reported cataract")],
    )


@pytest.fixture
def cataract_records():
    return [
        _grec("g_cat_ukbb#g0", "Cataract diagnosed", ["UKBB:cat"], ["UKBB"]),
        _grec("g_eye_aou#g0", "", ["AllOfUs:eye_cat_yes", "AllOfUs:eye_glauc_yes"], ["AllOfUs"], ideal="Eye checklist"),
        _grec("g_cat_clsa#g0", "Cataract CLSA", ["CLSA:cat"], ["CLSA"]),
    ]


@pytest.fixture
def cataract_field_index():
    return {
        "UKBB:cat": {"questionText": "Cataract diagnosed by doctor"},
        "AllOfUs:eye_cat_yes": {"questionText": "Eye condition cataract"},
        "AllOfUs:eye_glauc_yes": {"questionText": "Eye condition glaucoma"},
        "CLSA:cat": {"questionText": "Cataract eye lens clouding"},
    }


def test_coverage_is_union_while_surfacing_one_winner(cataract_records, cataract_field_index):
    complete = _fake_complete(
        _multi_match_json({"C1": [("UKBB:cat", 0.95), ("AllOfUs:eye_cat_yes", 0.85), ("CLSA:cat", 0.75)]})
    )
    result = derive_composite(
        _def_one(), cataract_records, complete, embed=None, top_k=10, field_index=cataract_field_index
    )
    m = {x.component: x for x in result.spec.matches}["Cataracts"]
    # winner group surfaces the concept/column ...
    assert m.concept_id == "g_cat_ukbb#g0"
    assert m.concept == "Cataract diagnosed"
    # ... but coverage is the UNION across all three reached groups / cohorts
    assert m.cohorts == ["AllOfUs", "CLSA", "UKBB"]
    # feasibility per-cohort present is rebuilt from the union
    present = {c.cohort: c.present for c in result.spec.feasibility.per_cohort}
    assert "Cataracts" in present["UKBB"]
    assert "Cataracts" in present["AllOfUs"]
    assert "Cataracts" in present["CLSA"]


def test_low_mean_group_still_credits_its_cohort_via_union(cataract_records, cataract_field_index):
    # The AoU eye checklist group is over-merged; the judge rates only its cataract option high and its
    # glaucoma option low. AoU must still be credited for Cataracts through that one on-topic member.
    complete = _fake_complete(
        _multi_match_json({"C1": [("UKBB:cat", 0.95), ("AllOfUs:eye_cat_yes", 0.90), ("AllOfUs:eye_glauc_yes", BELOW)]})
    )
    result = derive_composite(
        _def_one(), cataract_records, complete, embed=None, top_k=10, field_index=cataract_field_index
    )
    m = {x.component: x for x in result.spec.matches}["Cataracts"]
    assert "AllOfUs" in m.cohorts


def test_spec_to_dict_emits_x_of_y_and_coverage_members(cataract_records, cataract_field_index):
    complete = _fake_complete(
        _multi_match_json({"C1": [("UKBB:cat", 0.95), ("AllOfUs:eye_cat_yes", 0.85), ("CLSA:cat", 0.75)]})
    )
    result = derive_composite(
        _def_one(), cataract_records, complete, embed=None, top_k=10, field_index=cataract_field_index
    )
    blob = spec_to_dict(result.spec)
    m = next(x for x in blob["matches"] if x["component"] == "Cataracts")
    gc = {c["groupId"]: c for c in m["groupCandidates"]}
    assert gc["g_eye_aou#g0"]["nMatched"] == 1
    assert gc["g_eye_aou#g0"]["nTotal"] == 2  # the group has 2 members in the record
    assert set(m["coverageMembers"]) == {"AllOfUs", "CLSA", "UKBB"}


# --- enriched variable index + judge candidate label ---------------------------------------------


def test_answer_labels_parse_both_encodings_and_drop_sentinels():
    # UKBB/CLSA "code=label|…" with a negative sentinel code, plus a generic sentinel label.
    ukbb = {"valueEncoding": "-1=Do not know|1=Mouth ulcers|2=Painful gums|0=None of the above"}
    assert _answer_labels(ukbb) == ["Mouth ulcers", "Painful gums"]
    # AoU "code, label | …" shape.
    aou = {"valueEncoding": "DentalCare_Ulcers, Mouth ulcers | DentalCare_None, None"}
    assert _answer_labels(aou) == ["Mouth ulcers"]
    # structured responseOptions preferred over valueEncoding, negative sentinel code dropped.
    structured = {
        "responseOptions": [{"code": "1", "label": "Cataract"}, {"code": "-3", "label": "Prefer not to answer"}]
    }
    assert _answer_labels(structured) == ["Cataract"]


def test_answer_labels_are_capped():
    enc = "|".join(f"{i}=opt{i}" for i in range(1, 20))
    assert len(_answer_labels({"valueEncoding": enc})) == 8  # _MAX_ANSWER_LABELS


def test_variable_index_composes_name_question_and_answers():
    fi = {
        "UKBB:dental": {
            "name": "Mouth/teeth dental problems",
            "questionText": "Which of the following do you have?",
            "valueEncoding": "1=Mouth ulcers|2=Painful gums|0=None of the above",
            "dataType": "categorical",
        }
    }
    entry = build_variable_index(fi)[0]
    # name + generic stem are BOTH present (the specific name no longer shadowed by the generic stem)
    assert "Mouth/teeth dental problems" in entry.concept
    assert "Which of the following" in entry.concept
    # answer-option labels are held on the entry and folded into retrieval text (the strongest signal)
    assert entry.answer_text == "Mouth ulcers; Painful gums"
    assert "Painful gums" in entry.retrieval_text


def test_variable_index_falls_back_when_only_one_field_present():
    only_name = build_variable_index({"UKBB:x": {"name": "Nervous feelings"}})[0]
    assert only_name.concept == "Nervous feelings"
    only_q = build_variable_index({"UKBB:y": {"questionText": "Do you feel nervous?"}})[0]
    assert only_q.concept == "Do you feel nervous?"


# --- answer-option granularity — a checklist supplies several components ------------------------

_VASC = {
    "UKBB:vasc": {
        "name": "Vascular/heart problems diagnosed by doctor",
        "questionText": "Has a doctor told you that you have any of the following conditions?",
        "dataType": "Categorical multiple",
        "valueEncoding": "1=Heart attack|2=Angina|3=Stroke|4=High blood pressure|-7=None of the above",
    }
}


def test_multiselect_checklist_explodes_into_option_units():
    entries = build_variable_index(_VASC)
    ids = [e.concept_id for e in entries]
    # the base variable PLUS one coverage unit per non-sentinel option
    assert "UKBB:vasc" in ids
    assert "UKBB:vasc#opt=Heart attack" in ids
    assert "UKBB:vasc#opt=Stroke" in ids
    assert "UKBB:vasc#opt=None of the above" not in ids  # sentinel dropped
    opt = next(e for e in entries if e.concept_id == "UKBB:vasc#opt=Angina")
    assert opt.option_label == "Angina"
    assert "Angina" in opt.concept


def test_single_select_categorical_is_not_exploded():
    likert = {
        "UKBB:health": {
            "name": "Overall health rating",
            "dataType": "Categorical single",
            "valueEncoding": "1=Excellent|2=Good|3=Fair|4=Poor",
        }
    }
    entries = build_variable_index(likert)
    assert [e.concept_id for e in entries] == ["UKBB:health"]  # one concept, never fanned out


def test_base_variable_id_strips_the_option_suffix():
    assert _base_variable_id("UKBB:vasc#opt=Heart attack") == "UKBB:vasc"
    assert _base_variable_id("UKBB:plain") == "UKBB:plain"  # no suffix -> unchanged


def test_member_payload_names_the_option():
    assert _member_payload("UKBB:vasc#opt=Stroke", 0.9) == {
        "variableId": "UKBB:vasc",
        "confidence": 0.9,
        "optionLabel": "Stroke",
    }
    assert _member_payload("UKBB:plain", 0.5) == {"variableId": "UKBB:plain", "confidence": 0.5}


def _multiselect_def():
    return ScoreDefinition(
        name="vascular mini",
        kind=CompositeKind.DEFICIT_PROPORTION,
        components=[
            ScoreComponent(name="Myocardial infarction", definition="heart attack diagnosed by doctor"),
            ScoreComponent(name="Angina", definition="angina diagnosed by doctor"),
            ScoreComponent(name="Stroke", definition="stroke diagnosed by doctor"),
            ScoreComponent(name="Diabetes", definition="diabetes diagnosed by doctor"),
        ],
    )


def test_one_checklist_variable_supplies_several_components_by_option():
    records = [_grec("g_vasc#g0", "Vascular/heart problems", ["UKBB:vasc"], ["UKBB"])]
    # the judge binds a SPECIFIC option to each component; it rates nothing for Diabetes.
    complete = _fake_complete(
        _multi_match_json(
            {
                "C1": [("UKBB:vasc#opt=Heart attack", 0.95)],
                "C2": [("UKBB:vasc#opt=Angina", 0.92)],
                "C3": [("UKBB:vasc#opt=Stroke", 0.90)],
            }
        )
    )
    result = derive_composite(_multiselect_def(), records, complete, embed=None, top_k=25, field_index=_VASC)
    m = {x.component: x for x in result.spec.matches}
    # all three conditions are supplied by the SAME checklist variable, each via its own option
    for comp in ("Myocardial infarction", "Angina", "Stroke"):
        assert m[comp].matched, comp
        assert m[comp].concept_id == "g_vasc#g0"
        assert "UKBB" in m[comp].cohorts
    # precision: a condition NO option measures is NOT supplied by the checklist
    assert not m["Diabetes"].matched


def test_checklist_option_credits_its_cohort_and_names_the_option():
    records = [_grec("g_vasc#g0", "Vascular/heart problems", ["UKBB:vasc"], ["UKBB"])]
    complete = _fake_complete(_multi_match_json({"C1": [("UKBB:vasc#opt=Heart attack", 0.95)]}))
    definition = ScoreDefinition(
        name="mi only",
        kind=CompositeKind.DEFICIT_PROPORTION,
        components=[ScoreComponent(name="Myocardial infarction", definition="heart attack diagnosed by doctor")],
    )
    result = derive_composite(definition, records, complete, embed=None, top_k=25, field_index=_VASC)
    blob = spec_to_dict(result.spec)
    mi = next(x for x in blob["matches"] if x["component"] == "Myocardial infarction")
    assert "UKBB" in mi["cohorts"]
    cov = mi["coverageMembers"]["UKBB"]
    assert cov[0]["variableId"] == "UKBB:vasc"
    assert cov[0]["optionLabel"] == "Heart attack"  # the option that supplies it is named
