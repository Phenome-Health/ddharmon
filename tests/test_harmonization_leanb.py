"""Tests for the lean head/tail harmonization pipeline (split-aware, 3-stage).

Exercises prepare_leanb -> prepare_split -> prepare_group_assign -> assemble_leanb and the export
helpers on hand-built clusters (no BERTopic), with mock-LLM dicts at each stage. No
sentence-transformers, no real cohorts, no network.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ddharmon.clustering.topic_engine import collect_inputs
from ddharmon.harmonization.leanb import (
    DEFAULT_ADOPT_FLOOR,
    MAX_SHOW,
    CdeBackbone,
    PromptRecord,
    _clean_cde_text,
    _member_prompt_text,
    _member_text,
    _parse_split_groups,
    _value_set_text,
    assemble_leanb,
    export_leanb_eitl_queue,
    harmonize_leanb,
    prepare_group_assign,
    prepare_leanb,
    prepare_split,
    recover_outlier_clusters,
    write_records_json,
)
from ddharmon.harmonization.leanb_prompts import (
    SYS_GROUP_REASSIGN,
    SYS_SPLIT,
    group_reassign_system_prompt,
    split_system_prompt,
)
from ddharmon.harmonization.substrate import ClusteringSubstrate, build_substrate, clusters_from_substrate
from ddharmon.models.cluster import FieldCluster, FieldReference
from ddharmon.models.data_dictionary import Field, ResponseOption


@pytest.fixture
def world(hf):
    """Two cohorts + a 4-CDE backbone, positioned so age/smoke/zip concepts retrieve the right CDE.

    The "zip" pair is the split motivator: a home-address ZIP and an employer-address ZIP share surface
    wording (and embed near each other) but are distinct concepts on the object/referent axis.
    """
    a_fields = [
        hf.field("age", "Age in years"),
        hf.field("smoke", "Do you currently smoke"),
        hf.field("home_residence_zip", "ZIP code", question_text="ZIP code"),
        hf.field("employer_workplace_zip", "ZIP code", question_text="ZIP code"),
    ]
    b_fields = [hf.field("age_yrs", "Age in years"), hf.field("smoke_b", "Current smoker")]
    cde_fields = [
        hf.field(
            "AgeCDE",
            "Age of the participant in years",
            field_id="cde_age",
            question_text="What is your age in years?",
            encoding="years",
        ),
        hf.field(
            "SmokeCDE",
            "Current cigarette smoking status",
            field_id="cde_smoke",
            question_text="Do you currently smoke cigarettes?",
            encoding="1=Yes|0=No",
        ),
        hf.field(
            "HeightCDE",
            "Standing height in centimeters",
            field_id="cde_height",
            question_text="Standing height",
            encoding="cm",
        ),
        hf.field(
            "ZipCDE",
            "Postal ZIP code",
            field_id="cde_zip",
            question_text="ZIP code",
            encoding="text",
        ),
    ]
    # 5-d space: dim 0 age, dim 1 smoke, dim 2 height, dim 3 zip.
    ed_a = hf.embedded_dict(
        "CohortA",
        a_fields,
        sem_vecs=hf.l2(np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]], float)),
    )
    ed_b = hf.embedded_dict(
        "CohortB",
        b_fields,
        sem_vecs=hf.l2(np.array([[0.98, 0.02, 0, 0, 0], [0.02, 0.98, 0, 0, 0]], float)),
    )
    ed_cde = hf.embedded_dict(
        "NIH_CDE",
        cde_fields,
        sem_vecs=hf.l2(
            np.array(
                [[0.99, 0.01, 0, 0, 0], [0.01, 0.99, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
                float,
            )
        ),
    )
    embedded = [ed_a, ed_b, ed_cde]
    _docs, embeddings, field_refs, _cohorts = collect_inputs(embedded)
    by_key = {(r.dictionary_name, r.variable_name): r for r in field_refs}
    return embedded, embeddings, field_refs, by_key


def _cluster(cid: int, refs: list[FieldReference]) -> FieldCluster:
    return FieldCluster(cluster_id=cid, label="topic", members=refs)


def _ideal_resp(ideal_recs: list[PromptRecord]) -> dict[str, object]:
    """One mock ideal per cluster, keyed by the ideal prompt id."""
    return {r.id: {"ideal_cde": f"ideal for {r.context['cluster_id']}"} for r in ideal_recs}


# ── stage 1: prepare_leanb ──────────────────────────────────────


def test_cde_backbone_from_embedded(world):
    embedded, *_ = world
    cde = next(e for e in embedded if e.dictionary.cohort_name == "NIH_CDE")
    bb = CdeBackbone.from_embedded(cde, {f.variable_name: f for f in cde.dictionary.fields.values()})
    assert set(bb.ids) == {"AgeCDE", "SmokeCDE", "HeightCDE", "ZipCDE"}
    assert bb.vectors.shape == (4, 5)
    age_i = bb.ids.index("AgeCDE")
    assert bb.external_ids[age_i] == "cde_age"
    assert "age" in bb.rich_texts[age_i].lower() and "years" in bb.rich_texts[age_i].lower()


# ── M5 index hygiene (_clean_cde_text + CdeBackbone clean_text) ──


def test_clean_cde_text_strips_instruction_boilerplate():
    cleaned = _clean_cde_text("Ethnicity of participant READ IF NECESSARY select all that apply")
    assert "read if necessary" not in cleaned.lower()
    assert "select all that apply" not in cleaned.lower()
    assert "ethnicity" in cleaned.lower()


def test_clean_cde_text_drops_leading_opaque_code_keeps_concept():
    assert _clean_cde_text("PHX0001010203 Current cigarette smoking status") == "Current cigarette smoking status"


def test_clean_cde_text_keeps_opaque_code_when_no_concept_remains():
    # never blank out a candidate: a bare code with no concept text is left intact
    assert _clean_cde_text("PHX0001") == "PHX0001"


def test_clean_cde_text_noop_on_clean_text():
    assert _clean_cde_text("Age in years") == "Age in years"


def test_backbone_clean_text_cleans_candidate_pool(hf):
    # a CDE whose only distinctive text is instruction boilerplate — the M5 audit's ethnicity@0.45 case
    cde = hf.field("ETHN_01", "Ethnicity READ IF NECESSARY", question_text="Ethnicity", field_id="cde_ethn")
    ed = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=hf.l2(np.array([[1, 0, 0]], float)))
    fields = {f.variable_name: f for f in ed.dictionary.fields.values()}
    dirty = CdeBackbone.from_embedded(ed, fields).rich_texts[0]
    clean = CdeBackbone.from_embedded(ed, fields, clean_text=True).rich_texts[0]
    assert "read if necessary" in dirty.lower() and "ETHN_01" in dirty
    assert "read if necessary" not in clean.lower() and "ethnicity" in clean.lower()
    assert "ETHN_01" not in clean  # opaque lead code dropped once concept text remains


def test_prepare_leanb_builds_ideal_prompts_with_candidates(world):
    embedded, embeddings, field_refs, by_key = world
    age = _cluster(0, [by_key[("CohortA", "age")], by_key[("CohortB", "age_yrs")]])
    recs = prepare_leanb([age], embedded, embeddings, field_refs, top_k=4)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.context["cluster_id"].startswith("c")  # content-addressed id, not the HDBSCAN ordinal
    assert rec.id == f"leanb:ideal:{rec.context['cluster_id']}"
    assert "ideal" in rec.system_prompt.lower() and "Age in years" in rec.user_prompt
    cands = rec.context["candidates"]
    assert cands[0]["designation"] == "AgeCDE" and cands[0]["external_id"] == "cde_age"
    assert rec.context["cross_cohort"] is True and rec.context["n_members"] == 2
    assert rec.context["top1_cos"] > 0.9
    # the full ordered member list is carried with [mK] ids + embedding rows
    members = rec.context["members"]
    assert [m["member_id"] for m in members] == ["m1", "m2"]
    assert members[0]["variable_name"] == "age" and isinstance(members[0]["row"], int)


def test_member_text_includes_augmented_name_when_informative(world):
    embedded, embeddings, field_refs, by_key = world
    # employer_workplace_zip's question text ("ZIP code") omits the "employer" qualifier carried in the var name.
    zips = _cluster(3, [by_key[("CohortA", "home_residence_zip")], by_key[("CohortA", "employer_workplace_zip")]])
    recs = prepare_leanb([zips], embedded, embeddings, field_refs, top_k=4)
    members = {m["variable_name"]: m["text"] for m in recs[0].context["members"]}
    assert "employer" in members["employer_workplace_zip"].lower() and "[field:" in members["employer_workplace_zip"]
    assert "home" in members["home_residence_zip"].lower()


def test_cde_only_cluster_is_skipped(world):
    embedded, embeddings, field_refs, by_key = world
    cde_only = _cluster(9, [FieldReference("NIH_CDE", "AgeCDE", "Age of the participant in years")])
    assert prepare_leanb([cde_only], embedded, embeddings, field_refs) == []


def test_substrate_reload_reproduces_prompt_ids(world):
    """L2 cache-hit property: clusters reloaded from a frozen substrate yield IDENTICAL prompt ids."""
    from ddharmon.harmonization.substrate import build_substrate, clusters_from_substrate

    embedded, embeddings, field_refs, by_key = world
    clusters = [
        _cluster(0, [by_key[("CohortA", "age")], by_key[("CohortB", "age_yrs")]]),
        _cluster(1, [by_key[("CohortA", "home_residence_zip")]]),
    ]
    ids_before = [r.id for r in prepare_leanb(clusters, embedded, embeddings, field_refs, top_k=4)]

    sub = build_substrate(clusters, min_cluster_size=15)
    reloaded = clusters_from_substrate(sub, field_refs)  # the replay path (no re-clustering)
    ids_after = [r.id for r in prepare_leanb(reloaded, embedded, embeddings, field_refs, top_k=4)]

    assert ids_before == ids_after  # same partition -> same content-addressed ids -> the Batch cache hits
    assert ids_after and all(i.startswith("leanb:ideal:c") for i in ids_after)


def test_replay_from_frozen_substrate_is_byte_identical(world, tmp_path):
    """End-to-end replay determinism: a frozen partition + fixed LLM responses -> byte-identical records.json.

    ``test_substrate_reload_reproduces_prompt_ids`` proves the frozen substrate yields identical prompt ids
    (so the Batch cache hits and the responses are identical run-to-run). This proves the *rest* of the
    chain — prepare -> assemble -> serialize — introduces no dict/set-ordering or timestamp nondeterminism,
    so the full pipeline reproduces its output byte-for-byte. Together they back the "fully deterministic
    (by construction)" claim.
    """
    from ddharmon.harmonization.substrate import build_substrate, clusters_from_substrate

    embedded, embeddings, field_refs, by_key = world
    clusters = [
        _cluster(
            0,
            [
                by_key[("CohortA", "home_residence_zip")],
                by_key[("CohortA", "employer_workplace_zip")],
                by_key[("CohortB", "age_yrs")],
            ],
        )
    ]
    sub = build_substrate(clusters, min_cluster_size=15)

    def run_once(path):
        cl = clusters_from_substrate(sub, field_refs)  # the replay path — no re-clustering
        ideal = prepare_leanb(cl, embedded, embeddings, field_refs, top_k=4)
        split = prepare_split(ideal, _ideal_resp(ideal))
        split_resp = {
            split[0].id: {
                "groups": [{"member_ids": ["m1", "m2", "m3"], "concept": "ZIP code", "verdict": "adopt", "cde_id": "1"}]
            }
        }
        grp = prepare_group_assign(split, split_resp, embedded, embeddings, field_refs, top_k=4)
        responses = {r.id: {"verdict": "adopt", "cde_id": "1", "ranking": [1]} for r in grp}
        write_records_json(assemble_leanb(grp, responses, retrieval_floor=0.0), path)
        return path

    a, b = tmp_path / "records_a.json", tmp_path / "records_b.json"
    run_once(a)
    run_once(b)
    assert a.read_bytes() == b.read_bytes()  # byte-for-byte identical across independent replays


# ── stage 2: prepare_split ──────────────────────────────────────


def test_prepare_split_carries_members_and_candidates(world):
    embedded, embeddings, field_refs, by_key = world
    age = _cluster(0, [by_key[("CohortA", "age")], by_key[("CohortB", "age_yrs")]])
    ideal_recs = prepare_leanb([age], embedded, embeddings, field_refs, top_k=4)
    split_recs = prepare_split(ideal_recs, _ideal_resp(ideal_recs))
    assert len(split_recs) == 1
    sr = split_recs[0]
    cid = sr.context["cluster_id"]
    assert sr.id == f"leanb:split:{cid}"
    # ideal threaded into the prompt + context; the [mK]-prefixed members + candidates are in the prompt
    assert f"ideal for {cid}" in sr.user_prompt and sr.context["ideal_cde"] == f"ideal for {cid}"
    assert "[m1]" in sr.user_prompt and "Age in years" in sr.user_prompt
    assert "AgeCDE" in sr.user_prompt  # candidate block carried forward
    assert sr.context["members"] == ideal_recs[0].context["members"]


def test_prepare_split_respects_max_show(world):
    # M2: max_show caps the members shown to the split LLM in one call (chunking keeps every unit <= it).
    embedded, embeddings, field_refs, by_key = world
    zips = _cluster(
        3,
        [
            by_key[("CohortA", "home_residence_zip")],
            by_key[("CohortA", "employer_workplace_zip")],
            by_key[("CohortB", "age_yrs")],
        ],
    )
    ideal_recs = prepare_leanb([zips], embedded, embeddings, field_refs, top_k=4)
    split_recs = prepare_split(ideal_recs, _ideal_resp(ideal_recs), max_show=2)
    prompt = split_recs[0].user_prompt
    assert "[m1]" in prompt and "[m2]" in prompt and "[m3]" not in prompt  # only 2 of 3 members shown


# ── stage 3: prepare_group_assign ───────────────────────────────


def _split_recs_for_zip(world):
    embedded, embeddings, field_refs, by_key = world
    zips = _cluster(
        3,
        [
            by_key[("CohortA", "home_residence_zip")],
            by_key[("CohortA", "employer_workplace_zip")],
            by_key[("CohortB", "age_yrs")],  # filler member so each group is non-trivial
        ],
    )
    ideal_recs = prepare_leanb([zips], embedded, embeddings, field_refs, top_k=4)
    return embedded, embeddings, field_refs, prepare_split(ideal_recs, _ideal_resp(ideal_recs))


def test_prepare_group_assign_splits_into_groups_with_own_candidates_plus_residual(world):
    embedded, embeddings, field_refs, split_recs = _split_recs_for_zip(world)
    # mock split: two distinct-concept groups covering m1 (home) + m2 (employer). m3 (age filler) is
    # enumerated by NEITHER group -> M1 residual completion recovers it into a trailing residual group
    # instead of dropping it (the multi-group partial-coverage failure mode).
    split_resp = {
        split_recs[0].id: {
            "groups": [
                {"member_ids": ["m1"], "concept": "Home address ZIP code", "verdict": "refine", "cde_id": "1"},
                {"member_ids": ["m2"], "concept": "Employer address ZIP code", "verdict": "refine", "cde_id": "1"},
            ]
        }
    }
    grp_recs = prepare_group_assign(split_recs, split_resp, embedded, embeddings, field_refs, top_k=4)
    assert len(grp_recs) == 3  # 2 concept groups + 1 M1 residual
    cid = grp_recs[0].context["cluster_id"]
    assert [r.id for r in grp_recs] == [
        f"leanb:groupassign:{cid}:0",
        f"leanb:groupassign:{cid}:1",
        f"leanb:groupassign:{cid}:2",
    ]
    g0, g1, g2 = grp_recs
    assert g0.context["concept"] == "Home address ZIP code"
    assert g0.context["member_variable_names"] == ["CohortA:home_residence_zip"]
    assert g1.context["member_variable_names"] == ["CohortA:employer_workplace_zip"]
    # each concept group re-retrieved its OWN candidates and the concept reached the prompt
    assert g0.context["candidates"] and g0.context["candidates"][0]["designation"] == "ZipCDE"
    assert "Home address ZIP code" in g0.user_prompt
    assert g0.context["group_id"] == f"{cid}#g0" and g1.context["group_id"] == f"{cid}#g1"
    # M1: the uncovered member is recovered into a residual group (concept "") — not dropped
    assert g2.context["concept"] == "" and g2.context["member_variable_names"] == ["CohortB:age_yrs"]


def test_multigroup_split_covering_all_members_adds_no_residual(world):
    """M1 is a no-op when a multi-group split already covers every member — no spurious residual group."""
    embedded, embeddings, field_refs, split_recs = _split_recs_for_zip(world)
    split_resp = {
        split_recs[0].id: {
            "groups": [
                {"member_ids": ["m1", "m2"], "concept": "ZIP code", "verdict": "refine", "cde_id": "1"},
                {"member_ids": ["m3"], "concept": "Age", "verdict": "novel", "cde_id": None},
            ]
        }
    }
    grp_recs = prepare_group_assign(split_recs, split_resp, embedded, embeddings, field_refs, top_k=4)
    assert len(grp_recs) == 2  # every member covered -> no residual appended
    assert sum(r.context["n_members"] for r in grp_recs) == 3


def test_no_split_yields_single_group_over_all_members(world):
    embedded, embeddings, field_refs, split_recs = _split_recs_for_zip(world)
    # mock split: one group spanning every member (no over-split)
    split_resp = {
        split_recs[0].id: {
            "groups": [{"member_ids": ["m1", "m2", "m3"], "concept": "ZIP code", "verdict": "adopt", "cde_id": "1"}]
        }
    }
    grp_recs = prepare_group_assign(split_recs, split_resp, embedded, embeddings, field_refs, top_k=4)
    assert len(grp_recs) == 1
    assert grp_recs[0].context["n_members"] == 3
    assert len(grp_recs[0].context["member_variable_names"]) == 3


def test_unparseable_split_falls_back_to_single_group(world):
    embedded, embeddings, field_refs, split_recs = _split_recs_for_zip(world)
    grp_recs = prepare_group_assign(split_recs, {}, embedded, embeddings, field_refs, top_k=4)  # no split response
    assert len(grp_recs) == 1
    assert grp_recs[0].context["group_id"] == f"{grp_recs[0].context['cluster_id']}#g0"
    assert grp_recs[0].context["n_members"] == 3  # all members retained


# ── tolerant split parse: the Batch schema is soft, the LLM drops the {"groups":[…]} wrapper (~35%) ──


class TestSplitWrapperDrop:
    def test_parse_recovers_bare_single_group(self):
        """A bare single-group object (no ``groups`` wrapper) is recovered as one group, not discarded."""
        bare = {"member_ids": ["m1", "m2"], "concept": "Home ZIP", "verdict": "refine", "cde_id": "1"}
        groups = _parse_split_groups(bare)
        assert len(groups) == 1
        assert groups[0]["member_ids"] == ["m1", "m2"] and groups[0]["concept"] == "Home ZIP"

    def test_parse_still_reads_proper_wrapper(self):
        groups = _parse_split_groups({"groups": [{"member_ids": ["m1"], "concept": "ok"}]})
        assert len(groups) == 1 and groups[0]["concept"] == "ok"

    def test_parse_rejects_non_group_dict(self):
        assert _parse_split_groups({"rationale": "no groups here"}) == []

    def test_wrapper_drop_recovers_group_plus_residual(self, world):
        """The real bug: the split LLM returns a bare group covering a SUBSET. We recover that group AND add a
        residual group for the uncovered members — instead of collapsing the whole cluster to one un-split group."""
        embedded, embeddings, field_refs, split_recs = _split_recs_for_zip(
            world
        )  # 3 members: m1 home, m2 employer, m3 age
        bare = {split_recs[0].id: {"member_ids": ["m1"], "concept": "Home ZIP", "verdict": "refine", "cde_id": "1"}}
        grp_recs = prepare_group_assign(split_recs, bare, embedded, embeddings, field_refs, top_k=4)
        assert len(grp_recs) == 2  # recovered concept group + residual (NOT one 3-member fallback)
        g0, g1 = grp_recs
        assert g0.context["concept"] == "Home ZIP"
        assert g0.context["member_variable_names"] == ["CohortA:home_residence_zip"]
        assert g1.context["concept"] == ""  # residual group carries the uncovered members
        assert set(g1.context["member_variable_names"]) == {"CohortA:employer_workplace_zip", "CohortB:age_yrs"}
        # every member is retained across the two groups (nothing dropped)
        assert g0.context["n_members"] + g1.context["n_members"] == 3


# ── stage 4: assemble_leanb (multi-record) ──────────────────────


def _group_assign_recs(world):
    # Two concept groups cover m1 (home) + m2 (employer); m3 (age filler) is uncovered, so M1 residual
    # completion appends a trailing residual group (#g2) -> THREE group-assign records, not two.
    embedded, embeddings, field_refs, split_recs = _split_recs_for_zip(world)
    split_resp = {
        split_recs[0].id: {
            "groups": [
                {"member_ids": ["m1"], "concept": "Home address ZIP code", "verdict": "refine", "cde_id": "1"},
                {"member_ids": ["m2"], "concept": "Employer address ZIP code", "verdict": "novel", "cde_id": None},
            ]
        }
    }
    return prepare_group_assign(split_recs, split_resp, embedded, embeddings, field_refs, top_k=4)


def test_assemble_multi_record_two_groups_share_cluster_distinct_concepts(world):
    grp_recs = _group_assign_recs(world)
    cid = grp_recs[0].context["cluster_id"]  # content-addressed cluster id (shared by both groups)
    responses = {
        grp_recs[0].id: {"verdict": "adopt", "cde_id": "1", "ranking": [1, 2], "rationale": "home zip"},
        grp_recs[1].id: {"verdict": "novel", "cde_id": None, "ranking": [2, 1], "rationale": "no employer"},
    }
    result = assemble_leanb(grp_recs, responses, retrieval_floor=0.0)
    assert len(result.records) == 3  # 2 concept groups + the M1 residual (age)
    assert {r.cluster_id for r in result.records} == {cid}  # same cluster
    by_gid = {r.group_id: r for r in result.records}
    assert set(by_gid) == {f"{cid}#g0", f"{cid}#g1", f"{cid}#g2"}  # distinct groups + residual

    g0 = by_gid[f"{cid}#g0"]
    assert g0.verdict == "adopt" and g0.route == "assigned"
    assert g0.cde_id == "ZipCDE" and g0.cde_external_id == "cde_zip"
    assert g0.concept == "Home address ZIP code"
    assert g0.member_variable_names == ["CohortA:home_residence_zip"]
    assert g0.ranking == [0, 1]  # 1-based -> 0-based

    g1 = by_gid[f"{cid}#g1"]
    assert g1.verdict == "novel" and g1.route == "gencde_residual" and g1.cde_id is None
    assert g1.concept == "Employer address ZIP code"
    assert g1.member_variable_names == ["CohortA:employer_workplace_zip"]

    g2 = by_gid[f"{cid}#g2"]  # M1 residual: unanswered here -> empty verdict, routed to the residual
    assert g2.concept == "" and g2.member_variable_names == ["CohortB:age_yrs"]
    assert g2.route == "gencde_residual"


def test_assemble_persists_candidates(world):
    """The ranked candidate set is persisted on each record (for the review UI), best-first with flags."""
    grp_recs = _group_assign_recs(world)
    responses = {
        grp_recs[0].id: {"verdict": "adopt", "cde_id": "1", "ranking": [1, 2], "rationale": "home zip"},
        grp_recs[1].id: {"verdict": "novel", "cde_id": None, "ranking": [2, 1], "rationale": "no employer"},
    }
    result = assemble_leanb(grp_recs, responses, retrieval_floor=0.0)

    adopt = next(r for r in result.records if r.verdict == "adopt")
    assert adopt.candidates, "candidates should be persisted for the review UI"
    assert adopt.candidates[0].rank == 1  # best-first
    chosen = [c for c in adopt.candidates if c.is_chosen]
    assert len(chosen) == 1 and chosen[0].cde_id == adopt.cde_id
    assert chosen[0].cosine is not None
    assert any(c.llm_suggested for c in adopt.candidates)

    novel = next(r for r in result.records if r.verdict == "novel")
    assert novel.candidates and not any(c.is_chosen for c in novel.candidates)


def test_assemble_handles_missing_and_unparseable_responses(world):
    grp_recs = _group_assign_recs(world)
    result = assemble_leanb(grp_recs, {})  # no responses at all
    assert len(result.records) == 3  # 2 concept groups + M1 residual
    for rec in result.records:
        assert rec.verdict == "" and rec.route == "gencde_residual" and rec.cde_id is None


def test_novel_with_far_candidate_is_coverage_gap(world):
    embedded, embeddings, field_refs, split_recs = _split_recs_for_zip(world)
    split_resp = {split_recs[0].id: {"groups": [{"member_ids": ["m1"], "concept": "z", "verdict": "refine"}]}}
    grp_recs = prepare_group_assign(split_recs, split_resp, embedded, embeddings, field_refs, top_k=4)
    grp_recs[0].context["top1_cos"] = 0.10  # force the diagnostic deterministically
    result = assemble_leanb(grp_recs, {grp_recs[0].id: {"verdict": "novel", "cde_id": None}})
    assert result.records[0].coverage_gap is True


# ── retrieval floor (per-group context) ─────────────────────────


def _group_rec(gid: str, cands: list[dict]) -> PromptRecord:
    cluster_id, idx = gid.split("#g")
    return PromptRecord(
        id=f"leanb:groupassign:{cluster_id}:{idx}",
        system_prompt="s",
        user_prompt="u",
        schema="{}",
        model_tag="m",
        context={
            "cluster_id": cluster_id,
            "group_idx": int(idx),
            "group_id": gid,
            "concept": "c",
            "candidates": cands,
            "top1_cos": cands[0]["cos"],
            "ideal_cde": "i",
            "member_variable_names": ["A:v1"],
            "cohorts": ["A"],
            "cross_cohort": False,
            "n_members": 1,
        },
    )


def test_retrieval_floor_downgrades_far_match():
    rec = _group_rec("9#g0", [{"designation": "FarCDE", "cos": 0.12, "text": "t", "external_id": "x"}])
    resp = {rec.id: {"verdict": "adopt", "cde_id": "1"}}
    floored = assemble_leanb([rec], resp, retrieval_floor=0.30).records[0]
    assert floored.verdict == "novel" and floored.route == "gencde_residual"
    assert floored.cde_id is None and floored.floored is True and floored.chosen_cos == 0.12
    off = assemble_leanb([rec], resp, retrieval_floor=0.0).records[0]
    assert off.verdict == "adopt" and off.route == "assigned" and off.cde_id == "FarCDE" and off.floored is False


def test_retrieval_floor_keeps_close_match():
    rec = _group_rec("8#g0", [{"designation": "NearCDE", "cos": 0.88, "text": "t", "external_id": "x"}])
    r = assemble_leanb([rec], {rec.id: {"verdict": "adopt", "cde_id": "1"}}, retrieval_floor=0.30).records[0]
    assert r.verdict == "adopt" and r.floored is False and r.chosen_cos == 0.88


# ── M5 adopt-specific floor (demote weak-support adopt -> refine) ─


def test_adopt_floor_demotes_weak_support_adopt_to_refine():
    # cos 0.45 sits in [retrieval_floor 0.30, adopt_floor 0.55): the exact-equivalence claim is unsupported,
    # so the adopt is demoted to refine — still assigned (and eligible for a transform), not novel.
    rec = _group_rec("6#g0", [{"designation": "MidCDE", "cos": 0.45, "text": "t", "external_id": "x"}])
    resp = {rec.id: {"verdict": "adopt", "cde_id": "1"}}
    r = assemble_leanb([rec], resp, retrieval_floor=0.30, adopt_floor=0.55).records[0]
    assert r.verdict == "refine" and r.route == "assigned"
    assert r.adopt_demoted is True and r.floored is False
    assert r.cde_id == "MidCDE" and r.chosen_cos == 0.45


def test_adopt_floor_keeps_strong_adopt():
    rec = _group_rec("6#g1", [{"designation": "StrongCDE", "cos": 0.72, "text": "t", "external_id": "x"}])
    resp = {rec.id: {"verdict": "adopt", "cde_id": "1"}}
    r = assemble_leanb([rec], resp, retrieval_floor=0.30, adopt_floor=0.55).records[0]
    assert r.verdict == "adopt" and r.adopt_demoted is False and r.route == "assigned"


def test_adopt_floor_below_retrieval_floor_goes_novel_not_demoted():
    # cos 0.20 < retrieval_floor: the retrieval floor novels it first; the adopt-floor demotion never applies.
    rec = _group_rec("6#g2", [{"designation": "FarCDE", "cos": 0.20, "text": "t", "external_id": "x"}])
    resp = {rec.id: {"verdict": "adopt", "cde_id": "1"}}
    r = assemble_leanb([rec], resp, retrieval_floor=0.30, adopt_floor=0.55).records[0]
    assert r.verdict == "novel" and r.floored is True and r.adopt_demoted is False


def test_adopt_floor_does_not_touch_refine():
    # adopt_floor is adopt-specific: a refine at the same weak cosine is left as refine.
    rec = _group_rec("6#g3", [{"designation": "MidCDE", "cos": 0.45, "text": "t", "external_id": "x"}])
    resp = {rec.id: {"verdict": "refine", "cde_id": "1"}}
    r = assemble_leanb([rec], resp, retrieval_floor=0.30, adopt_floor=0.55).records[0]
    assert r.verdict == "refine" and r.adopt_demoted is False


def test_adopt_floor_none_leaves_adopt_unchanged():
    rec = _group_rec("6#g4", [{"designation": "MidCDE", "cos": 0.45, "text": "t", "external_id": "x"}])
    resp = {rec.id: {"verdict": "adopt", "cde_id": "1"}}
    r = assemble_leanb([rec], resp, retrieval_floor=0.30, adopt_floor=None).records[0]
    assert r.verdict == "adopt" and r.adopt_demoted is False


# ── ranking fallback when the assign LLM leaves cde_id null (regression) ──


def test_ranking_fallback_resolves_cde_when_cde_id_null():
    """Regression: the assign LLM commonly returns ``cde_id=null`` + a best-first ``ranking``.

    assemble_leanb must fall back to ``ranking[0]`` to resolve the chosen CDE + its cosine. Without it,
    ``chosen_cos`` stays None and the retrieval floor wrongly downgrades EVERY adopt/refine to novel —
    the production path emitted all-novel before this fix (CDEMapper bench bypasses assemble_leanb, so
    it never caught this). The top-ranked candidate here is index 1, NOT candidate 0, proving the
    fallback honors the ranking rather than blindly taking the first candidate.
    """
    cands = [
        {"designation": "FarCDE", "cos": 0.20, "text": "t", "external_id": "far"},
        {"designation": "NearCDE", "cos": 0.81, "text": "t", "external_id": "near"},
    ]
    rec = _group_rec("7#g0", cands)
    resp = {rec.id: {"verdict": "adopt", "cde_id": None, "ranking": ["2", "1"]}}  # top pick = candidate #2 (0-based 1)
    r = assemble_leanb([rec], resp, retrieval_floor=0.30).records[0]
    assert r.verdict == "adopt" and r.route == "assigned"
    assert r.cde_id == "NearCDE" and r.cde_external_id == "near"
    assert r.chosen_cos == 0.81 and r.floored is False


def test_ranking_fallback_still_respects_floor_when_top_ranked_is_far():
    """The fallback resolves the cosine; it does NOT bypass the floor. A far top-ranked candidate
    (cde_id null) still resolves chosen_cos and is then correctly downgraded to novel."""
    rec = _group_rec("7#g1", [{"designation": "FarCDE", "cos": 0.12, "text": "t", "external_id": "far"}])
    resp = {rec.id: {"verdict": "refine", "cde_id": None, "ranking": ["1"]}}
    r = assemble_leanb([rec], resp, retrieval_floor=0.30).records[0]
    assert r.verdict == "novel" and r.route == "gencde_residual"
    assert r.cde_id is None and r.floored is True and r.chosen_cos == 0.12


# ── exports ─────────────────────────────────────────────────────


def test_export_eitl_queue_and_records_json(world, tmp_path):
    grp_recs = _group_assign_recs(world)
    cid = grp_recs[0].context["cluster_id"]
    result = assemble_leanb(
        grp_recs,
        {
            grp_recs[0].id: {"verdict": "adopt", "cde_id": "1"},
            grp_recs[1].id: {"verdict": "refine", "cde_id": "1"},
        },
        retrieval_floor=0.0,
    )

    tsv = tmp_path / "eitl.tsv"
    n = export_leanb_eitl_queue(result, tsv)
    assert n == 3  # 2 answered groups + the M1 residual (empty verdict, sorts last)
    lines = tsv.read_text().splitlines()
    header = lines[0].split("\t")
    assert header[:5] == ["cluster_id", "group_id", "concept", "verdict", "route"]
    assert "members" in header
    # refine sorts before adopt/residual; the per-group rows carry the group id + members column
    assert lines[1].split("\t")[:2] == [cid, f"{cid}#g1"]
    members_col = header.index("members")
    assert lines[1].split("\t")[members_col] == "CohortA:employer_workplace_zip"
    assert len(lines) == 4  # header + 3

    js = tmp_path / "records.json"
    assert write_records_json(result, js) == 3
    loaded = json.loads(js.read_text())
    assert {r["group_id"] for r in loaded} == {f"{cid}#g0", f"{cid}#g1", f"{cid}#g2"}
    assert all(r["cluster_id"] == cid for r in loaded)


# ── prompt content ──────────────────────────────────────────────


def test_axis_preservation_clause_in_split_prompt():
    # the split system prompt instructs the model to split on the object/referent axis (preserve qualifier)
    s = SYS_SPLIT.lower()
    assert "object" in s and "referent" in s
    assert "split" in s and ("do not split" in s or "do not over-split" in s)


# ── M4 representation-mismatch clause (refine, not novel) ────────


def test_representation_refine_selectors_toggle_clause():
    # both stage prompts gain the clause only when the flag is on; base prompts are unchanged
    assert split_system_prompt(False) == SYS_SPLIT
    assert group_reassign_system_prompt(False) == SYS_GROUP_REASSIGN
    for on in (split_system_prompt(True), group_reassign_system_prompt(True)):
        low = on.lower()
        assert "representation mismatch is refine" in low
        assert "banding" in low and "composite" in low  # the recovered representation kinds
        assert "novel only when" in low  # still gated on a genuine concept/object change


def test_representation_refine_wired_into_split_and_group_assign(world):
    embedded, embeddings, field_refs, split_recs = _split_recs_for_zip(world)
    # stage 2 (split): flag flows from prepare_split into the emitted system prompt
    on_split = _split_recs_for_zip_with_flag(world, representation_refine=True)[0]
    assert "representation mismatch is refine" in on_split.system_prompt.lower()
    assert "representation mismatch is refine" not in split_recs[0].system_prompt.lower()
    # stage 3 (per-group assign): flag flows from prepare_group_assign into the per-group prompt
    split_resp = {split_recs[0].id: {"groups": [{"member_ids": ["m1"], "concept": "z", "verdict": "refine"}]}}
    grp_on = prepare_group_assign(
        split_recs, split_resp, embedded, embeddings, field_refs, top_k=4, representation_refine=True
    )
    grp_off = prepare_group_assign(split_recs, split_resp, embedded, embeddings, field_refs, top_k=4)
    assert "representation mismatch is refine" in grp_on[0].system_prompt.lower()
    assert "representation mismatch is refine" not in grp_off[0].system_prompt.lower()


def _split_recs_for_zip_with_flag(world, *, representation_refine: bool):
    embedded, embeddings, field_refs, by_key = world
    zips = _cluster(3, [by_key[("CohortA", "home_residence_zip")], by_key[("CohortB", "age_yrs")]])
    ideal_recs = prepare_leanb([zips], embedded, embeddings, field_refs, top_k=4)
    return prepare_split(ideal_recs, _ideal_resp(ideal_recs), representation_refine=representation_refine)


# ── 1a: values-aware prompts (retrieval text lean, prompt text value-aware) ──


class TestMemberPromptText:
    """The source variable's response options reach the PROMPT text but NOT the retrieval text."""

    def test_prompt_text_adds_value_options_retrieval_stays_lean(self):
        fld = Field(
            variable_name="smoke",
            description="Current smoking",
            question_text="Do you currently smoke",
            value_encoding_raw="1=Yes|2=No",
            data_type="categorical",
        )
        ref = FieldReference("CohortA", "smoke", "Do you currently smoke")
        retrieval = _member_text(fld, ref)
        prompt = _member_prompt_text(fld, ref)
        # retrieval text: concept only — no value codes (they are noise in BM25/dense)
        assert "1=Yes" not in retrieval and "values:" not in retrieval
        # prompt text: the source response options + data_type are visible to the LLM
        assert "1=Yes|2=No" in prompt
        assert "categorical" in prompt
        assert retrieval in prompt  # prompt = lean text + value metadata

    def test_prompt_text_includes_units(self):
        fld = Field(
            variable_name="wt",
            description="Body weight",
            question_text="Body weight",
            units="kg",
            data_type="continuous",
        )
        ref = FieldReference("CohortA", "wt", "Body weight")
        prompt = _member_prompt_text(fld, ref)
        assert "units kg" in prompt and "continuous" in prompt

    def test_value_set_text_falls_back_to_response_options(self):
        fld = Field(
            variable_name="x",
            description="d",
            response_options=[ResponseOption(code="0", label="No"), ResponseOption(code="1", label="Yes")],
        )
        assert _value_set_text(fld) == "0=No|1=Yes"

    def test_prompt_text_no_field_is_base(self):
        ref = FieldReference("CohortA", "x", "desc")
        assert _member_prompt_text(None, ref) == _member_text(None, ref)

    def test_prompt_text_drops_missing_sentinel_options(self):
        # The concept-identity route (judge/gen-ideal/split/assign) must NOT see missing/DK sentinels.
        fld = Field(
            variable_name="fhx",
            description="Family history of asthma",
            question_text="Family history of asthma",
            value_encoding_raw="-9=MISSING|0=No|1=Yes|9=Do not know",
            data_type="categorical",
        )
        ref = FieldReference("CohortA", "fhx", "Family history of asthma")
        prompt = _member_prompt_text(fld, ref)
        assert "0=No|1=Yes" in prompt
        assert "MISSING" not in prompt and "Do not know" not in prompt

    def test_prompt_text_numeric_sentinel_only_shows_no_value_tail(self):
        # MESA `-9=MISSING` on a numeric field must NOT render as a single-option categorical.
        fld = Field(
            variable_name="fhxasta2",
            description="Age at asthma diagnosis",
            question_text="Age at asthma diagnosis",
            value_encoding_raw="-9=MISSING",
            data_type="numeric",
        )
        ref = FieldReference("CohortA", "fhxasta2", "Age at asthma diagnosis")
        prompt = _member_prompt_text(fld, ref)
        assert "values:" not in prompt and "MISSING" not in prompt

    def test_value_set_text_keeps_sentinels_by_default_for_specgen(self):
        # spec-gen calls _value_set_text(fld) with the default -> sentinels PRESERVED (needed for recodes).
        fld = Field(variable_name="fhx", description="d", value_encoding_raw="-9=MISSING|0=No|1=Yes")
        assert _value_set_text(fld) == "-9=MISSING|0=No|1=Yes"
        assert _value_set_text(fld, drop_sentinels=True) == "0=No|1=Yes"

    def test_value_set_drop_sentinels_from_response_options(self):
        fld = Field(
            variable_name="x",
            description="d",
            response_options=[
                ResponseOption(code="-9", label="Missing"),
                ResponseOption(code="0", label="No"),
                ResponseOption(code="1", label="Yes"),
            ],
        )
        assert _value_set_text(fld, drop_sentinels=True) == "0=No|1=Yes"
        assert _value_set_text(fld) == "-9=Missing|0=No|1=Yes"


# ── M10: outlier recovery (recover_outlier_clusters) ─────────────


class TestRecoverOutlierClusters:
    def _refs(self, n):
        return [FieldReference("A", f"var_{i}", f"desc {i}") for i in range(n)]

    def test_no_outliers_is_noop(self):
        refs = self._refs(5)
        emb = np.zeros((5, 8), dtype=np.float32)
        sub = ClusteringSubstrate(
            clusters=[[("A", f"var_{i}") for i in range(5)]], min_cluster_size=15, n_fields=5, outlier=[]
        )
        clusters = clusters_from_substrate(sub, refs)
        out_clusters, out_sub = recover_outlier_clusters(clusters, sub, emb, refs)
        assert out_clusters is clusters and out_sub is sub  # unchanged (identity, no re-cluster)

    def test_below_threshold_outliers_recovered_and_folded_into_substrate(self):
        # 3 outliers (rows 7,8,9) < recluster_residual's threshold -> one recovered group, NO umap
        refs = self._refs(10)
        emb = np.zeros((10, 8), dtype=np.float32)
        sub = ClusteringSubstrate(
            clusters=[[("A", f"var_{i}") for i in range(7)]],
            min_cluster_size=15,
            n_fields=10,
            outlier=[("A", "var_7"), ("A", "var_8"), ("A", "var_9")],
        )
        clusters = clusters_from_substrate(sub, refs)
        out_clusters, out_sub = recover_outlier_clusters(clusters, sub, emb, refs, min_cluster_size=8)
        assert len(out_clusters) == len(clusters) + 1  # one recovered group appended
        assert out_sub.outlier == []  # all three recovered -> outlier list emptied
        assert out_sub.n_clusters == sub.n_clusters + 1  # recovered folded into the substrate for replay
        recovered = out_clusters[-1]
        assert {m.variable_name for m in recovered.members} == {"var_7", "var_8", "var_9"}

    def test_outliers_absent_from_field_refs_is_noop(self):
        refs = self._refs(3)
        emb = np.zeros((3, 8), dtype=np.float32)
        sub = ClusteringSubstrate(
            clusters=[[("A", "var_0")]], min_cluster_size=15, n_fields=3, outlier=[("A", "ghost")]
        )
        clusters = clusters_from_substrate(sub, refs)
        out_clusters, out_sub = recover_outlier_clusters(clusters, sub, emb, refs)
        assert out_clusters is clusters and out_sub is sub  # no rows map -> no-op


# ── productionization: the M-stack is ON by default (held-out full-5 validated 2026-07-04) ──


def test_release_defaults_enable_the_m_stack():
    """Guard the release decision: harmonize_leanb turns the validated M2/M3/M4/M5/M10 quality mods ON by
    default. A flip back to opt-in should be a deliberate change that trips this test, not a silent regression.
    """
    import inspect

    d = {k: v.default for k, v in inspect.signature(harmonize_leanb).parameters.items()}
    assert d["representation_refine"] is True  # M4: representation-mismatch -> refine
    assert d["clean_cde_text"] is True  # M5: CDE candidate-pool index hygiene
    assert d["adopt_floor"] == DEFAULT_ADOPT_FLOOR == 0.55  # M5: weak-adopt demote
    assert d["chunk_cap"] == MAX_SHOW  # M2: chunk oversized clusters (== max_show)
    assert d["chunk_skip_enumerated"] is True  # M2: keep enumerated families whole
    assert d["coherence_gate"] is True  # M3: NONE-fraction gate
    assert d["recover_outliers"] is True  # M10: HDBSCAN outlier recovery


def test_harmonize_leanb_default_run_threads_representation_refine(world):
    """A default (no-flag) harmonize_leanb call applies the release defaults end-to-end: M4's clause reaches
    the emitted split prompt — proving the orchestrator default flows into the stages, not just the signature.
    """
    embedded, embeddings, field_refs, by_key = world
    zips = _cluster(3, [by_key[("CohortA", "home_residence_zip")], by_key[("CohortB", "age_yrs")]])
    sub = build_substrate([zips], min_cluster_size=15, n_fields=len(field_refs))
    # generate set, split=None -> returns after building split prompts (no per-group assign LLM needed)
    result = harmonize_leanb(embedded, substrate=sub, generate=_ideal_resp, split=None)
    assert result.split_prompts
    assert "representation mismatch is refine" in result.split_prompts[0].system_prompt.lower()


# ── 08-04: staged-review stage ordering + the one resumable boundary (STGD-02, core half) ──
#
# The staged review flow pauses a run by EXITING at a stage boundary, so where a stage sits in the
# sequence is a product contract, not an implementation detail: Gate 1 pauses at the shipped
# ``classify=None`` early return and must be able to render each concept group's coherence verdict, which
# is only true if the judge's verdict pass runs BEFORE assign.


@pytest.fixture
def judgeable_world(hf):
    """Eight near-identical smoking variables across two cohorts + one smoking CDE.

    Large enough for the coherence judge to build a prompt (>= COHERENCE_MIN_MEMBERS members in the big
    group) and tight enough that the assign stage's chosen candidate clears the M5 adopt floor — so a
    default run reaches the categorical transform-spec stage with real prompts.
    """
    enc = "1=Yes|0=No"
    a_fields = [hf.field(f"smoke_a{i}", "Do you currently smoke cigarettes", encoding=enc) for i in range(4)]
    b_fields = [hf.field(f"smoke_b{i}", "Do you currently smoke cigarettes", encoding=enc) for i in range(4)]
    cde_fields = [
        hf.field(
            "SmokeCDE",
            "Current cigarette smoking status",
            field_id="cde_smoke",
            question_text="Do you currently smoke cigarettes?",
            encoding=enc,
        )
    ]
    ed_a = hf.embedded_dict("CohortA", a_fields, sem_vecs=hf.l2(np.array([[1, 0.01 * i] for i in range(4)], float)))
    ed_b = hf.embedded_dict(
        "CohortB", b_fields, sem_vecs=hf.l2(np.array([[1, 0.01 * (i + 4)] for i in range(4)], float))
    )
    ed_cde = hf.embedded_dict("NIH_CDE", cde_fields, sem_vecs=hf.l2(np.array([[1, 0.0]], float)))
    embedded = [ed_a, ed_b, ed_cde]
    _docs, embeddings, field_refs, _cohorts = collect_inputs(embedded)
    cohort_refs = [r for r in field_refs if r.dictionary_name != "NIH_CDE"]
    sub = build_substrate([_cluster(0, cohort_refs)], min_cluster_size=15, n_fields=len(field_refs))
    return embedded, sub


def _staged_stages(order: list[str], *, coherence_verdict: str = "split"):
    """One recording fake per injectable stage; each appends its own name then answers plausibly.

    The split fake carves the cluster into a big 7-member group (judgeable, routed ``adopt``) and a
    1-member group (routed ``novel``, so the GenCDE stage has work), which is what makes every stage in
    the sequence actually fire.
    """

    def generate(prompts):
        order.append("generate")
        return {p.id: {"ideal_cde": f"ideal for {p.context['cluster_id']}"} for p in prompts}

    def split(prompts):
        order.append("split")
        out = {}
        for p in prompts:
            ids = [m["member_id"] for m in p.context["members"]]
            out[p.id] = {
                "groups": [
                    {"member_ids": ids[:-1], "concept": "current smoking", "verdict": "adopt"},
                    {"member_ids": ids[-1:], "concept": "smoking outlier", "verdict": "novel"},
                ]
            }
        return out

    def classify(prompts):
        order.append("classify")
        out = {}
        for p in prompts:
            novel = p.context["n_members"] == 1
            out[p.id] = (
                {"verdict": "novel", "ranking": [], "rationale": "no candidate fits"}
                if novel
                else {"verdict": "adopt", "cde_id": "1", "ranking": [1], "rationale": "same concept"}
            )
        return out

    def coherence(prompts):
        order.append("coherence")
        return {
            p.id: {
                "summary": "current smoking",
                "coherent": False,
                "outliers": [1],
                "granularity": {
                    "verdict": coherence_verdict,
                    "axis": "measurand",
                    "distinct_values": ["cigarettes", "cigars"],
                },
            }
            for p in prompts
        }

    def gencde(prompts):
        order.append("gencde")
        return {
            p.id: {"preferred_name": "smoking_outlier", "definition": "d", "data_type": "categorical"} for p in prompts
        }

    def specgen(prompts):
        order.append("specgen")
        return {p.id: {"mappings": []} for p in prompts}

    def merge(prompts):
        order.append("merge")
        return {p.id: {"merge": False} for p in prompts}

    return {
        "generate": generate,
        "split": split,
        "classify": classify,
        "merge": merge,
        "coherence": coherence,
        "gencde": gencde,
        "specgen": specgen,
    }


def test_coherence_judge_runs_after_split_and_before_assign(judgeable_world, monkeypatch):
    """The ordering contract: judge verdicts exist by the time the assign boundary is reached.

    Gate 1 of the staged review flow pauses at the ``classify=None`` early return, so a verdict stamped
    after ``classify`` does not exist when the reviewer needs it.
    """
    from ddharmon.harmonization import leanb as leanb_mod

    embedded, sub = judgeable_world
    order: list[str] = []
    real_propagate = leanb_mod.propagate_coherence_review

    def spy_propagate(records):
        order.append("propagate")
        return real_propagate(records)

    monkeypatch.setattr(leanb_mod, "propagate_coherence_review", spy_propagate)
    result = harmonize_leanb(embedded, substrate=sub, **_staged_stages(order))

    assert "coherence" in order, order
    assert order.index("split") < order.index("coherence") < order.index("classify"), order
    # the propagation half stays late — it lands needs_review on artifacts that only exist after specgen/gencde
    assert order.index("gencde") < order.index("propagate"), order
    assert order.index("specgen") < order.index("propagate"), order
    assert result.records


def test_coherence_verdicts_are_available_at_the_gate_1_boundary(judgeable_world):
    """``classify=None`` returns with the judge already run — verdicts on the concept groups, no assign paid."""
    embedded, sub = judgeable_world
    order: list[str] = []
    stages = _staged_stages(order)
    result = harmonize_leanb(
        embedded,
        substrate=sub,
        generate=stages["generate"],
        split=stages["split"],
        classify=None,
        coherence=stages["coherence"],
    )

    assert "classify" not in order  # the assign stage was never paid for
    assert result.group_assign_prompts and result.concept_groups
    assert result.substrate is not None
    judged = [g for g in result.concept_groups if g.coherence_verdict]
    assert judged, [g.n_members for g in result.concept_groups]
    assert all(g.coherence_verdict == "split" and g.incoherent is True for g in judged)


def test_early_verdicts_are_transferred_onto_the_assembled_records(judgeable_world):
    """The verdicts stamped before assign must reach the real records once assemble_leanb builds them."""
    embedded, sub = judgeable_world
    order: list[str] = []
    result = harmonize_leanb(embedded, substrate=sub, **_staged_stages(order))

    flagged = [r for r in result.records if r.coherence_verdict == "split"]
    assert flagged, [(r.group_id, r.coherence_verdict, r.n_members) for r in result.records]
    for r in flagged:
        assert r.incoherent is True
        assert all(t.needs_review for t in r.transforms)  # propagation landed
        if r.gencde is not None:
            assert r.gencde.needs_review is True


def test_pipeline_never_auto_invokes_re_adjudication(judgeable_world, monkeypatch):
    """SPEC prohibition: the pipeline FLAGS an over-merge, a human resolves it. Nothing here re-splits."""
    from ddharmon.harmonization import leanb as leanb_mod

    embedded, sub = judgeable_world
    called: list[str] = []
    monkeypatch.setattr(leanb_mod, "prepare_readjudicate", lambda *a, **k: called.append("prepare_readjudicate") or [])
    monkeypatch.setattr(leanb_mod, "readjudicate", lambda *a, **k: called.append("readjudicate") or [])

    order: list[str] = []
    result = harmonize_leanb(embedded, substrate=sub, **_staged_stages(order))

    assert called == []
    assert any(r.incoherent for r in result.records)  # it DID flag — it just did not resolve
    assert all(r.readjudicated_from == "" for r in result.records)


# ── the one new named boundary: stop_after="gencde" (Gate 1 -> Gate 2) ──


def test_stop_after_is_feature_detectable():
    """The UI adapter guards on inspect.signature to degrade gracefully against an older pinned core."""
    import inspect

    params = inspect.signature(harmonize_leanb).parameters
    assert "stop_after" in params
    assert params["stop_after"].default is None  # additive: the default reproduces today's behaviour
    assert params["stop_after"].kind is inspect.Parameter.KEYWORD_ONLY


def test_stop_after_accepts_exactly_one_boundary_name():
    from ddharmon.harmonization.leanb import STOP_AFTER_BOUNDARIES

    assert STOP_AFTER_BOUNDARIES == ("gencde",)


@pytest.mark.parametrize("bad", ["gencd", "specgen", "cluster", "ideal", "split", "assign", "coherence", ""])
def test_unrecognised_stop_after_raises_before_any_work(judgeable_world, bad):
    """T-08-20: a typo must never be a silent no-op that runs the whole paid pipeline to completion."""
    embedded, sub = judgeable_world
    order: list[str] = []

    with pytest.raises(ValueError) as exc:
        harmonize_leanb(embedded, substrate=sub, stop_after=bad, **_staged_stages(order))

    assert "gencde" in str(exc.value)  # names the accepted set
    assert order == []  # rejected before a single stage ran


def test_stop_after_gencde_returns_a_resumable_partial(judgeable_world):
    embedded, sub = judgeable_world
    order: list[str] = []
    result = harmonize_leanb(embedded, substrate=sub, stop_after="gencde", **_staged_stages(order))

    assert order.index("gencde") == len(order) - 1  # gencde ran last
    assert "specgen" not in order  # and specgen was never paid for
    assert result.substrate is not None  # the resume key — without it a resumed run re-clusters
    assert result.records  # the assignment records are present
    assert result.specgen_prompts == []  # no spec output yet
    assert all(r.transforms == [] for r in result.records)
    assert result.concept_groups  # the Gate 1 rows travel with the partial


def test_every_early_return_carries_the_substrate(judgeable_world):
    """T-08-19: a partial result missing ``substrate`` strands every gate decision made against a partition."""
    embedded, sub = judgeable_world
    order: list[str] = []
    stages = _staged_stages(order)
    partials = [
        harmonize_leanb(embedded, substrate=sub, generate=None),
        harmonize_leanb(embedded, substrate=sub, generate=stages["generate"], split=None),
        harmonize_leanb(embedded, substrate=sub, generate=stages["generate"], split=stages["split"], classify=None),
        harmonize_leanb(embedded, substrate=sub, stop_after="gencde", **stages),
    ]
    assert all(p.substrate is not None for p in partials)


def test_stop_after_none_reproduces_the_default_run(judgeable_world):
    """T-08-20: the boundary is additive — an existing caller passing nothing is byte-for-byte unaffected."""
    embedded, sub = judgeable_world

    def run(**extra):
        order: list[str] = []
        res = harmonize_leanb(embedded, substrate=sub, **_staged_stages(order), **extra)
        return order, [
            (r.group_id, r.verdict, r.route, r.cde_id, r.coherence_verdict, r.incoherent, len(r.transforms))
            for r in res.records
        ]

    baseline_order, baseline_records = run()
    explicit_order, explicit_records = run(stop_after=None)

    assert explicit_order == baseline_order
    assert explicit_records == baseline_records


# ── forward-port guard: the keywords only the CLI passes must survive a port ──
#
# `max_clusters` is a `harmonize_leanb` keyword with exactly one consumer — `ddharmon.cli.harmonize`,
# which passes it in its `common` kwarg dict. `tests/test_cli.py` monkeypatches `harmonize_leanb` with a
# `lambda *a, **k`, so the CLI tests keep passing even if the keyword is deleted from the real function.
# That combination is how a forward-port can silently ship a package whose own CLI raises TypeError on
# every `ddharmon harmonize` invocation — permanently, once the version is on PyPI. These two tests are
# the trip-wire: one pins the signature, the other pins the behaviour.


def test_cli_kwargs_are_accepted_by_harmonize_leanb():
    """Every keyword `ddharmon harmonize` passes must still exist on `harmonize_leanb`."""
    import inspect

    params = inspect.signature(harmonize_leanb).parameters
    # Mirrors the `common` dict built in ddharmon.cli.harmonize.
    for kw in ("cde_cohort", "min_cluster_size", "top_k", "retrieval_floor", "model_tag", "max_clusters"):
        assert kw in params, f"harmonize_leanb lost the CLI keyword {kw!r} — `ddharmon harmonize` would break"
    assert params["max_clusters"].default is None  # uncapped by default


def test_max_clusters_caps_split_units_largest_first(world):
    """The cost cap keeps the LARGEST units (most members first) and drops the rest."""
    embedded, embeddings, field_refs, by_key = world
    big = _cluster(
        1,
        [
            by_key[("CohortA", "home_residence_zip")],
            by_key[("CohortA", "employer_workplace_zip")],
            by_key[("CohortA", "age")],
        ],
    )
    small = _cluster(2, [by_key[("CohortB", "smoke_b")], by_key[("CohortA", "smoke")]])
    sub = build_substrate([big, small], min_cluster_size=15, n_fields=len(field_refs))

    uncapped = harmonize_leanb(embedded, substrate=sub, generate=None)
    assert len(uncapped.ideal_prompts) == 2

    capped = harmonize_leanb(embedded, substrate=sub, generate=None, max_clusters=1)
    assert len(capped.ideal_prompts) == 1
    kept = capped.ideal_prompts[0]
    assert len(kept.context["members"]) == 3  # the larger unit survived, not merely the first one
