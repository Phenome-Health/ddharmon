"""Tests for the v2 lean head/tail harmonization pipeline (split-aware, 3-stage).

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
    CdeBackbone,
    PromptRecord,
    assemble_leanb,
    export_leanb_eitl_queue,
    prepare_group_assign,
    prepare_leanb,
    prepare_split,
    write_records_json,
)
from ddharmon.harmonization.leanb_prompts import SYS_SPLIT
from ddharmon.models.cluster import FieldCluster, FieldReference


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


def test_prepare_leanb_builds_ideal_prompts_with_candidates(world):
    embedded, embeddings, field_refs, by_key = world
    age = _cluster(0, [by_key[("CohortA", "age")], by_key[("CohortB", "age_yrs")]])
    recs = prepare_leanb([age], embedded, embeddings, field_refs, top_k=4)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.id == "leanb:ideal:0"
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


# ── stage 2: prepare_split ──────────────────────────────────────


def test_prepare_split_carries_members_and_candidates(world):
    embedded, embeddings, field_refs, by_key = world
    age = _cluster(0, [by_key[("CohortA", "age")], by_key[("CohortB", "age_yrs")]])
    ideal_recs = prepare_leanb([age], embedded, embeddings, field_refs, top_k=4)
    split_recs = prepare_split(ideal_recs, _ideal_resp(ideal_recs))
    assert len(split_recs) == 1
    sr = split_recs[0]
    assert sr.id == "leanb:split:0"
    # ideal threaded into the prompt + context; the [mK]-prefixed members + candidates are in the prompt
    assert "ideal for 0" in sr.user_prompt and sr.context["ideal_cde"] == "ideal for 0"
    assert "[m1]" in sr.user_prompt and "Age in years" in sr.user_prompt
    assert "AgeCDE" in sr.user_prompt  # candidate block carried forward
    assert sr.context["members"] == ideal_recs[0].context["members"]


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


def test_prepare_group_assign_splits_into_two_groups_with_own_candidates(world):
    embedded, embeddings, field_refs, split_recs = _split_recs_for_zip(world)
    # mock split: partition the cluster into 2 distinct-concept groups
    split_resp = {
        split_recs[0].id: {
            "groups": [
                {"member_ids": ["m1"], "concept": "Home address ZIP code", "verdict": "refine", "cde_id": "1"},
                {"member_ids": ["m2"], "concept": "Employer address ZIP code", "verdict": "refine", "cde_id": "1"},
            ]
        }
    }
    grp_recs = prepare_group_assign(split_recs, split_resp, embedded, embeddings, field_refs, top_k=4)
    assert len(grp_recs) == 2
    assert [r.id for r in grp_recs] == ["leanb:groupassign:3:0", "leanb:groupassign:3:1"]
    g0, g1 = grp_recs
    assert g0.context["concept"] == "Home address ZIP code"
    assert g0.context["member_variable_names"] == ["CohortA:home_residence_zip"]
    assert g1.context["member_variable_names"] == ["CohortA:employer_workplace_zip"]
    # each group re-retrieved its OWN candidates and the concept reached the prompt
    assert g0.context["candidates"] and g0.context["candidates"][0]["designation"] == "ZipCDE"
    assert "Home address ZIP code" in g0.user_prompt
    assert g0.context["group_id"] == "3#g0" and g1.context["group_id"] == "3#g1"


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
    assert grp_recs[0].context["group_id"] == "3#g0"
    assert grp_recs[0].context["n_members"] == 3  # all members retained


# ── stage 4: assemble_leanb (multi-record) ──────────────────────


def _group_assign_recs(world):
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
    responses = {
        "leanb:groupassign:3:0": {"verdict": "adopt", "cde_id": "1", "ranking": [1, 2], "rationale": "home zip"},
        "leanb:groupassign:3:1": {"verdict": "novel", "cde_id": None, "ranking": [2, 1], "rationale": "no employer"},
    }
    result = assemble_leanb(grp_recs, responses, retrieval_floor=0.0)
    assert len(result.records) == 2
    assert {r.cluster_id for r in result.records} == {"3"}  # same cluster
    by_gid = {r.group_id: r for r in result.records}
    assert set(by_gid) == {"3#g0", "3#g1"}  # distinct groups

    g0 = by_gid["3#g0"]
    assert g0.verdict == "adopt" and g0.route == "assigned"
    assert g0.cde_id == "ZipCDE" and g0.cde_external_id == "cde_zip"
    assert g0.concept == "Home address ZIP code"
    assert g0.member_variable_names == ["CohortA:home_residence_zip"]
    assert g0.ranking == [0, 1]  # 1-based -> 0-based

    g1 = by_gid["3#g1"]
    assert g1.verdict == "novel" and g1.route == "gencde_residual" and g1.cde_id is None
    assert g1.concept == "Employer address ZIP code"
    assert g1.member_variable_names == ["CohortA:employer_workplace_zip"]


def test_assemble_handles_missing_and_unparseable_responses(world):
    grp_recs = _group_assign_recs(world)
    result = assemble_leanb(grp_recs, {})  # no responses at all
    assert len(result.records) == 2
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
    result = assemble_leanb(
        grp_recs,
        {
            "leanb:groupassign:3:0": {"verdict": "adopt", "cde_id": "1"},
            "leanb:groupassign:3:1": {"verdict": "refine", "cde_id": "1"},
        },
        retrieval_floor=0.0,
    )

    tsv = tmp_path / "eitl.tsv"
    n = export_leanb_eitl_queue(result, tsv)
    assert n == 2
    lines = tsv.read_text().splitlines()
    header = lines[0].split("\t")
    assert header[:5] == ["cluster_id", "group_id", "concept", "verdict", "route"]
    assert "members" in header
    # refine sorts before adopt; the per-group rows carry the group id + members column
    assert lines[1].split("\t")[:2] == ["3", "3#g1"]
    members_col = header.index("members")
    assert lines[1].split("\t")[members_col] == "CohortA:employer_workplace_zip"
    assert len(lines) == 3  # header + 2

    js = tmp_path / "records.json"
    assert write_records_json(result, js) == 2
    loaded = json.loads(js.read_text())
    assert {r["group_id"] for r in loaded} == {"3#g0", "3#g1"}
    assert all(r["cluster_id"] == "3" for r in loaded)


# ── prompt content ──────────────────────────────────────────────


def test_axis_preservation_clause_in_split_prompt():
    # the split system prompt instructs the model to split on the object/referent axis (preserve qualifier)
    s = SYS_SPLIT.lower()
    assert "object" in s and "referent" in s
    assert "split" in s and ("do not split" in s or "do not over-split" in s)


def test_harmonize_leanb_clusters_cohorts_only_not_the_cde_backbone(monkeypatch):
    """harmonize_leanb must cluster the COHORT fields only — the CDE backbone is the retrieval
    target, not a clustered cohort. Regression guard for the cohorts-only clustering fix."""
    from types import SimpleNamespace

    import ddharmon.clustering.topic_engine as te
    import ddharmon.harmonization.leanb as leanb

    captured: dict[str, list[str]] = {}

    def fake_topic_model(dicts, **kwargs):
        captured["clustered"] = [d.dictionary.cohort_name for d in dicts]
        return SimpleNamespace(clusters=[], embeddings=np.zeros((0, 8), dtype=np.float32), field_refs=[])

    monkeypatch.setattr(te, "topic_model_dictionaries", fake_topic_model)
    monkeypatch.setattr(leanb, "prepare_leanb", lambda *a, **k: [])  # short-circuit downstream

    def mk(name: str):
        return SimpleNamespace(dictionary=SimpleNamespace(cohort_name=name, name=name, fields={}))

    embedded = [mk("AoU"), mk("CLSA"), mk("NIH_CDE")]
    result = leanb.harmonize_leanb(embedded, cde_cohort="NIH_CDE")  # generate=None -> $0 path

    assert captured["clustered"] == ["AoU", "CLSA"]  # CDE backbone excluded from clustering
    assert result.ideal_prompts == []
