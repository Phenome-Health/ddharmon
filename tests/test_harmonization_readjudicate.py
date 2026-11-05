"""Tests for human-triggered re-adjudication (re-split of a flagged over-merged group).

Exercises prepare_readjudicate (prompt shape + member reconstruction + judge hint) and the readjudicate
orchestrator (re-split -> reuse prepare_group_assign/assemble_leanb -> splice children over the parent),
with mock split/classify callables. No LLM, no network. Never auto-runs — always caller-triggered.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddharmon.clustering.topic_engine import collect_inputs
from ddharmon.harmonization.anchor import CDE_COHORT
from ddharmon.harmonization.leanb import LeanBResult, prepare_readjudicate, readjudicate, readjudicate_split_only
from ddharmon.harmonization.leanb_prompts import SYS_READJUDICATE
from ddharmon.harmonization.models import ConceptGroup, LeanBRecord


@pytest.fixture
def world(hf):
    """Cohort C (6 fields, two sub-concepts) + a 2-CDE backbone, so retrieval + assign resolve."""
    c_fields = [hf.field(f"v{i}", f"field {i} description") for i in range(6)]
    cde_fields = [
        hf.field("CDE_A", "Concept A", field_id="a"),
        hf.field("CDE_B", "Concept B", field_id="b"),
    ]
    c_vecs = hf.l2(
        np.array(
            [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0], [0, 0, 1, 0], [0, 0, 0.9, 0.1], [0, 0, 0.8, 0.2]],
            float,
        )
    )
    ed_c = hf.embedded_dict("C", c_fields, sem_vecs=c_vecs)
    ed_cde = hf.embedded_dict(CDE_COHORT, cde_fields, sem_vecs=hf.l2(np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)))
    _docs, embeddings, field_refs, _cn = collect_inputs([ed_c, ed_cde])
    return [ed_c, ed_cde], embeddings, field_refs


def _flagged(field_refs, n=6, group_id="c1#g0"):
    fids = [f"C:v{i}" for i in range(n)]
    return LeanBRecord(
        cluster_id="c1",
        verdict="adopt",
        route="assigned",
        group_id=group_id,
        member_variable_names=fids,
        n_members=n,
        incoherent=True,
        coherence_verdict="split",
        coherence_axis="concept measured",
        coherence_distinct_values=["A things", "B things"],
    )


# ── prepare_readjudicate ────────────────────────────────────────


def test_prepare_readjudicate_builds_split_shaped_prompt(world):
    embedded, embeddings, field_refs = world
    rec = _flagged(field_refs)
    prompts = prepare_readjudicate([rec], embedded, embeddings, field_refs, desired_n={"c1#g0": 2})

    assert len(prompts) == 1
    pr = prompts[0]
    assert pr.id == "leanb:readjudicate:c1#g0"
    assert pr.system_prompt == SYS_READJUDICATE
    assert pr.context["cluster_id"] == "c1#g0"  # namespaced under the parent group_id
    assert len(pr.context["members"]) == 6 and pr.context["candidates"]  # members + re-retrieved candidates
    assert pr.tool_schema is not None  # enforce_schema default True (M15)
    # the judge's hint + the desired-N target are woven into the prompt
    assert "concept measured" in pr.user_prompt
    assert "A things" in pr.user_prompt
    assert "2" in pr.user_prompt


def test_prepare_readjudicate_skips_unsplittable(world):
    embedded, embeddings, field_refs = world
    one = _flagged(field_refs, n=1, group_id="c1#g9")  # a single member -> nothing to re-partition
    assert prepare_readjudicate([one], embedded, embeddings, field_refs) == []


# ── readjudicate orchestrator (splice) ──────────────────────────


def _mock_split_into_two(prompts):
    """Every re-adjudication prompt splits into two child groups (m1-3 / m4-6)."""
    return {
        p.id: {
            "groups": [
                {"member_ids": ["m1", "m2", "m3"], "concept": "A", "verdict": "novel", "cde_id": None, "ranking": []},
                {"member_ids": ["m4", "m5", "m6"], "concept": "B", "verdict": "novel", "cde_id": None, "ranking": []},
            ]
        }
        for p in prompts
    }


def _mock_classify_novel(prompts):
    return {p.id: {"verdict": "novel", "ranking": [], "rationale": "child"} for p in prompts}


def test_readjudicate_splices_children_over_parent(world):
    embedded, embeddings, field_refs = world
    flagged = _flagged(field_refs)
    keeper = LeanBRecord(cluster_id="c2", verdict="adopt", route="assigned", group_id="c2#g0", n_members=3)
    result = LeanBResult(records=[flagged, keeper])

    readjudicate(result, embedded, embeddings, field_refs, split=_mock_split_into_two, classify=_mock_classify_novel)

    gids = [r.group_id for r in result.records]
    assert "c1#g0" not in gids  # the flagged parent was REPLACED
    assert "c2#g0" in gids  # the untouched record is kept
    children = [r for r in result.records if r.readjudicated_from == "c1#g0"]
    assert len(children) == 2  # re-split into two coherent children
    assert len({c.group_id for c in children}) == 2  # unique ids
    assert all(c.cluster_id == "c1#g0" for c in children)  # namespaced under the parent


def test_readjudicate_by_explicit_group_ids_only(world):
    embedded, embeddings, field_refs = world
    # a record that is NOT flagged incoherent, but the caller (human) explicitly selects it
    rec = _flagged(field_refs, group_id="c1#g0")
    rec.incoherent = False
    result = LeanBResult(records=[rec])

    readjudicate(
        result,
        embedded,
        embeddings,
        field_refs,
        split=_mock_split_into_two,
        classify=_mock_classify_novel,
        group_ids=["c1#g0"],
    )
    assert all(r.readjudicated_from == "c1#g0" for r in result.records)
    assert len(result.records) == 2


def test_readjudicate_noop_when_nothing_flagged(world):
    embedded, embeddings, field_refs = world
    clean = LeanBRecord(cluster_id="c1", verdict="adopt", route="assigned", group_id="c1#g0", incoherent=False)
    result = LeanBResult(records=[clean])

    readjudicate(result, embedded, embeddings, field_refs, split=_mock_split_into_two, classify=_mock_classify_novel)

    assert len(result.records) == 1 and result.records[0].group_id == "c1#g0"  # untouched


def test_readjudicate_single_group_replaces_parent_unchanged(world):
    embedded, embeddings, field_refs = world
    flagged = _flagged(field_refs)
    result = LeanBResult(records=[flagged])

    def split_into_one(prompts):
        return {
            p.id: {"groups": [{"member_ids": [f"m{i}" for i in range(1, 7)], "concept": "one", "verdict": "novel"}]}
            for p in prompts
        }

    readjudicate(result, embedded, embeddings, field_refs, split=split_into_one, classify=_mock_classify_novel)

    assert len(result.records) == 1  # one child replaces the parent
    assert result.records[0].readjudicated_from == "c1#g0"
    assert result.records[0].n_members == 6  # all members retained


# ── readjudicate_split_only (Option A: Gate-1 accept produces CHILD CONCEPT-GROUPS, unassigned) ──
#
# The workbench "accept the division" at Gate 1 is a GROUPING change, not an assignment. It must re-split
# the flagged over-merged group into child *concept-groups* that replace the parent in result.concept_groups
# (each tagged readjudicated_from = the parent group_id), WITHOUT running classify — the children are assigned
# later at the normal per-group assign stage (Gate 2). So this path takes `split` but NO `classify`, produces
# ConceptGroups (not LeanBRecords), and leaves the parent's record removed but no child records added.


def _parent_group(group_id="c1#g0", cluster_id="c1", n=6):
    return ConceptGroup(
        cluster_id=cluster_id,
        group_id=group_id,
        concept="mixed A+B",
        n_members=n,
        member_variable_names=[f"C:v{i}" for i in range(n)],
        incoherent=True,
        coherence_verdict="split",
    )


def test_readjudicate_split_only_replaces_parent_group_with_children(world):
    embedded, embeddings, field_refs = world
    flagged = _flagged(field_refs)  # record for c1#g0
    keeper_rec = LeanBRecord(cluster_id="c2", verdict="adopt", route="assigned", group_id="c2#g0", n_members=3)
    parent_group = _parent_group()
    keeper_group = ConceptGroup(cluster_id="c2", group_id="c2#g0", concept="keep", n_members=3)
    result = LeanBResult(records=[flagged, keeper_rec], concept_groups=[parent_group, keeper_group])

    # SPLIT-ONLY: no classify argument at all.
    readjudicate_split_only(result, embedded, embeddings, field_refs, split=_mock_split_into_two, group_ids=["c1#g0"])

    gids = [g.group_id for g in result.concept_groups]
    assert "c1#g0" not in gids  # the flagged parent GROUP was replaced
    assert "c2#g0" in gids  # the untouched group is kept
    children = [g for g in result.concept_groups if g.readjudicated_from == "c1#g0"]
    assert len(children) == 2  # re-split into two coherent child groups
    assert len({c.group_id for c in children}) == 2  # unique ids
    assert all(c.cluster_id == "c1#g0" for c in children)  # namespaced under the parent group
    assert {c.concept for c in children} == {"A", "B"}  # concept labels came from the split stage


def test_readjudicate_split_only_does_not_assign_children(world):
    embedded, embeddings, field_refs = world
    flagged = _flagged(field_refs)
    parent_group = _parent_group()
    result = LeanBResult(records=[flagged], concept_groups=[parent_group])

    readjudicate_split_only(result, embedded, embeddings, field_refs, split=_mock_split_into_two, group_ids=["c1#g0"])

    # The parent record is gone and NO child records were added — the children are unassigned ConceptGroups.
    assert result.records == []
    # The children exist only as groups (no verdict/route until Gate 2 assign).
    assert len(result.concept_groups) == 2
    assert all(getattr(g, "readjudicated_from", "") == "c1#g0" for g in result.concept_groups)


def test_readjudicate_split_only_noop_when_nothing_selected(world):
    embedded, embeddings, field_refs = world
    clean = LeanBRecord(cluster_id="c1", verdict="adopt", route="assigned", group_id="c1#g0", incoherent=False)
    group = ConceptGroup(cluster_id="c1", group_id="c1#g0", concept="clean", n_members=3)
    result = LeanBResult(records=[clean], concept_groups=[group])

    # explicit empty-intersection selection -> untouched
    readjudicate_split_only(result, embedded, embeddings, field_refs, split=_mock_split_into_two, group_ids=["nope"])

    assert [g.group_id for g in result.concept_groups] == ["c1#g0"]
    assert len(result.records) == 1
