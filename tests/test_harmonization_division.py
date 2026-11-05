"""Accepting a proposed division at Gate 1.

Gate 1 is the ``classify=None`` pause: the split has run, the judge has flagged an over-merged group, and nothing
has been assigned — so ``LeanBResult.records`` is EMPTY there. "Accept the division" used to pick its groups
from ``records`` and therefore re-split nothing (a paid button that silently did nothing). These tests drive the
real Gate-1 result: the division reads the Gate-1 concept groups, re-splits the group as the reviewer currently
sees it, and — carried as reviewer overrides — survives every later leg.

Synthetic embeddings only — no sentence-transformers, no network, no LLM.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddharmon.clustering.topic_engine import collect_inputs
from ddharmon.harmonization import GroupOverrides, division_overrides, harmonize_leanb, readjudicate_split_only
from ddharmon.harmonization.substrate import build_substrate
from ddharmon.models.cluster import FieldCluster

GLAUCOMA = ("CohortA:glaucoma", "CohortB:glaucoma_b", "CohortA:glaucoma_dx")
CATARACT = ("CohortA:cataract", "CohortB:cataract_b")


@pytest.fixture
def fused_world(hf):
    """ONE cluster fusing glaucoma and cataract items across two cohorts (the live walk's eye-condition
    over-merge), a second one-concept cluster, and one variable the clustering left in no group."""
    a = [
        hf.field("glaucoma", "Ever diagnosed with glaucoma", encoding="1=Yes|0=No"),
        hf.field("glaucoma_dx", "Glaucoma diagnosed by a doctor", encoding="1=Yes|0=No"),
        hf.field("cataract", "Ever diagnosed with cataract", encoding="1=Yes|0=No"),
        hf.field("macular", "Macular degeneration diagnosis", encoding="1=Yes|0=No"),
        hf.field("retina", "Detached retina", encoding="1=Yes|0=No"),
    ]
    b = [
        hf.field("glaucoma_b", "Glaucoma diagnosis", encoding="Y=Yes|N=No"),
        hf.field("cataract_b", "Cataract diagnosis", encoding="Y=Yes|N=No"),
        hf.field("macular_b", "Macular degeneration", encoding="Y=Yes|N=No"),
    ]
    cde = [
        hf.field("GlaucomaCDE", "Glaucoma diagnosis indicator", field_id="cde_g", encoding="1=Yes|0=No"),
        hf.field("CataractCDE", "Cataract diagnosis indicator", field_id="cde_c", encoding="1=Yes|0=No"),
        hf.field("MacularCDE", "Macular degeneration indicator", field_id="cde_m", encoding="1=Yes|0=No"),
        hf.field("RetinaCDE", "Retinal detachment indicator", field_id="cde_r", encoding="1=Yes|0=No"),
    ]
    ed_a = hf.embedded_dict(
        "CohortA",
        a,
        sem_vecs=hf.l2(
            np.array(
                [[1, 0.05, 0, 0, 0], [0.97, 0.06, 0, 0, 0], [0.05, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
                float,
            )
        ),
    )
    ed_b = hf.embedded_dict(
        "CohortB",
        b,
        sem_vecs=hf.l2(np.array([[0.98, 0.02, 0, 0, 0], [0.02, 0.98, 0, 0, 0], [0, 0, 0.98, 0.02, 0]], float)),
    )
    ed_cde = hf.embedded_dict(
        "NIH_CDE",
        cde,
        sem_vecs=hf.l2(np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], float)),
    )
    embedded = [ed_a, ed_b, ed_cde]
    _docs, _emb, field_refs, _cohorts = collect_inputs(embedded)
    by = {f"{r.dictionary_name}:{r.variable_name}": r for r in field_refs}
    fused = FieldCluster(cluster_id=0, label="eye", members=[by[k] for k in (*GLAUCOMA, *CATARACT)])
    macular = FieldCluster(cluster_id=1, label="macular", members=[by["CohortA:macular"], by["CohortB:macular_b"]])
    sub = build_substrate([fused, macular], min_cluster_size=15, n_fields=len(field_refs))
    return embedded, sub


class Stages:
    """Counting stand-ins. The pipeline's own split keeps every cluster whole (the over-merge Gate 1 shows);
    the re-split — the one the reviewer pays for by accepting — divides glaucoma from cataract."""

    def __init__(self) -> None:
        self.seen: dict[str, list[str]] = {}
        self.split_members: list[list[str]] = []

    def _log(self, name: str, prompts) -> None:
        self.seen.setdefault(name, []).extend(str(p.id) for p in prompts)

    def generate(self, prompts):
        self._log("generate", prompts)
        return {p.id: {"ideal_cde": f"ideal for {p.id}"} for p in prompts}

    def group_generate(self, prompts):
        self._log("group_generate", prompts)
        return {p.id: {"ideal_cde": f"regenerated for {sorted(p.context['member_variable_names'])}"} for p in prompts}

    def split(self, prompts):
        self._log("split", prompts)
        return {}

    def coherence(self, prompts):
        self._log("coherence", prompts)
        return {}

    def classify(self, prompts):
        self._log("classify", prompts)
        return {p.id: {"verdict": "adopt", "cde_id": "1", "ranking": [1], "rationale": "same"} for p in prompts}

    def resplit(self, prompts):
        """Divide each re-split prompt into its glaucoma and its cataract members."""
        self._log("resplit", prompts)
        out = {}
        for p in prompts:
            members = p.context["members"]
            self.split_members.append([f"{m['dictionary_name']}:{m['variable_name']}" for m in members])
            glau = [m["member_id"] for m in members if "glaucoma" in m["variable_name"]]
            cata = [m["member_id"] for m in members if "glaucoma" not in m["variable_name"]]
            out[p.id] = {
                "groups": [
                    {"member_ids": glau, "concept": "Glaucoma diagnosis"},
                    {"member_ids": cata, "concept": "Cataract diagnosis"},
                ]
            }
        return out

    def gate1(self) -> dict:
        return {"generate": self.generate, "split": self.split, "coherence": self.coherence}


def _gate1(world):
    embedded, sub = world
    stages = Stages()
    g1 = harmonize_leanb(embedded, substrate=sub, recover_outliers=False, classify=None, **stages.gate1())
    fused = next(g for g in g1.concept_groups if "CohortA:glaucoma" in g.member_variable_names)
    return g1, fused.group_id


def _inputs(world):
    embedded, _sub = world
    _docs, embeddings, field_refs, _cohorts = collect_inputs(embedded)
    return embedded, embeddings, field_refs


# ── the defect: at Gate 1 there are no records, and the division still has to happen ────────────────


def test_at_the_gate_1_pause_accepting_a_division_re_splits_the_concept_group(fused_world):
    """The live finding: ``records`` is empty at the Gate-1 pause, so selecting from it re-split nothing and the
    paid button did nothing. The division reads the Gate-1 groups."""
    g1, parent = _gate1(fused_world)
    assert g1.records == [], "fixture: Gate 1 is the classify=None pause, with nothing assigned"
    stages = Stages()
    out = readjudicate_split_only(g1, *_inputs(fused_world), split=stages.resplit, group_ids=[parent])

    assert stages.seen.get("resplit"), "the re-split the reviewer accepted was never asked"
    parts = [g for g in out.concept_groups if g.readjudicated_from == parent]
    assert sorted(sorted(p.member_variable_names) for p in parts) == [sorted(CATARACT), sorted(GLAUCOMA)]
    assert {p.concept for p in parts} == {"Glaucoma diagnosis", "Cataract diagnosis"}
    assert parent not in {g.group_id for g in out.concept_groups}, "the divided parent is still listed"
    assert out.records == [], "a division is a grouping change: nothing may be assigned here"


def test_a_group_the_reviewer_already_reshaped_is_divided_as_they_see_it(fused_world):
    """A reviewer who moved a variable out of the group (or one in) before accepting has the group as it now
    stands re-split — never the original membership, which would silently undo their move."""
    g1, parent = _gate1(fused_world)
    macular = next(g.group_id for g in g1.concept_groups if g.group_id != parent)
    current = GroupOverrides(moves={"CohortB:cataract_b": macular, "CohortA:retina": parent})
    stages = Stages()
    out = readjudicate_split_only(
        g1, *_inputs(fused_world), split=stages.resplit, group_ids=[parent], group_overrides=current
    )
    (asked,) = stages.split_members
    assert "CohortB:cataract_b" not in asked, "a variable the reviewer moved away was divided anyway"
    assert "CohortA:retina" in asked, "a variable the reviewer moved in was left out of the division"
    parts = {g.concept: g.member_variable_names for g in out.concept_groups if g.readjudicated_from == parent}
    assert sorted(parts["Cataract diagnosis"]) == ["CohortA:cataract", "CohortA:retina"]


def test_nothing_is_bought_for_a_group_with_fewer_than_two_variables_left(fused_world):
    g1, parent = _gate1(fused_world)
    macular = next(g.group_id for g in g1.concept_groups if g.group_id != parent)
    emptied = GroupOverrides(moves=dict.fromkeys((*GLAUCOMA, *CATARACT)[1:], macular))
    stages = Stages()
    out = readjudicate_split_only(
        g1, *_inputs(fused_world), split=stages.resplit, group_ids=[parent], group_overrides=emptied
    )
    assert "resplit" not in stages.seen
    assert not any(g.readjudicated_from for g in out.concept_groups)


# ── carried forward: an accepted division is a reviewer regrouping, so every later leg reproduces it ───────

PARTS = ("rev:00000000-0000-4000-8000-000000000001", "rev:00000000-0000-4000-8000-000000000002")


def _accepted(world):
    """Gate 1, the accept, and the division expressed as overrides — what the staged UI freezes at Continue."""
    g1, parent = _gate1(world)
    divided = readjudicate_split_only(g1, *_inputs(world), split=Stages().resplit, group_ids=[parent])
    parts = [g for g in divided.concept_groups if g.readjudicated_from == parent]
    return parent, parts, division_overrides(parts, PARTS)


def test_a_division_becomes_one_new_group_per_part_with_every_member_moved_into_it(fused_world):
    parent, parts, overrides = _accepted(fused_world)
    assert [g.group_id for g in overrides.new_groups] == list(PARTS)
    assert [g.name for g in overrides.new_groups] == [p.concept for p in parts]
    assert all(g.split_from == parent for g in overrides.new_groups), "a part lost the group it was divided from"
    assert overrides.moves == {m: gid for gid, p in zip(PARTS, parts, strict=True) for m in p.member_variable_names}


def test_the_next_leg_assigns_each_part_as_its_own_record_and_never_the_parent(fused_world):
    """The children of ``readjudicate_split_only`` exist on the Gate-1 result only; the next leg re-runs the split,
    which keeps the cluster whole. Carried as overrides, the division is applied there too."""
    parent, parts, overrides = _accepted(fused_world)
    embedded, sub = fused_world
    stages = Stages()
    result = harmonize_leanb(
        embedded,
        substrate=sub,
        recover_outliers=False,
        group_overrides=overrides,
        generate=stages.generate,
        split=stages.split,
        coherence=stages.coherence,
        classify=stages.classify,
        group_generate=stages.group_generate,
    )
    by = {r.group_id: r for r in result.records}
    assert parent not in by, "the divided parent was assigned"
    for gid, part in zip(PARTS, parts, strict=True):
        rec = by[gid]
        assert sorted(rec.member_variable_names) == sorted(part.member_variable_names)
        assert rec.concept == part.concept
        assert rec.readjudicated_from == parent, "the record does not say which group it was divided from"
        assert rec.ideal_cde.startswith("regenerated for"), "a part was assigned against the fused group's ideal"
    assert sorted(p.split("@")[0] for p in stages.seen["group_generate"]) == sorted(
        f"leanb:groupideal:{gid}" for gid in PARTS
    ), "each part buys exactly one ideal of its own"
    gate1_leg = Stages()
    harmonize_leanb(embedded, substrate=sub, recover_outliers=False, classify=None, **gate1_leg.gate1())
    for stage in ("generate", "split", "coherence"):
        assert stages.seen.get(stage) == gate1_leg.seen.get(stage), f"the division changed what {stage} was asked"


def test_a_division_merges_onto_the_reviewers_other_decisions(fused_world):
    parent, parts, _ = _accepted(fused_world)
    other = GroupOverrides(moves={"CohortA:retina": parent, "CohortA:macular": None})
    merged = division_overrides(parts, PARTS, base=other)
    assert merged.moves["CohortA:macular"] is None, "an unrelated move was dropped"
    assert all(merged.moves[m] in PARTS for p in parts for m in p.member_variable_names)


def test_each_part_needs_exactly_one_id(fused_world):
    _parent, parts, _ = _accepted(fused_world)
    with pytest.raises(ValueError):
        division_overrides(parts, PARTS[:1])
