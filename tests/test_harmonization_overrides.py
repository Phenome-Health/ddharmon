"""Reviewer group overrides: a Gate-1 regrouping is APPLIED before the paid assign.

``harmonize_leanb(group_overrides=...)`` takes the reviewer's moves (``"cohort:var" -> destination group``)
and their New groups, and re-shapes the post-split groups after the split replay and before assign:
moved members leave their origin, reviewer groups form (and may span clusters), emptied groups drop,
every changed or new group is re-retrieved (local, free), and each New group buys ONE generate-ideal call.

Synthetic embeddings only — no sentence-transformers, no network, no LLM.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ddharmon.clustering.topic_engine import collect_inputs
from ddharmon.harmonization import GroupOverrides, ReviewerGroup, harmonize_leanb
from ddharmon.harmonization.substrate import build_substrate
from ddharmon.models.cluster import FieldCluster

NEW = "rev:0f8fad5b-d9cb-469f-a165-70867728950e"


@pytest.fixture
def eye_world(hf):
    """Two eye clusters whose split leaves one group each, a cataract item in each, and one ungrouped field.

    Cluster X = {A:glaucoma, B:glaucoma_b, A:cataract}; cluster Y = {B:cataract_b, A:macular}; A:retina is in no
    cluster. The reviewer's case (the live walk's glaucoma/cataract spread): pull both cataract items into a
    New group that spans the two clusters.
    """
    a = [
        hf.field("glaucoma", "Ever diagnosed with glaucoma", encoding="1=Yes|0=No"),
        hf.field("cataract", "Ever diagnosed with cataract", encoding="1=Yes|0=No"),
        hf.field("macular", "Macular degeneration diagnosis", encoding="1=Yes|0=No"),
        hf.field("retina", "Detached retina", encoding="1=Yes|0=No"),
    ]
    b = [
        hf.field("glaucoma_b", "Glaucoma diagnosis", encoding="Y=Yes|N=No"),
        hf.field("cataract_b", "Cataract diagnosis", encoding="Y=Yes|N=No"),
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
        sem_vecs=hf.l2(np.array([[1, 0.05, 0, 0, 0], [0.05, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], float)),
    )
    ed_b = hf.embedded_dict(
        "CohortB", b, sem_vecs=hf.l2(np.array([[0.98, 0.02, 0, 0, 0], [0.02, 0.98, 0, 0, 0]], float))
    )
    ed_cde = hf.embedded_dict(
        "NIH_CDE",
        cde,
        sem_vecs=hf.l2(np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], float)),
    )
    embedded = [ed_a, ed_b, ed_cde]
    _docs, _emb, field_refs, _cohorts = collect_inputs(embedded)
    by = {(r.dictionary_name, r.variable_name): r for r in field_refs}
    x = FieldCluster(cluster_id=0, label="x", members=[by[("CohortA", "glaucoma")], by[("CohortB", "glaucoma_b")],
                                                       by[("CohortA", "cataract")]])  # fmt: skip
    y = FieldCluster(cluster_id=1, label="y", members=[by[("CohortB", "cataract_b")], by[("CohortA", "macular")]])
    sub = build_substrate([x, y], min_cluster_size=15, n_fields=len(field_refs))
    return embedded, sub


class Calls:
    """Counting stand-ins for every stage. The split answers nothing, so each cluster is one group."""

    def __init__(self) -> None:
        self.seen: dict[str, list[str]] = {}

    def _log(self, name: str, prompts) -> None:
        self.seen.setdefault(name, []).extend(str(p.id) for p in prompts)

    def stages(self, *, with_group_generate: bool = True) -> dict:
        def generate(prompts):
            self._log("generate", prompts)
            return {p.id: {"ideal_cde": f"ideal for {p.id}"} for p in prompts}

        def group_generate(prompts):
            self._log("group_generate", prompts)
            return {p.id: {"ideal_cde": "Cataract diagnosis (yes/no)"} for p in prompts}

        def split(prompts):
            self._log("split", prompts)
            return {}

        def classify(prompts):
            self._log("classify", prompts)
            return {p.id: {"verdict": "adopt", "cde_id": "1", "ranking": [1], "rationale": "same"} for p in prompts}

        def coherence(prompts):
            self._log("coherence", prompts)
            return {}

        out = {"generate": generate, "split": split, "classify": classify, "coherence": coherence}
        if with_group_generate:
            out["group_generate"] = group_generate
        return out


def _gate1(eye_world):
    embedded, sub = eye_world
    g1 = harmonize_leanb(embedded, substrate=sub, recover_outliers=False, classify=None, **{
        k: v for k, v in Calls().stages().items() if k in ("generate", "split", "coherence")})  # fmt: skip
    by_first = {g.member_variable_names[0]: g.group_id for g in g1.concept_groups}
    return by_first["CohortA:glaucoma"], by_first["CohortB:cataract_b"], g1


def _overrides(xg: str, yg: str, *, order: int = 0) -> GroupOverrides:
    moves = {
        "CohortA:cataract": NEW,  # X -> the New group
        "CohortB:cataract_b": NEW,  # Y -> the New group (so it spans both clusters)
        "CohortA:retina": xg,  # a field in NO group, placed into X
        "CohortA:macular": None,  # Y's last member, out of every group -> Y empties
    }
    if order:
        moves = dict(reversed(list(moves.items())))
    return GroupOverrides(moves=moves, new_groups=(ReviewerGroup(group_id=NEW, name="Cataract"),))


def _run(eye_world, overrides, calls: Calls | None = None, **kw):
    embedded, sub = eye_world
    calls = calls or Calls()
    stages = calls.stages(with_group_generate=kw.pop("with_group_generate", True))
    return harmonize_leanb(embedded, substrate=sub, recover_outliers=False, group_overrides=overrides, **stages, **kw)


def test_group_overrides_is_feature_detectable():
    """Additive, keyword-only, default None — the adapter guards on the signature like every other knob."""
    params = inspect.signature(harmonize_leanb).parameters
    for name in ("group_overrides", "group_generate"):
        assert name in params
        assert params[name].default is None
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_a_new_group_spans_clusters_and_is_assigned_as_its_own_record(eye_world):
    xg, yg, _g1 = _gate1(eye_world)
    result = _run(eye_world, _overrides(xg, yg))
    by = {r.group_id: r for r in result.records}
    assert NEW in by, sorted(by)
    rec = by[NEW]
    assert sorted(rec.member_variable_names) == ["CohortA:cataract", "CohortB:cataract_b"]
    assert rec.cohorts == ["CohortA", "CohortB"] and rec.cross_cohort
    assert rec.concept == "Cataract", "the reviewer's name is the group's concept"
    assert rec.ideal_cde == "Cataract diagnosis (yes/no)", "the New group's own generated ideal anchors it"
    assert rec.cde_id == "CataractCDE", "re-retrieved on its OWN members, so the cataract CDE ranks first"


def test_moved_members_leave_their_origin_and_reach_their_destination(eye_world):
    xg, yg, _g1 = _gate1(eye_world)
    by = {r.group_id: r for r in _run(eye_world, _overrides(xg, yg)).records}
    assert sorted(by[xg].member_variable_names) == ["CohortA:glaucoma", "CohortA:retina", "CohortB:glaucoma_b"]
    everywhere = [m for r in by.values() for m in r.member_variable_names]
    assert "CohortA:macular" not in everywhere, "a member moved to no group reached a record"
    assert len(everywhere) == len(set(everywhere)), "a member is in two groups"


def test_an_emptied_group_is_dropped_and_never_assigned(eye_world):
    xg, yg, _g1 = _gate1(eye_world)
    calls = Calls()
    result = _run(eye_world, _overrides(xg, yg), calls)
    assert yg not in {r.group_id for r in result.records}
    assert not any(pid.startswith(f"leanb:groupassign:{yg.split('#')[0]}:") for pid in calls.seen["classify"])


def test_each_new_or_changed_group_buys_exactly_one_ideal_on_its_own_runner(eye_world):
    """A New group's ideal, and a regenerated ideal for each existing group whose
    membership changed, are the ONLY new generate-style calls, sent to ``group_generate`` — never to
    ``generate``, whose prompts stay exactly the frozen partition's (a caller can keep that stage closed).
    Y is emptied by the overrides, so it has nothing to describe and buys nothing."""
    xg, yg, g1 = _gate1(eye_world)
    calls = Calls()
    result = _run(eye_world, _overrides(xg, yg), calls)
    asked = calls.seen["group_generate"]
    assert len(asked) == 2, asked
    assert sorted(pid.split("@")[0] for pid in asked) == sorted([f"leanb:groupideal:{NEW}", f"leanb:groupideal:{xg}"])
    assert all("@c" in pid for pid in asked), asked
    assert [p.id for p in result.group_ideal_prompts] == asked
    assert all(p.startswith("leanb:ideal:c") for p in calls.seen["generate"]), calls.seen["generate"]
    assert sorted(calls.seen["generate"]) == sorted(p.id for p in harmonize_leanb(
        eye_world[0], substrate=eye_world[1], recover_outliers=False, generate=None).ideal_prompts)  # fmt: skip


def test_group_generate_defaults_to_generate(eye_world):
    xg, yg, _g1 = _gate1(eye_world)
    calls = Calls()
    result = _run(eye_world, _overrides(xg, yg), calls, with_group_generate=False)
    assert any(p.startswith("leanb:groupideal:") for p in calls.seen["generate"])
    assert NEW in {r.group_id for r in result.records}


def test_the_ideal_prompt_id_is_content_addressed_on_the_new_groups_members(eye_world):
    """Same members -> same id (a later leg replays it for $0); different members -> a different id (a re-filled
    group can never be answered with an ideal written for other members)."""
    xg, yg, _g1 = _gate1(eye_world)
    a = _run(eye_world, _overrides(xg, yg)).group_ideal_prompts[0].id
    fewer = GroupOverrides(moves={"CohortA:cataract": NEW}, new_groups=(ReviewerGroup(NEW, "Cataract"),))
    b = _run(eye_world, fewer).group_ideal_prompts[0].id
    assert a != b and a.split("@")[0] == b.split("@")[0]


def test_the_same_overrides_reproduce_the_same_groups_and_prompt_ids(eye_world):
    """Replay-safe: a later leg with the same overrides (in any order) rebuilds byte-identical prompts."""
    xg, yg, _g1 = _gate1(eye_world)

    def fingerprint(order: int):
        calls = Calls()
        res = _run(eye_world, _overrides(xg, yg, order=order), calls)
        prompts = {p.id: (p.system_prompt, p.user_prompt) for p in res.group_assign_prompts}
        return calls.seen, prompts, [(r.group_id, r.member_variable_names) for r in res.records]

    assert fingerprint(0) == fingerprint(1)


def test_untouched_groups_keep_their_prompts_and_changed_ones_get_a_new_id(eye_world):
    """A group no override touched is byte-identical to the no-override run (its cached assign still hits);
    a changed group keeps its GROUP id (every decision keyed on it still applies) but its assign prompt id is
    content-addressed on the new membership, so a stale answer for the old members can never replay."""
    xg, yg, _g1 = _gate1(eye_world)
    only_x = GroupOverrides(moves={"CohortA:cataract": None})
    base = {p.context["group_id"]: p for p in _run(eye_world, None).group_assign_prompts}
    over = {p.context["group_id"]: p for p in _run(eye_world, only_x).group_assign_prompts}
    assert (over[yg].id, over[yg].user_prompt) == (base[yg].id, base[yg].user_prompt)
    assert over[xg].id.startswith(base[xg].id + "@c") and over[xg].id != base[xg].id
    assert "Ever diagnosed with cataract" not in over[xg].user_prompt, "the member sample still shows the moved member"
    assert "CohortA:cataract" not in over[xg].context["member_variable_names"]


def test_the_gate_1_view_is_unchanged_and_the_judge_is_not_re_asked(eye_world):
    """``concept_groups`` stays what Gate 1 showed (overrides change what is ASSIGNED, not what the earlier gate
    displayed), and no generate/split/judge prompt differs from the Gate-1 leg's."""
    xg, yg, g1 = _gate1(eye_world)
    base_calls, over_calls = Calls(), Calls()
    base = _run(eye_world, None, base_calls)
    over = _run(eye_world, _overrides(xg, yg), over_calls)
    view = lambda r: [(g.group_id, g.member_variable_names) for g in r.concept_groups]  # noqa: E731
    assert view(over) == view(base) == view(g1)
    for stage in ("generate", "split", "coherence"):
        assert over_calls.seen.get(stage) == base_calls.seen.get(stage), stage


def test_scope_applies_to_reviewer_groups_like_any_other(eye_world):
    xg, yg, _g1 = _gate1(eye_world)
    out = _run(eye_world, _overrides(xg, yg), assign_group_ids={xg})
    assert {r.group_id for r in out.records} == {xg}
    both = _run(eye_world, _overrides(xg, yg), assign_group_ids={xg, NEW})
    assert {r.group_id for r in both.records} == {xg, NEW}


def test_a_new_group_with_no_members_buys_nothing_and_forms_no_record(eye_world):
    calls = Calls()
    empty = GroupOverrides(new_groups=(ReviewerGroup(NEW, "Nothing here yet"),))
    result = _run(eye_world, empty, calls)
    assert "group_generate" not in calls.seen and not result.group_ideal_prompts
    assert NEW not in {r.group_id for r in result.records}


@pytest.mark.parametrize(
    "overrides",
    [
        GroupOverrides(moves={"CohortA:cataract": "rev:not-declared"}),
        GroupOverrides(moves={"CohortA:no_such_field": None}),
        GroupOverrides(moves={"NIH_CDE:CataractCDE": NEW}, new_groups=(ReviewerGroup(NEW, "x"),)),
        GroupOverrides(new_groups=(ReviewerGroup(NEW, "a"), ReviewerGroup(NEW, "b"))),
    ],
    ids=["unknown-destination", "unknown-member", "catalog-row", "duplicate-new-group"],
)
def test_an_override_that_names_nothing_real_raises_before_assign(eye_world, overrides):
    calls = Calls()
    with pytest.raises(ValueError):
        _run(eye_world, overrides, calls)
    assert "classify" not in calls.seen and "group_generate" not in calls.seen


def test_reviewer_shaped_groups_are_never_offered_to_the_model_merge(eye_world, monkeypatch):
    """The reviewer formed these groups by hand; a model merge folding one into another would undo that
    decision and move its members under a different group id. They are left out of merge candidacy, while an
    untouched group is still offered exactly as before."""
    import ddharmon.harmonization.merge as merge_mod

    xg, yg, _g1 = _gate1(eye_world)
    offered: list[str] = []
    real = merge_mod.prepare_merge

    def spy(records, *a, **kw):
        offered.extend(r.group_id for r in records)
        return real(records, *a, **kw)

    monkeypatch.setattr(merge_mod, "prepare_merge", spy)
    only_cataract = GroupOverrides(moves={"CohortA:cataract": NEW}, new_groups=(ReviewerGroup(NEW, "Cataract"),))
    result = _run(eye_world, only_cataract)
    assert {xg, NEW, yg} <= {r.group_id for r in result.records}
    assert offered == [yg], offered


# ── a group the reviewer EDITED gets its ideal regenerated for its final members ─────────────────────────────


def _regen(calls: Calls):
    """``group_generate`` answering with a description that names the group it was written for."""

    def group_generate(prompts):
        calls._log("group_generate", prompts)
        return {p.id: {"ideal_cde": f"regenerated for {p.context['group_id']}"} for p in prompts}

    return group_generate


def test_a_group_whose_membership_changed_is_judged_against_a_regenerated_ideal(eye_world):
    """The ideal anchors the novel decision, so a group the reviewer reshaped is assigned against a description of
    the members it now HAS — one paid call, on ``group_generate`` — while an untouched group keeps the ideal it
    was split with and buys nothing."""
    xg, yg, _g1 = _gate1(eye_world)
    base = {r.group_id: r for r in _run(eye_world, None).records}
    calls = Calls()
    embedded, sub = eye_world
    stages = {**calls.stages(), "group_generate": _regen(calls)}
    only_x = GroupOverrides(moves={"CohortA:cataract": None})
    result = harmonize_leanb(embedded, substrate=sub, recover_outliers=False, group_overrides=only_x, **stages)
    by = {r.group_id: r for r in result.records}
    assert by[xg].ideal_cde == f"regenerated for {xg}", "the reshaped group kept the ideal of its old members"
    assert by[yg].ideal_cde == base[yg].ideal_cde, "an untouched group's ideal changed"
    (pid,) = calls.seen["group_generate"]
    assert pid.startswith(f"leanb:groupideal:{xg}@c"), pid
    (prompt,) = result.group_ideal_prompts
    assert prompt.context["member_variable_names"] == by[xg].member_variable_names
    assert "cataract" not in prompt.user_prompt.lower(), "the regenerated ideal was shown the member that left"
    assign = next(p for p in result.group_assign_prompts if p.context["group_id"] == xg)
    assert assign.context["ideal_cde"] == f"regenerated for {xg}"


def test_a_regenerated_ideal_is_content_addressed_on_the_final_members(eye_world):
    """Same final members -> the same prompt id (a later leg replays it for $0); other members -> another id."""
    xg, _yg, _g1 = _gate1(eye_world)
    a = _run(eye_world, GroupOverrides(moves={"CohortA:cataract": None})).group_ideal_prompts
    b = _run(eye_world, GroupOverrides(moves={"CohortA:cataract": None})).group_ideal_prompts
    c = _run(eye_world, GroupOverrides(moves={"CohortA:glaucoma": None})).group_ideal_prompts
    assert [p.id for p in a] == [p.id for p in b]
    assert a[0].id != c[0].id and a[0].id.split("@")[0] == c[0].id.split("@")[0] == f"leanb:groupideal:{xg}"


def test_no_ideal_is_bought_for_a_group_that_will_not_be_assigned(eye_world):
    """The ideal is consumed only by the assign, so a group the reviewer left OUT of scope — reshaped or New —
    buys none (Gate 1's quote prices the in-scope groups only)."""
    xg, yg, _g1 = _gate1(eye_world)
    calls = Calls()
    result = _run(eye_world, _overrides(xg, yg), calls, assign_group_ids={yg})
    assert "group_generate" not in calls.seen, calls.seen
    assert not result.group_ideal_prompts
    in_x = Calls()
    _run(eye_world, _overrides(xg, yg), in_x, assign_group_ids={xg})
    assert [pid.split("@")[0] for pid in in_x.seen["group_generate"]] == [f"leanb:groupideal:{xg}"]


def test_a_reshaped_group_with_no_regenerated_answer_keeps_its_original_ideal(eye_world):
    """A regenerate call that came back empty must not leave the group with NO anchor: it keeps the one it was
    split with (the adapter separately fails a leg whose deciding stage came back short)."""
    xg, _yg, _g1 = _gate1(eye_world)
    base = {r.group_id: r for r in _run(eye_world, None).records}
    embedded, sub = eye_world
    calls = Calls()
    stages = {**calls.stages(), "group_generate": lambda prompts: {}}
    result = harmonize_leanb(
        embedded,
        substrate=sub,
        recover_outliers=False,
        group_overrides=GroupOverrides(moves={"CohortA:cataract": None}),
        **stages,
    )
    assert {r.group_id: r for r in result.records}[xg].ideal_cde == base[xg].ideal_cde
