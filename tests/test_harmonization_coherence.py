"""Tests for the step-2 dual-sample coherence judge (post-assign, read-only).

Exercises the sampling math (``_adaptive_k2`` / ``_dual_sample`` — cosine-medoid core/periphery), the $0
matrix pre-filter (``_matrix_suspect``), and ``prepare_coherence`` / ``assemble_coherence`` on hand-built
records with mock-LLM responses. No sentence-transformers, no real cohorts, no network. Flag-not-gate:
an incoherent verdict must set ``incoherent`` + ``needs_review`` and NEVER change verdict/route.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddharmon.clustering.topic_engine import collect_inputs
from ddharmon.harmonization.leanb import (
    COHERENCE_K1,
    _adaptive_k2,
    _dual_sample,
    _matrix_suspect,
    assemble_coherence,
    prepare_coherence,
)
from ddharmon.harmonization.leanb_prompts import SYS_COHERENCE
from ddharmon.harmonization.models import GenCDE, LeanBRecord, TransformKind, TransformSpec
from ddharmon.harmonization.pipeline import PromptRecord

# 2-D layout: 6 "core" members near [1,0], 2 outliers near [0,1]. Medoid falls in the core; the two
# outliers are the furthest members, so any periphery sample must include them.
_CORE_OUT_VECS = np.array(
    [[1, 0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [0.96, 0.04], [0.95, 0.05], [0, 1], [0.05, 0.99]],
    dtype=float,
)


def _world(hf, specs, vecs):
    """One cohort 'C' with fields ``specs`` = [(var, desc), …] and L2-normalized ``vecs`` (n × d)."""
    fields = [hf.field(v, d) for v, d in specs]
    ed = hf.embedded_dict("C", fields, sem_vecs=hf.l2(np.asarray(vecs, float)))
    _docs, embeddings, field_refs, _cohorts = collect_inputs([ed])
    return [ed], embeddings, field_refs


def _record(field_refs, idxs, group_id, *, verdict="adopt"):
    fids = [f"{field_refs[i].dictionary_name}:{field_refs[i].variable_name}" for i in idxs]
    return LeanBRecord(
        cluster_id=group_id.split("#")[0],
        verdict=verdict,
        route="assigned" if verdict in ("adopt", "refine") else "gencde_residual",
        group_id=group_id,
        member_variable_names=fids,
        n_members=len(fids),
    )


# ── _adaptive_k2 ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "n,expected",
    [
        (6, 1),  # n<10: remainder after the 5 core
        (7, 2),
        (9, 4),
        (10, 5),  # n>=10: ceil(0.2n) clipped up to the floor of 5
        (24, 5),  # ceil(4.8)=5
        (26, 6),  # ceil(5.2)=6
        (50, 10),
        (100, 20),  # clipped down to the ceiling of 20
        (200, 20),
    ],
)
def test_adaptive_k2(n, expected):
    assert _adaptive_k2(n) == expected


# ── _dual_sample ────────────────────────────────────────────────


def test_dual_sample_selects_core_and_disjoint_periphery():
    emb = _CORE_OUT_VECS / np.linalg.norm(_CORE_OUT_VECS, axis=1, keepdims=True)
    sample = _dual_sample(list(range(8)), emb)
    assert sample is not None
    k1, k2 = sample
    assert len(k1) == COHERENCE_K1  # 5 centroid-closest
    assert len(k2) == _adaptive_k2(8) == 3  # 3 centroid-furthest
    assert set(k1).isdisjoint(set(k2))  # disjoint (the fix for self-fulfilling verification)
    assert {6, 7} <= set(k2)  # the two off-theme outliers are the furthest -> in the periphery
    assert 6 not in set(k1) and 7 not in set(k1)


def test_dual_sample_none_for_small_group():
    emb = _CORE_OUT_VECS / np.linalg.norm(_CORE_OUT_VECS, axis=1, keepdims=True)
    assert _dual_sample(list(range(5)), emb) is None  # < COHERENCE_MIN_MEMBERS (6)


# ── _matrix_suspect ($0 pre-filter) ─────────────────────────────


def test_matrix_suspect_regression():
    # Pin: an open-vocabulary matrix (one template x entity fillers) must flag; distinct texts must not.
    matrix = [
        f"are you still seeing a doctor or health care provider for {c}?"
        for c in ("adhd", "depression", "asthma", "diabetes", "ptsd", "epilepsy")
    ]
    distinct = [
        "what is your age?",
        "country of birth",
        "annual household income",
        "do you currently smoke cigarettes?",
        "weight in kilograms",
        "highest level of education completed",
    ]
    assert _matrix_suspect(matrix)
    assert not _matrix_suspect(distinct)


def test_matrix_suspect_below_min_members():
    assert not _matrix_suspect(["seeing a provider for asthma"])  # single member -> not a matrix


# ── prepare_coherence ───────────────────────────────────────────


def test_prepare_coherence_builds_prompt_for_large_group_only(hf):
    specs = [(f"v{i}", f"field {i} description") for i in range(8)]
    embedded, embeddings, field_refs = _world(hf, specs, _CORE_OUT_VECS)
    big = _record(field_refs, list(range(8)), "c1#g0")
    small = _record(field_refs, [0, 1, 2], "c2#g0")  # 3 members -> not judged

    prompts = prepare_coherence(big_and_small := [big, small], embedded, embeddings, field_refs)

    assert len(prompts) == 1  # only the 8-member group
    pr = prompts[0]
    assert pr.id == "leanb:coherence:c1#g0"
    assert pr.system_prompt == SYS_COHERENCE
    assert "CORE" in pr.user_prompt and "PERIPHERY" in pr.user_prompt
    # the periphery member fids are carried for outlier resolution in assemble
    assert pr.context["group_id"] == "c1#g0"
    assert len(pr.context["periphery_fids"]) == _adaptive_k2(8)
    assert big_and_small[1].group_id == "c2#g0"  # small record untouched (no prompt)


def test_prepare_coherence_stamps_matrix_suspect(hf):
    conditions = ("adhd", "depression", "asthma", "diabetes", "ptsd", "epilepsy")
    specs = [(f"cond_{c}", f"still seeing a provider for {c}") for c in conditions]
    embedded, embeddings, field_refs = _world(hf, specs, _CORE_OUT_VECS[:6])
    rec = _record(field_refs, list(range(6)), "m1#g0")

    prepare_coherence([rec], embedded, embeddings, field_refs, pre_filter=True)

    assert rec.matrix_suspect is True  # $0 detector stamped even before any LLM verdict


def test_prepare_coherence_pre_filter_off(hf):
    conditions = ("adhd", "depression", "asthma", "diabetes", "ptsd", "epilepsy")
    specs = [(f"cond_{c}", f"still seeing a provider for {c}") for c in conditions]
    embedded, embeddings, field_refs = _world(hf, specs, _CORE_OUT_VECS[:6])
    rec = _record(field_refs, list(range(6)), "m1#g0")

    prepare_coherence([rec], embedded, embeddings, field_refs, pre_filter=False)

    assert rec.matrix_suspect is False  # left at the default when the pre-filter is disabled


# ── assemble_coherence (flag-not-gate) ──────────────────────────


def _coh_prompt(group_id, periphery_fids):
    return PromptRecord(
        id=f"leanb:coherence:{group_id}",
        system_prompt=SYS_COHERENCE,
        user_prompt="…",
        schema="{}",
        model_tag="test",
        context={"group_id": group_id, "periphery_fids": periphery_fids},
    )


def test_assemble_coherence_flags_split_and_propagates_needs_review():
    rec = LeanBRecord(
        cluster_id="c1",
        verdict="adopt",
        route="assigned",
        group_id="c1#g0",
        cde_id="BP_CDE",
        transforms=[TransformSpec(source_variable="C:sys", target_cde_id="BP_CDE", kind=TransformKind.CATEGORICAL)],
        gencde=GenCDE(gencde_id="GENCDE:c1#g0"),
    )
    pr = _coh_prompt("c1#g0", periphery_fids=["C:pulse", "C:method"])
    resp = {
        pr.id: {
            "summary": "Blood pressure measurement",
            "coherent": False,
            "outliers": [1, 2],
            "granularity": {"verdict": "split", "axis": "measurand", "distinct_values": ["systolic", "diastolic"]},
        }
    }

    assemble_coherence([pr], resp, [rec])

    assert rec.incoherent is True
    assert rec.coherent is False
    assert rec.coherence_verdict == "split"
    assert rec.coherence_axis == "measurand"
    assert rec.coherence_distinct_values == ["systolic", "diastolic"]
    assert rec.coherence_summary == "Blood pressure measurement"
    assert rec.coherence_outliers == ["C:pulse", "C:method"]  # 1-indexed positions -> fids
    # flag propagates to the recodes + GenCDE for human re-adjudication
    assert rec.transforms[0].needs_review is True
    assert rec.gencde.needs_review is True
    # never auto-splits: verdict + route are unchanged
    assert rec.verdict == "adopt" and rec.route == "assigned"


def test_assemble_coherence_qualify_is_advisory_not_flagged():
    # RECALIBRATED (full-5 §6.11): qualify over-fired 46% on coherent-with-qualifier groups -> ADVISORY,
    # not a hard flag. The axis/distinct_values are recorded (the qualifier value-set hint) but incoherent stays False.
    rec = LeanBRecord(
        cluster_id="c1",
        verdict="adopt",
        route="assigned",
        group_id="c1#g0",
        transforms=[TransformSpec(source_variable="C:milk", target_cde_id="MILK", kind=TransformKind.CATEGORICAL)],
    )
    pr = _coh_prompt("c1#g0", periphery_fids=["C:a"])
    resp = {
        pr.id: {
            "summary": "milk consumption varying by fat content",
            "coherent": True,
            "outliers": [],
            "granularity": {"verdict": "qualify", "axis": "fat content", "distinct_values": ["whole", "skim"]},
        }
    }

    assemble_coherence([pr], resp, [rec])

    assert rec.coherence_verdict == "qualify"  # advisory payload recorded
    assert rec.coherence_axis == "fat content" and rec.coherence_distinct_values == ["whole", "skim"]
    assert rec.incoherent is False  # NOT a hard flag


# ── R2: distinct-KINDS discriminator (opt-in, ADDITIVE over the R0 split-only flag; §6.16) ──────


def _qualify_rec(gid, summary, axis, distinct_values, *, verdict="adopt"):
    return LeanBRecord(
        cluster_id=gid.split("#")[0],
        verdict=verdict,
        route="assigned",
        group_id=gid,
        coherence_verdict="qualify",
        coherence_summary=summary,
        coherence_axis=axis,
        coherence_distinct_values=distinct_values,
        transforms=[TransformSpec(source_variable="C:x", target_cde_id="CDE", kind=TransformKind.CATEGORICAL)],
        gencde=GenCDE(gencde_id=f"GENCDE:{gid}"),
    )


def test_prepare_kinds_selects_only_nonpositional_qualify():
    from ddharmon.harmonization.leanb import prepare_kinds

    recs = [
        LeanBRecord(cluster_id="a", verdict="novel", route="novel", group_id="a#g0", coherence_verdict="single"),
        LeanBRecord(cluster_id="b", verdict="novel", route="novel", group_id="b#g0", coherence_verdict="split"),
        _qualify_rec("q#g0", "milk by fat", "fat", ["whole", "skim"]),
        _qualify_rec("p#g0", "height trials", "trial", ["1", "2", "3"]),  # positional -> skipped
    ]
    prompts = prepare_kinds(recs)
    assert [p.context["group_id"] for p in prompts] == ["q#g0"]  # only the non-positional qualify group
    assert "milk by fat" in prompts[0].user_prompt and "whole" in prompts[0].user_prompt  # judge outputs carried


def test_assemble_kinds_flags_distinct_kinds_and_propagates():
    from ddharmon.harmonization.leanb import assemble_kinds, prepare_kinds

    rec = _qualify_rec("q#g0", "ECG measures + completion status", "kind", ["QRS interval", "ECG done"])
    prompts = prepare_kinds([rec])
    assemble_kinds(prompts, {prompts[0].id: {"kind": "distinct_kinds", "rationale": "a duration vs a status"}}, [rec])
    assert rec.coherence_kind == "distinct_kinds"
    assert rec.incoherent is True  # R2 upgrades the flag
    assert rec.transforms[0].needs_review is True and rec.gencde.needs_review is True
    assert rec.verdict == "adopt" and rec.route == "assigned"  # flag-not-gate: verdict/route unchanged


def test_assemble_kinds_values_of_one_property_not_flagged():
    from ddharmon.harmonization.leanb import assemble_kinds, prepare_kinds

    rec = _qualify_rec("q#g0", "milk by fat content", "fat", ["whole", "skim"])
    prompts = prepare_kinds([rec])
    assemble_kinds(prompts, {prompts[0].id: {"kind": "values_of_one_property", "rationale": "one attribute"}}, [rec])
    assert rec.coherence_kind == "values_of_one_property"
    assert rec.incoherent is False
    assert rec.transforms[0].needs_review is False


def test_r2_positional_qualify_never_flagged():
    # A positional/repeating-measure qualify group is skipped by prepare_kinds (no prompt built) → never
    # flagged. Reproduces the exclude_positional guard the distinct-kinds rule was validated with.
    from ddharmon.harmonization.leanb import _is_positional, prepare_kinds

    rec = _qualify_rec("p#g0", "systolic reading, trials", "trial", ["1", "2", "3", "4"])
    assert _is_positional(rec.coherence_distinct_values) is True
    assert prepare_kinds([rec]) == []
    assert rec.incoherent is False
    assert rec.transforms[0].needs_review is False  # advisory -> nothing flagged for re-adjudication


def test_assemble_coherence_coherent_false_nonsplit_not_hard_flagged():
    # coherent=False with a qualify/single verdict is a SECONDARY outlier signal, not a hard flag.
    rec = LeanBRecord(cluster_id="c1", verdict="novel", route="gencde_residual", group_id="c1#g0")
    pr = _coh_prompt("c1#g0", periphery_fids=["C:a", "C:b"])
    resp = {
        pr.id: {
            "summary": "provider seen for [condition]",
            "coherent": False,
            "outliers": [2],
            "granularity": {"verdict": "qualify", "axis": "condition", "distinct_values": ["adhd", "asthma"]},
        }
    }

    assemble_coherence([pr], resp, [rec])

    assert rec.coherent is False
    assert rec.coherence_outliers == ["C:b"]  # secondary outlier signal recorded
    assert rec.incoherent is False  # not split -> not a hard flag


def test_assemble_coherence_single_is_not_flagged():
    rec = LeanBRecord(
        cluster_id="c1",
        verdict="adopt",
        route="assigned",
        group_id="c1#g0",
        transforms=[TransformSpec(source_variable="C:age", target_cde_id="AGE", kind=TransformKind.CATEGORICAL)],
    )
    pr = _coh_prompt("c1#g0", periphery_fids=["C:x"])
    resp = {
        pr.id: {
            "summary": "age in years",
            "coherent": True,
            "outliers": [],
            "granularity": {"verdict": "single", "axis": None, "distinct_values": []},
        }
    }

    assemble_coherence([pr], resp, [rec])

    assert rec.coherent is True
    assert rec.coherence_verdict == "single"
    assert rec.coherence_axis == ""  # null axis normalized to ""
    assert rec.incoherent is False
    assert rec.transforms[0].needs_review is False  # coherent -> nothing flagged


def test_assemble_coherence_tolerates_missing_and_unparseable():
    rec = LeanBRecord(cluster_id="c1", verdict="adopt", route="assigned", group_id="c1#g0")
    pr = _coh_prompt("c1#g0", periphery_fids=["C:x"])

    assemble_coherence([pr], {}, [rec])  # no response for this id
    assert rec.incoherent is False and rec.coherence_verdict == ""  # defaults preserved

    assemble_coherence([pr], {pr.id: "not json at all"}, [rec])
    assert rec.incoherent is False and rec.coherence_verdict == ""


# ── 08-04: the judge is SPLIT into a verdict pass (early, pre-assign) + a propagation pass (late) ──
#
# Gate 1 of the staged review flow pauses at the shipped ``classify=None`` early return — BEFORE
# ``assemble_leanb`` exists to build a LeanBRecord. So the verdict pass must run on the pre-record
# concept-group shape (:class:`ConceptGroup`) and must stamp ``incoherent`` itself; only the genuinely
# post-assign half (``needs_review`` on the recodes / GenCDE) may be deferred.


def _coh_prompt_ids(group_id, cluster_id, periphery_fids):
    """A coherence prompt whose context carries BOTH identifiers, as prepare_coherence emits."""
    return PromptRecord(
        id=f"leanb:coherence:{group_id or cluster_id}",
        system_prompt=SYS_COHERENCE,
        user_prompt="…",
        schema="{}",
        model_tag="test",
        context={"group_id": group_id, "cluster_id": cluster_id, "periphery_fids": periphery_fids},
    )


def _split_payload(pr, *, verdict="split"):
    return {
        pr.id: {
            "summary": "Blood pressure measurement",
            "coherent": False,
            "outliers": [1],
            "granularity": {"verdict": verdict, "axis": "measurand", "distinct_values": ["systolic", "diastolic"]},
        }
    }


def test_assemble_coherence_verdicts_stamps_incoherent_without_propagating():
    """The VERDICT pass owns ``incoherent`` — Gate 1 renders before assign and must show that flag.

    Propagation (``needs_review`` on recodes / GenCDE) is the only genuinely post-assign half and must
    NOT have happened yet.
    """
    from ddharmon.harmonization.leanb import assemble_coherence_verdicts

    rec = LeanBRecord(
        cluster_id="c1",
        verdict="adopt",
        route="assigned",
        group_id="c1#g0",
        transforms=[TransformSpec(source_variable="C:sys", target_cde_id="BP", kind=TransformKind.CATEGORICAL)],
        gencde=GenCDE(gencde_id="GENCDE:c1#g0"),
    )
    pr = _coh_prompt_ids("c1#g0", "c1", ["C:pulse"])

    assemble_coherence_verdicts([pr], _split_payload(pr), [rec])

    assert rec.incoherent is True  # stamped by the VERDICT pass
    assert rec.coherence_verdict == "split"
    assert rec.coherence_axis == "measurand"
    assert rec.coherence_outliers == ["C:pulse"]
    assert rec.transforms[0].needs_review is False  # propagation has NOT run
    assert rec.gencde.needs_review is False
    assert rec.verdict == "adopt" and rec.route == "assigned"  # flag-not-gate


def test_assemble_coherence_verdicts_folds_on_cluster_id_when_group_id_is_absent():
    """No silent drop: today's ``by_group`` fold discards an item carrying only ``cluster_id``."""
    from ddharmon.harmonization.leanb import ConceptGroup, assemble_coherence_verdicts

    grp = ConceptGroup(cluster_id="c9", group_id="", n_members=8)
    pr = _coh_prompt_ids("", "c9", ["C:x"])

    assemble_coherence_verdicts([pr], _split_payload(pr), [grp])

    assert grp.coherence_verdict == "split"
    assert grp.incoherent is True


def test_assemble_coherence_verdicts_accepts_a_pre_record_concept_group():
    """The verdict pass is typed structurally, so it works on the pre-record Gate 1 row shape."""
    from ddharmon.harmonization.leanb import ConceptGroup, assemble_coherence_verdicts

    grp = ConceptGroup(cluster_id="c1", group_id="c1#g0", n_members=9)
    pr = _coh_prompt_ids("c1#g0", "c1", ["C:pulse"])

    assemble_coherence_verdicts([pr], _split_payload(pr, verdict="qualify"), [grp])

    assert grp.coherence_verdict == "qualify"
    assert grp.coherence_distinct_values == ["systolic", "diastolic"]
    assert grp.incoherent is False  # qualify is ADVISORY under R0


def test_prepare_coherence_stamps_matrix_suspect_on_a_concept_group(hf):
    """The $0 §29.1 matrix pre-filter works on the pre-record shape too (no LeanBRecord required)."""
    from ddharmon.harmonization.leanb import ConceptGroup
    from ddharmon.harmonization.leanb import prepare_coherence as _prep

    conditions = ("adhd", "depression", "asthma", "diabetes", "ptsd", "epilepsy")
    specs = [(f"cond_{c}", f"still seeing a provider for {c}") for c in conditions]
    embedded, embeddings, field_refs = _world(hf, specs, _CORE_OUT_VECS[:6])
    fids = [f"{r.dictionary_name}:{r.variable_name}" for r in field_refs]
    grp = ConceptGroup(cluster_id="m1", group_id="m1#g0", member_variable_names=fids, n_members=len(fids))

    prompts = _prep([grp], embedded, embeddings, field_refs)

    assert len(prompts) == 1
    assert prompts[0].context["group_id"] == "m1#g0" and prompts[0].context["cluster_id"] == "m1"
    assert grp.matrix_suspect is True


def test_sub_threshold_group_is_explicitly_unjudged_not_coherent(hf):
    """A group below COHERENCE_MIN_MEMBERS gets no prompt and no verdict — never recorded as ``single``."""
    from ddharmon.harmonization.leanb import COHERENCE_MIN_MEMBERS, ConceptGroup, assemble_coherence_verdicts
    from ddharmon.harmonization.leanb import prepare_coherence as _prep

    specs = [(f"v{i}", f"desc {i}") for i in range(3)]
    embedded, embeddings, field_refs = _world(hf, specs, _CORE_OUT_VECS[:3])
    fids = [f"{r.dictionary_name}:{r.variable_name}" for r in field_refs]
    grp = ConceptGroup(cluster_id="m1", group_id="m1#g0", member_variable_names=fids, n_members=len(fids))
    assert len(fids) < COHERENCE_MIN_MEMBERS

    prompts = _prep([grp], embedded, embeddings, field_refs)
    assemble_coherence_verdicts(prompts, {}, [grp])

    assert prompts == []
    assert grp.coherence_verdict == ""  # explicitly unjudged
    assert grp.coherence_verdict != "single"
    assert grp.incoherent is False


@pytest.mark.parametrize("bad", [None, "not json at all", 42, [], {"error": "judge timed out"}])
def test_judge_failure_leaves_the_group_unjudged_and_never_single(bad):
    """T-08-17 / T-08-18: a missing, malformed or error response degrades to unjudged, never to ``single``."""
    from ddharmon.harmonization.leanb import ConceptGroup, assemble_coherence_verdicts

    grp = ConceptGroup(cluster_id="c1", group_id="c1#g0", n_members=8)
    pr = _coh_prompt_ids("c1#g0", "c1", ["C:x"])

    assemble_coherence_verdicts([pr], {pr.id: bad}, [grp])

    assert grp.coherence_verdict == ""
    assert grp.coherence_verdict != "single"
    assert grp.incoherent is False


def test_transfer_coherence_verdicts_moves_group_flags_onto_records():
    """The early verdicts must reach the real records once assemble_leanb has built them."""
    from ddharmon.harmonization.leanb import ConceptGroup, transfer_coherence_verdicts

    grp = ConceptGroup(
        cluster_id="c1",
        group_id="c1#g0",
        n_members=8,
        coherent=False,
        coherence_verdict="split",
        coherence_summary="BP measurement",
        coherence_axis="measurand",
        coherence_distinct_values=["systolic", "diastolic"],
        coherence_outliers=["C:pulse"],
        incoherent=True,
        matrix_suspect=True,
    )
    rec = LeanBRecord(
        cluster_id="c1",
        verdict="adopt",
        route="assigned",
        group_id="c1#g0",
        transforms=[TransformSpec(source_variable="C:sys", target_cde_id="BP", kind=TransformKind.CATEGORICAL)],
    )

    transfer_coherence_verdicts([grp], [rec])

    assert rec.incoherent is True and rec.coherent is False
    assert rec.coherence_verdict == "split" and rec.coherence_axis == "measurand"
    assert rec.coherence_distinct_values == ["systolic", "diastolic"]
    assert rec.coherence_outliers == ["C:pulse"]
    assert rec.coherence_summary == "BP measurement"
    assert rec.matrix_suspect is True
    assert rec.transforms[0].needs_review is False  # transfer does not propagate
    assert rec.verdict == "adopt" and rec.route == "assigned"  # flag-not-gate


def test_transfer_coherence_verdicts_matches_on_cluster_id_fallback():
    from ddharmon.harmonization.leanb import ConceptGroup, transfer_coherence_verdicts

    grp = ConceptGroup(cluster_id="c7", group_id="", coherence_verdict="split", incoherent=True)
    rec = LeanBRecord(cluster_id="c7", verdict="novel", route="gencde_residual", group_id="")

    transfer_coherence_verdicts([grp], [rec])

    assert rec.coherence_verdict == "split" and rec.incoherent is True


def test_propagate_coherence_review_is_idempotent():
    """A resumed run may replay propagation — applying it twice must equal applying it once."""
    from ddharmon.harmonization.leanb import propagate_coherence_review

    def build():
        return LeanBRecord(
            cluster_id="c1",
            verdict="adopt",
            route="assigned",
            group_id="c1#g0",
            incoherent=True,
            transforms=[TransformSpec(source_variable="C:sys", target_cde_id="BP", kind=TransformKind.CATEGORICAL)],
            gencde=GenCDE(gencde_id="GENCDE:c1#g0"),
        )

    once, twice = build(), build()
    propagate_coherence_review([once])
    propagate_coherence_review([twice])
    propagate_coherence_review([twice])

    assert once.transforms[0].needs_review is True and once.gencde.needs_review is True
    assert (twice.transforms[0].needs_review, twice.gencde.needs_review) == (
        once.transforms[0].needs_review,
        once.gencde.needs_review,
    )
    assert twice.verdict == once.verdict and twice.route == once.route


def test_propagate_coherence_review_leaves_coherent_records_untouched():
    from ddharmon.harmonization.leanb import propagate_coherence_review

    rec = LeanBRecord(
        cluster_id="c1",
        verdict="adopt",
        route="assigned",
        group_id="c1#g0",
        coherence_verdict="single",
        transforms=[TransformSpec(source_variable="C:age", target_cde_id="AGE", kind=TransformKind.CATEGORICAL)],
        gencde=GenCDE(gencde_id="GENCDE:c1#g0"),
    )

    propagate_coherence_review([rec])

    assert rec.transforms[0].needs_review is False and rec.gencde.needs_review is False


def test_assemble_coherence_wrapper_equals_verdicts_then_propagation():
    """The shipped entry point survives as a thin wrapper so notebooks and the Batch driver keep working."""
    from ddharmon.harmonization.leanb import assemble_coherence_verdicts, propagate_coherence_review

    def build():
        return LeanBRecord(
            cluster_id="c1",
            verdict="adopt",
            route="assigned",
            group_id="c1#g0",
            transforms=[TransformSpec(source_variable="C:sys", target_cde_id="BP", kind=TransformKind.CATEGORICAL)],
            gencde=GenCDE(gencde_id="GENCDE:c1#g0"),
        )

    wrapped, manual = build(), build()
    pr = _coh_prompt_ids("c1#g0", "c1", ["C:pulse"])

    assemble_coherence([pr], _split_payload(pr), [wrapped])
    assemble_coherence_verdicts([pr], _split_payload(pr), [manual])
    propagate_coherence_review([manual])

    assert (wrapped.coherence_verdict, wrapped.incoherent) == (manual.coherence_verdict, manual.incoherent)
    assert wrapped.transforms[0].needs_review is manual.transforms[0].needs_review is True
    assert wrapped.gencde.needs_review is manual.gencde.needs_review is True


def test_assemble_kinds_folds_on_cluster_id_when_group_id_is_absent():
    """R2's fold had the same no-silent-drop defect as the coherence fold.

    ``prepare_kinds`` already builds the prompt id as ``leanb:kinds:{group_id or cluster_id}``, so a record
    carrying only ``cluster_id`` got a prompt built AND PAID FOR, then had its verdict discarded by a fold
    keyed on ``group_id`` alone.
    """
    from ddharmon.harmonization.leanb import assemble_kinds, prepare_kinds

    rec = LeanBRecord(
        cluster_id="c7",
        verdict="adopt",
        route="assigned",
        group_id="",  # cluster-granularity record: no group id
        coherence_verdict="qualify",
        coherence_summary="ECG measures + completion status",
        coherence_axis="kind",
        coherence_distinct_values=["QRS interval", "ECG done"],
        transforms=[TransformSpec(source_variable="C:x", target_cde_id="CDE", kind=TransformKind.CATEGORICAL)],
    )
    prompts = prepare_kinds([rec])
    assert [p.id for p in prompts] == ["leanb:kinds:c7"]

    assemble_kinds(prompts, {prompts[0].id: {"kind": "distinct_kinds", "rationale": "a duration vs a status"}}, [rec])

    assert rec.coherence_kind == "distinct_kinds"
    assert rec.incoherent is True
    assert rec.verdict == "adopt" and rec.route == "assigned"  # flag-not-gate
