"""Tests for M2 cross-record redundancy merge (candidate generation, union-find, assemble).

The deterministic core (centroid ∪ signature candidates, union-find, merged-record construction) is tested
without an LLM; the adjudication verdicts are supplied as mock responses.
"""

from __future__ import annotations

import numpy as np

from ddharmon.harmonization.merge import (
    _merge_group,
    _parse_merge,
    _reps,
    assemble_merge,
    merge_candidate_pairs,
    union_find,
)
from ddharmon.harmonization.models import LeanBRecord
from ddharmon.harmonization.pipeline import PromptRecord
from ddharmon.models.cluster import FieldReference


def _refs(names: list[str]) -> list[FieldReference]:
    return [FieldReference("C", n, "") for n in names]


def _row_of(refs):
    return {(r.dictionary_name, r.variable_name): i for i, r in enumerate(refs)}


def _rec(gid: str, members: list[str], *, verdict: str = "refine") -> LeanBRecord:
    return LeanBRecord(
        cluster_id="c",
        verdict=verdict,
        route="assigned",
        group_id=gid,
        member_variable_names=members,
        n_members=len(members),
        cohorts=sorted({m.split(":")[0] for m in members}),
    )


def _merge_prompt(a: str, b: str) -> PromptRecord:
    return PromptRecord(
        id=f"leanb:merge:{a}|{b}", system_prompt="", user_prompt="", schema="", model_tag="x", context={"a": a, "b": b}
    )


class TestCandidatePairs:
    def test_centroid_pair_and_orthogonal_non_pair(self):
        refs = _refs(["a1", "a2", "b1", "b2", "c1", "c2"])
        emb = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)  # A,B ∥ ; C ⟂
        text_of = {(r.dictionary_name, r.variable_name): "" for r in refs}
        recs = [_rec("A", ["C:a1", "C:a2"]), _rec("B", ["C:b1", "C:b2"]), _rec("C", ["C:c1", "C:c2"])]
        pairs = merge_candidate_pairs(recs, emb, _row_of(refs), text_of, tau=0.85)
        assert ("A", "B") in pairs and "centroid" in pairs[("A", "B")]["via"]
        assert ("A", "C") not in pairs and ("B", "C") not in pairs

    def test_signature_pair_even_when_centroids_far(self):
        refs = _refs(["a1", "a2", "a3", "b1", "b2", "b3"])
        emb = np.array([[1, 0]] * 3 + [[0, 1]] * 3, dtype=np.float32)  # A far from B
        text_of = {
            ("C", "a1"): "Medication 1",
            ("C", "a2"): "Medication 2",
            ("C", "a3"): "Medication 3",
            ("C", "b1"): "Medication 20",
            ("C", "b2"): "Medication 21",
            ("C", "b3"): "Medication 22",
        }
        recs = [_rec("A", ["C:a1", "C:a2", "C:a3"]), _rec("B", ["C:b1", "C:b2", "C:b3"])]
        pairs = merge_candidate_pairs(recs, emb, _row_of(refs), text_of, tau=0.85)
        assert ("A", "B") in pairs and "signature" in pairs[("A", "B")]["via"]  # reunited below τ


def test_reps_returns_closest_members_as_text():
    refs = _refs(["m1", "m2", "m3"])
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)  # m1,m2 at centroid; m3 far
    text_of = {("C", "m1"): "apple", ("C", "m2"): "banana", ("C", "m3"): "outlier"}
    out = _reps(_rec("A", ["C:m1", "C:m2", "C:m3"]), emb, _row_of(refs), text_of, k=2)
    assert out == ["apple", "banana"]  # 2 closest to the centroid (tie broken deterministically by id)


def test_union_find_transitive_closure():
    groups = union_find([("A", "B"), ("B", "C")], ["A", "B", "C", "D"])
    assert sorted(len(g) for g in groups) == [1, 3]
    big = next(g for g in groups if len(g) == 3)
    assert set(big) == {"A", "B", "C"}


def test_parse_merge_tolerant():
    assert _parse_merge({"merge": "true"}) and _parse_merge({"merge": True}) and _parse_merge({"merge": "yes"})
    assert not _parse_merge({"merge": "false"}) and not _parse_merge({"merge": "no"})
    assert not _parse_merge(None) and not _parse_merge({"reason": "no verdict key"})


class TestAssembleMerge:
    def test_confirmed_pair_merges_with_adopt_primary(self):
        recs = [_rec("A", ["C:a1", "C:a2"], verdict="adopt"), _rec("B", ["C:b1"], verdict="novel"), _rec("C", ["C:c1"])]
        out = assemble_merge(recs, [_merge_prompt("A", "B")], {"leanb:merge:A|B": {"merge": "true"}})
        assert len(out) == 2  # {A,B} merged + C
        merged = next(r for r in out if set(r.member_variable_names) == {"C:a1", "C:a2", "C:b1"})
        assert merged.verdict == "adopt"  # adopt primary beats novel
        assert merged.n_members == 3 and merged.raw["merged_from"] == ["A", "B"]
        assert "merged 2 records" in merged.rationale

    def test_no_confirmation_passes_through(self):
        recs = [_rec("A", ["C:a1"]), _rec("B", ["C:b1"])]
        out = assemble_merge(recs, [_merge_prompt("A", "B")], {"leanb:merge:A|B": {"merge": "false"}})
        assert len(out) == 2 and {r.group_id for r in out} == {"A", "B"}


def test_merge_group_recomputes_cohorts_and_picks_priority_verdict():
    merged = _merge_group([_rec("A", ["X:a1", "X:a2"], verdict="novel"), _rec("B", ["Y:b1"], verdict="refine")])
    assert merged.verdict == "refine"  # refine beats novel
    assert merged.cohorts == ["X", "Y"] and merged.cross_cohort is True
    assert merged.member_variable_names == ["X:a1", "X:a2", "Y:b1"]
