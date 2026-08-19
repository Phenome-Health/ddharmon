"""Selective transform-spec export — mark-for-export concept filtering.

Covers the headless selection primitives (:mod:`ddharmon.harmonization.selection`) and the ``select=``
plumbing on :func:`export_transform_review`.
"""

from __future__ import annotations

import csv as _csv

import numpy as np

from ddharmon.harmonization import (
    concepts_matching,
    export_transform_review,
    list_export_concepts,
    select_records,
)
from ddharmon.harmonization.models import LeanBRecord, TransformKind, TransformSpec


def _rec(
    group_id: str,
    cluster_id: str,
    *,
    verdict: str = "refine",
    route: str = "assigned",
    cohorts: list[str] | None = None,
    cross_cohort: bool = False,
    n_specs: int = 1,
    identity: int = 0,
    concept: str = "",
) -> LeanBRecord:
    """A LeanBRecord with ``n_specs`` categorical (non-identity) recodes + ``identity`` no-op specs."""
    r = LeanBRecord(
        cluster_id=cluster_id,
        verdict=verdict,
        route=route,
        group_id=group_id,
        concept=concept or group_id,
        cde_id="CDE1",
        cohorts=cohorts or ["CohortA"],
        cross_cohort=cross_cohort,
        n_members=len(cohorts or ["CohortA"]),
    )
    r.transforms = [
        TransformSpec(source_variable=f"CohortA:v{i}", target_cde_id="CDE1", kind=TransformKind.CATEGORICAL)
        for i in range(n_specs)
    ] + [
        TransformSpec(source_variable=f"CohortA:id{i}", target_cde_id="CDE1", kind=TransformKind.IDENTITY)
        for i in range(identity)
    ]
    return r


class TestListExportConcepts:
    def test_opt_in_default_marks_nothing(self):
        concepts = list_export_concepts([_rec("c0#g0", "c0"), _rec("c1#g0", "c1")])
        assert len(concepts) == 2
        assert all(c.export is False for c in concepts)  # lean by default

    def test_opt_out_default_marks_all(self):
        concepts = list_export_concepts([_rec("c0#g0", "c0")], default_export=True)
        assert concepts[0].export is True

    def test_only_with_specs_drops_specless(self):
        recs = [
            _rec("c0#g0", "c0", n_specs=2),
            _rec("c1#g0", "c1", n_specs=0, verdict="novel", route="gencde_residual"),
        ]
        # default drops the spec-less novel; opt-out keeps everything
        assert {c.group_id for c in list_export_concepts(recs)} == {"c0#g0"}
        assert {c.group_id for c in list_export_concepts(recs, only_with_specs=False)} == {"c0#g0", "c1#g0"}

    def test_n_specs_counts_non_identity_only(self):
        c = list_export_concepts([_rec("c0#g0", "c0", n_specs=3, identity=2)])[0]
        assert c.n_specs == 3  # the 2 identity no-ops are not export payload

    def test_eligible_gates_marking(self):
        recs = [_rec("c0#g0", "c0"), _rec("c1#g0", "c1")]
        accepted = {"c0#g0"}
        concepts = list_export_concepts(recs, default_export=True, eligible=lambda r: r.group_id in accepted)
        by_id = {c.group_id: c for c in concepts}
        assert by_id["c0#g0"].eligible is True and by_id["c0#g0"].export is True
        # ineligible: cannot be exported even under opt-out
        assert by_id["c1#g0"].eligible is False and by_id["c1#g0"].export is False


class TestSelectRecords:
    def setup_method(self):
        self.recs = [_rec("c0#g0", "c0"), _rec("c0#g1", "c0"), _rec("c1#g0", "c1")]

    def test_none_returns_all(self):
        assert select_records(self.recs, None) == self.recs

    def test_predicate(self):
        picked = select_records(self.recs, lambda r: r.cluster_id == "c1")
        assert [r.group_id for r in picked] == ["c1#g0"]

    def test_group_id_set(self):
        picked = select_records(self.recs, {"c0#g1"})
        assert [r.group_id for r in picked] == ["c0#g1"]

    def test_cluster_id_selects_all_its_groups(self):
        # a coarse *cluster* id transparently grabs every concept-group inside it
        picked = select_records(self.recs, {"c0"})
        assert [r.group_id for r in picked] == ["c0#g0", "c0#g1"]

    def test_dict_mapping(self):
        picked = select_records(self.recs, {"c0#g0": True, "c0#g1": False, "c1#g0": True})
        assert [r.group_id for r in picked] == ["c0#g0", "c1#g0"]

    def test_export_concept_list_uses_export_flag(self):
        concepts = list_export_concepts(self.recs)  # all export=False
        concepts[1].export = True  # user marks the second concept
        picked = select_records(self.recs, concepts)
        assert [r.group_id for r in picked] == ["c0#g1"]

    def test_empty_selection_picks_nothing(self):
        assert select_records(self.recs, set()) == []
        assert select_records(self.recs, []) == []


class TestConceptsMatching:
    def setup_method(self):
        self.recs = [
            _rec("c0#g0", "c0", verdict="refine", cohorts=["UKBB", "AoU"], cross_cohort=True),
            _rec("c0#g1", "c0", verdict="adopt", cohorts=["UKBB"]),
            _rec("c1#g0", "c1", verdict="novel", route="gencde_residual", n_specs=0),
        ]

    def test_by_verdict(self):
        assert concepts_matching(self.recs, verdict="adopt") == {"c0#g1"}

    def test_by_cohort_membership(self):
        assert concepts_matching(self.recs, cohort="AoU") == {"c0#g0"}

    def test_by_cluster(self):
        assert concepts_matching(self.recs, cluster_id="c0") == {"c0#g0", "c0#g1"}

    def test_by_cross_cohort(self):
        assert concepts_matching(self.recs, cross_cohort=True) == {"c0#g0"}

    def test_with_specs_excludes_specless(self):
        # the novel record has no spec, so even matching its cluster returns nothing spec-bearing
        assert concepts_matching(self.recs, cluster_id="c1") == set()
        assert concepts_matching(self.recs, cluster_id="c1", with_specs=False) == {"c1#g0"}


def _export_world(hf):
    """Two 1-member refine records (distinct clusters) sharing one coded CDE — for the export-count test."""
    s1 = hf.field("smoke", "Do you smoke", encoding="1=Yes|2=No", data_type="categorical", question_text="Smoke?")
    s2 = hf.field("drink", "Do you drink", encoding="1=Yes|2=No", data_type="categorical", question_text="Drink?")
    cde = hf.field("CDE1", "Current status", field_id="TINY1", encoding="1=Yes|0=No")
    ed_a = hf.embedded_dict("CohortA", [s1, s2], sem_vecs=np.array([[1.0], [1.0]]))
    ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
    cde_fields = dict(ed_cde.dictionary.fields)

    def mk(gid: str, cluster: str, sv: str) -> LeanBRecord:
        r = LeanBRecord(
            cluster_id=cluster,
            verdict="refine",
            route="assigned",
            group_id=gid,
            cde_id="CDE1",
            cde_external_id="TINY1",
            member_variable_names=[sv],
        )
        r.transforms = [
            TransformSpec(
                source_variable=sv,
                target_cde_id="CDE1",
                kind=TransformKind.CATEGORICAL,
                code_map={"1": "1", "2": "0"},
                coverage=1.0,
            )
        ]
        return r

    return [ed_a, ed_cde], cde_fields, [mk("c0#g0", "c0", "CohortA:smoke"), mk("c1#g0", "c1", "CohortA:drink")]


class TestExportTransformReviewSelect:
    def _rows(self, path):
        with open(path, encoding="utf-8") as f:
            return list(_csv.DictReader(f))

    def test_none_exports_all(self, hf, tmp_path):
        embedded, cde_fields, recs = _export_world(hf)
        n = export_transform_review(recs, embedded, cde_fields, tmp_path / "all.csv")
        assert n == 2  # one categorical row per record

    def test_select_group_id_scopes_export(self, hf, tmp_path):
        embedded, cde_fields, recs = _export_world(hf)
        path = tmp_path / "one.csv"
        n = export_transform_review(recs, embedded, cde_fields, path, select={"c0#g0"})
        assert n == 1
        assert self._rows(path)[0]["source_id"] == "CohortA:smoke"

    def test_select_by_cluster_is_coarse(self, hf, tmp_path):
        embedded, cde_fields, recs = _export_world(hf)
        path = tmp_path / "clust.csv"
        n = export_transform_review(recs, embedded, cde_fields, path, select={"c1"})
        assert n == 1 and self._rows(path)[0]["source_id"] == "CohortA:drink"

    def test_select_empty_exports_nothing(self, hf, tmp_path):
        embedded, cde_fields, recs = _export_world(hf)
        assert export_transform_review(recs, embedded, cde_fields, tmp_path / "none.csv", select=set()) == 0

    def test_select_edited_concept_table(self, hf, tmp_path):
        embedded, cde_fields, recs = _export_world(hf)
        concepts = list_export_concepts(recs)  # opt-in: nothing marked
        next(c for c in concepts if c.group_id == "c1#g0").export = True
        n = export_transform_review(recs, embedded, cde_fields, tmp_path / "edit.csv", select=concepts)
        assert n == 1
