"""Tests for the L2 frozen clustering substrate (content-addressed ids + persist/reload partition)."""

from __future__ import annotations

from ddharmon.harmonization.substrate import (
    ClusteringSubstrate,
    build_substrate,
    cluster_content_id,
    clusters_from_substrate,
    content_token,
    load_substrate,
    save_substrate,
)
from ddharmon.models.cluster import FieldCluster, FieldReference


def _ref(d: str, v: str) -> FieldReference:
    return FieldReference(d, v, f"{v} description")


def _cluster(cid: int, keys: list[tuple[str, str]]) -> FieldCluster:
    return FieldCluster(cluster_id=cid, label="t", members=[_ref(d, v) for d, v in keys])


class TestContentIds:
    def test_cluster_content_id_is_order_independent(self):
        a = cluster_content_id([("A", "x"), ("B", "y")])
        b = cluster_content_id([("B", "y"), ("A", "x")])
        assert a == b and a.startswith("c")

    def test_cluster_content_id_is_membership_sensitive(self):
        base = cluster_content_id([("A", "x"), ("B", "y")])
        assert cluster_content_id([("A", "x")]) != base  # dropped a member
        assert cluster_content_id([("A", "x"), ("B", "y"), ("C", "z")]) != base  # added one
        assert cluster_content_id([("A", "x"), ("B", "Y")]) != base  # different variable

    def test_content_token_is_ordered_and_stable(self):
        assert content_token("CDE1", "1=Yes|2=No") == content_token("CDE1", "1=Yes|2=No")
        assert content_token("CDE1", "1=Yes|2=No") != content_token("1=Yes|2=No", "CDE1")  # order matters
        assert content_token("CDE1", "a") != content_token("CDE2", "a")


class TestSubstrateRoundTrip:
    def _substrate(self) -> ClusteringSubstrate:
        clusters = [_cluster(0, [("A", "x"), ("B", "y")]), _cluster(1, [("A", "z")])]
        outlier = _cluster(-1, [("B", "w")])
        return build_substrate(clusters, min_cluster_size=15, outlier=outlier, n_fields=4)

    def test_build_extracts_partition(self):
        sub = self._substrate()
        assert sub.clusters == [[("A", "x"), ("B", "y")], [("A", "z")]]
        assert sub.outlier == [("B", "w")]
        assert sub.min_cluster_size == 15 and sub.n_fields == 4 and sub.n_clusters == 2

    def test_save_load_round_trip(self, tmp_path):
        sub = self._substrate()
        p = save_substrate(sub, tmp_path / "sub.json")
        back = load_substrate(p)
        assert back.clusters == sub.clusters  # tuples survive the JSON round trip
        assert back.outlier == sub.outlier
        assert back.min_cluster_size == sub.min_cluster_size and back.n_fields == sub.n_fields
        assert back.substrate_id == sub.substrate_id  # content id stable across the round trip

    def test_substrate_id_is_sensitive_to_partition(self):
        a = build_substrate([_cluster(0, [("A", "x")]), _cluster(1, [("B", "y")])], min_cluster_size=15)
        b = build_substrate([_cluster(0, [("A", "x"), ("B", "y")])], min_cluster_size=15)  # merged
        assert a.substrate_id != b.substrate_id  # same fields, different grouping -> different id


class TestClustersFromSubstrate:
    def test_reconstructs_members_and_coverage(self):
        sub = ClusteringSubstrate(clusters=[[("A", "x"), ("B", "y")], [("A", "z")]], min_cluster_size=15)
        refs = [_ref("A", "x"), _ref("B", "y"), _ref("A", "z")]
        clusters = clusters_from_substrate(sub, refs)
        assert len(clusters) == 2
        assert [(m.dictionary_name, m.variable_name) for m in clusters[0].members] == [("A", "x"), ("B", "y")]
        assert clusters[0].cohort_coverage == {"A": 1, "B": 1}
        assert clusters[1].cohort_coverage == {"A": 1}

    def test_drops_missing_members_and_empty_clusters(self):
        sub = ClusteringSubstrate(clusters=[[("A", "x"), ("GONE", "g")], [("GONE", "h")]], min_cluster_size=15)
        clusters = clusters_from_substrate(sub, [_ref("A", "x")])
        assert len(clusters) == 1  # the all-missing cluster is skipped
        assert [(m.dictionary_name, m.variable_name) for m in clusters[0].members] == [("A", "x")]

    def test_reconstructed_cluster_has_same_content_id(self):
        keys = [("A", "x"), ("B", "y")]
        sub = ClusteringSubstrate(clusters=[keys], min_cluster_size=15)
        [cluster] = clusters_from_substrate(sub, [_ref("A", "x"), _ref("B", "y")])
        member_keys = [(m.dictionary_name, m.variable_name) for m in cluster.members]
        assert cluster_content_id(member_keys) == cluster_content_id(keys)  # the property that makes the cache hit
