"""Tests for BERTopic-based topic modeling.

Unit tests (no bertopic required):
- extract_topic_clusters: synthetic topic IDs -> FieldCluster conversion
- TopicModelResult: dataclass construction and properties
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from ddharmon.clustering.topic_engine import extract_topic_clusters, recluster_residual
from ddharmon.models.cluster import FieldCluster, FieldReference, TopicModelResult

# ── helpers ─────────────────────────────────────────────────────


def _make_refs(n: int, cohorts: list[str]) -> list[FieldReference]:
    """Create synthetic FieldReferences cycling through cohorts."""
    return [
        FieldReference(
            dictionary_name=cohorts[i % len(cohorts)],
            variable_name=f"var_{i}",
            description=f"Description for variable {i}",
        )
        for i in range(n)
    ]


# ── extract_topic_clusters ──────────────────────────────────────


def test_extract_topic_clusters_basic():
    """Basic topic extraction with 3 topics and no outliers."""
    refs = _make_refs(9, ["A", "B", "C"])
    topics = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    clusters, outlier = extract_topic_clusters(topics, refs, ["A", "B", "C"])

    assert outlier is None
    assert len(clusters) == 3
    assert all(isinstance(c, FieldCluster) for c in clusters)
    assert all(len(c.members) == 3 for c in clusters)


def test_extract_topic_clusters_with_outliers():
    """Topic -1 should be separated into outlier_cluster."""
    refs = _make_refs(6, ["A", "B"])
    topics = [0, 0, 1, 1, -1, -1]

    clusters, outlier = extract_topic_clusters(topics, refs, ["A", "B"])

    assert len(clusters) == 2
    assert outlier is not None
    assert outlier.cluster_id == -1
    assert len(outlier.members) == 2


def test_extract_topic_clusters_cohort_coverage():
    """Verify cohort coverage and missing_cohorts tracking."""
    refs = [
        FieldReference("A", "v1", "desc1"),
        FieldReference("A", "v2", "desc2"),
        FieldReference("B", "v3", "desc3"),
    ]
    topics = [0, 0, 0]

    clusters, _ = extract_topic_clusters(topics, refs, ["A", "B", "C"])

    assert len(clusters) == 1
    c = clusters[0]
    assert c.cohort_coverage == {"A": 2, "B": 1}
    assert c.missing_cohorts == ["C"]


def test_extract_topic_clusters_all_outliers():
    """All fields assigned to outlier topic."""
    refs = _make_refs(5, ["A"])
    topics = [-1, -1, -1, -1, -1]

    clusters, outlier = extract_topic_clusters(topics, refs, ["A"])

    assert len(clusters) == 0
    assert outlier is not None
    assert len(outlier.members) == 5


def test_extract_topic_clusters_single_topic():
    """Single topic with no outliers."""
    refs = _make_refs(10, ["A", "B"])
    topics = [0] * 10

    clusters, outlier = extract_topic_clusters(topics, refs, ["A", "B"])

    assert len(clusters) == 1
    assert outlier is None
    assert len(clusters[0].members) == 10


# ── recluster_residual (tail re-clustering) ─────────────────────


def _blob_embeddings(seed: int = 0) -> np.ndarray:
    """60 vectors: rows 0-14 = head (diffuse), rows 15-59 = residual in 3 tight, far-apart blobs."""
    rng = np.random.RandomState(seed)
    head = rng.normal(0.0, 1.0, size=(15, 16)).astype(np.float32)
    blobs = []
    for k in range(3):
        center = np.zeros(16, dtype=np.float32)
        center[k] = 10.0  # orthogonal directions → cosine-separable
        blobs.append(center + rng.normal(0.0, 0.05, size=(15, 16)).astype(np.float32))
    return np.vstack([head, *blobs]).astype(np.float32)


def test_recluster_residual_empty():
    """No residual indices → empty result, no UMAP."""
    refs = _make_refs(5, ["A"])
    clusters, outlier = recluster_residual(np.zeros((5, 8), dtype=np.float32), refs, [])
    assert clusters == [] and outlier is None


def test_recluster_residual_small_residual_single_group():
    """Below the clustering threshold → one group over EXACTLY the residual (head excluded), no UMAP."""
    refs = _make_refs(20, ["A", "B"])
    res_idx = [10, 12, 14, 16]  # 4 ≤ max(min_cluster_size, umap_n_neighbors)
    clusters, outlier = recluster_residual(np.zeros((20, 8), dtype=np.float32), refs, res_idx)
    assert outlier is None and len(clusters) == 1
    assert {m.variable_name for m in clusters[0].members} == {"var_10", "var_12", "var_14", "var_16"}


def test_recluster_residual_isolates_tail_and_conserves_members():
    """Full UMAP+HDBSCAN path: clusters ONLY the residual (head excluded), every residual field accounted
    for, and 3 far-apart blobs separate into multiple clusters."""
    emb = _blob_embeddings()
    refs = _make_refs(60, ["A", "B", "C"])
    res_idx = list(range(15, 60))  # the 45 residual rows; head rows 0-14 must be excluded
    clusters, outlier = recluster_residual(emb, refs, res_idx, min_cluster_size=5, umap_n_neighbors=10)

    placed = [m for c in clusters for m in c.members] + (outlier.members if outlier else [])
    assert len(placed) == 45  # every residual field placed, none dropped
    assert {m.variable_name for m in placed} == {f"var_{i}" for i in res_idx}  # residual-only, no head
    assert len(clusters) >= 2  # the tail separates into distinct concepts


# ── TopicModelResult ────────────────────────────────────────────


def test_topic_model_result_n_topics():
    """n_topics property returns cluster count."""
    refs = _make_refs(3, ["A"])
    clusters = [
        FieldCluster(cluster_id=0, label="a", members=refs[:2], cohort_coverage={"A": 2}, missing_cohorts=[]),
        FieldCluster(cluster_id=1, label="b", members=refs[2:], cohort_coverage={"A": 1}, missing_cohorts=[]),
    ]

    result = TopicModelResult(
        model=MagicMock(),
        docs=["d1", "d2", "d3"],
        embeddings=np.zeros((3, 8)),
        field_refs=refs,
        clusters=clusters,
        outlier_cluster=None,
        all_cohort_names=["A"],
    )

    assert result.n_topics == 2
    assert result.outlier_cluster is None


def test_topic_model_result_topic_info_delegates():
    """topic_info property delegates to model.get_topic_info()."""
    mock_model = MagicMock()
    mock_model.get_topic_info.return_value = "fake_df"

    result = TopicModelResult(
        model=mock_model,
        docs=[],
        embeddings=np.zeros((0, 8)),
        field_refs=[],
        clusters=[],
        outlier_cluster=None,
    )

    assert result.topic_info == "fake_df"
    mock_model.get_topic_info.assert_called_once()


def test_topic_model_result_topic_info_graceful_failure():
    """topic_info returns None if model raises."""
    mock_model = MagicMock()
    mock_model.get_topic_info.side_effect = RuntimeError("no model")

    result = TopicModelResult(
        model=mock_model,
        docs=[],
        embeddings=np.zeros((0, 8)),
        field_refs=[],
        clusters=[],
        outlier_cluster=None,
    )

    assert result.topic_info is None
