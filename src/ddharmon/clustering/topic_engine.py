"""BERTopic clustering orchestrator: embeddings -> topics -> labeled clusters.

Thin orchestrator that builds a BERTopic model, fits it on pre-computed
embeddings, and converts results to FieldCluster objects for cohort
coverage tracking.  All heavy lifting is delegated to BERTopic itself —
use ``result.model.visualize_*()`` for built-in interactive Plotly viz.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ddharmon.clustering.labeling import derive_cluster_label, label_clusters_llm
from ddharmon.embedding.service import EmbeddedDictionary
from ddharmon.llm.base import BaseLLMClient
from ddharmon.models.cluster import FieldCluster, FieldReference, TopicModelResult

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────


def collect_inputs(
    embedded_dicts: list[EmbeddedDictionary],
) -> tuple[list[str], NDArray[np.float32], list[FieldReference], list[str]]:
    """Flatten embedded dicts into parallel lists for BERTopic.

    Args:
        embedded_dicts: List of EmbeddedDictionary from embed_dictionary().

    Returns:
        Tuple of (docs, embeddings, field_refs, cohort_names) where:
        - docs: per-field semantic text (mirrors the embedded vector —
          question_text-preferred, _embed_variable_name-respecting) for c-TF-IDF
        - embeddings: (N, D) stacked semantic vectors
        - field_refs: FieldReference per field (for cohort coverage)
        - cohort_names: unique cohort names in order
    """
    docs: list[str] = []
    vectors: list[NDArray[np.float32]] = []
    refs: list[FieldReference] = []
    cohort_names: list[str] = []

    for ed in embedded_dicts:
        cohort = ed.dictionary.cohort_name or ed.dictionary.name
        if cohort not in cohort_names:
            cohort_names.append(cohort)

        var_names = ed.get_variable_names()
        vecs = ed.get_all_vectors()
        for i, var in enumerate(var_names):
            fld = ed.dictionary.fields[var]
            # Mirror the embedded semantic text so c-TF-IDF top terms reflect the
            # clustered concept, not the raw variable name. Respects
            # _embed_variable_name and prefers question_text; include=set() drops
            # the category suffix to keep terms field-level. (Previously
            # f"{var}: {fld.description}", which always prepended the name and
            # ignored the flag — the main source of variable-name tokens in top
            # terms, e.g. CLSA "medidecisionrule13com".)
            docs.append(fld.to_embedding_text(include=set()))
            refs.append(FieldReference(cohort, var, fld.description))
            vectors.append(vecs[i])

    return docs, np.stack(vectors), refs, cohort_names


def extract_topic_clusters(
    topics: list[int],
    field_refs: list[FieldReference],
    all_cohort_names: list[str],
) -> tuple[list[FieldCluster], FieldCluster | None]:
    """Convert BERTopic topic assignments to FieldCluster list.

    Produces the same FieldCluster format as hierarchical extract_clusters(),
    so downstream code (cohort coverage, cluster inspection) works unchanged.

    Args:
        topics: Per-document topic IDs from BERTopic (-1 = outlier).
        field_refs: Ordered list of FieldReference matching topic indices.
        all_cohort_names: All cohort names for coverage tracking.

    Returns:
        Tuple of (clusters, outlier_cluster).
    """
    all_cohorts_set = set(all_cohort_names)
    groups: dict[int, list[FieldReference]] = defaultdict(list)
    for ref, topic_id in zip(field_refs, topics, strict=True):
        groups[topic_id].append(ref)

    clusters: list[FieldCluster] = []
    outlier_cluster: FieldCluster | None = None

    for topic_id, members in sorted(groups.items()):
        coverage: dict[str, int] = defaultdict(int)
        for m in members:
            coverage[m.dictionary_name] += 1
        missing = sorted(all_cohorts_set - set(coverage.keys()))

        fc = FieldCluster(
            cluster_id=topic_id,
            label="",
            members=members,
            cohort_coverage=dict(coverage),
            missing_cohorts=missing,
        )
        if topic_id == -1:
            outlier_cluster = fc
        else:
            clusters.append(fc)

    return clusters, outlier_cluster


# ── main entry point ───────────────────────────────────────────


def topic_model_dictionaries(
    embedded_dicts: list[EmbeddedDictionary],
    *,
    min_cluster_size: int = 15,
    umap_n_components: int = 5,
    umap_n_neighbors: int = 15,
    nr_topics: int | None = None,
    reduce_outliers: bool = False,
    llm_client: BaseLLMClient | None = None,
    random_state: int = 42,
) -> TopicModelResult:
    """Cluster fields using BERTopic for large-scale topic discovery.

    Uses pre-computed embeddings from EmbeddedDictionary.  The returned
    ``TopicModelResult.model`` is the fitted BERTopic instance — call its
    ``visualize_*()`` methods directly for interactive Plotly charts::

        result = topic_model_dictionaries(embedded_dicts)
        result.model.visualize_documents(result.docs, embeddings=result.embeddings)
        result.model.visualize_hierarchy()
        result.model.visualize_topics()
        result.model.visualize_heatmap()

    Args:
        embedded_dicts: List of EmbeddedDictionary to cluster.
        min_cluster_size: HDBSCAN minimum cluster size.
        umap_n_components: UMAP dimensions for HDBSCAN (not visualization).
        umap_n_neighbors: UMAP n_neighbors for clustering step.
        nr_topics: If set, reduce to this many topics post-hoc.
        reduce_outliers: If True, reassign outlier fields to nearest topic.
        llm_client: Optional LLM client for upgraded cluster labels.
        random_state: Seed for reproducibility.

    Returns:
        TopicModelResult with FieldCluster list, fitted BERTopic model,
        docs, and embeddings for native visualization.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    t0 = time.perf_counter()

    # Step 1: Collect inputs
    docs, embeddings, field_refs, cohort_names = collect_inputs(embedded_dicts)
    logger.info("topic_model_dictionaries: %d fields from %d dicts", len(field_refs), len(embedded_dicts))

    # Step 2: Build and fit
    model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            metric="cosine",
            random_state=random_state,
            low_memory=False,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            prediction_data=True,
        ),
        calculate_probabilities=True,
    )
    topics, probs = model.fit_transform(docs, embeddings)

    # Step 3: Optional post-hoc reduction. reduce_topics() mutates the model in
    # place and returns self, so read the reduced assignments off the model.
    if nr_topics is not None:
        model.reduce_topics(docs, nr_topics=nr_topics)
        # topics_ is List[int]|None on the stub but always populated after fit + reduce
        topics = cast("list[int]", model.topics_)
        logger.info("Reduced to %d topics", nr_topics)

    if reduce_outliers:
        topics = model.reduce_outliers(docs, topics)
        logger.info("Outliers reassigned to nearest topics")

    # Step 4: Convert to FieldCluster for cohort coverage tracking
    clusters, outlier_cluster = extract_topic_clusters(topics, field_refs, cohort_names)

    # Step 5: Label clusters
    for cluster in clusters:
        cluster.label = derive_cluster_label([m.description for m in cluster.members])
    if outlier_cluster:
        outlier_cluster.label = "Outliers"

    if llm_client is not None:
        try:
            label_clusters_llm(clusters, llm_client.complete)
            logger.info("LLM labels applied to %d topic clusters", len(clusters))
        except NotImplementedError:
            logger.info("LLM client does not implement complete(); using derived labels")

    elapsed = time.perf_counter() - t0
    n_outliers = len(outlier_cluster.members) if outlier_cluster else 0
    logger.info(
        "topic_model_dictionaries complete: %d fields, %d topics, %d outliers in %.2fs",
        len(field_refs),
        len(clusters),
        n_outliers,
        elapsed,
    )

    return TopicModelResult(
        model=model,
        docs=docs,
        embeddings=embeddings,
        field_refs=field_refs,
        clusters=clusters,
        outlier_cluster=outlier_cluster,
        all_cohort_names=cohort_names,
    )


# ── residual (tail) re-clustering ──────────────────────────────


def recluster_residual(
    embeddings: NDArray[np.float32],
    field_refs: list[FieldReference],
    residual_indices: list[int],
    *,
    min_cluster_size: int = 8,
    min_samples: int = 4,
    umap_n_components: int = 5,
    umap_n_neighbors: int = 15,
    random_state: int = 42,
) -> tuple[list[FieldCluster], FieldCluster | None]:
    """Re-cluster the uncovered TAIL (residual) fields IN ISOLATION, not off the global clustering.

    In the head/tail architecture the fields not assigned to a CDE (the novel / coverage-gap residual) are
    re-clustered among themselves: global clustering shatters or absorbs rare tail concepts under the
    dominant head concepts, whereas isolating the tail lets HDBSCAN find them at the right density.
    ``residual_indices`` select the residual rows of ``embeddings`` / ``field_refs`` (e.g. the novel-routed
    fields). Returns ``(clusters, outlier_cluster)`` in the same FieldCluster format as
    :func:`topic_model_dictionaries` (via :func:`extract_topic_clusters`); cluster ids are local to the
    residual subset.

    RECALL/PRECISION TRADEOFF (held-out PhenX, BioLORD encoder; benchmarks/BENCHMARK-HISTORY.md): vs reading
    the global clustering, isolating the tail LIFTS cross-cohort recall (macro +0.19-0.23; micro-recall
    ~0.10->0.42 at the 30% tail) but OVER-MERGES — cross-study precision ~halves at the 20-30% tail
    (0.44->0.23 at 30%). Net micro-F1 still improves (0.16->0.29). It is therefore a RECALL-favoring COARSE
    pass meant to FEED the split-aware concept-resolution stage (which partitions over-merged groups) + EITL
    review, NOT to emit final GenCDE groups directly, and NOT enabled by default. The asymmetry justifies it
    behind split-aware: a shattered tail concept (global clustering) is unrecoverable, an over-merged group
    is split downstream.
    """
    from hdbscan import HDBSCAN
    from umap import UMAP

    residual_indices = list(residual_indices)
    refs = [field_refs[i] for i in residual_indices]
    cohort_names: list[str] = []
    for r in refs:
        if r.dictionary_name not in cohort_names:
            cohort_names.append(r.dictionary_name)

    n = len(residual_indices)
    if n == 0:
        return [], None
    # Too few points for a UMAP neighborhood / any HDBSCAN cluster: keep the residual together as one group
    # so it still flows to the split-aware stage rather than being dropped.
    if n <= max(min_cluster_size, umap_n_neighbors):
        logger.info("recluster_residual: %d residual fields below clustering threshold -> single group", n)
        return extract_topic_clusters([0] * n, refs, cohort_names)

    sub = np.asarray(embeddings, dtype=np.float32)[residual_indices]
    red = np.asarray(
        UMAP(
            n_components=min(umap_n_components, n - 1),
            n_neighbors=min(umap_n_neighbors, n - 1),
            metric="cosine",
            random_state=random_state,
            low_memory=False,
        ).fit_transform(sub)
    )
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(red)
    clusters, outlier_cluster = extract_topic_clusters([int(x) for x in labels], refs, cohort_names)
    logger.info(
        "recluster_residual: %d residual fields -> %d clusters (%d outliers)",
        n,
        len(clusters),
        len(outlier_cluster.members) if outlier_cluster else 0,
    )
    return clusters, outlier_cluster
