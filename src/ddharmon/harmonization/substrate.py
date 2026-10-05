"""L2 — frozen clustering substrate: persist + reload the field->cluster partition (cheap replay).

``topic_model_dictionaries`` (UMAP+HDBSCAN) is not bit-reproducible across processes — a re-run reshuffles
every cluster, so each leanb prompt (keyed by its cluster's member set) changes and the Batch response
cache misses on *everything*, forcing a full re-pay even for a one-line downstream change.

Freezing the partition as a :class:`ClusteringSubstrate` breaks that: the partition is computed once and
saved; a re-run *loads* it (``harmonize_leanb(substrate=...)``) instead of re-clustering, so the clusters —
and the content-addressed prompt ids derived from their member sets (:func:`cluster_content_id`) — are
identical run-to-run and the cached LLM responses hit. Only stages whose *inputs* actually changed re-pay.

Cache-key semantics: prompt ids are keyed by the SEMANTIC IDENTITY of the unit of work (a cluster's member
set; a (cde_id, source-encoding) signature), not the full prompt text. That is stable given a frozen
substrate *and* deterministic prompt construction (embeddings come from the SQLite cache; members are
frozen). If you change a prompt's WORDING, mint a fresh substrate/cache (the id won't move on its own).

Clustering mode: a substrate records whether the CDE catalog's rows entered the clustering that produced it
(:attr:`ClusteringSubstrate.clustered_with_catalog`; cohort-only is the ``harmonize_leanb`` default), and a
replay runs in that RECORDED mode. A file saved before the field existed loads as catalog-in.

This lives in ``harmonization`` (not ``clustering``) so importing it doesn't pull in the heavy
clustering/scipy stack — it depends only on the light ``models.cluster`` dataclasses.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ddharmon.models.cluster import FieldCluster, FieldReference

_SEP = "\x1f"  # unit separator between keys
_PAIR = "\x1e"  # record separator within a (dictionary, variable) key


def _sha(s: str, length: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:length]


def cluster_content_id(member_keys: list[tuple[str, str]]) -> str:
    """Stable, order-independent content id for a cluster's member set (``"c" + sha1(sorted keys)``).

    A cluster's identity is its membership, not the ephemeral ordinal HDBSCAN assigns. Two runs that
    produce the same member set get the same id (so the LLM cache hits); adding or removing one member
    changes it (so a genuinely-changed cluster re-pays).
    """
    joined = _SEP.join(sorted(f"{d}{_PAIR}{v}" for d, v in member_keys))
    return "c" + _sha(joined)


def content_token(*parts: str, length: int = 12) -> str:
    """Stable id for an ORDERED tuple of strings (e.g. a (cde_id, source-encoding) spec-gen signature)."""
    return _sha(_SEP.join(parts), length)


@dataclass
class ClusteringSubstrate:
    """A frozen field->cluster partition: each cluster as its member ``(dictionary_name, variable_name)`` keys."""

    clusters: list[list[tuple[str, str]]]
    min_cluster_size: int
    #: Rows that ENTERED clustering: every embedded dictionary's rows when ``clustered_with_catalog``, only the
    #: non-catalog dictionaries' rows otherwise. Informational (0 = unknown): a replay compares it, per mode,
    #: with the rows it would cluster and logs a difference; it never rejects one.
    n_fields: int = 0
    outlier: list[tuple[str, str]] = field(default_factory=list)
    #: Whether M10 outlier recovery has ALREADY been applied to this partition (its recovered clusters are
    #: in ``clusters`` and ``outlier`` holds only what recovery left as noise). Recovery is applied to a
    #: partition at most once: re-applying it re-clusters the LEFTOVERS, which are a different, smaller
    #: residual - ``recluster_residual`` lumps any residual of <= 15 rows into one group - so a replay would
    #: grow a cluster the original run never had. ``False`` for a partition straight out of
    #: clustering (:func:`build_substrate`); set by ``recover_outlier_clusters`` / ``harmonize_leanb``.
    outliers_recovered: bool = False
    #: Whether the CDE catalog's own rows ENTERED the clustering that produced this partition.
    #: ``False`` = cohort-only clustering, the ``harmonize_leanb`` default: no catalog row is in
    #: ``clusters`` or ``outlier``, and a replay never lets one into M2 chunking or M10 outlier recovery.
    #: ``True`` = the earlier catalog-in clustering: catalog rows sit inside cohort clusters, form catalog-only
    #: clusters and appear among the noise, and a replay keeps them exactly where the run had them. A replay
    #: always runs in the mode RECORDED here, never in the mode a caller asks for. Defaults to ``True``: every
    #: substrate built or saved before the field existed came from catalog-in clustering or holds no catalog
    #: row at all, and for the latter the two modes replay identically.
    clustered_with_catalog: bool = True

    @property
    def substrate_id(self) -> str:
        """Content id of the whole PARTITION — sensitive to how members are grouped, not just which exist."""
        return _sha(_SEP.join(sorted(cluster_content_id(cl) for cl in self.clusters)))

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)


def build_substrate(
    clusters: list[FieldCluster],
    *,
    min_cluster_size: int,
    outlier: FieldCluster | None = None,
    n_fields: int = 0,
    clustered_with_catalog: bool = True,
) -> ClusteringSubstrate:
    """Extract the reusable partition from a clustering result (keeps only member keys — no vectors/labels).

    ``clustered_with_catalog`` records whether the catalog's rows entered that clustering (see
    :attr:`ClusteringSubstrate.clustered_with_catalog`); ``harmonize_leanb`` always passes it. The default is the
    legacy reading (catalog-in), which is also behaviour-neutral for a partition holding no catalog row.
    """
    return ClusteringSubstrate(
        clusters=[[(m.dictionary_name, m.variable_name) for m in cl.members] for cl in clusters],
        min_cluster_size=min_cluster_size,
        n_fields=n_fields,
        outlier=[(m.dictionary_name, m.variable_name) for m in outlier.members] if outlier else [],
        clustered_with_catalog=clustered_with_catalog,
    )


def save_substrate(substrate: ClusteringSubstrate, path: str | Path) -> Path:
    """Write the substrate to JSON (keys as ``[dictionary, variable]`` pairs)."""
    payload = {
        # 2 adds `outliers_recovered`, 3 adds `clustered_with_catalog`; older files lack them (see load_substrate)
        "version": 3,
        "substrate_id": substrate.substrate_id,
        "min_cluster_size": substrate.min_cluster_size,
        "n_fields": substrate.n_fields,
        "n_clusters": substrate.n_clusters,
        "clusters": [[[d, v] for d, v in cl] for cl in substrate.clusters],
        "outlier": [[d, v] for d, v in substrate.outlier],
        "outliers_recovered": substrate.outliers_recovered,
        "clustered_with_catalog": substrate.clustered_with_catalog,
    }
    p = Path(path)
    p.write_text(json.dumps(payload, indent=2))
    return p


def load_substrate(path: str | Path) -> ClusteringSubstrate:
    """Load a substrate previously written by :func:`save_substrate`.

    A file written before ``outliers_recovered`` existed (version 1) loads as ALREADY RECOVERED. That is the
    partition its run SHOWED: M10 recovery is on by default, and what gets saved is the substrate
    ``harmonize_leanb`` returns, which is captured AFTER recovery folded its clusters in. So the file's
    clusters already include whatever recovery found, and its ``outlier`` list is the leftover noise.
    Defaulting to ``False`` would re-run recovery on those leftovers at every replay and mint a cluster the
    run never had (a paused review run resumed on a partition the reviewer never saw). The
    cost of ``True`` is that a pre-M10 file replays without recovery, which is also the partition it
    recorded; a caller who wants M10 applied to such a base partition can set the flag to ``False``.

    A file written before ``clustered_with_catalog`` existed (versions 1-2) loads as CATALOG-IN, the only mode
    ``harmonize_leanb`` had then, so a parked review run or a frozen experiment resumes on exactly the
    partition, chunks and outlier recovery it was built with. (A partition clustered on the cohort
    dictionaries alone holds no catalog row; the two modes replay it identically.)
    """
    payload = json.loads(Path(path).read_text())
    return ClusteringSubstrate(
        clusters=[[(d, v) for d, v in cl] for cl in payload["clusters"]],
        min_cluster_size=int(payload["min_cluster_size"]),
        n_fields=int(payload.get("n_fields", 0)),
        outlier=[(d, v) for d, v in payload.get("outlier", [])],
        outliers_recovered=bool(payload.get("outliers_recovered", True)),
        clustered_with_catalog=bool(payload.get("clustered_with_catalog", True)),
    )


def clusters_from_substrate(substrate: ClusteringSubstrate, field_refs: list[FieldReference]) -> list[FieldCluster]:
    """Reconstruct :class:`FieldCluster`s from a frozen partition + the (deterministic) field refs.

    ``field_refs`` come from :func:`~ddharmon.clustering.topic_engine.collect_inputs` (reproducible from the
    embedding cache), so reconstruction needs no re-clustering. Members absent from ``field_refs`` are
    dropped and an empty cluster is skipped; ``cohort_coverage`` is recomputed. ``cluster_id`` is the
    enumeration index — the leanb chain keys its prompts off the member set's :func:`cluster_content_id`,
    not this ordinal.
    """
    ref_of = {(r.dictionary_name, r.variable_name): r for r in field_refs}
    out: list[FieldCluster] = []
    for keys in substrate.clusters:
        members = [ref_of[k] for k in keys if k in ref_of]
        if not members:
            continue
        cov = Counter(m.dictionary_name for m in members)
        out.append(
            FieldCluster(cluster_id=len(out), label="", members=members, cohort_coverage=dict(cov), missing_cohorts=[])
        )
    return out
