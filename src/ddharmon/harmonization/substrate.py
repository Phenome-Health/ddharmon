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
    n_fields: int = 0
    outlier: list[tuple[str, str]] = field(default_factory=list)

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
) -> ClusteringSubstrate:
    """Extract the reusable partition from a clustering result (keeps only member keys — no vectors/labels)."""
    return ClusteringSubstrate(
        clusters=[[(m.dictionary_name, m.variable_name) for m in cl.members] for cl in clusters],
        min_cluster_size=min_cluster_size,
        n_fields=n_fields,
        outlier=[(m.dictionary_name, m.variable_name) for m in outlier.members] if outlier else [],
    )


def save_substrate(substrate: ClusteringSubstrate, path: str | Path) -> Path:
    """Write the substrate to JSON (keys as ``[dictionary, variable]`` pairs)."""
    payload = {
        "version": 1,
        "substrate_id": substrate.substrate_id,
        "min_cluster_size": substrate.min_cluster_size,
        "n_fields": substrate.n_fields,
        "n_clusters": substrate.n_clusters,
        "clusters": [[[d, v] for d, v in cl] for cl in substrate.clusters],
        "outlier": [[d, v] for d, v in substrate.outlier],
    }
    p = Path(path)
    p.write_text(json.dumps(payload, indent=2))
    return p


def load_substrate(path: str | Path) -> ClusteringSubstrate:
    """Load a substrate previously written by :func:`save_substrate`."""
    payload = json.loads(Path(path).read_text())
    return ClusteringSubstrate(
        clusters=[[(d, v) for d, v in cl] for cl in payload["clusters"]],
        min_cluster_size=int(payload["min_cluster_size"]),
        n_fields=int(payload.get("n_fields", 0)),
        outlier=[(d, v) for d, v in payload.get("outlier", [])],
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
