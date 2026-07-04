"""Deterministic, coherence-aware chunking of oversized clusters (M2, Phase 2a).

The split LLM sees only ``MAX_SHOW`` members of a cluster, and on very large clusters it also
under-enumerates (answers only a subset of a long list). So an oversized cluster loses most of its members
at the split step. This module partitions an oversized cluster into sub-chunks each ``<= cap`` — the
OPERATIONAL count the LLM reliably enumerates — so every member is actually shown and placed.

The cap is a BATCHING bound, NOT a semantic target. A genuinely large *coherent* family (e.g. 438 FFQ
food items, one rollup concept) must NOT be shattered into 438 concepts: chunking only bounds how many
members reach the LLM per call, and the cross-chunk merge (Phase 2b) re-unifies chunks of the same concept.
Chunking never decides the final concept grouping — the split + merge do.

Coherence-aware: chunks are formed by deterministic average-linkage agglomerative clustering (cosine) on
the members' cached embeddings — the SAME scipy family the top-level clustering uses — so each sub-chunk is
internally homogeneous and the LLM's within-chunk split is easy (usually one group). This is "clustering as
batching" (per the frozen v3 decision to demote clustering to retrieval/batching), NOT a semantic hierarchy
(refuted as the assignment engine, Run 021b).

Deterministic: scipy ``linkage``/``fcluster`` carry no RNG; given the frozen substrate + cached embeddings
+ a stable member ordering, the chunk partition — and hence each chunk's content-addressed prompt id
(:func:`~ddharmon.harmonization.substrate.cluster_content_id`) — is identical run-to-run, so the Batch
response cache stays valid on replay. Member conservation is total: every input member lands in exactly one
output cluster.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from ddharmon.harmonization.positional import detect_enumerated_family
from ddharmon.models.cluster import FieldCluster

logger = logging.getLogger(__name__)

# Cap = the members shown to the split LLM in one call (the reliable-enumeration limit). Keep it <= leanb
# MAX_SHOW so an un-chunked unit is always shown in FULL. Threshold defaults to the cap: anything larger is
# chunked; a caller may raise the threshold to chunk only much-larger clusters in a cautious first phase.
DEFAULT_CHUNK_CAP = 45


def _l2(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _partition_labels(vectors: NDArray[np.float32], cap: int) -> list[int]:
    """Deterministic member -> chunk labels: recursively BISECT any sub-group that exceeds ``cap``.

    Average-linkage agglomerative bisection (cosine): a group over the cap is split in two at its highest
    merge and each half recursed until every chunk fits. Unlike a single global ``maxclust`` cut — which
    isolates outliers as singletons before it breaks a tight over-cap blob, shattering the well-sized parts —
    this splits ONLY the oversized structure, so chunks stay near the cap. RNG-free; the grouping (not the
    label order) is what content-addresses each chunk.
    """
    n = len(vectors)
    if n <= cap:
        return [0] * n
    groups: list[list[int]] = []
    stack: list[list[int]] = [list(range(n))]
    while stack:
        idx = stack.pop()
        if len(idx) <= cap:
            groups.append(idx)
            continue
        sub = _l2(vectors[np.asarray(idx, dtype=np.intp)])
        two = fcluster(linkage(pdist(sub, metric="cosine"), method="average"), t=2, criterion="maxclust")
        g1 = [idx[i] for i in range(len(idx)) if two[i] == 1]
        g2 = [idx[i] for i in range(len(idx)) if two[i] == 2]
        if not g1 or not g2:  # degenerate (identical vectors can't bisect) -> hard-cut to the cap
            groups.append(idx[:cap])
            if idx[cap:]:
                stack.append(idx[cap:])
        else:
            stack.extend((g1, g2))
    labels = [0] * n
    for lab, idx in enumerate(groups):
        for i in idx:
            labels[i] = lab
    return labels


def chunk_oversized(
    clusters: list[FieldCluster],
    embeddings: NDArray[np.float32],
    field_refs: list,
    *,
    cap: int = DEFAULT_CHUNK_CAP,
    threshold: int | None = None,
    skip_enumerated: bool = False,
) -> list[FieldCluster]:
    """Replace every cluster with ``> threshold`` members by coherence-aware sub-chunks each ``<= cap``.

    Returns a NEW flat cluster list (clusters at/under the threshold pass through unchanged; oversized ones
    are replaced by their chunks). Members are sorted by ``(dictionary_name, variable_name)`` before linkage
    so the partition is order-independent and deterministic. A cluster whose members lack embedding rows
    (can't be placed geometrically) is passed through intact rather than dropped.

    ``threshold`` defaults to ``cap`` (chunk anything the split LLM couldn't show in full). Downstream
    stages are unchanged — they simply see more, smaller units, each with its own content-addressed id.

    ``skip_enumerated`` (opt-in): keep an oversized ENUMERATED-ENTITY family (same template / different
    entity — a food-frequency battery, a medication checklist; :func:`detect_enumerated_family`) as ONE
    cluster instead of chunking it. Per the split rule the family is one rollup concept, so chunk-splitting
    only inflates the LLM budget; the cross-chunk :mod:`~ddharmon.harmonization.merge` stage collapses the
    split+residual back into a single record. Conservative detector (high precision); a false positive
    degrades to a large blob the M3 coherence gate flags, not silent corruption.
    """
    thresh = cap if threshold is None else threshold
    row_of = {(r.dictionary_name, r.variable_name): i for i, r in enumerate(field_refs)}
    out: list[FieldCluster] = []
    n_chunked = 0
    n_family = 0
    for cluster in clusters:
        members = list(cluster.members)
        if len(members) <= thresh:
            out.append(cluster)
            continue
        if skip_enumerated and detect_enumerated_family([m.description for m in members]) is not None:
            out.append(cluster)  # one rollup concept — don't shatter it; merge reunites any split residual
            n_family += 1
            continue
        ordered = sorted(members, key=lambda m: (m.dictionary_name, m.variable_name))
        rows = [row_of.get((m.dictionary_name, m.variable_name)) for m in ordered]
        if any(r is None for r in rows):  # missing embeddings -> can't chunk geometrically, keep intact
            out.append(cluster)
            continue
        idx = np.asarray(rows, dtype=np.intp)  # all ints here (guarded above)
        labels = _partition_labels(embeddings[idx], cap)
        buckets: dict[int, list] = {}
        for member, lab in zip(ordered, labels, strict=True):
            buckets.setdefault(lab, []).append(member)
        for lab in sorted(buckets):
            grp = buckets[lab]
            cov = Counter(m.dictionary_name for m in grp)
            out.append(
                FieldCluster(
                    cluster_id=len(out), label=cluster.label, members=grp, cohort_coverage=dict(cov), missing_cohorts=[]
                )
            )
        n_chunked += 1
    logger.info(
        "chunk_oversized: %d/%d clusters chunked (cap=%d, threshold=%d) -> %d units; %d enumerated families kept whole",
        n_chunked,
        len(clusters),
        cap,
        thresh,
        len(out),
        n_family,
    )
    return out
