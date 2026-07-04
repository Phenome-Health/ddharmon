"""Tests for M2 deterministic coherence-aware cluster chunking (`chunk_oversized`).

The chunker bounds how many members reach the split LLM (an operational cap, not a semantic target) while
keeping chunks internally coherent, deterministic (cache-safe replay), and member-conserving.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ddharmon.harmonization.chunk import chunk_oversized
from ddharmon.models.cluster import FieldCluster, FieldReference


def _refs(n: int, prefix: str = "v") -> list[FieldReference]:
    return [FieldReference("C", f"{prefix}{i}", f"field {i}") for i in range(n)]


def _cluster(members: list[FieldReference]) -> FieldCluster:
    return FieldCluster(
        cluster_id=0, label="lbl", members=list(members), cohort_coverage={"C": len(members)}, missing_cohorts=[]
    )


def _spread(n: int) -> np.ndarray:
    """n unit vectors spread over a quarter circle — deterministic, with real cosine structure."""
    theta = np.linspace(0.0, np.pi / 2, n)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)


def _keyset(out: list[FieldCluster]) -> list[tuple]:
    return sorted((m.dictionary_name, m.variable_name) for c in out for m in c.members)


def test_small_cluster_passes_through_unchanged():
    refs = _refs(10)
    cl = _cluster(refs)
    out = chunk_oversized([cl], _spread(10), refs, cap=45)
    assert len(out) == 1 and out[0] is cl  # same object, not re-chunked


def test_oversized_cluster_chunked_to_cap_and_conserves_members():
    refs = _refs(100)
    out = chunk_oversized([_cluster(refs)], _spread(100), refs, cap=45)
    assert len(out) >= 3  # ceil(100/45) = 3 units at minimum
    assert all(len(c.members) <= 45 for c in out)  # the cap invariant
    keys = _keyset(out)
    assert keys == sorted((r.dictionary_name, r.variable_name) for r in refs)  # conservation
    assert len(keys) == len(set(keys)) == 100  # nothing dropped, nothing duplicated


def test_deterministic_and_order_independent():
    refs = _refs(60)
    emb = _spread(60)

    def sig(out):
        return sorted(tuple(sorted((m.dictionary_name, m.variable_name) for m in c.members)) for c in out)

    out1 = chunk_oversized([_cluster(refs)], emb, refs, cap=25)
    out2 = chunk_oversized([_cluster(list(reversed(refs)))], emb, refs, cap=25)  # shuffled member order
    assert sig(out1) == sig(out2)  # partition is order-independent (internal sort) and reproducible


def test_coherence_aware_keeps_blobs_together():
    a, b = _refs(30, "a"), _refs(30, "b")
    refs = a + b
    emb = np.array([[1.0, 0.0]] * 30 + [[0.0, 1.0]] * 30, dtype=np.float32)  # two orthogonal blobs
    out = chunk_oversized([_cluster(refs)], emb, refs, cap=45)  # 60 > 45 -> must chunk
    assert len(out) == 2
    for c in out:  # each chunk is pure (all "a" or all "b") — coherence preserved, blobs not mixed
        assert len({m.variable_name[0] for m in c.members}) == 1


def test_cluster_with_missing_embeddings_kept_intact():
    refs = _refs(60)
    ghost = FieldReference("C", "ghost", "no embedding row")
    cl = _cluster(refs + [ghost])  # 61 members, one absent from field_refs
    out = chunk_oversized([cl], _spread(60), refs, cap=45)
    assert len(out) == 1 and out[0] is cl  # kept whole (not chunked, not dropped)


def test_threshold_above_cap_defers_chunking():
    refs = _refs(48)
    out = chunk_oversized([_cluster(refs)], _spread(48), refs, cap=45, threshold=50)
    assert len(out) == 1  # 48 <= threshold(50) -> passes through despite exceeding cap
    assert Counter(len(c.members) for c in out) == {48: 1}


def test_skip_enumerated_family_kept_whole():
    # 60 members of a food-frequency battery (one enumerated family) — the description carries the template
    foods = [f"food{i}" for i in range(60)]
    refs = [FieldReference("C", f, f"How often do you eat {f}") for f in foods]
    cl = _cluster(refs)
    emb = _spread(60)  # embeddings are irrelevant to the family decision (which reads descriptions)
    kept = chunk_oversized([cl], emb, refs, cap=45, skip_enumerated=True)
    assert len(kept) == 1 and kept[0] is cl  # recognized as one rollup -> not chunked
    chunked = chunk_oversized([cl], emb, refs, cap=45, skip_enumerated=False)
    assert len(chunked) >= 2  # default: still chunked (the merge stage would reunite it)
