"""Tests for lexical (BM25) and hybrid lexical+dense retrieval."""

from __future__ import annotations

import numpy as np

from ddharmon.matching.lexical import (
    BM25,
    hybrid_topk,
    reciprocal_rank_fusion,
    tokenize,
)

CORPUS = [
    "Current Educational Attainment",  # 0
    "Address City Name",  # 1
    "Body Mass Index BMI",  # 2
    "Date of Birth Month",  # 3
    "Annual Household Income",  # 4
]


def test_tokenize_lowercases_splits_and_drops_short():
    assert tokenize("City of Residence!") == ["city", "of", "residence"]
    assert tokenize("BMI (kg/m2)") == ["bmi", "kg", "m2"]
    # single-char tokens dropped by default min_len=2
    assert "a" not in tokenize("a big test")


def test_bm25_ranks_lexically_overlapping_doc_first():
    bm25 = BM25(CORPUS)
    # query shares tokens with doc 0 ("educational attainment")
    scores = bm25.scores("Educational Attainment")
    assert scores.shape == (len(CORPUS),)
    assert int(np.argmax(scores)) == 0
    # a query with no shared vocabulary scores all zeros
    assert np.allclose(bm25.scores("zzz qqq"), 0.0)


def test_bm25_top_k_returns_requested_count_and_order():
    bm25 = BM25(CORPUS)
    top = bm25.top_k("city name address", k=2)
    assert top[0] == 1  # "Address City Name"
    assert len(top) == 2


def test_bm25_accepts_pretokenized_documents():
    bm25 = BM25([["body", "mass", "index"], ["annual", "income"]])
    assert int(np.argmax(bm25.scores(["body", "mass"]))) == 0


def test_reciprocal_rank_fusion_rewards_agreement():
    # item 2 is top of both rankings -> should win; item 0 only leads ranking A
    fused = reciprocal_rank_fusion([[2, 0, 1], [2, 1, 0]], n_items=3)
    assert int(np.argmax(fused)) == 2
    assert fused[2] > fused[0] > 0


def test_reciprocal_rank_fusion_absent_items_score_zero():
    fused = reciprocal_rank_fusion([[0, 1]], n_items=4)
    assert fused[2] == 0.0 and fused[3] == 0.0
    assert fused[0] > fused[1] > 0


def test_hybrid_topk_fuses_dense_and_lexical():
    # dense favors index 0; lexical favors index 3. With small rrf_k, rank-1 dominates so each
    # modality's top item surfaces (the point of fusion). (Larger rrf_k rewards consistent mid-rank.)
    dense = np.array([0.9, 0.5, 0.1, 0.2])
    lexical = np.array([0.0, 0.4, 0.1, 0.9])
    top = hybrid_topk(dense, lexical, top_k=4, rrf_k=1)
    assert set(top[:2]) == {0, 3}  # the two single-modality winners
    assert len(top) == 4


def test_hybrid_topk_rejects_misaligned_arrays():
    import pytest

    with pytest.raises(ValueError):
        hybrid_topk(np.zeros(3), np.zeros(4), top_k=2)


def test_hybrid_topk_pool_bounds_considered_items():
    # with pool=1, only each modality's #1 enters fusion; everything else ties at 0
    dense = np.array([0.1, 0.2, 0.9, 0.3])
    lexical = np.array([0.9, 0.1, 0.0, 0.0])
    top = hybrid_topk(dense, lexical, top_k=2, pool=1)
    assert set(top) == {0, 2}  # lexical#1=idx0, dense#1=idx2
