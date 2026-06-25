"""Lexical (BM25) and hybrid lexical+dense retrieval.

Dense embedding retrieval (cosine over sentence-transformer vectors) misses lexical near-matches —
e.g. a field "Educational Attainment" vs a CDE "Current Educational Attainment", or "City of Residence"
vs "Address City Name" — that share vocabulary but sit apart in embedding space. Benchmarking against the
CDEMapper gold (494 field->CDE pairs) showed BM25 alone beats dense at every recall@k, and a Reciprocal
Rank Fusion (RRF) of the two beats either alone (recall@5: dense 0.447 -> BM25 0.611 -> hybrid 0.632),
which lifts end-to-end CDE-assignment accuracy 0.358 -> 0.458.

This module provides:
  * ``BM25`` — Okapi BM25 with an inverted index over a fixed corpus.
  * ``reciprocal_rank_fusion`` — combine several ranked candidate lists into one fused ranking.
  * ``hybrid_topk`` — fuse a dense-cosine score array with a BM25 score array and return top-k indices.

All retrieval is deterministic. BM25 is dependency-free (no rank_bm25 needed).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def tokenize(text: str, min_len: int = 2) -> list[str]:
    """Lowercase, split on non-alphanumeric runs, drop tokens shorter than ``min_len``."""
    return [t for t in _TOKEN_RE.split(str(text or "").lower()) if len(t) >= min_len]


class BM25:
    """Okapi BM25 ranking over a fixed document corpus.

    Builds an inverted index at construction so per-query scoring touches only the postings of the
    query terms. Parameters follow the standard Okapi defaults (``k1=1.5``, ``b=0.75``).

    Args:
        documents: The corpus, either as raw strings (tokenized with :func:`tokenize`) or as
            pre-tokenized token lists.
        k1: Term-frequency saturation parameter.
        b: Document-length normalization parameter.
    """

    def __init__(self, documents: list[str] | list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        docs_tokens = [d if isinstance(d, list) else tokenize(d) for d in documents]
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: Counter[str] = Counter()
        for i, toks in enumerate(docs_tokens):
            tf = Counter(toks)
            for term, count in tf.items():
                self._postings[term].append((i, count))
            doc_freq.update(tf.keys())
        # BM25+ idf (always positive): log(1 + (N - n + 0.5) / (n + 0.5))
        self._idf = {term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for term, n in doc_freq.items()}

    def scores(self, query: str | list[str]) -> np.ndarray:
        """Return a BM25 score for every document against ``query`` (length ``n_docs``)."""
        tokens = query if isinstance(query, list) else tokenize(query)
        out = np.zeros(self.n_docs, dtype=np.float64)
        if not self.avgdl:
            return out
        for term in set(tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_id, count in postings:
                denom = count + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                out[doc_id] += idf * (count * (self.k1 + 1)) / denom
        return out

    def top_k(self, query: str | list[str], k: int) -> list[int]:
        """Return the indices of the top-``k`` documents for ``query`` by BM25 score."""
        return np.argsort(-self.scores(query))[:k].tolist()


def reciprocal_rank_fusion(rankings: list[list[int]], n_items: int, *, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of several ranked index lists into one fused score per item.

    For each ranking, item at rank ``r`` (0-based) contributes ``1 / (k + r + 1)``. Items absent from a
    ranking contribute nothing for that ranking. Returns an array of length ``n_items`` of fused scores
    (higher = better); sort descending for the fused order.

    Args:
        rankings: Each element is a list of item indices in descending preference order.
        n_items: Total number of items (length of the returned array).
        k: RRF damping constant (standard default 60).
    """
    fused = np.zeros(n_items, dtype=np.float64)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + rank + 1)
    return fused


def hybrid_topk(
    dense_scores: np.ndarray,
    lexical_scores: np.ndarray,
    top_k: int,
    *,
    rrf_k: int = 60,
    pool: int = 1000,
) -> list[int]:
    """Fuse dense-cosine and lexical (BM25) score arrays via RRF and return the top-k item indices.

    Both arrays must be aligned to the same candidate space (index i = the same candidate). Each array is
    converted to a rank order (truncated to ``pool`` to bound work on large corpora), fused with RRF, and
    the top-``top_k`` fused indices returned.

    Args:
        dense_scores: Per-candidate dense similarity (e.g. cosine). Higher = better.
        lexical_scores: Per-candidate lexical score (e.g. BM25). Higher = better.
        top_k: Number of indices to return.
        rrf_k: RRF damping constant.
        pool: Truncate each ranking to this many items before fusion (caps cost on 20k+ corpora).
    """
    if dense_scores.shape != lexical_scores.shape:
        raise ValueError(f"dense {dense_scores.shape} and lexical {lexical_scores.shape} must align")
    n = dense_scores.shape[0]
    dense_order = np.argsort(-dense_scores)[:pool].tolist()
    lexical_order = np.argsort(-lexical_scores)[:pool].tolist()
    fused = reciprocal_rank_fusion([dense_order, lexical_order], n, k=rrf_k)
    return np.argsort(-fused)[:top_k].tolist()
