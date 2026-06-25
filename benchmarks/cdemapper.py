"""Benchmark A — CDEMapper gold: var -> CDE retrieval recall@k (dense vs BM25 vs hybrid).

"Are we matching the right CDE to a variable?" Ground truth: the Yale CDE-Mapping-Tool EvaluationData
(494 field->CDE pairs; GoldID is an NIH tinyId = our backbone id-space). Only the rows whose GoldID is
present in our backbone snapshot are scorable for retrieval (the rest target CDEs we never flattened ->
their correct verdict is `novel`, an assignment-arm concern, not a retrieval hit).

This is the $0, portable, reproducible retrieval gate. The LLM assignment arm (does the model pick the
retrieved gold CDE) lives in the sandbox (`bench_assign.py`) — it needs API keys, so it is not part of
the $0 gate. Retrieval recall@5 bounds assignment accuracy from above, so this is the leading indicator.

  PYTHONHASHSEED=0 python -m benchmarks.cdemapper
"""

from __future__ import annotations

import csv
import json
import os
import tempfile

import numpy as np

from benchmarks import _common as common
from ddharmon.matching.lexical import BM25, hybrid_topk, tokenize

KS = (1, 5, 10, 20, 50, 100)


def load_gold() -> list[dict]:
    rows: list[dict] = []
    for name, path in common.ensure_cdemapper_gold().items():
        with open(path) as f:
            for r in csv.DictReader(f):
                r = {k: (v or "").strip() for k, v in r.items()}
                if r.get("GoldID"):
                    r["_set"] = name
                    rows.append(r)
    return rows


def embed_gold(rows: list[dict], provider) -> dict[str, np.ndarray]:
    """Embed each gold field through the pipeline ingestion path (Element->question_text), keyed by row_id."""
    from ddharmon.embedding import embed_dictionary
    from ddharmon.ingestion import load_dictionary, preprocess_dictionary

    fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="gold_")
    os.close(fd)
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "Element", "Question_Text", "Values"])
        for i, r in enumerate(rows):
            w.writerow([f"g{i}", r["Element"], r.get("Question_Text", ""), r.get("Values", "")])
    dd = preprocess_dictionary(
        load_dictionary(
            tmp,
            cohort_name="GOLD",
            variable_name="row_id",
            embed_variable_name=False,
            question_text="Element",
            description="Question_Text",
            value_encoding="Values",
        )
    )
    ed = embed_dictionary(dd, provider=provider)
    names, vecs = ed.get_variable_names(), ed.get_all_vectors()
    os.unlink(tmp)
    return {n: vecs[i] for i, n in enumerate(names)}


def _recall(rank_per_row: list[int | None], scored: int) -> dict[str, float]:
    return {f"@{k}": round(sum(1 for r in rank_per_row if r is not None and r <= k) / scored, 3) for k in KS}


def main() -> None:
    provider = common.make_provider()
    cde_ids, cde_vecs, rich_corpus, tiny2des = common.load_cde_backbone(provider)
    cde_u = common.normalize_rows(cde_vecs.astype(np.float32))
    des_lower = [c.lower() for c in cde_ids]
    print("  building BM25 over rich CDE text…")
    bm25 = BM25(rich_corpus)

    gold = load_gold()
    gold_emb = embed_gold(gold, provider)
    keep = [i for i in range(len(gold)) if f"g{i}" in gold_emb]
    gold = [gold[i] for i in keep]
    gold_mat = common.normalize_rows(np.stack([gold_emb[f"g{i}"] for i in keep]).astype(np.float32))
    dense_sims = gold_mat @ cde_u.T

    backbone_tiny = set(tiny2des)
    n_in = sum(1 for r in gold if r["GoldID"] in backbone_tiny)
    topk = max(KS)
    dense_rank, bm25_rank, hybrid_rank = [], [], []
    scored = 0
    for gi, r in enumerate(gold):
        if r["GoldID"] not in backbone_tiny:
            continue
        scored += 1
        target = tiny2des[r["GoldID"]].lower()
        q = tokenize(" ".join(x for x in [r["Element"], r.get("Question_Text", ""), r.get("Values", "")] if x))
        lex = bm25.scores(q)
        d_order = np.argsort(-dense_sims[gi])[:topk]
        b_order = np.argsort(-lex)[:topk]
        h_order = hybrid_topk(dense_sims[gi], lex, topk)

        def rank(order, tgt=target):
            top = [des_lower[j] for j in order]
            return top.index(tgt) + 1 if tgt in top else None

        dense_rank.append(rank(d_order))
        bm25_rank.append(rank(b_order))
        hybrid_rank.append(rank(h_order))

    dense_r, bm25_r, hybrid_r = _recall(dense_rank, scored), _recall(bm25_rank, scored), _recall(hybrid_rank, scored)
    print(f"\n{'='*60}\nCDEMapper retrieval recall@k  ({scored}/{len(gold)} in-backbone scorable)\n{'='*60}")
    print(f"  {'k':>4} {'dense':>8} {'bm25':>8} {'hybrid':>8}")
    for k in KS:
        kk = f"@{k}"
        print(f"  {k:>4} {dense_r[kk]:>8.3f} {bm25_r[kk]:>8.3f} {hybrid_r[kk]:>8.3f}")

    result = {
        "benchmark": "cdemapper_retrieval",
        "gold_total": len(gold),
        "in_backbone": n_in,
        "scored": scored,
        "dense": dense_r,
        "bm25": bm25_r,
        "hybrid": hybrid_r,
    }
    out = common.CACHE_DIR / "cdemapper_result.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"\n{json.dumps(result)}\n  wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
