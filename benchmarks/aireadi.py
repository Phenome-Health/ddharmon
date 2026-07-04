"""Benchmark D — AI-READI gold: survey var -> standard concept retrieval recall@k.

"When a survey field carries a known standardized concept, can we retrieve it?" Ground truth: the
AI-READI project's REDCap -> OMOP/CDE value-set mapping (MIT; `AI-READI/DataElementMaps`), shipped as
``data/examples/aireadi_surveys.csv`` — each survey field is mapped to an OMOP/CDE concept
(LOINC / SNOMED / PhenX / OMOP) with a concept code. This is a HELD-OUT generalization check (we never
tuned on it), complementary to Benchmark A:
  * CDEMapper (A) — retrieve the right CDE from the large NIH-CDE backbone (our assignment target; DEV).
  * AI-READI (D) — retrieve the right standardized concept from AI-READI's own concept catalog (held-out,
    a different target vocabulary — tests that the encoder generalizes beyond the NIH backbone).

Self-contained and $0: the candidate pool is the set of distinct AI-READI gold concepts; for each field
we measure whether its mapped concept is retrieved in the top-k (dense / BM25 / hybrid), mirroring the
CDEMapper retrieval gate. Both query (field) and candidate (concept) text go through the same pipeline
ingestion + embedding path, so this measures the shipped encoder, not a bespoke one. Deterministic under
PYTHONHASHSEED=0.

  PYTHONHASHSEED=0 python -m benchmarks.aireadi
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
    """AI-READI survey fields that carry a gold concept. concept_id = vocab:code (else the name)."""
    rows: list[dict] = []
    with open(common.AIREADI_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = (r.get("field_label") or "").strip()
            concept = (r.get("cde_concept_name") or "").strip()
            if not (label and concept):
                continue
            vocab = (r.get("cde_vocabulary") or "").strip()
            code = (r.get("cde_concept_code") or "").strip()
            rows.append(
                {
                    "variable_name": (r.get("variable_name") or "").strip(),
                    "field_label": label,
                    "value_encoding": (r.get("value_encoding") or "").strip(),
                    "concept_name": concept,
                    "concept_id": f"{vocab}:{code}" if (vocab and code) else concept,
                    "vocabulary": vocab,
                }
            )
    return rows


def embed_texts(items: list[tuple[str, str, str]], provider) -> dict[str, np.ndarray]:
    """Embed (row_id, question_text, values) triples through the pipeline ingestion path; keyed by row_id."""
    from ddharmon.embedding import embed_dictionary
    from ddharmon.ingestion import load_dictionary, preprocess_dictionary

    fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="aireadi_")
    os.close(fd)
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "qt", "vals"])
        for row_id, qt, vals in items:
            w.writerow([row_id, qt, vals])
    dd = preprocess_dictionary(
        load_dictionary(
            tmp,
            cohort_name="AIREADI",
            variable_name="row_id",
            embed_variable_name=False,
            question_text="qt",
            value_encoding="vals",
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
    gold = load_gold()

    # Candidate pool: distinct gold concepts (concept_id -> display name for embedding + BM25).
    concept_name: dict[str, str] = {}
    for r in gold:
        concept_name.setdefault(r["concept_id"], r["concept_name"])
    concept_ids = list(concept_name)
    cid_index = {cid: i for i, cid in enumerate(concept_ids)}
    print(f"  AI-READI gold: {len(gold)} mapped survey fields -> {len(concept_ids)} distinct concepts")

    # Embed candidates (concept names) and queries (field label + value encoding).
    cand_emb = embed_texts([(f"c{i}", concept_name[cid], "") for i, cid in enumerate(concept_ids)], provider)
    cand_mat = common.normalize_rows(np.stack([cand_emb[f"c{i}"] for i in range(len(concept_ids))]).astype(np.float32))
    query_emb = embed_texts([(f"q{i}", r["field_label"], r["value_encoding"]) for i, r in enumerate(gold)], provider)

    bm25 = BM25([concept_name[cid] for cid in concept_ids])
    topk = max(KS)
    dense_rank, bm25_rank, hybrid_rank = [], [], []
    scored = 0
    for qi, r in enumerate(gold):
        if f"q{qi}" not in query_emb:  # dropped by preprocessing (empty text) — unscorable
            continue
        scored += 1
        gold_idx = cid_index[r["concept_id"]]
        qv = common.normalize_rows(query_emb[f"q{qi}"][None, :].astype(np.float32))[0]
        dense = cand_mat @ qv
        lex = bm25.scores(tokenize(" ".join(x for x in [r["field_label"], r["value_encoding"]] if x)))

        def rank(order, tgt=gold_idx):
            order = list(order)
            return order.index(tgt) + 1 if tgt in order else None

        dense_rank.append(rank(np.argsort(-dense)[:topk]))
        bm25_rank.append(rank(np.argsort(-lex)[:topk]))
        hybrid_rank.append(rank(hybrid_topk(dense, lex, topk)))

    dense_r, bm25_r, hybrid_r = _recall(dense_rank, scored), _recall(bm25_rank, scored), _recall(hybrid_rank, scored)
    print(
        f"\n{'='*60}\nAI-READI var->concept retrieval recall@k  ({scored} fields, {len(concept_ids)} concepts)\n{'='*60}"
    )
    print(f"  {'k':>4} {'dense':>8} {'bm25':>8} {'hybrid':>8}")
    for k in KS:
        kk = f"@{k}"
        print(f"  {k:>4} {dense_r[kk]:>8.3f} {bm25_r[kk]:>8.3f} {hybrid_r[kk]:>8.3f}")

    result = {
        "benchmark": "aireadi_retrieval",
        "gold_fields": len(gold),
        "n_concepts": len(concept_ids),
        "scored": scored,
        "dense": dense_r,
        "bm25": bm25_r,
        "hybrid": hybrid_r,
    }
    out = common.CACHE_DIR / "aireadi_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\n{json.dumps(result)}\n  wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
