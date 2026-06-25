"""Benchmark B — PhenX<->dbGaP crosswalk: cross-cohort co-clustering.

"Do variables that measure the same concept across DIFFERENT cohorts end up in the same cluster?"
Ground truth: the PhenX dbGaP Variable Crosswalk (Pan et al., Sci Data 2022, doi 10.1038/s41597-022-01660-4)
— the public PhenX Toolkit Variable_cross_reference export. dbGaP STUDY_ID = the cohort; variables sharing
a PhenX VARIABLE_ID / PROTOCOL across studies should co-cluster.

Method (faithful to the pipeline core): embed dbGaP variable descriptions, UMAP(5D,cosine,seed42) ->
HDBSCAN, score cross-study co-clustering recall (micro = pair-weighted; macro = mean per-concept; any-link
= share of concepts at least partially bridged) + a cut-independent embedding-separability signal. $0,
reproducible. Re-run as the embedding / clustering matures.

  PYTHONHASHSEED=0 python -m benchmarks.phenx
  PYTHONHASHSEED=0 python -m benchmarks.phenx --min-cluster-size 15 --level protocol
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from benchmarks import _common as common


def _crossstudy_recall(groups: dict, var_study: dict, cluster_of: dict) -> dict:
    """Cross-study co-clustering recall over concept groups. Outliers (-1) treated as singletons."""
    total_cross = co_cross = n_multi = 0
    per_concept: list[float] = []
    uid = [0]

    def clu(dvar):
        c = cluster_of.get(dvar, -1)
        if c == -1:
            uid[0] += 1
            return ("out", uid[0])
        return c

    for members in groups.values():
        members = [m for m in members if m in var_study]
        studies = Counter(var_study[m] for m in members)
        if len(studies) < 2:
            continue
        n_multi += 1
        n_total = len(members)
        cross = n_total * (n_total - 1) // 2 - sum(n * (n - 1) // 2 for n in studies.values())
        by_cl: dict = defaultdict(Counter)
        for m in members:
            by_cl[clu(m)][var_study[m]] += 1
        co = 0
        for stc in by_cl.values():
            mc = sum(stc.values())
            co += mc * (mc - 1) // 2 - sum(n * (n - 1) // 2 for n in stc.values())
        total_cross += cross
        co_cross += co
        if cross:
            per_concept.append(co / cross)
    return {
        "n_concepts_multistudy": n_multi,
        "cross_study_pairs": total_cross,
        "co_clustered": co_cross,
        "micro_recall": round(co_cross / total_cross, 4) if total_cross else None,
        "macro_recall": round(float(np.mean(per_concept)), 4) if per_concept else None,
        "concepts_any_link": (
            round(float(np.mean([1.0 if r > 0 else 0.0 for r in per_concept])), 4) if per_concept else None
        ),
    }


def _embedding_signal(by_pxvar, var_study, umat, idx, seed=0, n_sample=20000):
    """Cut-independent: mean cosine of same-concept cross-study pairs vs random cross-study pairs."""
    rng = np.random.RandomState(seed)
    dvars = list(idx)
    pos = []
    for members in by_pxvar.values():
        by_study: dict = defaultdict(list)
        for m in members:
            if m in idx:
                by_study[var_study[m]].append(m)
        if len(by_study) < 2:
            continue
        studs = list(by_study)
        for _ in range(min(8, sum(len(v) for v in by_study.values()))):
            a, b = rng.choice(len(studs), 2, replace=False)
            x = by_study[studs[a]][rng.randint(len(by_study[studs[a]]))]
            y = by_study[studs[b]][rng.randint(len(by_study[studs[b]]))]
            pos.append((idx[x], idx[y]))
        if len(pos) >= n_sample:
            break
    pos_cos = float(np.mean([umat[i] @ umat[j] for i, j in pos])) if pos else None
    neg = []
    while len(neg) < len(pos):
        i, j = rng.randint(len(dvars)), rng.randint(len(dvars))
        if i != j and var_study[dvars[i]] != var_study[dvars[j]]:
            neg.append((i, j))
    neg_cos = float(np.mean([umat[i] @ umat[j] for i, j in neg])) if neg else None
    return {
        "same_concept_xstudy_cos": round(pos_cos, 4) if pos_cos else None,
        "random_xstudy_cos": round(neg_cos, 4) if neg_cos else None,
        "separation": round(pos_cos - neg_cos, 4) if (pos_cos and neg_cos) else None,
        "n_pairs_sampled": len(pos),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cluster-size", type=int, default=8)
    ap.add_argument("--min-samples", type=int, default=4)
    ap.add_argument("--level", choices=["pxvar", "protocol", "both"], default="both")
    args = ap.parse_args()

    var_text, var_study, by_pxvar, by_proto = common.coclustering_inputs()
    dvars = sorted(var_text)
    print(
        f"dbGaP vars: {len(dvars)} · studies: {len(set(var_study.values()))} · "
        f"PhenX vars: {len(by_pxvar)} · protocols: {len(by_proto)}"
    )

    provider = common.make_provider()
    print("  embedding dbGaP variable descriptions…")
    vecs = np.asarray(provider.embed([var_text[d] for d in dvars]), dtype=np.float32)
    umat = common.normalize_rows(vecs)
    idx = {d: i for i, d in enumerate(dvars)}

    from hdbscan import HDBSCAN
    from umap import UMAP

    print(f"  UMAP(5D,cosine,seed42) → HDBSCAN(mcs={args.min_cluster_size},ms={args.min_samples},eom)…")
    red = UMAP(n_components=5, n_neighbors=15, metric="cosine", random_state=42).fit_transform(vecs)
    labels = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(red)
    cluster_of = {dvars[i]: int(labels[i]) for i in range(len(dvars))}
    n_out = int(np.sum(labels == -1))
    n_clusters = len({c for c in labels.tolist() if c != -1})
    print(f"  clusters: {n_clusters} · outliers: {n_out} ({n_out / len(dvars):.1%})")

    result = {
        "benchmark": "phenx_coclustering",
        "n_vars": len(dvars),
        "n_studies": len(set(var_study.values())),
        "n_clusters": n_clusters,
        "outlier_pct": round(n_out / len(dvars), 4),
        "clusterer": f"umap5d_cosine_s42+hdbscan_mcs{args.min_cluster_size}_ms{args.min_samples}_eom",
        "embedding_signal": _embedding_signal(by_pxvar, var_study, umat, idx),
    }
    sig = result["embedding_signal"]
    print(f"\n{'='*60}\nCROSS-STUDY CO-CLUSTERING\n{'='*60}")
    print(
        f"  embedding separability (cut-independent): same-concept {sig['same_concept_xstudy_cos']} "
        f"vs random {sig['random_xstudy_cos']}  (Δ={sig['separation']})"
    )
    if args.level in ("pxvar", "both"):
        result["pxvar"] = _crossstudy_recall(by_pxvar, var_study, cluster_of)
        r = result["pxvar"]
        print(
            f"  PhenX-VARIABLE: micro={r['micro_recall']} · macro={r['macro_recall']} · "
            f"any-link={r['concepts_any_link']}  ({r['n_concepts_multistudy']} concepts)"
        )
    if args.level in ("protocol", "both"):
        result["protocol"] = _crossstudy_recall(by_proto, var_study, cluster_of)
        r = result["protocol"]
        print(
            f"  PhenX-PROTOCOL: micro={r['micro_recall']} · macro={r['macro_recall']} · "
            f"any-link={r['concepts_any_link']}  ({r['n_concepts_multistudy']} concepts)"
        )
    out = common.CACHE_DIR / "phenx_result.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"\n{json.dumps(result)}\n  wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
