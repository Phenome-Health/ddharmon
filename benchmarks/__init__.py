"""Standing external-ground-truth benchmarks for the harmonization pipeline.

Three benchmarks — the var level (A/B) and the value level (C) — run as the pipeline matures:
  * ``benchmarks.cdemapper`` — var -> CDE assignment vs the Yale CDEMapper gold (retrieval recall@k;
    optional LLM assignment). "Are we matching the right CDE?"
  * ``benchmarks.phenx``     — cross-cohort co-clustering vs the PhenX<->dbGaP crosswalk.
    "Do same-concept variables from different cohorts land in the same cluster?"
  * ``benchmarks.athlos``    — value-recode / transform-spec correctness vs the ATHLOS harmonisation
    scripts (AGPL-3). "Are the source-value -> target-value recodes generated correctly?"

Portable: public gold sets are fetched on demand into ``.cache/benchmarks/``; the CDE backbone is the
shipped ``data/examples/all_cdes_flat.tsv``. Reproducible under ``PYTHONHASHSEED=0``.

  PYTHONHASHSEED=0 python -m benchmarks.cdemapper          # retrieval recall@k (dense/bm25/hybrid), $0
  PYTHONHASHSEED=0 python -m benchmarks.phenx              # cross-cohort co-clustering, $0
  PYTHONHASHSEED=0 python -m benchmarks.athlos             # value-recode gold + baselines, $0
"""
