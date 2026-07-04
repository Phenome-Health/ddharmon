"""Standing external-ground-truth benchmarks for the harmonization pipeline.

Four benchmarks — the var level (A/B/D) and the value level (C) — run as the pipeline matures:
  * ``benchmarks.cdemapper`` — var -> CDE assignment vs the Yale CDEMapper gold (retrieval recall@k;
    optional LLM assignment). "Are we matching the right CDE?"  (DEV set)
  * ``benchmarks.phenx``     — cross-cohort co-clustering vs the PhenX<->dbGaP crosswalk.
    "Do same-concept variables from different cohorts land in the same cluster?"  (held-out)
  * ``benchmarks.athlos``    — value-recode / transform-spec correctness vs the ATHLOS harmonisation
    scripts (AGPL-3). "Are the source-value -> target-value recodes generated correctly?"
  * ``benchmarks.aireadi``   — var -> standardized concept retrieval vs AI-READI's gold OMOP/CDE
    mappings (MIT). "Can we retrieve a field's mapped concept?"  (held-out; different target vocabulary)

Portable: public gold sets are fetched on demand into ``.cache/benchmarks/`` (AI-READI's gold ships
in-repo); the CDE backbone is the shipped ``data/examples/all_cdes_flat.tsv``. Reproducible under
``PYTHONHASHSEED=0``.

  PYTHONHASHSEED=0 python -m benchmarks.cdemapper          # retrieval recall@k (dense/bm25/hybrid), $0
  PYTHONHASHSEED=0 python -m benchmarks.phenx              # cross-cohort co-clustering, $0
  PYTHONHASHSEED=0 python -m benchmarks.athlos             # value-recode gold + baselines, $0
  PYTHONHASHSEED=0 python -m benchmarks.aireadi            # var->concept retrieval recall@k, $0
"""
