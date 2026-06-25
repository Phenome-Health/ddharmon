# Benchmarks — three external ground-truth yardsticks

Re-run these as the pipeline matures. They measure harmonization against *external, curated* ground
truth (not self-defined metrics): the **variable level** (A/B) and the **value level** (C).

| Benchmark | Question | Ground truth | Module |
|-----------|----------|--------------|--------|
| **CDEMapper** | Are we matching the **right CDE** to a variable? | Yale CDE-Mapping-Tool (494 field→CDE pairs) | `benchmarks.cdemapper` |
| **PhenX** | Do same-concept vars from **different cohorts** co-cluster? | PhenX↔dbGaP crosswalk (10k vars / 440 studies) | `benchmarks.phenx` |
| **ATHLOS** | Are the **value recodes** (source value → target value) generated correctly? | ATHLOS harmonisation scripts (AGPL-3; 284 recode golds) | `benchmarks.athlos` |

All three are **portable** (public gold fetched on demand into `.cache/benchmarks/`; CDE backbone is the
shipped `data/examples/all_cdes_flat.tsv`) and **$0** (local embeddings / parsing; no LLM in the gate).
Reproducible under `PYTHONHASHSEED=0`. Run from the repo root:

```bash
PYTHONHASHSEED=0 python -m benchmarks.cdemapper   # retrieval recall@k: dense vs bm25 vs hybrid
PYTHONHASHSEED=0 python -m benchmarks.phenx       # cross-cohort co-clustering vs PhenX concepts
PYTHONHASHSEED=0 python -m benchmarks.phenx --min-cluster-size 15 --level protocol
PYTHONHASHSEED=0 python -m benchmarks.athlos      # value-recode gold + identity/label-sim baselines
```

## Release gates (WS4)
`benchmarks.gate` runs the two $0 benchmarks and asserts **regression floors**, exiting non-zero on a
breach — the tracked, reproducible release gate.

```bash
PYTHONHASHSEED=0 python -m benchmarks.gate     # or: ./scripts/bench_gate.sh
RUN_BENCHMARKS=1 ./scripts/check.sh            # fold the gate into the full check (else skipped — see below)
```

It is **excluded from the default `./scripts/check.sh`** (it loads the encoder + fetches public gold,
~3–5 min, network once) — set `RUN_BENCHMARKS=1` (CI / pre-release) to include it. **Hard gates are
DETERMINISTIC signals only** (CDEMapper hybrid recall; the PhenX cut-independent separability Δ). The PhenX
co-clustering macro/micro use UMAP+HDBSCAN (not bit-reproducible across processes) and are **advisory** —
reported, never gating. The LLM assignment / recode arms need API keys and are not in the $0 gate.

**Committed baselines + floors** — encoder default **`FremyCompany/BioLORD-2023`** (2026-06-24):

| metric | baseline | floor | gating |
|---|---|---|---|
| CDEMapper hybrid recall@5 | 0.674 | ≥ 0.63 | hard |
| CDEMapper hybrid recall@100 | 0.926 | ≥ 0.90 | hard |
| PhenX separability Δ (cut-independent) | 0.611 | ≥ 0.55 | hard |
| PhenX VAR macro recall | 0.472 | ≥ 0.38 | advisory |

Floors are regression guards with margin below baseline, not targets; lowering one requires a re-baseline
+ a written reason.

## CDEMapper (var → CDE retrieval)
Fetches `EvaluationData/{ADRD,Eye,Stroke,covid-19}.csv` from `BIDS-Xu-Lab/CDE-Mapping-Tool`. GoldID is an
NIH tinyId = our backbone id-space; only rows whose GoldID is in our snapshot are scorable (the rest
target CDEs we never flattened → their correct verdict is `novel`). Reports retrieval recall@k for
**dense** (BioLORD-2023 cosine — the committed encoder default), **BM25** (lexical, over rich CDE text), and **hybrid** (RRF of the two — the
adopted candidate generator, `ddharmon.matching.hybrid_topk`). Retrieval recall@5 bounds end-to-end
assignment accuracy. The LLM assignment arm (does the model pick the gold CDE) lives in the sandbox
(`bench_assign.py`) — it needs API keys, so it is not part of this $0 gate.

## PhenX (cross-cohort co-clustering)
Fetches the PhenX Toolkit `Variable_cross_reference.xlsx`. Embeds dbGaP variable descriptions →
UMAP(5D,cosine,seed42)→HDBSCAN (the pipeline core) → cross-study co-clustering recall vs PhenX concepts:
**micro** (pair-weighted; dominated by giant demographics concepts), **macro** (mean per-concept — the
fair number), **any-link** (share of concepts at least partially bridged), plus a cut-independent
embedding-separability Δ (same-concept-cross-study cosine vs random).

## ATHLOS (value-recode / transform-spec correctness)
Fetches + extracts the ATHLOS harmonisation-script repo (`athlosproject/athlos-project.github.io`,
**AGPL-3** — fine for evaluation; matters only if redistributing derived code). Parses ~1,900 per-variable
`.Rmd` scripts (all three Categories layouts + the `car::recode` mini-language), applies each recode to its
source value set → a `{source_code → target_code}` gold, deduped to **284 unique recode entries** (from
1,213 cohort-wave instances) across 41 target variables (**88% non-identity**). Scope = clean *categorical*
recodes; continuous / quantile / derived / multi-response transforms are detected and skipped. Reports an
oracle self-check (1.0) plus **identity** and **label-similarity** baselines (entry-exact / pair-acc
incl-NA / pair-acc excl-NA). The LLM recode-generator arm (the real number — predict the mapping from
source+target value sets) lives in the sandbox (`bench_athlos.py --llm`); it needs API keys, so it is not
part of this $0 gate. Pairs with the EITL `transform_review` campaign as the in-domain acceptance gate.

## Provenance / notes
- The reusable hybrid retrieval is `src/ddharmon/matching/lexical.py` (`BM25`, `reciprocal_rank_fusion`,
  `hybrid_topk`), unit-tested in `tests/test_lexical_retrieval.py`.
- The LLM-dependent arms (assignment accuracy, the ATHLOS recode generator) need API keys and are run
  manually, separate from this $0 deterministic gate.
