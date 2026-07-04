# Benchmarks — external ground-truth yardsticks

Re-run these as the pipeline matures. They measure harmonization against *external, curated* ground
truth (not self-defined metrics): the **variable level** (A/B/D) and the **value level** (C categorical,
C2 numeric).

| Benchmark | Question | Ground truth | Module |
|-----------|----------|--------------|--------|
| **CDEMapper** | Are we matching the **right CDE** to a variable? | Yale CDE-Mapping-Tool (494 field→CDE pairs; DEV) | `benchmarks.cdemapper` |
| **PhenX** | Do same-concept vars from **different cohorts** co-cluster? | PhenX↔dbGaP crosswalk (10k vars / 440 studies; held-out) | `benchmarks.phenx` |
| **ATHLOS** | Are the **categorical value recodes** (source value → target value) generated correctly? | ATHLOS harmonisation scripts (AGPL-3; 284 recode golds) | `benchmarks.athlos` |
| **AI-READI** | Can we retrieve a survey field's **mapped standard concept**? | AI-READI REDCap→OMOP/CDE mappings (MIT; 615 fields → 249 concepts; held-out) | `benchmarks.aireadi` |
| **units (C2)** | Are **numeric transform specs** correct — unit conversions (N1) + arithmetic (N2)? | Curated authoritative conversion factors + canonical derivations ($0, in-code) | `benchmarks.units` |

All four are **portable** (public gold fetched on demand into `.cache/benchmarks/`, except AI-READI's
which ships in-repo; CDE backbone is the shipped `data/examples/all_cdes_flat.tsv`) and **$0** (local
embeddings / parsing; no LLM in the gate).
Reproducible under `PYTHONHASHSEED=0`. Run from the repo root:

```bash
PYTHONHASHSEED=0 python -m benchmarks.cdemapper   # retrieval recall@k: dense vs bm25 vs hybrid
PYTHONHASHSEED=0 python -m benchmarks.phenx       # cross-cohort co-clustering vs PhenX concepts
PYTHONHASHSEED=0 python -m benchmarks.phenx --min-cluster-size 15 --level protocol
PYTHONHASHSEED=0 python -m benchmarks.athlos      # value-recode gold + identity/label-sim baselines
PYTHONHASHSEED=0 python -m benchmarks.aireadi     # var→concept retrieval recall@k (dense/bm25/hybrid)
PYTHONHASHSEED=0 python -m benchmarks.units       # C2 numeric: N1 unit-conversion gold + N2 formula verify
```

## Release gates (WS4)
`benchmarks.gate` runs the deterministic $0 benchmarks and asserts **regression floors**, exiting
non-zero on a breach — the tracked, reproducible release gate.

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
| AI-READI dense recall@5 (held-out) | 0.655 | ≥ 0.58 | hard |
| AI-READI dense recall@100 (held-out) | 0.914 | ≥ 0.85 | hard |
| C2 N1 unit-conversion accuracy | 1.000 | ≥ 0.95 | hard |
| C2 N2 formula-verify oracle self-check | 1.000 | ≥ 0.99 | hard |

Floors are regression guards with margin below baseline, not targets; lowering one requires a re-baseline
+ a written reason. The C2 `units` benchmark is deterministic and encoder-free (no fetch, no keys), so it
also runs inline in the fast suite (`tests/test_benchmarks_units.py`). Full per-encoder history: `.planning/experiments/nb05-recursion/BENCHMARK-HISTORY.md`.

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

## AI-READI (var → standardized concept retrieval)
Reads the shipped `data/examples/aireadi_surveys.csv` (built by `scripts/build_aireadi_csv.py` from the
MIT `AI-READI/DataElementMaps` repo). AI-READI is **CDE-forward**: each REDCap survey field is mapped to a
standardized concept (LOINC / SNOMED / PhenX / OMOP) with a concept code. The benchmark takes the 615
fields that carry a concept and the 249 distinct concepts they map to, then measures retrieval recall@k —
for each field, is its mapped concept retrieved in the top-k (**dense** / **BM25** / **hybrid**) when the
candidate pool is the AI-READI concept catalog. Both field text and concept text go through the same
pipeline ingestion + embedding path, so this scores the shipped encoder. **Held-out** (we never tuned on
it) and a *different* target vocabulary than the NIH-CDE backbone, so it complements CDEMapper as a
generalization check: held-out dense recall@5 ≈ 0.66 sits alongside CDEMapper's DEV hybrid recall@5 ≈
0.67. Note **dense beats hybrid here** — BM25 lexical overlap hurts on short standardized concept names —
so the gate floors the dense signal (unlike CDEMapper, where hybrid wins). The LLM assignment arm (does
the model pick the gold concept; and does it match the concept *code*) is a natural follow-up but needs
API keys, so it is not part of this $0 gate.

## units (C2 — numeric transform specs: unit conversion + arithmetic)
Two $0, deterministic, encoder-free checks over the C2 numeric engine. **N1** applies the curated unit
canonicalizer (`ddharmon.values.convert_units`) to a hand-verified gold of `(source_unit, target_unit,
sample_value → expected_value)` cases whose expected values come from **authoritative** factors (1 kg =
2.2046226 lb; °C = (°F−32)·5/9; …) written independently of the implementation table — so a perfect score
is real external agreement, not a tautology. **N2** exercises the deterministic formula verifier
(`ddharmon.harmonization.verify_formula`) on canonical derivations (BMI, months→years, %→fraction): the
correct formula must verify at 1.0 (oracle self-check) and a deliberately wrong foil must score low — the
harness must distinguish a right spec from a wrong one. The LLM formula-**generator** arm (vs the ATHLOS-55
arithmetic gold) needs the ATHLOS arithmetic parser + API keys, so — like the categorical LLM recode arm —
it is **not** in the $0 gate (a documented follow-up).

## Provenance / notes
- The reusable hybrid retrieval is `src/ddharmon/matching/lexical.py` (`BM25`, `reciprocal_rank_fusion`,
  `hybrid_topk`), unit-tested in `tests/test_lexical_retrieval.py`.
- Baselines and the full experimental narrative: `.planning/experiments/nb05-recursion/BENCHMARKS.md`
  + `LEDGER.md` (Runs 019–024 for A/B; Benchmark C added 2026-06-18).
- The original exploratory harnesses are in the gitignored `notebooks/clustering/sandbox/`
  (`bench_cdemapper.py`, `bench_assign.py`, `bench_phenx.py`, `bench_athlos.py`); these `benchmarks/`
  modules are the tracked, portable productionization.
