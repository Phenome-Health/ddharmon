# Changelog

All notable changes to ddharmon are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [0.9.0] — 2026-07-04

The **v3 harmonization milestone**: the split-aware pipeline hardened into a production-quality assignment
+ transform-spec engine, expanded to N≥3 bundled cohorts, gated by a second held-out CDE benchmark, and
documented with an honest measured-quality + limitations statement. Validated end-to-end on a held-out
5-cohort public run (18,128 fields): fields reaching a record 62%→98%, real-concept grouping 30%→61%,
CDE assignment 25%→42%. All new pipeline behavior is **on by default** (each mod has an opt-out flag).

### Added

- **Split-aware grouping at scale.** Oversized clusters are chunked into coherence-aware sub-units
  (recursive average-linkage bisection) so the split LLM sees every member; a cross-record **merge** stage
  reunites same-concept records over-split across clusters; a NONE-fraction **coherence gate** demotes
  records whose coded edges are mostly unmappable. Eliminates the large-cluster member-drop that stranded
  most fields in un-differentiated blobs.
- **Outlier recovery** — HDBSCAN noise is re-clustered in isolation at a lower density and folded back in,
  recovering coherent sub-threshold families (recovered 86% of outliers on the validation run).
- **Transform-spec generation** (`ddharmon.harmonization.transform`, `ddharmon.values`) — categorical
  value-recode specs, N1 unit conversions (curated UCUM-factor table, no `pint` dependency), N2 arithmetic
  formulas (safe-eval verified), and wide→long specs for repeating-measure families, with units-driven
  confidence routed through the same review layer as mappings. Emits specs (including parameterized
  data-dependent specs); does not execute them on data.
- **Concept-match gate** — flags an adopt/refine whose assigned CDE fails a same-concept check (the
  full-coverage-wrong-concept case), routing it to review without flipping the verdict.
- **N≥3 bundled example cohorts** — UK Biobank (Showcase schema), MESA (dbGaP), and AI-READI added
  alongside All of Us + CLSA, each with a reproduce-script and provenance (`data/examples/README.md`).
- **AI-READI CDE benchmark** — a second held-out variable→concept assignment gold (OMOP/CDE anchors) wired
  into `benchmarks/gate.py`, plus a C2 unit-conversion gate. A scheduled/manual **benchmark CI workflow**
  (`.github/workflows/benchmarks.yml`) runs the full $0 gate.
- **Reproducible-by-construction runs** — a frozen clustering **substrate** (`ddharmon.harmonization.substrate`)
  + content-addressed prompt ids + `temperature=0` on the batch stages let a re-run replay byte-identically
  from cached responses (UMAP/HDBSCAN are not bit-reproducible on their own).
- **Docs** — a **Command-line usage** section and a **Limitations & measured quality** section (dev-vs-held-out
  benchmark table) in the README.

### Changed

- **Retrieval matching quality** — same-concept candidates in a different encoding (banding, flag, composite,
  unit) now route to *refine* (a transform bridges them) rather than *novel*; the CDE candidate pool gets
  index hygiene (boilerplate + opaque-code stripping); a weak-support adopt is demoted to refine.
- `harmonize_leanb` turns the above quality mods on by default (each has an opt-out flag) and keeps the
  `max_clusters` cost cap. The conservative adopt/refine/novel router and 0.30 retrieval floor are unchanged;
  threshold calibration from expert-review verdicts remains a future release.

## [0.7.0] — 2026-06-24

The **v2 split-aware harmonization pipeline** — a new architecture that leads with *assignment to the
given CDE backbone* for the covered head and routes only the uncovered tail to generation/clustering.
The earlier sub-cluster-anchored pipeline (`harmonization/pipeline.py` + `anchor.py`) is retained for
reference. See [`docs/v2_methods.md`](docs/v2_methods.md).

### Added

- **v2 lean head/tail pipeline** (`ddharmon.harmonization.leanb`) — `harmonize_leanb()` runs
  cluster → hybrid retrieve → generate-ideal → split into concept-groups → per-group re-retrieve +
  assign → route adopt/refine/novel. Emits one `LeanBRecord` per distinct concept-group within a
  cluster (split-aware "adopt-with-context"), never silently pooling distinct concepts. Helpers:
  `prepare_leanb`, `prepare_split`, `prepare_group_assign`, `assemble_leanb`, `CdeBackbone`,
  `LeanBResult`, `export_leanb_eitl_queue`, `write_records_json`.
- **Hybrid lexical+dense retrieval** (`ddharmon.matching.lexical`) — a self-contained `BM25`,
  `reciprocal_rank_fusion`, and `hybrid_topk` (RRF of BM25 + dense cosine), the adopted candidate
  generator. No new dependencies. Unit-tested.
- **EITL campaign export** (`ddharmon.export.eitl`) — `export_split_eitl_campaign` / `build_cde_lookup`
  emit reviewer-ready campaigns honoring the import contract (U+2028-escaped source text, no raw
  newlines, `QUOTE_ALL`, csv-module chunking under the single-file size limit).
- **Residual (tail) re-clustering** (`ddharmon.clustering.recluster_residual`) — re-clusters the
  uncovered tail in isolation to feed the split-aware stage. Recall-favoring and **off by default**
  (it over-merges standalone — see the docstring for the measured precision/recall tradeoff).
- **Standing evaluation benchmarks** (`benchmarks/`) — reproducible, $0, no-proprietary-data gates:
  CDEMapper (var→CDE retrieval recall), PhenX (cross-cohort co-clustering), and ATHLOS (value-recode
  correctness), plus `benchmarks/gate.py` asserting hard floors on the deterministic signals. See
  [`benchmarks/README.md`](benchmarks/README.md).
- **v2 end-to-end notebook** — `notebooks/clustering/v2_harmonization_pipeline.ipynb`, the new default
  entry point, runs the full v2 flow on the bundled AoU + CLSA + CDE example data.

### Changed

- **Default embedding encoder → `FremyCompany/BioLORD-2023`** (was `all-mpnet-base-v2`). A
  concept↔definition contrastive encoder; the only model of seven evaluated that wins **both** the
  CDEMapper retrieval benchmark (hybrid recall@5 0.637→0.679) and held-out PhenX cross-cohort
  separability (Δ 0.536→0.611). Still 768-d; first use downloads ~440 MB from HuggingFace. Pair with
  `hybrid_topk`; do not ensemble with mpnet. This changes embedding/clustering output vs prior releases.
- **c-TF-IDF cluster labels** now mirror the embedded semantic text (`Field.to_embedding_text`) instead
  of the raw `name: description` string, so top terms reflect the clustered concept.

### Fixed

- A latent crash in `topic_model_dictionaries` when `nr_topics` is set (`reduce_topics` mutates the model
  in place and returns `self`; the reduced assignments are now read off the model).

## [0.6.1] — 2026-06-22

### Changed

- **`publish-to-pypi` skill docs** — auto-suggest the next semver version, document the
  PR → merge → prune → local-sync release flow, and clarify the canonical-vs-mirror remote
  split. Documentation only; no changes to the `ddharmon` package itself.

## [0.6.0] — 2026-06-22

### Added

- **`ddharmon` CLI entry point** — a minimal `click` console script (`src/ddharmon/cli.py`)
  wired up as the `ddharmon` command; shows help when invoked bare and reports the version.
- **Post-publish release verification** (`scripts/verify_release.py` + `scripts/smoke_test.py`)
  — an automated three-stage gate that polls PyPI for propagation, installs the exact pinned
  version into an ephemeral `uv` venv, and runs a self-contained smoke test (version,
  value-encoding parsing, ingestion/preprocessing, CLI) against the installed wheel; `--full`
  additionally exercises the `[all]` embedding stack. Driven from the `publish-to-pypi` skill.

### Changed

- **`__version__` is now single-sourced** from installed distribution metadata
  (`importlib.metadata`), so the package and distribution versions can never drift.
- **Dependencies bumped** — pyarrow 23.0.1 → 24.0.0 and the python-minor-and-patch group
  (18 updates); added Dependabot config and patched dev/notebook vulnerabilities.
- **CI:** publish workflow actions bumped to Node 24 versions.

### Fixed

- `scripts/check.sh` now quotes venv paths so repositories whose path contains spaces work.

## [0.5.0] — 2026-05-29

First public release of the v1 harmonization pipeline (supersedes the early 0.x placeholder
releases). v1 is a deliberately-scoped **sub-cluster-anchored CDE harmonization** pipeline — an
extension of the embedding-clustering-for-variable-harmonization line of work
(see [`docs/v1_methods.md`](docs/v1_methods.md)).

### Added

- **`ddharmon.harmonization`** — the v1 pipeline package:
  - `harmonize_dictionaries(...)` — end-to-end orchestration: semantic cluster → value
    sub-cluster → CDE anchor → adopt/refine/novel classify.
  - `prepare_from_clusters(...)` / `assemble_verdicts(...)` — split clustering from LLM so the
    sub-cluster → anchor → gate → prompt logic is testable and the LLM call can run inline or
    via the offline Batch API workflow.
  - `find_anchor_cde(...)` — per-sub-cluster CDE recommendation (medoid → best in-cluster CDE,
    ranked by similarity, then canonicalness, then metadata richness; GenCDE fallback).
  - Classify-only adopt/refine/novel prompts (`harmonize` and concept-only `kg_only` modes),
    robust JSON response parsing, and `HarmonizationVerdict` / `AnchorResult` models.
  - Exporters: `write_prompts_jsonl`, `write_buckets`, `export_eitl_queue` (EITL review TSV).
- **`ddharmon.clustering.value_subcluster`** — HDBSCAN sub-clustering on value-encoding vectors
  within each semantic topic (Phase 1b), plus `build_value_vector_lookup`.
- **Multi-cohort + CDE ingestion** verified through the existing generic loader for the NIH CDE
  export (flatten via `scripts/flatten_cde_repo.py`; full repo ~22.7k CDEs or endorsed-only ~174).
- **Canonical notebook** `notebooks/clustering/v1_harmonization_pipeline.ipynb` — thin
  orchestration over the new API (cohorts + CDE → embed → cluster → sub-cluster → anchor →
  classify → EITL).
- **Docs**: `docs/v1_methods.md` (methods + citation lineage); README v1 section.
- Tests: `test_subcluster`, `test_anchor`, `test_harmonization_prompts`, `test_harmonization_pipeline`.

### Notes / scope

- The single LLM call is the **classify-only** adopt/refine/novel pass; recommendations are
  routed to **EITL** for human verification (no spec is auto-authored, nothing is auto-applied).
- CDEs participate in clustering (loaded as cohort `NIH_CDE`); the anchor is the best CDE that
  lands in a sub-cluster, with a GenCDE-needed (`novel`) fallback when none does.

### Deferred (publication-pending / future)

- LLM **coherence judging** (a dual-sample coherence pass).
- LLM **concept-labeling** (v1 uses derived c-TF-IDF labels) and **spec authoring** / transformation specs.
- **Granularity-loss** detection.
- **Deep recursive clustering** (v1 is topic → semantic split → value sub-cluster, bounded depth).
- Pairwise 1:1 matching (built, not in the v1 release surface), standards/KG mapping, and the
  Click CLI orchestrator (`ddharmon.cli:main` is declared but unimplemented).

[1.0.0]: https://github.com/Phenome-Health/ddharmon/releases/tag/v1.0.0
