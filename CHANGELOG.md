# Changelog

All notable changes to ddharmon are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [1.0.0]

First public release. ddharmon harmonizes biomedical data-dictionary variables across studies by
assigning each concept to a Common Data Element (CDE) backbone — leading with assignment for the covered
head and routing only the uncovered tail to generation and clustering — and sends every recommendation to
expert review. Full algorithm, parameters, and design rationale are in [`docs/methods.md`](docs/methods.md).

Validated end-to-end on a held-out 5-cohort public run (18,128 fields): 97.5% of fields reach a record,
60.7% land in a real-concept group, and 41.9% of records are assigned to a CDE.

### Harmonization pipeline

- **Lean head/tail pipeline** (`ddharmon.harmonization.leanb`) — `harmonize_leanb()` runs
  cluster → hybrid retrieve → generate-ideal → split into concept-groups → per-group re-retrieve +
  assign → route adopt/refine/novel. Emits one record per distinct concept-group within a cluster,
  never silently pooling distinct concepts. Helpers: `prepare_leanb`, `prepare_split`,
  `prepare_group_assign`, `assemble_leanb`, `CdeBackbone`, `LeanBResult`, `export_leanb_eitl_queue`,
  `write_records_json`.
- **Split-aware grouping at scale** — oversized clusters are chunked into coherence-aware sub-units so
  the split step sees every member; a cross-record **merge** stage reunites the same concept over-split
  across clusters; a NONE-fraction **coherence gate** demotes records whose coded edges are mostly
  unmappable. Eliminates the large-cluster member-drop that otherwise strands fields in un-differentiated
  blobs.
- **Outlier recovery** — density-clustering noise is re-clustered in isolation at a lower density and
  folded back in, recovering coherent sub-threshold families.
- **Hybrid lexical+dense retrieval** (`ddharmon.matching.lexical`) — a self-contained `BM25`,
  `reciprocal_rank_fusion`, and `hybrid_topk` (RRF of BM25 + dense cosine), the candidate generator.
  No extra dependencies.
- **Retrieval matching quality** — same-concept candidates in a different encoding (banding, flag,
  composite, unit) route to *refine* (a transform bridges them) rather than *novel*; the CDE candidate
  pool gets index hygiene (boilerplate + opaque-code stripping); a weak-support adopt is demoted to
  refine. The conservative adopt/refine/novel router and 0.30 retrieval floor keep review recall-favoring.
- **Concept-match gate** — flags an adopt/refine whose assigned CDE fails a same-concept check (the
  full-coverage-wrong-concept case), routing it to review without flipping the verdict.

### Transform specs

- **Transform-spec generation** (`ddharmon.harmonization.transform`, `ddharmon.values`) — categorical
  value-recode specs, N1 unit conversions (curated UCUM-factor table, no `pint` dependency), N2
  arithmetic formulas (safe-eval verified), and wide→long specs for repeating-measure families, with
  units-driven confidence routed through the same review layer as mappings. Emits specs (including
  parameterized, data-dependent specs); does not execute them on data.

### Review, CLI, and export

- **`ddharmon` CLI** — `harmonize` (full split-aware pipeline) and `cluster` ($0, no LLM) console
  subcommands; dictionaries taken inline as `NAME=path` or via a JSON `--config`; `--dry-run` builds the
  stage-1 prompts at $0.
- **EITL campaign export** (`ddharmon.export.eitl`) — `export_split_eitl_campaign` / `build_cde_lookup`
  emit reviewer-ready campaigns honoring the import contract (escaped source text, no raw newlines,
  `QUOTE_ALL`, size-limited chunking). Nothing is auto-applied.

### Data, benchmarks, reproducibility

- **Bundled public example cohorts** — All of Us, CLSA, UK Biobank (Showcase schema), MESA (dbGaP), and
  AI-READI, each with a reproduce-script and provenance (`data/examples/README.md`). Metadata only —
  never participant-level data.
- **Standing evaluation benchmarks** (`benchmarks/`) — reproducible, `$0`, no-proprietary-data gates:
  CDEMapper (var→CDE retrieval), PhenX (cross-cohort co-clustering), ATHLOS (value-recode correctness),
  and AI-READI (variable→concept assignment), with `benchmarks/gate.py` asserting hard floors on the
  deterministic signals and a benchmark CI workflow (`.github/workflows/benchmarks.yml`).
- **Default embedding encoder** `FremyCompany/BioLORD-2023` — a concept↔definition contrastive encoder
  (768-d; first use downloads ~440 MB from HuggingFace). Pair with `hybrid_topk`.
- **Reproducible-by-construction runs** — a frozen clustering **substrate**
  (`ddharmon.harmonization.substrate`) + content-addressed prompt ids + `temperature=0` on the batch
  stages let a re-run replay byte-identically from cached responses (UMAP/HDBSCAN are not
  bit-reproducible on their own).
- **Post-publish release verification** (`scripts/verify_release.py` + `scripts/smoke_test.py`) and a
  single-sourced `__version__` (from installed distribution metadata).

### Scope (held for a forthcoming paper / future releases)

- LLM coherence judging, LLM concept-labeling, granularity-loss detection, and deep recursive clustering.
- Pairwise 1:1 matching as a first-class surface (the engine is built), and standards/KG mapping
  (LOINC / SNOMED / OMOP).
- Expert-review threshold calibration for the adopt/refine/novel cutoff.

[1.0.0]: https://github.com/Phenome-Health/ddharmon/releases/tag/v1.0.0
