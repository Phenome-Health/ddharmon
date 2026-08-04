# Changelog

All notable changes to ddharmon are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

### Added

- **Composite / derived-variable builder** (`ddharmon.harmonization.composite`) — answers, for a finished
  run, whether a *published composite score* (a frailty index, a PHQ-9 sum, an intrinsic-capacity score) can
  be computed from the concepts that run harmonized, which concepts compose it, and how. Four stages, only
  two of which cost an LLM call: transcribe the score from its source document → hybrid-retrieval shortlist
  per component + one judge pass → deterministic per-cohort feasibility → the derivation recipe.
  `derive_composite()` is the entry point; `spec_to_dict()` / `spec_to_json()` serialize the result.

  Both dominant published shapes are first-class: **criteria-count** (Fried phenotype — k of n criteria) and
  **deficit-accumulation** (a frailty index — deficits ÷ items considered), plus sum / weighted-sum /
  z-composite forms (`CompositeKind`).

  Grounding is structural rather than merely instructed: the judge may only choose among the concept ids
  RETRIEVED for that component, and anything else is discarded with the component reported MISSING. Concepts
  are referenced by record id, not label. Nothing is invented — a cutoff the source does not state stays
  `unstated` and flagged for review, a band list yields no cut-point, a run where nothing matched emits
  "NOT COMPUTABLE" rather than a runnable-looking division by zero, and a document claiming more items than
  it lists is reported as incomplete instead of being filled in from prior knowledge.

  Feasibility is honest about its limits: data-dictionary metadata gives per-cohort *presence*, so the report
  names which cohorts are computable and explicitly does not claim effective N.

  Reviewer overrides (`overrides={component: concept_id | None}`) pin or drop a match and skip the judge, so
  a fully-pinned re-derive costs **zero** LLM calls.

- **Score-source ingestion** (`ddharmon.harmonization.score_sources`) — the builder's document front door:
  pasted text, a PDF, a Word (`.docx`) supplement, or a fetched URL / bare DOI / GitHub repo (README +
  selected docs), each returning a `ScoreSource` with the extracted text, its provenance, and a sha256 of
  exactly what was read. A score's definition must come from a real document, never from a model's
  recollection of it.

  The Word path exists because a published score's item table usually lives in the **supplement**, and
  supplements are routinely `.docx`. `docx_to_text` walks the document body's XML children so paragraphs
  and **tables** stay interleaved in document order — `python-docx`'s `Document.paragraphs` omits table
  content entirely, which would silently drop the very item list the caller came for. Tables are rendered
  as pipe-separated rows so the grid survives into the extraction prompt. A legacy binary `.doc` is
  identified as a different format needing re-saving, not as a broken `.docx`.

  Fetching is deliberately bounded: http(s) only, every redirect hop re-validated against private /
  loopback / link-local address space, a byte cap, and a timeout. A body advertised as PDF without a `%PDF`
  header falls back to HTML extraction rather than surfacing a parser stack trace — publishers commonly
  answer a `.pdf` URL with an interstitial page.

  Every fetch failure a caller can act on is raised as a `ValueError` naming the recovery, never as a raw
  `httpx` exception — so a hosted caller maps it to a readable 4xx instead of a 500. That matters most for
  the common case: a DOI resolving to a paywalled journal that refuses automated readers (403) is reported
  as exactly that, with "upload the PDF instead", rather than as a crash.

- New optional extra **`sources`** (`pypdf`, `python-docx`) for the PDF and Word paths, folded into `all`.
  Both parsers are imported lazily, so paste / URL / repo ingestion works without it.

- `parse.salvage_objects()` — shared rescue for a long JSON array truncated at the token cap (a 68-item
  component list makes this a real risk).

## [1.1.0]

Adds **GenCDE** — generated Common Data Elements for the *novel* route — together with transform-spec
generation for those generated targets. Every new stage is **opt-in and default OFF**, so 1.1.0 is a
drop-in, backward-compatible upgrade over 1.0.0: existing `harmonize_leanb()` callers behave identically
until they enable the flags.

### Added

- **GenCDE synthesis** (`ddharmon.harmonization.gencde`, opt-in `gencde` stage) — when no existing CDE
  fits a concept group, synthesize a structured GenCDE (preferred name, definition, data type, permissible
  values / units, aliases, provenance) from the members' *pooled cross-cohort evidence*, giving the novel
  route a harmonization target instead of a dead end. Deterministic and cache-stable (temp 0). New helpers
  `prepare_gencde`, `assemble_gencde`, `observed_answer_labels`; new `LeanBRecord.gencde` field and EITL
  export columns; adds Benchmark E (FAIRkit reproducibility protocol).
- **GenCDE transform-spec generation** (opt-in `gencde_specgen` stage) — novel records now carry
  member→GenCDE transform specs, mirroring the adopt/refine path:
  - **C1 — categorical recodes** against the GenCDE's permissible values (`prepare_gencde_specgen` /
    `assemble_gencde_specgen`, reusing the shared recode machinery).
  - **N1 — unit/scale specs** for numeric source edges (`generate_gencde_unit_specs`).
  - **N2 — arithmetic recodes**, an LLM formula upgrade of the numeric residuals
    (`prepare_gencde_arith_specgen` / `assemble_gencde_arith_specgen`); arithmetic **always** routes to
    review (never auto-approved).
- **Numeric GenCDE calibration** — `GenCDE.value_coverage` is now `float | None` (numeric domains report
  N/A rather than a vacuous 1.0); numeric confidence rests on the LLM score (penalized without units/bounds),
  and a missing numeric domain trips `needs_review`.
- **Measurand-axis split** (opt-in `measurand_split`, default OFF) — a measurand clause on the split and
  generate-ideal prompts so distinct quantities (e.g. systolic / diastolic / pulse) partition instead of
  fusing. Off by default pending further validation.

New public symbols are exported from `ddharmon.harmonization`; the full harmonization suite is green.

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
- **Analysis ideas** (`ddharmon.harmonization.analysis_ideas`) — `generate_analysis_ideas()` runs one
  grounded LLM pass over a run's cross-cohort concepts (metadata only) to *suggest* downstream analyses a
  harmonization newly enables (association tests, replication/meta-analysis, pooled prevalence, …). Every
  idea is grounded in concepts actually present in the run (hallucinated concepts dropped) and the parser
  salvages a token-cap-truncated response. Proposes hypotheses; never runs them. Helpers:
  `build_concept_digest`, `AnalysisIdea`, `AnalysisIdeasResult`.
- **Bring-your-own-key (BYOK)** — `AnthropicClient` and the batch helpers (`submit_batch`,
  `retrieve_batch`, `submit_and_wait`, `resume_and_wait`) take an optional keyword-only `api_key`; the
  batch helpers also take an optional `base_url` for an on-prem Anthropic-passthrough proxy. Both default to
  `None`, preserving the unchanged `ANTHROPIC_API_KEY`-from-env behavior. A supplied key is scoped to that
  client/call and never written to `os.environ` (no cross-job leak), letting a web backend thread a
  per-request key through to the harmonization run.

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
