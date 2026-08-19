# Changelog

All notable changes to ddharmon are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [1.3.0]

Adds the **coherence judge** as a public capability, moves its verdicts ahead of assignment, and adds a
named **resumable stage boundary** so a caller can stop a run at a decided point and resume it against the
identical clustering partition. Both additions are opt-in and default OFF.

This is also the first release published to PyPI since 1.1.0, so it delivers everything recorded under
1.2.0 (realized cost accounting) plus the composite/derived-variable builder and score-source ingestion
below, which had accumulated unreleased.

### Added

- **The coherence judge.** `harmonize_leanb` can now be given a `coherence` stage callable. The judge
  examines each post-split concept group and records whether the group holds a single concept or has pooled
  more than one, together with a short summary, the axis along which the group varies, the distinct values
  it saw and any members it considered outliers.

  The judge **flags; it does not act.** A group judged over-merged is marked, and that mark travels with the
  record so a downstream reviewer can see it. Nothing is re-split, re-routed or re-assigned on the judge's
  word, and the judge cannot reach a group's route or its assignment verdict — the type it is given exposes
  neither.

  A group the judge could not evaluate is recorded as **unjudged**, never as coherent. A malformed or missing
  response degrades to unjudged rather than unwinding the run.

  Public names, all exported from `ddharmon.harmonization`:

  | Name | What it is |
  | --- | --- |
  | `assemble_coherence_verdicts` | the verdict pass — folds judge responses onto the groups |
  | `transfer_coherence_verdicts` | carries those verdicts onto the assembled records |
  | `propagate_coherence_review` | marks the affected transforms and generated CDEs for review |
  | `assemble_coherence` | the combined entry point — verdicts, then propagation |
  | `ConceptGroup` | the post-split, pre-assignment group shape |
  | `CoherenceTarget` | the structural protocol both group and record satisfy |
  | `concept_groups_from_prompts` | builds the group shape from prepared assignment prompts |
  | `prepare_coherence` | builds the judge prompts ($0 — no LLM call) |
  | `prepare_kinds` / `assemble_kinds` | the opt-in second read that widens the flag rule |
  | `prepare_readjudicate` / `readjudicate` | an opt-in second look at a flagged group, driven by the caller |

  `LeanBResult` gains `concept_groups`, so the groups and their verdicts are available at the point where
  assignment has not yet run. `LeanBRecord` and `ConceptGroup` carry the verdict fields themselves
  (`coherent`, `coherence_verdict`, `coherence_summary`, `coherence_axis`, `coherence_distinct_values`,
  `coherence_outliers`, `coherence_kind`, `incoherent`, `matrix_suspect`), and `LeanBRecord` also carries
  `readjudicated_from` for a record carved out of a re-adjudicated parent group.

- **A named resumable stage boundary.** `harmonize_leanb` accepts a new argument:

  ```python
  harmonize_leanb(..., stop_after="gencde")
  ```

  The named stage runs to completion and the pipeline then stops, returning a partial result that carries
  the clustering substrate along with the records produced so far and the prompts prepared for the stages
  that did not run. Because the substrate is on the result, a later call can replay the exact same partition
  rather than re-deriving one.

  **The accepted vocabulary is exactly one name.**

  ```python
  STOP_AFTER_BOUNDARIES == ("gencde",)
  ```

  Any other non-`None` value raises `ValueError` naming the accepted set, and it raises before any stage
  runs — a mistyped boundary cannot silently execute a whole billable pipeline past the requested stop.
  Callers that want to feature-detect the argument can look for `stop_after` in
  `inspect.signature(harmonize_leanb).parameters`.

  Do not assume any other boundary name exists. One more can be added later without breaking any caller, so
  the vocabulary starts at the single name that is actually needed.

- **`ddharmon.text_hygiene`** — one lightweight, cohort-agnostic home (standard library only) for the
  source-data artifacts that pollute what the model and the embeddings see: instrument-administration
  wrappers and help-message markup (`clean_field_text`), missing/refused/don't-know sentinel response codes
  (`is_sentinel_label`, `strip_sentinel_encodings`), and the generic survey/CDE instruction boilerplate list
  `CDE_TEXT_BOILERPLATE`, which previously lived inside the harmonization module and is re-exported from
  there unchanged.

- **Export selection** (`ddharmon.harmonization.selection`) — `select_records`, `list_export_concepts`,
  `concepts_matching` and the `ExportConcept` row shape let a caller scope an export to the concepts a
  reviewer marked. `export_transform_review` takes a matching `select=` argument; the default (`None`)
  exports every concept, exactly as before.

- **Opt-in prompt levers**, each default OFF and individually switchable so an A/B can attribute them
  separately: `prepare_leanb(debias_ideal=...)` (a generate-ideal variant that drops the one-concept
  presumption and enumerates distinct measurands), `prepare_split(bundle_guard=...)` /
  `prepare_group_assign(bundle_guard=...)` (a candidate that bundles several measured quantities does not
  license a shared adopt), and `prepare_split(enforce_schema=...)` (issue the split as a forced tool call so
  the `{groups: [...]}` wrapper is structurally guaranteed instead of instructed).

- **`PromptRecord` gains `tool_schema`, `tool_name` and `max_tokens`**, emitted into
  `to_jsonl_record()` only when set. They carry the forced-tool-call request through to whichever driver
  submits the prompts. **The batch submitter bundled in this release does not read them yet** — a driver
  that ignores them degrades to the previous soft-schema behaviour rather than failing, and a record that
  sets none of them serializes byte-for-byte as before.

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

### Changed

- **The judge now evaluates groups before assignment, not after.** Verdicts are stamped immediately after
  the split stage and before assignment begins; the review marks they imply are applied later, after
  transform specification and generated-CDE synthesis.

  One consequence is worth stating plainly for anyone comparing runs: the judge now sees groups **as the
  split stage produced them**, before the cross-record merge reunites members that were separated. A re-run
  therefore scores a slightly different population of groups than a run judged under the previous ordering.
  This is the intended ordering — the judge's calibration was measured at post-split group granularity,
  which is now what it is given — but two runs across the change are not directly comparable
  group-for-group.

- **The review-mark pass runs unconditionally.** It does nothing when no group was flagged, and running it
  twice has the same effect as running it once, so replaying a stage is safe.

- **Ingestion strips administrative text by default.** `preprocess_dictionary` gains a step that removes
  instrument-administration wrappers, trailing help-message markup, HTML tags/entities and survey
  boilerplate from `description` and `question_text`, reported as `admin_text_stripped`. It never blanks a
  field: a cleaned value is applied only when it is a non-empty change. This is the one change here that is
  ON by default and alters preprocessed text — pass `strip_administrative_text=False` for the previous
  behaviour.

- **Concept-identity prompts drop sentinel response codes.** The value set rendered into the
  generate-ideal / split / assign / judge prompts now omits missing/refused/don't-know codes, so a numeric
  field encoded only as `-9=MISSING` reads as numeric rather than as a single-option categorical. Transform
  specification still sees the full value set, because it needs those codes. Set
  `DDHARMON_PROMPT_HYGIENE=0` to reproduce the previous prompt text.

- **The review-queue export gains four columns** — `incoherent`, `coherence_verdict`, `coherence_axis` and
  `matrix_suspect` — so a consumer that pins the queue's column set needs updating.

### Compatibility

The new stages and arguments are additive. `coherence`, `distinct_kinds` and `stop_after` all default to
`None`, and no existing argument changed name, position or meaning — `max_clusters`, the cost cap the
bundled CLI passes, is unchanged and now has a regression test of its own. No stage was promoted to a
public function: `harmonize_leanb` remains the single owner of stage sequencing.

Two defaults do change what a run produces without any flag being set: ingestion's administrative-text
strip and the sentinel-code drop in the concept-identity prompts, both listed under **Changed** above with
the switch that restores the previous behaviour.

## [1.2.0]

Adds **realized cost accounting** — report what a run *actually* cost from the real token usage each LLM
call returns, rather than estimating from a hardcoded per-token table. Additive and backward-compatible over
1.1.0: existing callers are unaffected; the cost is captured automatically and exposed for callers that want
it.

### Added

- **`ddharmon.llm.cost`** — the source of truth for run cost:
  - `TokenUsage`, and `price_usage(model, input_tokens, output_tokens, *, batch=False)` — prices captured
    tokens against **LiteLLM's maintained model→price map** when `litellm` is installed, otherwise against a
    small built-in Anthropic/OpenAI rate table (so cost works without the optional dependency). The Anthropic
    Batch API's 50% discount is applied; an unpriceable model reports `$0` rather than guessing.
  - `CostLedger` — accumulates realized cost + token totals per pipeline stage and per run.
  - Exported from `ddharmon.llm` (`CostLedger`, `TokenUsage`, `price_usage`).
- **Token-usage capture on the clients** — `AnthropicClient` / `OpenAIClient` accumulate each call's real
  input/output token usage into a `usage_log`, drained per stage via the new `BaseLLMClient.drain_usage()`.
- **Batch usage preserved** — `retrieve_batch` now records each succeeded response's realized token usage and
  the model that ran, in the responses JSONL (previously discarded), so the Batch path can be priced too.
  Backward-compatible: response files written before this — and any existing reader — ignore the added keys.

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
