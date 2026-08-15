# ddharmon — Data Dictionary Harmonization Tool

> ⚠️ **Actively under development and internal review.** ddharmon is under active development and
> its methods, outputs, and reported figures are undergoing internal review. Interfaces and results
> may change without notice, and its harmonization recommendations are **not yet validated** — every
> recommendation is meant to be checked by a domain expert before it is acted on. Treat anything the
> tool produces as provisional. Feedback and issues are welcome.

ddharmon harmonizes biomedical **data dictionaries**. It clusters equivalent variables across many
studies, assigns each concept to an existing **Common Data Element (CDE)** from the NIH catalog — or
flags it for a newly generated CDE when nothing fits — and routes every recommendation to expert review.

**Try it in your browser — no install — at [ddharmon.io](https://ddharmon.io).**

Prefer to run it yourself? ddharmon is a Python package: reach for the **hosted app** above for a quick look,
the **`ddharmon` CLI** for scriptable and batch/headless runs, and the **Python library** (see the
[example notebook](notebooks/clustering/harmonization_pipeline.ipynb)) to build it into your own pipelines.
It reads data-dictionary *metadata only* — never participant-level data — so restricted cohorts can run
entirely within their own infrastructure.

## Quick start

Requirements: **Python 3.12+** and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Phenome-Health/ddharmon.git
cd ddharmon
uv sync --extra all
cp .env.example .env        # set ANTHROPIC_API_KEY for the LLM passes (sync / batch)
```

Then open **`notebooks/clustering/harmonization_pipeline.ipynb`** — it runs the full pipeline end to
end on bundled example data (All of Us + CLSA dictionaries against the NIH CDE catalog) and is the best
place to start. A `MAX_CLUSTERS` cap keeps the demo cheap; the embedding and clustering stages run with no
API key.

> The bundled `data/examples/all_cdes_flat.tsv` is a flattened snapshot of the NIH CDE catalog. To refresh
> it or use your own, run `scripts/flatten_cde_repo.py <All-CDEs.json> <out.tsv>`. Without a CDE catalog,
> ddharmon still clusters your variables — it just won't assign CDEs.

## Command-line usage

Besides the notebook, ddharmon ships two subcommands. Both take dictionaries inline as `NAME=path` or via
a JSON `--config`.

**`ddharmon harmonize`** — the full split-aware pipeline (embed → cluster → retrieve → generate-ideal →
split → assign/route → transform specs → EITL campaign):

```bash
# harmonize two cohorts against a CDE backbone
ddharmon harmonize AoU=all_of_us_surveys.csv CLSA=clsa_baseline.csv --cde all_cdes_flat.tsv -o out/

# reproduce the bundled example (All of Us + CLSA vs the NIH CDE catalog)
ddharmon harmonize --config data/examples/harmonize_example.json -o out/

# $0 dry run: build the generate-ideal prompts without calling the LLM
ddharmon harmonize --config data/examples/harmonize_example.json --dry-run -o out/
```

Set `ANTHROPIC_API_KEY` for the full run; without it (or with `--dry-run`) ddharmon writes the stage-1
prompts for later batch submission. Outputs a `records.json` plus an `eitl/` expert-review campaign. Cost
knobs: `--max-clusters N` (harmonize the N largest clusters first), `--top-k`, `--min-cluster-size`.

**`ddharmon cluster`** — cluster equivalent variables only ($0, no LLM):

```bash
ddharmon cluster AoU=all_of_us_surveys.csv CLSA=clsa_baseline.csv -o clusters.json
```

Columns are auto-detected; override them per dictionary in the `--config` `columns` block (keys are
`load_dictionary`'s kwargs).

## How it works

ddharmon leads with **assignment to the existing CDE backbone** for concepts that are already covered, and
only sends the *uncovered tail* to generation/clustering:

```
ingest (study dictionaries + NIH CDE catalog)
  → embed variables (BioLORD semantic vectors, SQLite-cached)
    → cluster equivalent variables (coarse semantic clusters)
      → per cluster: retrieve candidate CDEs (hybrid: BM25 + dense, fused with RRF)
        → generate an "ideal" CDE (an independent coverage anchor)
          → split the cluster into distinct concept-groups
            → per group: rank the candidates and decide adopt / refine / novel
              → route:  adopt / refine → assign that CDE
                        novel          → generate a CDE / re-cluster the tail
                → expert-in-the-loop (EITL) review queue
```

Nothing is auto-applied — every adopt/refine/novel recommendation lands in a review queue. Full algorithm,
parameters, and design rationale are in [`docs/methods.md`](docs/methods.md).

## What ddharmon adds

Most tools cluster *either* a CDE repository *or* a single cohort, use a single semantic vector, and map
element-by-element. ddharmon targets harmonizing **many studies at once**:

- **Assignment-first, across many studies.** Variables from many study dictionaries and the NIH CDE
  catalog are embedded together, so equivalent variables surface as one cluster and each concept is
  assigned to a real CDE — not one pairwise lookup at a time.
- **Hybrid retrieval.** Candidate CDEs come from a fusion (reciprocal rank fusion) of lexical **BM25** and
  dense similarity using **BioLORD-2023**, a concept↔definition biomedical encoder — stronger candidates
  than dense-only retrieval.
- **Split-aware concept resolution.** A coarse cluster that happens to pool more than one concept is split
  into distinct concept-groups, so each gets its own CDE decision — distinct concepts are never silently
  merged onto a single CDE.
- **A novel-CDE path.** When no existing CDE fits, the concept is routed to a generated (GenCDE) candidate
  and/or re-clustered with the rest of the uncovered tail, instead of being forced onto a poor match.
- **adopt / refine / novel → expert review.** Every recommendation is routed to expert-in-the-loop (EITL)
  review; nothing is applied automatically.
- **Reproducible evaluation.** Standing, $0, no-proprietary-data benchmarks (see [`benchmarks/`](benchmarks))
  gate retrieval and clustering quality on every change.

## Limitations & measured quality

ddharmon is a **data-*dictionary* harmonizer**: it works on metadata (field names, descriptions, value
codings) from public catalogs, never participant-level data. It **proposes; humans decide** — every
recommendation carries a confidence and verdict and is routed to expert-in-the-loop review. It **emits**
transform specs (including parameterized specs for data-dependent transforms) but does not execute them on
data. And it scores **cross-dictionary harmonization** (pooling equivalent fields across ≥2 dictionaries)
separately from **single-dictionary CDE assignment** (the CDEMapper / DIVER task).

**Measured quality** — BioLORD-2023 encoder + hybrid BM25⊕dense retrieval + split-aware LLM assignment.
These are current baselines, not ceilings; dev-vs-held-out is labelled, and dev numbers (measured on the
set we tuned against) should be read as optimistic:

| Signal | Number | Gold set | Held-out? |
|--------|--------|----------|-----------|
| Candidate retrieval recall@5 / @100 | 0.674 / 0.926 | CDEMapper | dev (tuned on it) |
| In-backbone CDE assignment | 0.521 | CDEMapper | dev |
| Cross-cohort separability Δ | 0.611 | PhenX | held-out |
| Variable → concept recall@5 / @100 | 0.655 / 0.914 | AI-READI OMOP/CDE | held-out |
| Value-recode pair accuracy | 0.869 | ATHLOS | held-out |
| N1 unit-conversion accuracy | 1.00 | curated UCUM gold | deterministic |

On a full public 5-cohort run (18,128 fields), 97.5% of fields reach a record, 60.7% land in a
real-concept group, and 41.9% of records are assigned to a CDE — a coverage snapshot of what a run
produces, not a gold-accuracy score.

**Known limitations:**

- **We do not outperform dedicated CDE mappers on retrieval recall, and don't claim to.** ddharmon is a
  convergent method with extended scope (cross-dictionary pooling + transform-spec generation). CDEMapper is
  also our tuning set, so its numbers are optimistic; PhenX and AI-READI are the held-out generalization checks.
- **The adopt / refine / novel router is conservative and uncalibrated, by design** — a strict cutoff plus a
  0.30 retrieval floor. Expect over-routing to review (novel-precision ≈ 0.5): a recall-favoring safety bias.
  Calibrating the thresholds against human review verdicts is a planned follow-up, not part of this release.
- **Numeric transforms are partial by construction:** unit (N1) and arithmetic (N2) specs are authored and
  gold-gated; data-dependent transforms (quantile binning, normalization) are *detected* and emitted as a
  *parameterized* spec routed to review — not authored with final values.
- **Pairwise 1:1 mapping** is not yet a first-class surface (the engine exists).
- **Clustering is not bit-reproducible** across machines (UMAP / HDBSCAN); reproducibility comes from freezing
  the clustering substrate and caching the LLM responses.

## Related work

Framing biomedical variable/CDE harmonization as an **embedding → clustering → LLM** problem is an active
line of work; ddharmon builds directly on it:

- **CDEMapper** (Wang et al., *JAMIA* 2025) — LLM-powered mapping of local data elements to NIH CDEs via
  semantic indexing + BM25 + GPT candidates + human review. *Per-element lookup.*
- **DataTecnica — DIVER** (Long et al., [*npj Digital Medicine* 2026](https://www.nature.com/articles/s41746-026-02795-z))
  — an AI-assisted pipeline that generates and audits Common Data Elements at scale (GenCDEs) to align data
  standards. *Closest to generating novel CDEs.*
- **Krishnamurthy et al., 2025** (arXiv:2506.02160) — embeds ~24k NIH CDEs and clusters them with HDBSCAN,
  then LLM-labels the clusters. *Clusters the target CDE repository.*
- **PASSIONATE** (Salimi et al., *Sci Rep* 2025) — a Parkinson's variable-mapping ground truth; shows
  language-model embeddings beat fuzzy string matching for pairwise cohort harmonization.
- **Harmony** (McElroy et al., *BMC Psychiatry* 2024), **Semantic Search Helper** (Gottfried 2025),
  **datastew**, and **BDI-Kit** (Lopez et al., *Patterns* 2026) — embedding/LLM harmonization and
  schema/value-matching siblings.

## Installation

The core install is lightweight; optional extras unlock additional capabilities.

| Extra | Adds |
|-------|------|
| *(none)* | Core ingestion + lexical matching |
| `embeddings` | sentence-transformers + faiss-cpu (semantic embedding + vector search) |
| `llm` | openai + anthropic (LLM assignment / reranking) |
| `clustering` | scikit-learn, UMAP, plotly |
| `viz` | matplotlib, seaborn, scipy |
| `bertopic` | BERTopic topic modeling |
| `all` | everything above — required to run the full pipeline + notebook |
| `notebooks` | `all` + Jupyter |
| `dev` | `all` + pytest, ruff, black, pyright |

```bash
pip install "ddharmon[all]"          # full pipeline
# or, for development:
uv sync --extra dev
```

The package also installs a `ddharmon` console command (`ddharmon --help`); the notebook is the primary
end-to-end entry point.

## Development

```bash
./scripts/check.sh     # lint, format, typecheck, test
./scripts/fix.sh       # auto-fix lint + format
pytest                 # tests
```

### Project structure

```
src/ddharmon/
├── models/         # data models (dataclasses)
├── ingestion/      # study + CDE dictionary parsers, preprocessing
├── embedding/      # BioLORD semantic embedding, SQLite cache
├── clustering/     # semantic clustering (BERTopic / HDBSCAN) + residual re-clustering
├── matching/       # hybrid retrieval (BM25 + dense, RRF) + pairwise matching
├── harmonization/  # split-aware assignment → adopt/refine/novel + EITL export
├── llm/            # LLM clients + Anthropic Batch API
├── values/         # value-encoding parsing
└── export/         # EITL campaigns + visualization
benchmarks/         # reproducible $0 eval gates (CDEMapper, PhenX, ATHLOS) + gate.py
notebooks/clustering/harmonization_pipeline.ipynb   # end-to-end entry point
data/examples/      # bundled dictionaries (All of Us, CLSA) + flattened NIH CDE catalog
scripts/            # data builders, prompt runners, release verification, check/fix
tests/
```

### Benchmarks

`benchmarks/` runs standing, reproducible, $0 evaluations on public data — CDEMapper (variable→CDE
retrieval), PhenX (cross-cohort co-clustering), and ATHLOS (value-recode correctness) — with
`benchmarks/gate.py` asserting hard floors on the deterministic signals. See
[`benchmarks/README.md`](benchmarks/README.md).

## Roadmap

- **Pairwise 1:1 mapping** as a first-class surface (the engine is built).
- **Standards mapping** (LOINC / SNOMED / OMOP).
- **Calibrated adopt/refine/novel thresholds** — tuned from expert-review verdicts (the router ships
  conservative today; transform-spec authoring already generates categorical + unit/arithmetic specs).

## License

MIT — see [LICENSE](LICENSE). © 2026 Phenome Health.
