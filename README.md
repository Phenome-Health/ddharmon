# ddharmon — Data Dictionary Harmonization Tool

ddharmon harmonizes biomedical **data dictionaries**. It clusters equivalent variables across many
studies, assigns each concept to an existing **Common Data Element (CDE)** from the NIH catalog — or
flags it for a newly generated CDE when nothing fits — and routes every recommendation to expert review.

![ddharmon in the Phenome Health data-harmonization ecosystem](docs/ph-ecosystem-v1.png)

*Where ddharmon sits: cluster equivalent variables across studies → assign each to an NIH CDE (or generate
a novel one) → expert review — alongside related public tools and the broader Phenome Health stack.*

## Quick start

Requirements: **Python 3.12+** and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Phenome-Health/ddharmon.git
cd ddharmon
uv sync --extra all
cp .env.example .env        # set ANTHROPIC_API_KEY for the LLM passes (sync / batch)
```

Then open **`notebooks/clustering/v2_harmonization_pipeline.ipynb`** — it runs the full pipeline end to
end on bundled example data (All of Us + CLSA dictionaries against the NIH CDE catalog) and is the best
place to start. A `MAX_CLUSTERS` cap keeps the demo cheap; the embedding and clustering stages run with no
API key.

> The bundled `data/examples/all_cdes_flat.tsv` is a flattened snapshot of the NIH CDE catalog. To refresh
> it or use your own, run `scripts/flatten_cde_repo.py <All-CDEs.json> <out.tsv>`. Without a CDE catalog,
> ddharmon still clusters your variables — it just won't assign CDEs.

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
parameters, and design rationale are in [`docs/v2_methods.md`](docs/v2_methods.md).

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
notebooks/clustering/v2_harmonization_pipeline.ipynb   # end-to-end entry point
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

- **Web GUI** — a point-and-click app (upload dictionaries → run with live progress → review → export),
  shipping as part of **biomapper-ui**.
- **Pairwise 1:1 mapping** as a first-class surface (the engine is built).
- **Standards mapping** (LOINC / SNOMED / OMOP).
- **Value-recode / transform-spec authoring**, with adopt/refine/novel thresholds calibrated from expert
  review verdicts.

## License

MIT — see [LICENSE](LICENSE). © 2026 Phenome Health.
