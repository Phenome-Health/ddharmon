# ddharmon — Data Dictionary Harmonization Tool

ddharmon harmonizes biomedical **data dictionaries**: it identifies clusters of equivalent
variables across studies and recommends a Common Data Element (CDE) anchor for each — routing
every recommendation to expert review.

![ddharmon in the Phenome Health data-harmonization ecosystem](docs/ph-ecosystem-v1.png)

*Where ddharmon sits: cluster variables + CDEs → anchor a CDE per sub-cluster (or generate a novel
one) → expert review — alongside related public tools and the broader Phenome Health stack.*

## v1 — Sub-cluster-anchored CDE harmonization

**ddharmon v1** clusters cohort variables *and* NIH Common Data Elements (CDEs) together using
**dual vectors** (separate semantic + value-encoding embeddings), **sub-clusters by value
vectors**, **recommends a CDE per sub-cluster**, and emits an **LLM adopt / refine / novel**
recommendation per sub-cluster — routed to expert-in-the-loop (EITL) review.

```
ingest (cohorts + CDE)
  → dual-vector embed (semantic + value)
    → semantic cluster (BERTopic)
      → value sub-cluster (HDBSCAN on value vectors, per topic)
        → CDE anchor per sub-cluster (medoid → best in-cluster CDE; GenCDE fallback)
          → adopt / refine / novel  (single classify-only LLM call)
            → EITL review queue
```

Run it end-to-end in **`notebooks/clustering/v1_harmonization_pipeline.ipynb`**, or call the API:

```python
from ddharmon.embedding import SentenceTransformerProvider, embed_dictionary
from ddharmon.harmonization import harmonize_dictionaries, export_eitl_queue

provider = SentenceTransformerProvider()
embedded = [embed_dictionary(dd, provider=provider) for dd in dictionaries]  # cohorts + NIH_CDE
result = harmonize_dictionaries(embedded, classify=classify_via_batch)        # batch-backed LLM
export_eitl_queue(result, "eitl_queue.tsv")
```

**Lineage.** v1 extends the embedding-clustering-for-variable-harmonization line of work — see
[`docs/v1_methods.md`](docs/v1_methods.md) for methods and citations (Krishnamurthy 2025; Salimi
2025; and related). We cluster the *source* (cohort variables), sub-cluster by value encoding, and
anchor each sub-cluster to a CDE.

**In v1:** multi-cohort + CDE ingestion · dual-vector embedding · BERTopic · value sub-clustering ·
per-sub-cluster CDE anchoring · adopt/refine/novel classify → EITL.
**Deferred (publication-pending):** LLM coherence judging, recursive clustering, LLM spec authoring,
granularity-loss detection, CDE common data model, the pairwise 1:1 matching surface, standards
mapping, and the CLI orchestrator. See [`CHANGELOG.md`](CHANGELOG.md).

## Installation

The core install is lightweight; optional extras unlock additional capabilities.

| Extra | Use case |
|-------|----------|
| *(none)* | Core ingestion + lexical matching |
| `embeddings` | sentence-transformers + faiss-cpu — semantic embedding + vector search |
| `clustering` | scikit-learn, UMAP, plotly |
| `bertopic` | BERTopic topic modeling |
| `llm` | openai, anthropic — LLM classify / rerank |
| `all` | everything above (required to run the full pipeline + notebook) |
| `dev` | `all` + pytest, ruff, black, pyright |

```bash
pip install "ddharmon[all]"          # full pipeline
# or, with uv (recommended for development):
uv sync --extra all
```

## Quick start

Requirements: **Python 3.12+** and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Phenome-Health/ddharmon.git
cd ddharmon
uv sync --extra all
cp .env.example .env        # set ANTHROPIC_API_KEY for the classify pass (sync/batch)
```

Then open `notebooks/clustering/v1_harmonization_pipeline.ipynb`, or use the Python API above.

> The NIH CDE catalog is not bundled. To anchor against CDEs, flatten the CDE repository locally with
> `scripts/flatten_cde_repo.py <All-CDEs.json> <out.tsv>`; without it, the pipeline still clusters
> and sub-clusters cohort variables (`cdeSet = none`).

## Development

```bash
./scripts/check.sh     # lint, format, typecheck, test
./scripts/fix.sh       # auto-fix lint + format
pytest                 # tests
```

### Project structure

```
src/ddharmon/
├── models/          # data models (plain dataclasses)
├── ingestion/       # multi-cohort + CDE dictionary parsers
├── embedding/       # dual-vector embedding (semantic + value), SQLite cache
├── clustering/      # BERTopic semantic clustering + value sub-clustering
├── harmonization/   # v1: CDE anchoring + adopt/refine/novel + EITL export
├── matching/        # pairwise 1:1 matching (built; not in the v1 surface)
├── llm/             # LLM clients + Anthropic Batch API
├── values/          # value-encoding parsing
└── export/          # visualization (dendrograms, UMAP, Plotly)
notebooks/clustering/v1_harmonization_pipeline.ipynb
scripts/             # flatten_cde_repo, build_clsa_csv, prompt runners, check/fix
tests/
```

## Roadmap (beyond v1)

- **Web GUI** — under development; the point-and-click app (upload dictionaries → run with live
  progress → review recommendations → export) will ship as added functionality of **biomapper-ui**.
- Pairwise **1:1 mapping** as a first-class surface (the engine is built).
- **Standards mapping** (LOINC / SNOMED / OMOP).
- LLM coherence judging, concept labeling, and transformation-spec authoring (publication-pending).

## License

MIT — see [LICENSE](LICENSE). © 2026 Phenome Health.
