# ddharmon — Data Dictionary Harmonization Tool

Part of **ARPA-H Activity 2**: Multi-omics data harmonization across Arivale, HPP, UKBB, TwinsUK,
All of Us, and CLSA cohorts.

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
**Publication-pending / deferred:** LLM coherence judging, recursive clustering, LLM spec authoring,
granularity-loss detection, CDE CDM, pairwise 1:1 matching, standards/KG mapping, CLI orchestrator.
See [`CHANGELOG.md`](CHANGELOG.md).

## Roadmap (beyond v1)

The broader tool targets four capabilities; v1 ships the clustering/harmonization core (#2):

1. **1:1 Semantic Mapping** — Match fields between two dictionaries *(built; not in v1 release surface)*
2. **Semantic Clustering + CDE harmonization** — **v1**
3. **Standards Mapping** — Map to LOINC, SNOMED, OMOP *(roadmap)*
4. **Knowledge Graph Integration** — Map to KRAKEN KG via biomapper2 *(roadmap)*

**Strategy**: Develop standalone → validate patterns → contribute to biomapper2 via PR

## Installation

The core install is intentionally lightweight. Optional dependency groups unlock additional capabilities:

| Extra | Packages | Use case |
|-------|----------|----------|
| *(none)* | pandas, numpy, httpx, click, rapidfuzz, openpyxl | Core harmonization, lexical matching |
| `embeddings` | sentence-transformers, faiss-cpu | Semantic embedding generation and vector search |
| `llm` | openai, anthropic | LLM-based re-ranking and annotation |
| `viz` | matplotlib, seaborn, scipy | Visualization and statistical analysis |
| `all` | all of the above + nest-asyncio | Complete feature set (required for notebooks) |
| `notebooks` | `all` + Jupyter, JupyterLab, plotly | Full notebook environment |
| `dev` | `all` + pytest, ruff, black, pyright | Development and testing |

```bash
pip install ddharmon                    # core only
pip install "ddharmon[embeddings]"      # + semantic embeddings
pip install "ddharmon[llm]"             # + LLM re-ranking
pip install "ddharmon[all]"             # everything

# With uv (recommended for development)
uv sync                     # core only
uv sync --extra embeddings  # + semantic embeddings
uv sync --extra all         # everything
uv sync --extra notebooks   # full notebook environment
```

> **Note**: `notebooks/00_setup_validation.ipynb` imports `sentence_transformers` and `faiss`.
> Run `uv sync --extra notebooks` (or `--extra all`) before opening notebooks.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- biomapper2 repository (see setup below)

### Setup

1. **Clone this repository**:
   ```bash
   cd ~/Insync/projects/
   git clone git@github.com:Phenome-Health/ph-arpa-data-harmonization.git
   cd ph-arpa-data-harmonization
   ```

2. **Install dependencies**:
   ```bash
   # Core install (pandas, numpy, httpx, click, fuzzy matching)
   uv sync

   # For embedding generation and vector search (sentence-transformers, faiss)
   uv sync --extra embeddings

   # For LLM-based re-ranking (openai, anthropic)
   uv sync --extra llm

   # For visualization and stats (matplotlib, seaborn, scipy)
   uv sync --extra viz

   # All optional deps (required for notebooks)
   uv sync --extra all

   # Full notebook environment (Jupyter + all optional deps)
   uv sync --extra notebooks
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (Biomapper2, Kestrel, OpenAI, etc.)
   ```

4. **Verify setup**:
   ```bash
   ./scripts/check.sh
   # Or run the setup validation notebook (requires --extra notebooks)
   uv sync --extra notebooks
   jupyter lab notebooks/00_setup_validation.ipynb
   ```

### Architecture Note

ddharmon uses the **Biomapper2 API** (not the library) for entity resolution. This keeps concerns separated:
- **ddharmon**: Data dictionary harmonization
- **Biomapper2 API**: Entity resolution service

When we identify patterns that should live in biomapper2, we'll contribute them via PR to that repo.

## Development

### Running checks

```bash
# All checks (lint, format, typecheck, test)
./scripts/check.sh

# Auto-fix issues
./scripts/fix.sh

# Individual tools
ruff check src/
black --check src/ tests/
pyright
pytest
```

### Project Structure

```
ph-arpa-data-harmonization/
├── pyproject.toml              # Package config
├── .env.example                # API credentials template
├── CLAUDE.md                   # Claude Code conventions
├── PLAN.md                     # Architecture & implementation plan
├── notebooks/
│   ├── 00_setup_validation.ipynb
│   ├── demographics/           # Demographics harmonization notebooks
│   └── questionnaires/         # Questionnaire harmonization notebooks
├── src/ddharmon/
│   ├── models/                 # Data models (dataclasses)
│   ├── ingestion/              # Data dictionary parsers (multi-cohort + CDE)
│   ├── embedding/              # Dual-vector embedding layer (semantic + value), SQLite cache
│   ├── clustering/             # BERTopic semantic clustering + value sub-clustering
│   ├── harmonization/          # v1: CDE anchoring + adopt/refine/novel + EITL export
│   ├── matching/               # Pairwise 1:1 matching (built; roadmap for release)
│   ├── llm/                    # LLM clients + Anthropic Batch API
│   ├── values/                 # Value-encoding parsing
│   └── export/                 # Visualization (dendrograms, UMAP, Plotly)
│   # roadmap: cli.py, harmonizer.py, config.py, standards/, kg/, review/
├── data/
│   ├── examples/               # Sample data dictionaries
│   └── review/                 # Review artifacts (JSON, TSV)
├── scripts/
│   ├── check.sh                # Run all checks
│   └── fix.sh                  # Auto-fix linting
└── tests/                      # pytest tests
```

### Collaboration Workflow

This project uses two Claude Code instances working in parallel:

1. **Branch** from `plan/ddharmon-architecture`
2. **Develop** in notebooks first, then extract to `src/`
3. **PR** with Greptile + manual review

See `CLAUDE.md` for detailed conventions (naming, APIs, review artifacts).

## CLI Usage (roadmap)

> The Click CLI below is the **roadmap** target; v1 ships as a Python API + the canonical
> notebook (`notebooks/clustering/v1_harmonization_pipeline.ipynb`). `ddharmon.cli:main` is
> declared but not implemented in v1.

```bash
# Load a data dictionary
ddharmon load my_dict path/to/dictionary.csv

# Map between two dictionaries
ddharmon map source_dict target_dict --output mappings.json

# Cluster fields across all loaded dictionaries
ddharmon cluster --output clusters.json --visualize dendrogram.png

# Map to standards
ddharmon standards my_dict --systems loinc,snomed --output standards.json

# Map to KRAKEN KG
ddharmon kg my_dict --output kg_mappings.json

# Review pending mappings
ddharmon review list
ddharmon review approve <mapping_id> --reviewer "Your Name"
```

## Related Projects

- [biomapper2](https://github.com/Phenome-Health/biomapper2) — Entity annotation/mapping library
- [biovector-eval](https://github.com/Phenome-Health/biovector-eval) — Reference implementations for harmonization

## License

MIT — see [LICENSE](LICENSE). © 2026 Phenome Health.

> Note: third-party datasets under `data/benchmarks/raw/` retain their own upstream
> licenses (see `data/benchmarks/LICENSE_NOTES.md`) and are not covered by the MIT grant.
