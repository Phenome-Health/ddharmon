# Example data dictionaries

These are **publicly-available** data dictionaries bundled so the v1 harmonization pipeline
notebook (`notebooks/clustering/v1_harmonization_pipeline.ipynb`) runs end-to-end on a fresh
clone. Each ingested CSV is reproducible from its public source via a script in `scripts/` —
the source workbook is included alongside it for full transparency.

## Provenance

| Cohort | Ingested file | Public source | Build script |
|--------|---------------|---------------|--------------|
| **All of Us** | `all_of_us_surveys.csv` | [All of Us Survey Data Codebooks](https://docs.google.com/spreadsheets/d/1pODkE2bFN-kmVtYp89rtrJg7oXck4Fsex58237x47mA/edit) → `all_of_us_survey_codebooks.xlsx` | `scripts/build_all_of_us_csv.py` |
| **CLSA** | `clsa_baseline.csv` | [CLSA Data Dictionaries](https://www.clsa-elcv.ca/resource-types/data-dictionaries/) → `clsa_baseline.xlsx` | `scripts/build_clsa_csv.py` |
| **NIH CDEs** | `all_cdes_flat.tsv` | [NIH CDE Repository](https://cde.nlm.nih.gov/) (repo JSON export) | `scripts/flatten_cde_repo.py` |

## Reproduce

The bundled CSVs/TSV are exactly the output of these commands (run from the repo root):

```bash
# All of Us — concatenate the per-survey codebook tabs into one flat CSV
python scripts/build_all_of_us_csv.py

# CLSA — join the Variables + Categories sheets into code=label|... value encodings
python scripts/build_clsa_csv.py

# NIH CDEs — flatten the CDE Repository JSON export to a TSV
python scripts/flatten_cde_repo.py <cde_repo.json> data/examples/all_cdes_flat.tsv
```

The `*.xlsx` source workbooks for All of Us and CLSA are committed here. The NIH CDE Repository
JSON export is not bundled (it is large and regenerated from the public repository); the
flattener accepts any CDE Repository JSON.

## Bring your own

To run the pipeline on your own cohort, drop a CSV/TSV in this directory and add a loader entry
in step 1 of the notebook, mapping your columns to `load_dictionary`'s named kwargs
(`variable_name=`, `description=`, `value_encoding=`, …). No source workbook or build script is
required — those are only here to document how the bundled examples were produced.
