# Example data dictionaries

These are **publicly-available** data dictionaries bundled so the harmonization pipeline runs
end-to-end on a fresh clone — via the v2 notebook
(`notebooks/clustering/v2_harmonization_pipeline.ipynb`) or the CLI
(`ddharmon harmonize --config data/examples/harmonize_example.json`). Each ingested CSV is
reproducible from its public source via a script in `scripts/` — the source workbook is included
alongside it for full transparency.

These are **metadata only** — field names, descriptions, and value codings (data
*dictionaries*), never participant-level data — drawn from each project's openly published,
no-login data catalog and used here for non-commercial research. The build scripts below
document exactly how each file was constructed from its public source.

## Provenance

| Cohort | Ingested file | Public source | Build script |
|--------|---------------|---------------|--------------|
| **All of Us** | `all_of_us_surveys.csv` | [All of Us Survey Data Codebooks](https://docs.google.com/spreadsheets/d/1pODkE2bFN-kmVtYp89rtrJg7oXck4Fsex58237x47mA/edit) → `all_of_us_survey_codebooks.xlsx` | `scripts/build_all_of_us_csv.py` |
| **CLSA** | `clsa_baseline.csv` | [CLSA Data Dictionaries](https://www.clsa-elcv.ca/resource-types/data-dictionaries/) → `clsa_baseline.xlsx` | `scripts/build_clsa_csv.py` |
| **UK Biobank** | `ukbb_showcase.csv` | [UKB Showcase Schema](https://biobank.ndph.ox.ac.uk/showcase/schema.cgi) (public field + encoding metadata, no login) | `scripts/build_ukbb_csv.py` |
| **AI-READI** | `aireadi_surveys.csv` | [AI-READI/DataElementMaps](https://github.com/AI-READI/DataElementMaps) (MIT — REDCap survey → OMOP/CDE value-set mappings) | `scripts/build_aireadi_csv.py` |
| **MESA** | `mesa_dbgap.csv` | [dbGaP phs000209](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000209) public variable summaries (`data_dict.xml`, no login) | `scripts/build_dbgap_csv.py` |
| **NIH CDEs** | `all_cdes_flat.tsv` | [NIH CDE Repository](https://cde.nlm.nih.gov/) (repo JSON export) | `scripts/flatten_cde_repo.py` |

## Reproduce

The bundled CSVs/TSV are exactly the output of these commands (run from the repo root):

```bash
# All of Us — concatenate the per-survey codebook tabs into one flat CSV
python scripts/build_all_of_us_csv.py

# CLSA — join the Variables + Categories sheets into code=label|... value encodings
python scripts/build_clsa_csv.py

# UK Biobank — download the public Showcase schema files and join field + encoding metadata.
# Default scope = survey/questionnaire/demographic + physical-measure fields (~2.7k), aligned with
# the other survey-scoped example cohorts. UKBB's full catalog (~11.8k) is mostly imaging/omics/genomics/EHR.
python scripts/build_ukbb_csv.py
python scripts/build_ukbb_csv.py --all-fields        # full ~11.8k-field catalog instead
python scripts/build_ukbb_csv.py --list-categories   # discover category ids for a custom --include-category-subtree

# AI-READI — join the public REDCap→OMOP/CDE value-set mapping into one flat survey CSV.
# Each field also carries its mapped OMOP/CDE concept (a gold anchor for CDE-assignment).
python scripts/build_aireadi_csv.py

# MESA — read dbGaP's PUBLIC data_dict.xml variable summaries (data is controlled; dictionaries aren't).
# Clinical/exam/questionnaire core; --exclude-datasets CT drops the imaging-derived CT metrics (~54%).
python scripts/build_dbgap_csv.py --exclude-datasets CT

# FHS — the SAME generic dbGaP builder handles Framingham (and any dbGaP study). NOT bundled: the full
# pull is ~93k variables / 14 MB — too large for a demo example (no natural scope axis to trim). Regenerate:
python scripts/build_dbgap_csv.py --release phs000007.v35.p16 --label fhs   # -> gitignored data/examples/fhs_dbgap.csv

# NIH CDEs — flatten the CDE Repository JSON export to a TSV
python scripts/flatten_cde_repo.py <cde_repo.json> data/examples/all_cdes_flat.tsv
```

The `*.xlsx` source workbooks for All of Us and CLSA are committed here. UK Biobank's source is
the live public Showcase Schema (`scdown.cgi?fmt=txt&id=N`); `build_ukbb_csv.py` downloads it to
a gitignored `.ukbb_schema_cache/` (~30 MB) rather than bundling the raw TSVs. AI-READI's source
is the MIT-licensed `DataElementMaps` repo; `build_aireadi_csv.py` downloads the single mapping
CSV to a gitignored `.aireadi_cache/`. MESA's source is dbGaP's public `data_dict.xml` variable
summaries (the participant *data* is controlled, but the *dictionaries* are open); the XML files
download to a gitignored `.dbgap_cache/`. The same `build_dbgap_csv.py` handles any dbGaP study —
**FHS** is verified but not bundled (its ~93k-variable / 14 MB dictionary is too large for a demo;
regenerate with the command above). The NIH CDE Repository JSON export is not bundled (it is
large and regenerated from the public repository); the flattener accepts any CDE Repository JSON.

## Bring your own

To run the pipeline on your own cohort, drop a CSV/TSV in this directory and add a loader entry
in step 1 of the notebook, mapping your columns to `load_dictionary`'s named kwargs
(`variable_name=`, `description=`, `value_encoding=`, …). No source workbook or build script is
required — those are only here to document how the bundled examples were produced.

**Access-gated cohorts (builder ships, data doesn't).** Some dictionaries are governed by a Data
Use Agreement and can't be redistributed here. For those we ship the *builder* but not the output.
Example: **PPMI** (Parkinson's Progression Markers Initiative) — `scripts/build_ppmi_csv.py` joins the
PPMI `Data_Dictionary` + `Code_List` files you download under your own DUA (from
[the PPMI portal](https://www.ppmi-info.org/access-data-specimens/download-data)) into the same
`code=label|…` shape. Its output lands in a gitignored `data/ppmi/` and is never committed.
