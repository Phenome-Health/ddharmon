"""Build the All of Us flat survey CSV from the public multi-sheet codebook workbook.

Source (public): the *All of Us Research Program* Survey Data Codebooks, published as a
Google Sheet and downloadable as `.xlsx`:
    https://docs.google.com/spreadsheets/d/1pODkE2bFN-kmVtYp89rtrJg7oXck4Fsex58237x47mA/edit

The workbook has one `ReadMe` tab plus one tab per survey (Basics, Lifestyle, Overall
Health, ...). Every survey tab is a REDCap-style codebook with the same column shape:
`Item Concept | Form Name | Section Header | Field Type | Field Label |
Choices, Calculations, OR Slider Labels | ...`.

This script concatenates the survey tabs into a single flat CSV, prepending a `Survey`
column (the tab name) so the cohort-of-origin survives the flattening. Output columns match
what ddharmon's `load_dictionary` reads in the v1 pipeline notebook (`Item Concept`,
`Field Label`, `Field Type`, `Choices, Calculations, OR Slider Labels`, `Survey`).

Two tabs need light normalization, handled automatically:
  - `COPE` carries a `Version` column and a `Generalized Answer Codes/Rules` header
    (renamed to the canonical `Generalized Answer Codes`).
  - the COVID-19 vaccine "Minute Survey" tab has three blank rows above its header and
    embedded newlines in header cells; the header row is detected dynamically and column
    names are whitespace-normalized.

Usage:
    python scripts/build_all_of_us_csv.py [--xlsx PATH] [--out PATH]

Idempotent — overwrites the output CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_XLSX = Path("data/examples/all_of_us_survey_codebooks.xlsx")
DEFAULT_OUT = Path("data/examples/all_of_us_surveys.csv")

# Canonical output column order (anything not listed is appended in first-seen order).
CANONICAL_COLS = [
    "Survey",
    "Version",
    "Item Concept",
    "Form Name",
    "Section Header",
    "Field Type",
    "Field Label",
    "Choices, Calculations, OR Slider Labels",
    "Text Validation Type OR Show Slider Number",
    "Text Validation Min",
    "Text Validation Max",
    "Branching Logic (Show field only if...)",
    "Required Field?",
    "Custom Alignment",
    "Question Number (surveys only)",
    "Matrix Group Name",
    "Matrix Ranking?",
    "Field Annotation",
    "Generalized Answer Codes",
    "Registered Tier Rules",
    "Controlled Tier Rules",
]

# Per-tab header variants -> canonical name.
COLUMN_ALIASES = {"Generalized Answer Codes/Rules": "Generalized Answer Codes"}


def _norm(col: object) -> str:
    """Collapse embedded newlines / repeated whitespace in a header cell."""
    return " ".join(str(col).split())


def _find_header_row(xlsx_path: Path, sheet: str, scan: int = 8) -> int:
    """Return the 0-based row index whose cells contain the 'Item Concept' header."""
    probe = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, nrows=scan, dtype=str)
    for i in range(len(probe)):
        if any(_norm(v) == "Item Concept" for v in probe.iloc[i].tolist()):
            return i
    raise ValueError(f"No 'Item Concept' header found in tab {sheet!r}")


def build_all_of_us_csv(xlsx_path: Path, out_path: Path) -> None:
    xl = pd.ExcelFile(xlsx_path)
    survey_tabs = [s for s in xl.sheet_names if s.strip().lower() != "readme"]
    print(f"Survey tabs: {len(survey_tabs)}")

    frames = []
    for tab in survey_tabs:
        header_row = _find_header_row(xlsx_path, tab)
        df = pd.read_excel(xlsx_path, sheet_name=tab, header=header_row, dtype=str)
        df.columns = [COLUMN_ALIASES.get(_norm(c), _norm(c)) for c in df.columns]
        df = df.dropna(axis=1, how="all")  # drop fully-empty 'Unnamed' columns
        df.insert(0, "Survey", tab.strip())
        frames.append(df)
        print(f"  {tab.strip()!r}: {len(df)} rows (header @ row {header_row})")

    merged = pd.concat(frames, ignore_index=True, sort=False)

    # drop rows with no content beyond the Survey label
    content_cols = [c for c in merged.columns if c != "Survey"]
    merged = merged.dropna(axis=0, how="all", subset=content_cols).reset_index(drop=True)

    ordered = [c for c in CANONICAL_COLS if c in merged.columns]
    ordered += [c for c in merged.columns if c not in ordered]
    merged = merged[ordered]

    n_enc = int(merged["Choices, Calculations, OR Slider Labels"].notna().sum())
    print(
        f"Merged: {merged.shape[0]} items x {merged.shape[1]} cols — "
        f"{n_enc} with response options ({n_enc / len(merged):.1%})"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.xlsx.exists():
        raise SystemExit(f"Source not found: {args.xlsx}")
    build_all_of_us_csv(args.xlsx, args.out)


if __name__ == "__main__":
    main()
