"""Build a PPMI data-dictionary flat CSV from the PPMI Data Dictionary + Code List.

IMPORTANT — ACCESS TIER. Unlike the All of Us / CLSA / UK Biobank / AI-READI example
dictionaries (which come from openly downloadable, no-login or openly-licensed public
sources), **PPMI data and its documentation are governed by a Data Use Agreement (DUA)**.
The PPMI Data Dictionary and Code List are obtained by registering at the PPMI portal,
accepting the DUA, and downloading from the Guidance Resources / download page:
    https://www.ppmi-info.org/access-data-specimens/download-data

Because of that, this is a **builder only** — it ships in the repo, but the **output CSV
is NOT bundled** (it goes to a gitignored directory). It is the "bring your own cohort"
pattern from data/examples/README.md: you download the dictionary with your own
credentials, point this script at it, and it produces a CSV the pipeline can ingest. No
PPMI data is downloaded, mirrored, or redistributed here.

Inputs (download from the PPMI portal under your DUA):
  - Data_Dictionary[_Annotated].csv — one row per (MOD_NAME table, ITM_NAME variable):
        MOD_NAME, ITM_NAME, SEQ_NO, DSCR (description), ITM_TYPE, ..., CODELIST, ...
    Rows with a blank ITM_NAME are table-header rows whose DSCR is the table's label
    (e.g. MOD_NAME "AE" -> "Adverse Event Log"); we use those as the human form label.
  - Code_List.csv — the code book: PAG_NAME, ITM_NAME, CDL_NAME, CODE, DECODE.
    A variable's Data_Dictionary.CODELIST names the code list (== Code_List.CDL_NAME)
    whose CODE/DECODE pairs are its value set.

Output: one row per (form, variable) with `code=label|code=label|...` value encodings —
the shape ddharmon's `load_dictionary` expects (compare build_clsa/ukbb/aireadi).

Usage:
    python scripts/build_ppmi_csv.py --data-dictionary PATH --code-list PATH [--out PATH]
                                     [--max-encoding-members N]

Idempotent — overwrites the output CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DD = Path("data/ppmi/Data_Dictionary_Annotated.csv")
DEFAULT_CL = Path("data/ppmi/Code_List.csv")
DEFAULT_OUT = Path("data/ppmi/ppmi_dictionary.csv")  # gitignored — DUA-gated, not shipped


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a PPMI export CSV (try UTF-8, fall back to Latin-1)."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"Could not decode {path} as utf-8 or latin-1")


def build_value_encodings(code_list: pd.DataFrame, max_members: int) -> dict[str, str]:
    """CDL_NAME -> 'code=label|...' (dedup CODE=DECODE; shared lists like YN repeat across items)."""
    code_list = code_list.copy()
    code_list["pair"] = code_list["CODE"].str.strip() + "=" + code_list["DECODE"].str.strip()
    encodings: dict[str, str] = {}
    for cdl, grp in code_list.groupby("CDL_NAME", sort=False):
        seen: list[str] = []
        for p in grp["pair"]:
            if p not in seen:
                seen.append(p)
        if len(seen) <= max_members:
            encodings[cdl] = "|".join(seen)
    return encodings


def build_ppmi_csv(dd_path: Path, cl_path: Path, out_path: Path, max_members: int) -> None:
    for p in (dd_path, cl_path):
        if not p.exists():
            raise SystemExit(
                f"Input not found: {p}\n"
                "Download the PPMI Data Dictionary + Code List from the PPMI portal "
                "(requires a Data Use Agreement) and pass --data-dictionary / --code-list."
            )

    dd = _read_csv(dd_path)
    cl = _read_csv(cl_path)
    print(f"Data dictionary: {len(dd)} rows, {dd['MOD_NAME'].nunique()} modules")
    print(f"Code list: {len(cl)} rows, {cl['CDL_NAME'].nunique()} code lists")

    # Table-header rows (blank ITM_NAME): DSCR is the human label for the MOD_NAME table.
    mod_label = {
        r["MOD_NAME"]: r["DSCR"].strip()
        for _, r in dd[dd["ITM_NAME"].str.strip() == ""].iterrows()
        if r["DSCR"].strip()
    }
    variables = dd[dd["ITM_NAME"].str.strip() != ""].copy()

    encodings = build_value_encodings(cl, max_members)

    out = pd.DataFrame(
        {
            "variable_name": variables["ITM_NAME"].str.strip(),
            "form_name": variables["MOD_NAME"].str.strip(),
            "form_label": variables["MOD_NAME"].map(lambda m: mod_label.get(m, m)),
            "description": variables["DSCR"].str.strip(),
            "data_type": variables["ITM_TYPE"].str.strip(),
            "codelist_name": variables["CODELIST"].str.strip(),
            "value_encoding": variables["CODELIST"].map(lambda c: encodings.get(c.strip(), "")),
        }
    )
    out = out.sort_values(["form_name", "variable_name"], kind="stable")

    n_enc = int((out["value_encoding"] != "").sum())
    print(
        f"\nOutput: {out.shape} — {n_enc} variables with value_encoding "
        f"({n_enc / len(out):.1%}), {len(out) - n_enc} numeric/free-text/date"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    print("NOTE: PPMI is DUA-governed — do not commit or redistribute this output CSV.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dictionary", type=Path, default=DEFAULT_DD)
    parser.add_argument("--code-list", type=Path, default=DEFAULT_CL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-encoding-members",
        type=int,
        default=200,
        help="Code lists larger than this are not expanded inline (default: 200).",
    )
    args = parser.parse_args()
    build_ppmi_csv(args.data_dictionary, args.code_list, args.out, args.max_encoding_members)


if __name__ == "__main__":
    main()
