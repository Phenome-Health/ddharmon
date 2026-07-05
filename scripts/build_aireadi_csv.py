"""Build an AI-READI survey flat CSV from the public Common Data Element mappings.

Source (public, MIT-licensed): the AI-READI project (Bridge2AI flagship Type 2
Diabetes dataset) publishes its REDCap survey -> OMOP/CDE value-set mappings in the
`AI-READI/DataElementMaps` GitHub repo. This is *metadata* — survey field names,
question text, response-option code/label pairs, and the OMOP/CDE concept each
maps to — NOT the participant data (which is controlled access via FAIRhub). See:
    https://github.com/AI-READI/DataElementMaps   (MIT)
    https://docs.aireadi.org/

The single source file `aireadi2omop_redcap_valueset_mapping.csv` is a (field, value)
table across ~43 REDCap survey forms. Two row kinds:
  - field-def rows  (FIELD_ID empty): SRC_CODE = variable name, SRC_CD_DESCRIPTION =
    the question/label, and TARGET_CONCEPT_* = the field's OMOP/CDE anchor.
  - value rows      (FIELD_ID set):   FIELD_ID = variable name, one row per response
    option (SRC_CODE = code, SRC_CD_DESCRIPTION = label).

This script joins the two into one flat CSV — one row per (form, variable) — with
`code=label|code=label|...` value encodings (the shape ddharmon's `load_dictionary`
expects; compare `build_clsa_csv.py` / `build_ukbb_csv.py`). Because AI-READI is
CDE-forward, each field also carries its mapped OMOP/CDE concept (name + vocabulary +
code) — a gold anchor for validating ddharmon's CDE-assignment, not just an input.

Unlike UKBB, no category/scope filter is needed: this file is *only* the survey/
clinical-form data elements (imaging / omics / sensor data live elsewhere and are
out of ddharmon's questionnaire scope).

Usage:
    python scripts/build_aireadi_csv.py [--out PATH] [--cache-dir DIR]
                                        [--ref GIT_REF] [--max-encoding-members N] [--refresh]

Idempotent — re-downloads only if missing (use --refresh to force) and overwrites the CSV.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_OUT = Path("data/examples/aireadi_surveys.csv")
DEFAULT_CACHE = Path("data/examples/.aireadi_cache")
SOURCE_FILE = "aireadi2omop_redcap_valueset_mapping.csv"
RAW_URL = "https://raw.githubusercontent.com/AI-READI/DataElementMaps/{ref}/" + SOURCE_FILE


def _read_csv(path: Path) -> pd.DataFrame:
    """Read the mapping CSV. It is Windows-1252 (non-breaking spaces / smart quotes)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="cp1252")


def download_source(cache_dir: Path, ref: str, refresh: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / SOURCE_FILE
    if dest.exists() and not refresh:
        print(f"  cached  {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
    else:
        url = RAW_URL.format(ref=ref)
        print(f"  fetch   {url}")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted public host)
        print(f"          -> {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


def build_aireadi_csv(out_path: Path, cache_dir: Path, ref: str, max_members: int, refresh: bool) -> None:
    print("Downloading AI-READI CDE mapping from the public DataElementMaps repo ...")
    src = download_source(cache_dir, ref, refresh)
    df = _read_csv(src)
    print(f"\nSource rows: {len(df)} across {df['FORM_NAME'].nunique()} REDCap forms")

    # Split the two row kinds.
    defs = df[df["FIELD_ID"] == ""].copy()  # field-definition rows (varname in SRC_CODE)
    vals = df[df["FIELD_ID"] != ""].copy()  # value rows (varname in FIELD_ID)

    # Value encodings: collapse each (form, variable)'s response options to code=label|...
    vals["pair"] = vals["SRC_CODE"].str.strip() + "=" + vals["SRC_CD_DESCRIPTION"].str.strip()
    venc = (
        vals.groupby(["FORM_NAME", "FIELD_ID"], sort=False)
        .agg(
            value_encoding=("pair", lambda s: "|".join(s)),
            n_codes=("pair", "size"),
            field_type_v=("FIELD_TYPE", "first"),
        )
        .reset_index()
        .rename(columns={"FIELD_ID": "variable_name"})
    )
    # Don't let an oversized value set swamp the geometric value vector (cf. build_ukbb_csv).
    oversized = venc["n_codes"] > max_members
    venc.loc[oversized, "value_encoding"] = ""

    # Field-level info (question text + OMOP/CDE anchor) from the definition rows.
    dinfo = (
        defs.groupby(["FORM_NAME", "SRC_CODE"], sort=False)
        .first()
        .reset_index()
        .rename(columns={"SRC_CODE": "variable_name"})
    )

    merged = dinfo.merge(venc, on=["FORM_NAME", "variable_name"], how="outer")

    out = pd.DataFrame(
        {
            "variable_name": merged["variable_name"],
            "form_name": merged["FORM_NAME"],
            "field_label": merged["SRC_CD_DESCRIPTION"].fillna(""),
            "field_type": merged["FIELD_TYPE"].fillna("").where(merged["FIELD_TYPE"].notna(), merged["field_type_v"]),
            "value_encoding": merged["value_encoding"].fillna(""),
            "n_codes": merged["n_codes"].fillna(0).astype(int).astype(str).replace("0", ""),
            # CDE-forward bonus: the OMOP/CDE concept each field is mapped to (gold anchor).
            "cde_concept_name": merged["TARGET_CONCEPT_NAME"].fillna(""),
            "cde_vocabulary": merged["TARGET_VOCABULARY_ID"].fillna(""),
            "cde_concept_code": merged["TARGET_CONCEPT_CODE"].fillna(""),
            "cde_domain": merged["TARGET_DOMAIN_ID"].fillna(""),
            "mapping_confidence": merged["CONFIDENCE"].fillna(""),
        }
    )
    # Every field needs some embeddable text: blank label -> CDE concept name -> variable name.
    blank = out["field_label"].str.strip() == ""
    fallback = out["cde_concept_name"].where(out["cde_concept_name"].str.strip() != "", out["variable_name"])
    out.loc[blank, "field_label"] = fallback[blank]

    out = out.sort_values(["form_name", "variable_name"], kind="stable")

    n_enc = int((out["value_encoding"] != "").sum())
    n_cde = int(out["cde_concept_name"].str.strip().ne("").sum())
    print(
        f"\nOutput: {out.shape} — "
        f"{n_enc} fields with value_encoding ({n_enc / len(out):.1%}), "
        f"{n_cde} with an OMOP/CDE anchor ({n_cde / len(out):.1%}), "
        f"max value-set size {int(venc['n_codes'].max())}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--ref", default="main", help="Git ref/branch/commit of DataElementMaps to pull (default: main)."
    )
    parser.add_argument(
        "--max-encoding-members",
        type=int,
        default=200,
        help="Value sets larger than this are recorded by n_codes only, not expanded inline (default: 200).",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-download the source CSV even if cached.")
    args = parser.parse_args()
    build_aireadi_csv(args.out, args.cache_dir, args.ref, args.max_encoding_members, args.refresh)


if __name__ == "__main__":
    main()
