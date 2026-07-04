"""Build a flat data-dictionary CSV from a dbGaP study's PUBLIC variable summaries.

dbGaP has two access tiers: the participant *data* is controlled (DAC approval), but the
**variable-level data dictionaries are public** — published on the open FTP site so the
community can see what variables exist before applying. This builder reads those public
`*.data_dict.xml` files (no login, no DUA) and joins them into the `code=label|...` shape
ddharmon's `load_dictionary` expects (compare build_clsa/ukbb/aireadi). See:
    https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/about.html  (open vs controlled access)
    https://ftp.ncbi.nlm.nih.gov/dbgap/studies/

Generic across dbGaP studies — MESA is the default; FHS (or any study) is one flag away:
    python scripts/build_dbgap_csv.py                               # MESA (phs000209.v13.p3)
    python scripts/build_dbgap_csv.py --release phs000007.v32.p13 --label fhs   # Framingham

Each dataset's `data_dict.xml` is a `<data_table>` of `<variable>`s: name, description, type,
unit, and `<value code="X">label</value>` codings. One output row per (dataset, variable);
`dataset` (e.g. MESA_Exam1Main) becomes the category.

Large code systems (> --max-encoding-members) are recorded by `n_codes` only, not expanded
inline, so they don't swamp the geometric value vector (cf. build_ukbb_csv).

The shipped `data/examples/mesa_dbgap.csv` is the clinical/exam/questionnaire core — produced with
`--exclude-datasets CT` to drop the 12 imaging-derived datasets (lung / coronary-calcium / body-comp CT),
which are ~54% of MESA's variables and off-domain for questionnaire harmonization (same call as UKBB's
imaging exclusion). The builder itself defaults to ALL datasets (generic across dbGaP studies).

Usage:
    python scripts/build_dbgap_csv.py --exclude-datasets CT          # shipped MESA clinical core
    python scripts/build_dbgap_csv.py                                # full MESA (incl. imaging CT)
    python scripts/build_dbgap_csv.py --release phs000007.v32.p13 --label fhs   # Framingham
    # plus [--out PATH] [--cache-dir DIR] [--datasets SUBSTR ...] [--max-encoding-members N] [--refresh]

Idempotent — re-downloads only missing dictionaries (use --refresh to force) and overwrites the CSV.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

DEFAULT_RELEASE = "phs000209.v13.p3"  # MESA — Multi-Ethnic Study of Atherosclerosis
DEFAULT_LABEL = "mesa"
FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/dbgap/studies"
DATA_DICT_RE = re.compile(r'href="([^"]*\.data_dict\.xml)"')
DATASET_RE = re.compile(r"pht\d+\.v\d+\.(.+?)\.data_dict\.xml$")


def dataset_name(filename: str) -> str:
    """Dataset label from a data_dict filename, e.g. ...pht001116.v10.MESA_Exam1Main.data_dict.xml -> MESA_Exam1Main."""
    m = DATASET_RE.search(filename)
    return m.group(1) if m else filename


def summaries_url(release: str) -> str:
    study = release.split(".")[0]
    return f"{FTP_BASE}/{study}/{release}/pheno_variable_summaries/"


def list_data_dicts(release: str) -> list[str]:
    """Scrape the public pheno_variable_summaries listing for *.data_dict.xml filenames."""
    with urllib.request.urlopen(summaries_url(release), timeout=120) as resp:  # noqa: S310 (trusted NCBI host)
        html = resp.read().decode("utf-8", "replace")
    return sorted(set(DATA_DICT_RE.findall(html)))


def download_dicts(
    release: str, cache_dir: Path, refresh: bool, datasets: list[str] | None, exclude: list[str] | None
) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = list_data_dicts(release)
    # Case-sensitive substring match on the dataset name (dbGaP names are consistently cased;
    # e.g. exclude "CT" drops imaging-derived MESA_*LungExam*CT without hitting "Function").
    if datasets:
        names = [n for n in names if any(d in dataset_name(n) for d in datasets)]
    if exclude:
        names = [n for n in names if not any(d in dataset_name(n) for d in exclude)]
    print(f"  {len(names)} data_dict.xml files to read{' (filtered)' if (datasets or exclude) else ''}")
    base = summaries_url(release)
    paths: list[Path] = []
    for i, name in enumerate(names, 1):
        dest = cache_dir / name
        if not (dest.exists() and not refresh):
            urllib.request.urlretrieve(base + name, dest)  # noqa: S310 (trusted NCBI host)
            if i % 20 == 0 or i == len(names):
                print(f"    fetched {i}/{len(names)}")
        paths.append(dest)
    return paths


def parse_dict(path: Path, max_members: int) -> list[dict]:
    """Parse one dbGaP data_dict.xml -> list of variable rows."""
    dataset = dataset_name(path.name)
    root = ET.parse(path).getroot()
    study = root.get("study_id", "")
    rows: list[dict] = []
    for v in root.findall(".//variable"):
        values = [(val.get("code", ""), (val.text or "").strip()) for val in v.findall("value")]
        n = len(values)
        encoding = "|".join(f"{c}={lbl}" for c, lbl in values) if 0 < n <= max_members else ""
        rows.append(
            {
                "variable_name": (v.findtext("name", "") or "").strip(),
                "dataset": dataset,
                "description": (v.findtext("description", "") or "").strip(),
                "data_type": (v.findtext("type", "") or "").strip(),
                "units": (v.findtext("unit", "") or "").strip(),
                "value_encoding": encoding,
                "n_codes": str(n) if n else "",
                "dbgap_variable_id": v.get("id", ""),
                "dbgap_study": study,
            }
        )
    return rows


def build_dbgap_csv(
    release: str,
    label: str,
    out_path: Path,
    cache_dir: Path,
    datasets: list[str] | None,
    exclude: list[str] | None,
    max_members: int,
    refresh: bool,
) -> None:
    print(f"dbGaP study release {release} (label={label}) — reading PUBLIC variable summaries …")
    paths = download_dicts(release, cache_dir, refresh, datasets, exclude)

    rows: list[dict] = []
    for p in paths:
        rows.extend(parse_dict(p, max_members))
    if not rows:
        raise SystemExit("No variables parsed — check --datasets / --exclude-datasets (nothing matched).")
    out = pd.DataFrame(rows)
    out = out[out["variable_name"] != ""].sort_values(["dataset", "variable_name"], kind="stable")

    n_enc = int((out["value_encoding"] != "").sum())
    print(
        f"\nOutput: {out.shape} — {out['dataset'].nunique()} datasets, "
        f"{n_enc} variables with value_encoding ({n_enc / len(out):.1%}), "
        f"{len(out) - n_enc} numeric/free-text"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release", default=DEFAULT_RELEASE, help="dbGaP study-version dir (default: MESA phs000209.v13.p3)."
    )
    parser.add_argument("--label", default=DEFAULT_LABEL, help="Cohort label for the output filename (default: mesa).")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV (default: data/examples/<label>_dbgap.csv).")
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Download cache (default: data/examples/.dbgap_cache/<release>)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="SUBSTR",
        help="Only datasets whose name contains one of these substrings (case-sensitive).",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="+",
        metavar="SUBSTR",
        help="Drop datasets whose name contains one of these substrings, case-sensitive "
        "(e.g. CT drops imaging-derived MESA_*CT metrics; the shipped mesa_dbgap.csv uses this).",
    )
    parser.add_argument(
        "--max-encoding-members",
        type=int,
        default=200,
        help="Value sets larger than this are not expanded inline (default: 200).",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-download dictionaries even if cached.")
    args = parser.parse_args()
    out = args.out or Path(f"data/examples/{args.label}_dbgap.csv")
    cache = args.cache_dir or Path(f"data/examples/.dbgap_cache/{args.release}")
    build_dbgap_csv(
        args.release,
        args.label,
        out,
        cache,
        args.datasets,
        args.exclude_datasets,
        args.max_encoding_members,
        args.refresh,
    )


if __name__ == "__main__":
    main()
