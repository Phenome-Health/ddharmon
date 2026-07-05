"""Build a UK Biobank flat CSV from the public Showcase **Schema** downloads.

Source (public, no login / no application required): the UK Biobank Data Showcase
exposes its full data dictionary — field definitions, categories and value
encodings — as machine-readable schema files. These describe the *metadata* only
(field ids, titles, descriptions, code->label mappings); they are NOT the
participant data, which requires an approved application + the Research Analysis
Platform. See:
    https://biobank.ndph.ox.ac.uk/showcase/schema.cgi
    https://community.ukbiobank.ac.uk/hc/en-gb/articles/15955597101085

Each schema is a tab-separated file downloadable at:
    https://biobank.ndph.ox.ac.uk/showcase/scdown.cgi?fmt=txt&id=<N>

This script downloads the handful we need and joins them into one flat CSV with
`code=label|code=label|...` style value encodings — the same shape ddharmon's
`load_dictionary` expects (compare CLSA's `build_clsa_csv.py` and the
`value_encoding` column it emits).

The join, by schema id:
  1   field       field_id, title, value_type, units, main_category, encoding_id, notes
  3   category    category_id -> title  (resolve field.main_category)
  2   encoding    encoding_id -> (title, num_members)
  5   esimpint  ┐
  6   esimpstring│ encoding_id -> [(value, meaning), ...]  collapsed to value=meaning|...
  7   esimpreal  │
  8   esimpdate  │
 11   ehierint   │ (hierarchical: same value/meaning columns, code systems like ICD-10)
 12   ehierstring┘

Large code systems (ICD-10, OPCS, job/medication codes — hundreds-to-thousands of
members) are NOT expanded into `value_encoding`: dumping them would swamp the
geometric value vector with noise. Encodings with more than `--max-encoding-members`
members are recorded by reference instead (`encoding_name` + `n_codes` columns) so
nothing is lost. This is a generic size rule, not a per-cohort hardcode.

Scope: by DEFAULT only survey / questionnaire / demographic fields are extracted
(~2.4k of the ~11.8k Showcase fields), keeping UKBB's content domain aligned with the
other survey-scoped example cohorts (All of Us, CLSA). UKBB's catalog is far broader —
most fields are imaging / omics / genomics / EHR-linkage, which have no analog in those
cohorts. The default scope is the category subtrees in DEFAULT_SURVEY_CATEGORIES; pass
`--all-fields` for the full catalog, or `--include-category-subtree` to choose your own
roots (run `--list-categories` to discover ids).

Usage:
    python scripts/build_ukbb_csv.py                       # default: survey/demographic scope
    python scripts/build_ukbb_csv.py --all-fields          # full ~11.8k-field catalog
    python scripts/build_ukbb_csv.py --list-categories     # discover category ids
    python scripts/build_ukbb_csv.py --include-category-subtree 1 100089  # custom scope
    # plus [--out PATH] [--cache-dir DIR] [--max-encoding-members N] [--refresh]

Idempotent — re-downloads only missing schema files (use --refresh to force) and
overwrites the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_OUT = Path("data/examples/ukbb_showcase.csv")
DEFAULT_CACHE = Path("data/examples/.ukbb_schema_cache")
BASE_URL = "https://biobank.ndph.ox.ac.uk/showcase/scdown.cgi?fmt=txt&id="

# Schema id -> local filename stem.
SCHEMAS = {
    1: "field",
    3: "category",
    2: "encoding",
    5: "esimpint",
    6: "esimpstring",
    7: "esimpreal",
    8: "esimpdate",
    11: "ehierint",
    12: "ehierstring",
    13: "catbrowse",  # category tree (parent_id -> child_id), for --include-category-subtree
}
# Tables holding (encoding_id, value, meaning, showcase_order) value->label pairs.
VALUE_TABLE_IDS = (5, 6, 7, 8, 11, 12)

# Default scope: survey / questionnaire / demographic / physical-measure Showcase category roots
# (with descendants) — Population characteristics + the Assessment-centre Touchscreen / Cognitive
# function / Verbal interview / Physical measures branches + all Online follow-up. This keeps
# UKBB's content domain aligned with the other survey-scoped example cohorts (All of Us, CLSA) —
# including the anthropometrics (height/weight/BMI) and clinical measures that are universal
# cross-cohort harmonization targets — and excludes the imaging / omics / genomics / EHR-linkage
# bulk that has no analog there. Pass --all-fields for the full catalog, or
# --include-category-subtree to choose your own roots (see --list-categories).
DEFAULT_SURVEY_CATEGORIES = ["1", "100025", "100026", "100071", "100006", "100089"]

# UK Biobank value_type codes -> human-readable data type.
VALUE_TYPE = {
    "11": "Integer",
    "21": "Categorical single",
    "22": "Categorical multiple",
    "31": "Continuous",
    "41": "Text",
    "51": "Date",
    "61": "Time",
    "101": "Compound",
}


def _read_tsv(path: Path) -> pd.DataFrame:
    """Read a Showcase schema TSV.

    They are unquoted (disable quote handling) and Windows-1252 encoded
    (em-dashes / smart quotes in field notes), not UTF-8.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        encoding="cp1252",
        on_bad_lines="warn",
    )


def download_schemas(cache_dir: Path, refresh: bool) -> dict[int, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for schema_id, stem in SCHEMAS.items():
        dest = cache_dir / f"{stem}.tsv"
        if dest.exists() and not refresh:
            print(f"  cached  schema {schema_id:>2} -> {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
        else:
            url = f"{BASE_URL}{schema_id}"
            print(f"  fetch   schema {schema_id:>2} <- {url}")
            urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted public host)
            print(f"          -> {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
        paths[schema_id] = dest
    return paths


def _children_map(catbrowse: pd.DataFrame) -> dict[str, list[str]]:
    """parent_category_id -> [child_category_id, ...] from the catbrowse tree."""
    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in zip(catbrowse["parent_id"], catbrowse["child_id"], strict=True):
        children[parent].append(child)
    return children


def _subtree_categories(roots: list[str], children: dict[str, list[str]]) -> set[str]:
    """All category ids in the subtree(s) rooted at `roots`, inclusive of the roots."""
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        cid = stack.pop()
        if cid in seen:
            continue
        seen.add(cid)
        stack.extend(children.get(cid, []))
    return seen


def list_categories(paths: dict[int, Path]) -> None:
    """Print the UKBB category tree (roots + direct children) with subtree field counts.

    Use the printed ids with --include-category-subtree to scope the extract. Survey/
    demographic content lives mainly under 'Population characteristics', the Touchscreen/
    Cognitive function/Verbal interview children of 'Assessment centre', and all of
    'Online follow-up'; imaging/omics/genomics/EHR-linkage sit under their own roots.
    """
    titles = _read_tsv(paths[3]).set_index("category_id")["title"].to_dict()
    catbrowse = _read_tsv(paths[13])
    field_counts = _read_tsv(paths[1])["main_category"].value_counts().to_dict()
    children = _children_map(catbrowse)

    def subtree_field_count(cid: str) -> int:
        return sum(field_counts.get(c, 0) for c in _subtree_categories([cid], children))

    roots = sorted(set(catbrowse["parent_id"]) - set(catbrowse["child_id"]), key=int)
    print("UKBB category tree (id  title  [fields in subtree]):\n")
    for root in roots:
        print(f"{root:>7}  {titles.get(root, '?')}  [{subtree_field_count(root)}]")
        for child in children.get(root, []):
            print(f"     {child:>7}  {titles.get(child, '?')}  [{subtree_field_count(child)}]")
        print()


def build_value_encodings(paths: dict[int, Path], max_members: int) -> tuple[dict[str, str], dict[str, int]]:
    """Return (encoding_id -> 'value=meaning|...', encoding_id -> n_members) for codeable encodings.

    Only encodings with <= max_members members are expanded; larger ones are
    omitted from the expanded map (caller falls back to a reference).
    """
    frames = []
    for schema_id in VALUE_TABLE_IDS:
        df = _read_tsv(paths[schema_id])[["encoding_id", "value", "meaning", "showcase_order"]]
        frames.append(df)
    values = pd.concat(frames, ignore_index=True)
    values["order"] = pd.to_numeric(values["showcase_order"], errors="coerce").fillna(0)
    # Deterministic order: by encoding, then the Showcase display order.
    values = values.sort_values(["encoding_id", "order"], kind="stable")

    members = values.groupby("encoding_id", sort=False).size().to_dict()
    small = {enc for enc, n in members.items() if n <= max_members}

    expanded: dict[str, str] = {}
    small_vals = values[values["encoding_id"].isin(small)].copy()
    small_vals["pair"] = small_vals["value"].str.strip() + "=" + small_vals["meaning"].str.strip()
    for enc, pairs in small_vals.groupby("encoding_id", sort=False)["pair"]:
        expanded[enc] = "|".join(pairs)
    return expanded, {k: int(v) for k, v in members.items()}


def build_ukbb_csv(
    out_path: Path,
    cache_dir: Path,
    max_members: int,
    refresh: bool,
    include_subtree: list[str] | None = None,
) -> None:
    print(f"Downloading Showcase schema files to {cache_dir} ...")
    paths = download_schemas(cache_dir, refresh)

    fields = _read_tsv(paths[1])
    print(f"\nFields: {len(fields)} rows, {len(fields.columns)} cols")

    if include_subtree:
        children = _children_map(_read_tsv(paths[13]))
        keep = _subtree_categories(include_subtree, children)
        before = len(fields)
        fields = fields[fields["main_category"].isin(keep)]
        print(
            f"Category-subtree filter {include_subtree}: kept {len(fields)}/{before} fields "
            f"({len(keep)} categories in subtree)"
        )

    categories = _read_tsv(paths[3]).set_index("category_id")["title"].to_dict()
    enc_meta = _read_tsv(paths[2]).set_index("encoding_id")
    enc_title = enc_meta["title"].to_dict()

    expanded, members = build_value_encodings(paths, max_members)
    n_codeable = sum(1 for v in members.values() if v > 0)
    print(
        f"Encodings: {len(members)} codeable; "
        f"{len(expanded)} expanded inline (<= {max_members} members), "
        f"{n_codeable - len(expanded)} referenced (too large)"
    )

    def value_encoding(enc: str) -> str:
        return expanded.get(enc, "") if enc not in ("", "0") else ""

    def encoding_name(enc: str) -> str:
        # Only label the reference case (large encoding, not expanded inline).
        if enc in ("", "0") or enc in expanded:
            return ""
        return enc_title.get(enc, "")

    def n_codes(enc: str) -> str:
        return str(members.get(enc, "")) if enc not in ("", "0") else ""

    out = pd.DataFrame(
        {
            "field_id": fields["field_id"],
            "field_name": fields["title"],
            "description": fields["notes"],
            "category": fields["main_category"].map(lambda c: categories.get(c, "")),
            "data_type": fields["value_type"].map(lambda t: VALUE_TYPE.get(t, t)),
            "units": fields["units"],
            "value_encoding": fields["encoding_id"].map(value_encoding),
            "encoding_id": fields["encoding_id"].map(lambda e: "" if e in ("", "0") else e),
            "encoding_name": fields["encoding_id"].map(encoding_name),
            "n_codes": fields["encoding_id"].map(n_codes),
            "participant_count": fields["num_participants"],
        }
    )

    n_with_enc = int((out["value_encoding"] != "").sum())
    n_ref = int((out["encoding_name"] != "").sum())
    print(
        f"\nOutput: {out.shape} — "
        f"{n_with_enc} fields with inline value_encoding ({n_with_enc / len(out):.1%}), "
        f"{n_ref} with a large-encoding reference, "
        f"{len(out) - n_with_enc - n_ref} numeric/open-text"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--max-encoding-members",
        type=int,
        default=200,
        help="Encodings with more members than this are referenced, not expanded inline (default: 200).",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-download schema files even if cached.")
    parser.add_argument(
        "--include-category-subtree",
        nargs="+",
        metavar="CATEGORY_ID",
        help=(
            "Override the default scope: keep only fields under these category ids (and all "
            "descendants). Run --list-categories to discover ids."
        ),
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="Extract the FULL Showcase catalog (~11.8k fields) instead of the default survey/demographic scope.",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print the category tree with subtree field counts and exit (to pick --include-category-subtree ids).",
    )
    args = parser.parse_args()

    if args.list_categories:
        list_categories(download_schemas(args.cache_dir, args.refresh))
        return

    # Default scope = survey/demographic; --all-fields disables filtering;
    # --include-category-subtree overrides with explicit roots.
    if args.all_fields:
        include_subtree = None
    elif args.include_category_subtree:
        include_subtree = args.include_category_subtree
    else:
        include_subtree = DEFAULT_SURVEY_CATEGORIES

    build_ukbb_csv(
        args.out,
        args.cache_dir,
        args.max_encoding_members,
        args.refresh,
        include_subtree=include_subtree,
    )


if __name__ == "__main__":
    main()
