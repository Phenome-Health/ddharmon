"""Build unified variable-matching benchmark from Tier-1 public sources.

Normalizes 5 public ground-truth datasets into one TSV with a shared schema.
Emits a SSSOM-conformant subset for rows where both source and target have
stable CURIEs.

Sources (all downloads preserved under data/benchmarks/raw/):
  - PASSIONATE (Salimi 2025)            — PD cohort variables (CC-BY-4.0)
  - CDEMapper (Wang 2025)               — Study vars → NIH CDEs (repo, no LICENSE)
  - Hao 2024 (BMC MIDM)                 — NACC↔ADNI↔NIH-CDE AD pairs (repo, no LICENSE)
  - McElroy/Harmony 2024 (BMC Psych)    — Mental health questionnaires (CC-BY)
  - Zhang 2024 (Sci Data)               — Cross-standard SSSOM (MIT)

Run:  uv run python scripts/build_benchmark.py
"""

from __future__ import annotations

import glob
from itertools import combinations
from pathlib import Path

import pandas as pd

RAW = Path("data/benchmarks/raw")
OUT = Path("data/benchmarks")

COLUMNS = [
    "source_id",
    "source_label",
    "source_description",
    "source_cohort",
    "target_id",
    "target_label",
    "target_description",
    "target_cohort",
    "relation",
    "confidence",
    "mapping_source",
    "domain",
    "notes",
]


def _truncate(s: object, n: int = 200) -> str | None:
    if pd.isna(s) or s is None:
        return None
    s = str(s).replace("\n", " ").replace("\t", " ").strip()
    return s[:n] if len(s) > n else s


def normalize_passionate() -> list[dict]:
    """PASSIONATE: wide cohort matrix. Emit all cross-cohort pairs per concept.

    Row semantics: one row per harmonized PD concept (Feature) with optional
    CURIE and the cohort-specific variable name in each cohort column. Two
    cohort variables tied to the same Feature are treated as 'exact' — they
    were manually harmonized as semantically equivalent.
    """
    df = pd.read_csv(RAW / "passionate" / "PASSIONATE.csv")
    non_cohort = {"Feature", "CURIE", "Definition", "Synonyms"}
    cohort_cols = [c for c in df.columns if c not in non_cohort]

    def _fmt(v: object) -> str:
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    rows: list[dict] = []
    for _, r in df.iterrows():
        mapped = [(c, _fmt(r[c])) for c in cohort_cols if pd.notna(r[c])]
        if len(mapped) < 2:
            continue
        concept_curie = r["CURIE"] if pd.notna(r["CURIE"]) else None
        concept_desc = r.get("Definition") if pd.notna(r.get("Definition")) else r["Feature"]

        for (cohort_a, var_a), (cohort_b, var_b) in combinations(mapped, 2):
            rows.append(
                {
                    "source_id": f"PASSIONATE-{cohort_a}:{var_a}",
                    "source_label": var_a,
                    "source_description": _truncate(concept_desc),
                    "source_cohort": cohort_a,
                    "target_id": f"PASSIONATE-{cohort_b}:{var_b}",
                    "target_label": var_b,
                    "target_description": _truncate(concept_desc),
                    "target_cohort": cohort_b,
                    "relation": "exact",
                    "confidence": None,
                    "mapping_source": "10.5281/zenodo.12743988",
                    "domain": "PD",
                    "notes": f"Harmonized concept: {r['Feature']}"
                    + (f" ({concept_curie})" if concept_curie else ""),
                }
            )
    return rows


def normalize_cdemapper() -> list[dict]:
    """CDEMapper: 4 domain CSVs. Each row = study variable → NIH CDE mapping.

    ADRD/Eye/Stroke use columns (Element, Question_Text, Values, GoldName,
    GoldID). covid-19 uses (Query, Element, Variable_Label, Question, Response,
    Source, GoldName, GoldID). GoldID values of 'Mapping_Not_Found' are
    explicit unmatched; values containing '||' are ambiguous multi-candidate
    cases (preserved as-is in target_id).
    """
    files = {
        "ADRD.csv": "AD",
        "Eye.csv": "eye",
        "Stroke.csv": "stroke",
        "covid-19.csv": "covid",
    }
    rows: list[dict] = []
    for fname, domain in files.items():
        df = pd.read_csv(RAW / "cdemapper" / "EvaluationData" / fname)
        question_col = "Question_Text" if "Question_Text" in df.columns else "Question"
        values_col = "Values" if "Values" in df.columns else "Response"
        for _, r in df.iterrows():
            gold_id = r.get("GoldID")
            is_unmatched = pd.isna(gold_id) or str(gold_id).strip().lower() == "mapping_not_found"
            rows.append(
                {
                    "source_id": f"CDEMapper-{domain}:{r['Element']}",
                    "source_label": r["Element"],
                    "source_description": _truncate(r.get(question_col)),
                    "source_cohort": f"CDEMapper-{domain}",
                    "target_id": None if is_unmatched else f"NIH-CDE:{gold_id}",
                    "target_label": None if is_unmatched else _truncate(r.get("GoldName")),
                    "target_description": None,
                    "target_cohort": None if is_unmatched else "NIH-CDE",
                    "relation": "unmatched" if is_unmatched else "exact",
                    "confidence": None,
                    "mapping_source": "10.1093/jamia/ocaf064",
                    "domain": domain,
                    "notes": _truncate(r.get(values_col)),
                }
            )
    return rows


def _hao_relation(r: pd.Series) -> str:
    """Infer relation from Hao match columns.

    DE_Match  = data-element match (concept-level)
    ValueType_Match / Unit_Match / PerValue_Match = structural agreement
    Values 'T' / 'F' / 'P' / 'S' / 'P/S' / NaN.
    Treat DE_Match=T as 'exact' if all struct match, else 'broader' (concept
    agrees but scale/unit/values differ). DE_Match=F → 'unmatched'.
    """
    de = str(r.get("DE_Match", "")).strip().upper()
    if de == "F":
        return "unmatched"
    if de != "T":
        return "related"  # unknown label, treat as weak
    struct_cols = ("ValueType_Match", "Unit_Match", "PerValue_Match")
    all_t = all(str(r.get(c, "")).strip().upper() == "T" for c in struct_cols if pd.notna(r.get(c)))
    return "exact" if all_t else "broader"


def normalize_hao() -> list[dict]:
    """Hao 2024: 3 pairwise AD files (NACC↔ADNI, NACC↔CDE, ADNI↔CDE)."""
    rows: list[dict] = []

    # 1-NACC-ADNI: NACC VariableName/ShortDescriptor ↔ ADNI FLDNAME/TEXT
    df = pd.read_excel(RAW / "hao_2024" / "MapResult" / "1-NACC-ADNI.xlsx")
    for _, r in df.iterrows():
        rows.append(
            {
                "source_id": f"NACC:{r['VariableName']}",
                "source_label": r["VariableName"],
                "source_description": _truncate(r.get("ShortDescriptor")),
                "source_cohort": "NACC",
                "target_id": f"ADNI:{r['FLDNAME']}",
                "target_label": r["FLDNAME"],
                "target_description": _truncate(r.get("TEXT")),
                "target_cohort": "ADNI",
                "relation": _hao_relation(r),
                "confidence": float(r["COSINE"]) if pd.notna(r.get("COSINE")) else None,
                "mapping_source": "10.1186/s12911-024-02500-8",
                "domain": "AD",
                "notes": f"pair_id={r.get('ID1-ID2')}",
            }
        )

    # 2-NACC-CDE: NACC ↔ NIH CDE
    df = pd.read_excel(RAW / "hao_2024" / "MapResult" / "2-NACC-CDE.xlsx")
    cde_name_col = next((c for c in df.columns if "cde" in c.lower() or "gold" in c.lower() or c == "Name"), None)
    cde_id_col = next((c for c in df.columns if c.lower().endswith("id") and c != "rowid"), None)
    for _, r in df.iterrows():
        tgt_id_val = r.get(cde_id_col) if cde_id_col else None
        tgt_label = r.get(cde_name_col) if cde_name_col else None
        rows.append(
            {
                "source_id": f"NACC:{r['VariableName']}",
                "source_label": r["VariableName"],
                "source_description": _truncate(r.get("ShortDescriptor")),
                "source_cohort": "NACC",
                "target_id": f"NIH-CDE:{tgt_id_val}" if pd.notna(tgt_id_val) else None,
                "target_label": _truncate(tgt_label),
                "target_description": None,
                "target_cohort": "NIH-CDE",
                "relation": _hao_relation(r),
                "confidence": float(r["COSINE"]) if pd.notna(r.get("COSINE")) else None,
                "mapping_source": "10.1186/s12911-024-02500-8",
                "domain": "AD",
                "notes": f"pair_id={r.get('ID1-ID2')}",
            }
        )

    # 3-ADNI-CDE: ADNI ↔ NIH CDE
    df = pd.read_excel(RAW / "hao_2024" / "MapResult" / "3-ADNI-CDE.xlsx")
    cde_name_col = next((c for c in df.columns if "cde" in c.lower() or "gold" in c.lower() or c == "Name"), None)
    cde_id_col = next((c for c in df.columns if c.lower().endswith("id") and c not in ("rowid", "rowid.1")), None)
    for _, r in df.iterrows():
        tgt_id_val = r.get(cde_id_col) if cde_id_col else None
        tgt_label = r.get(cde_name_col) if cde_name_col else None
        rows.append(
            {
                "source_id": f"ADNI:{r['FLDNAME']}",
                "source_label": r["FLDNAME"],
                "source_description": _truncate(r.get("TEXT")),
                "source_cohort": "ADNI",
                "target_id": f"NIH-CDE:{tgt_id_val}" if pd.notna(tgt_id_val) else None,
                "target_label": _truncate(tgt_label),
                "target_description": None,
                "target_cohort": "NIH-CDE",
                "relation": _hao_relation(r),
                "confidence": float(r.get("similary") or r.get("COSINE") or 0) or None,
                "mapping_source": "10.1186/s12911-024-02500-8",
                "domain": "AD",
                "notes": f"pair_id={r.get('ID1_ID2') or r.get('ID1-ID2')}",
            }
        )

    return rows


def normalize_mcelroy() -> list[dict]:
    """McElroy 2024: 39 mental-health items, Walktrap cluster assignments.

    Uses the 'Corr' sheet (empirical correlation network — the gold standard).
    For each pair of items in the same cluster, emit as 'related' with the
    cluster description in notes. Cross-cluster pairs are NOT emitted as
    'unmatched' — cluster boundaries are coarse and false negatives would
    dominate.
    """
    df = pd.read_excel(
        RAW / "harmony_mcelroy" / "supplementary_file_2.xlsx",
        sheet_name="Corr",
        header=1,
    )
    df["Measure"] = df["Measure"].ffill()

    rows: list[dict] = []
    for cid, grp in df.groupby("Cluster"):
        items = list(
            zip(
                grp["Node name"].astype(str),
                grp["Question content"].astype(str),
                grp["Measure"].astype(str),
            )
        )
        cluster_desc = grp["Cluster description"].iloc[0]
        for (a_name, a_q, a_scale), (b_name, b_q, b_scale) in combinations(items, 2):
            rows.append(
                {
                    "source_id": f"Harmony:{a_name}",
                    "source_label": a_name,
                    "source_description": _truncate(a_q),
                    "source_cohort": a_scale,
                    "target_id": f"Harmony:{b_name}",
                    "target_label": b_name,
                    "target_description": _truncate(b_q),
                    "target_cohort": b_scale,
                    "relation": "related",
                    "confidence": None,
                    "mapping_source": "10.1186/s12888-024-05954-2",
                    "domain": "mental_health",
                    "notes": f"Walktrap-Corr cluster {cid}: {cluster_desc}",
                }
            )
    return rows


_PREDICATE_MAP = {
    "skos:exactMatch": "exact",
    "skos:broadMatch": "broader",
    "skos:narrowMatch": "narrower",
    "skos:relatedMatch": "related",
    "skos:closeMatch": "related",
}


def normalize_zhang() -> list[dict]:
    """Zhang 2024: 7 topic SSSOM TSVs (FHIR/OMOP/CDISC/openEHR/Phenopackets)."""
    topic_files = sorted(glob.glob(str(RAW / "zhang_2024" / "sssom-mappings" / "tables" / "*_sssom.xlsx")))
    rows: list[dict] = []
    for f in topic_files:
        topic = Path(f).stem.replace("_sssom", "")
        df = pd.read_excel(f)
        for _, r in df.iterrows():
            src_id = str(r["subject_id"])
            tgt_id = str(r["object_id"])
            rows.append(
                {
                    "source_id": src_id,
                    "source_label": _truncate(r.get("subject_label")),
                    "source_description": None,
                    "source_cohort": src_id.split(":")[0].upper(),
                    "target_id": tgt_id,
                    "target_label": _truncate(r.get("object_label")),
                    "target_description": None,
                    "target_cohort": tgt_id.split(":")[0].upper(),
                    "relation": _PREDICATE_MAP.get(str(r["predicate_id"]), "related"),
                    "confidence": None,
                    "mapping_source": "10.1038/s41597-024-04168-1",
                    "domain": "cross_standard",
                    "notes": f"Topic: {topic}",
                }
            )
    return rows


def _is_curie(x: object) -> bool:
    """A stable CURIE is one whose prefix is a registered standard/vocab — not
    a cohort-specific synthetic prefix we invented here (PASSIONATE-*,
    CDEMapper-*, Harmony, NACC, ADNI)."""
    if pd.isna(x) or x is None:
        return False
    s = str(x)
    if ":" not in s:
        return False
    prefix = s.split(":", 1)[0]
    synthetic = {
        "PASSIONATE-PPMI", "PASSIONATE-BIOFIND", "PASSIONATE-LuxPARK",
        "PASSIONATE-LCC", "PASSIONATE-PRoBaND", "PASSIONATE-OPDC",
        "PASSIONATE-OMOP", "PASSIONATE-Fox Insight", "PASSIONATE-DATATOP",
        "PASSIONATE-PINE",
        "CDEMapper-AD", "CDEMapper-eye", "CDEMapper-stroke", "CDEMapper-covid",
        "Harmony", "NACC", "ADNI", "NIH-CDE",
    }
    return prefix not in synthetic


def _vc_to_md(s: pd.Series, label: str) -> str:
    out = [f"| {label} | count |", "|---|---|"]
    for k, v in s.items():
        out.append(f"| {k} | {v} |")
    return "\n".join(out)


def _write_stats(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Benchmark Stats",
        "",
        f"Total rows: {len(df)}",
        "",
        "## By mapping_source",
        _vc_to_md(df["mapping_source"].value_counts(), "mapping_source"),
        "",
        "## By domain",
        _vc_to_md(df["domain"].value_counts(), "domain"),
        "",
        "## By relation",
        _vc_to_md(df["relation"].value_counts(), "relation"),
        "",
        "## By (domain, relation)",
        "",
    ]
    pivot = df.groupby(["domain", "relation"]).size().unstack(fill_value=0)
    header = "| domain | " + " | ".join(str(c) for c in pivot.columns) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(pivot.columns) + 1))
    for idx, row in pivot.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")

    lines.append("")
    lines.append(f"## Source cohorts: {df['source_cohort'].nunique()} unique")
    lines.append(_vc_to_md(df["source_cohort"].value_counts().head(20), "source_cohort"))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    all_rows: list[dict] = []
    for fn in (
        normalize_passionate,
        normalize_cdemapper,
        normalize_hao,
        normalize_mcelroy,
        normalize_zhang,
    ):
        rows = fn()
        print(f"  {fn.__name__}: {len(rows)} rows")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "normalized.tsv", sep="\t", index=False)
    print(f"\nTotal: {len(df)} rows → {OUT / 'normalized.tsv'}")

    sssom = df[df["source_id"].apply(_is_curie) & df["target_id"].apply(_is_curie)].copy()
    sssom.to_csv(OUT / "sssom_subset.sssom.tsv", sep="\t", index=False)
    print(f"SSSOM-conformant subset: {len(sssom)} rows → {OUT / 'sssom_subset.sssom.tsv'}")

    _write_stats(df, OUT / "stats.md")
    print(f"Stats → {OUT / 'stats.md'}")


if __name__ == "__main__":
    main()
