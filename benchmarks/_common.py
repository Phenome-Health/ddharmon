"""Shared helpers for the standing benchmarks: data fetch, CDE backbone, embedding.

Keeps the two benchmark entry points (``cdemapper``, ``phenx``) thin and portable. Public gold data is
fetched on demand into ``.cache/benchmarks/`` (gitignored); the CDE backbone is the shipped
``data/examples/all_cdes_flat.tsv``.
"""

from __future__ import annotations

import csv
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache" / "benchmarks"
CDE_FLAT = REPO_ROOT / "data" / "examples" / "all_cdes_flat.tsv"
# AI-READI survey gold ships in-repo (built by scripts/build_aireadi_csv.py); no fetch needed.
AIREADI_CSV = REPO_ROOT / "data" / "examples" / "aireadi_surveys.csv"

CDEMAPPER_RAW = "https://raw.githubusercontent.com/BIDS-Xu-Lab/CDE-Mapping-Tool/main/EvaluationData"
CDEMAPPER_SETS = ("ADRD", "Eye", "Stroke", "covid-19")
PHENX_XLSX_URL = "https://www.phenxtoolkit.org/toolkit_content/documents/resources/Variable_cross_reference.xlsx"
ATHLOS_TARBALL = "https://github.com/athlosproject/athlos-project.github.io/archive/refs/heads/master.tar.gz"


def _fetch(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` (cached; skips if present)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url} -> {dest.relative_to(REPO_ROOT)}")
    req = urllib.request.Request(url, headers={"User-Agent": "ddharmon-benchmarks"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:  # noqa: S310
        f.write(resp.read())
    return dest


def ensure_cdemapper_gold() -> dict[str, Path]:
    """Fetch the 4 CDEMapper EvaluationData CSVs into the cache; return {set_name: path}."""
    return {
        name: _fetch(f"{CDEMAPPER_RAW}/{name}.csv", CACHE_DIR / "cdemapper" / f"{name}.csv") for name in CDEMAPPER_SETS
    }


def ensure_phenx_crosswalk() -> Path:
    """Fetch the PhenX<->dbGaP Variable_cross_reference.xlsx into the cache; return its path."""
    return _fetch(PHENX_XLSX_URL, CACHE_DIR / "phenx" / "Variable_cross_reference.xlsx")


def ensure_athlos_repo() -> Path:
    """Fetch + extract the ATHLOS harmonisation-script repo (AGPL-3); return the extracted root.

    The repo is the value-recode gold for Benchmark C: ~1,900 per-variable .Rmd scripts each carrying a
    source value set + target value set + car::recode algorithm. Cached as a tarball, extracted once.
    """
    root = CACHE_DIR / "athlos" / "athlos-project.github.io-master"
    if root.exists() and any(root.rglob("*.Rmd")):
        return root
    tarball = _fetch(ATHLOS_TARBALL, CACHE_DIR / "athlos" / "athlos-master.tar.gz")
    print(f"  extracting {tarball.relative_to(REPO_ROOT)} …")
    with tarfile.open(tarball) as tf:
        tf.extractall(tarball.parent, filter="data")  # py3.12 safe extraction
    return root


def make_provider():
    """The pipeline's embedding provider (local sentence-transformers)."""
    from ddharmon.embedding import SentenceTransformerProvider

    return SentenceTransformerProvider()


def load_cde_backbone(provider) -> tuple[list[str], np.ndarray, list[str], dict[str, str]]:
    """Load + embed the shipped CDE backbone, faithfully to the pipeline (state.load_cde).

    Returns:
        cde_ids: designations (the candidate space, aligned to cde_vecs).
        cde_vecs: embedding matrix (one row per designation).
        rich_corpus: richer per-designation text (designation + question + definition + permissible
            values) for BM25, aligned to cde_ids.
        tiny2des: NIH tinyId -> designation (for scoring against GoldID / dbGaP id-spaces).
    """
    from ddharmon.embedding import embed_dictionary
    from ddharmon.ingestion import load_dictionary, preprocess_dictionary

    dd = preprocess_dictionary(
        load_dictionary(
            str(CDE_FLAT),
            cohort_name="NIH_CDE",
            variable_name="designation",
            embed_variable_name=True,
            description="definition",
            question_text="question_text",
            data_type="datatype",
            value_encoding="permissible_values",
        )
    )
    ed = embed_dictionary(dd, provider=provider)
    cde_ids = ed.get_variable_names()
    cde_vecs = ed.get_all_vectors()

    rich: dict[str, str] = {}
    tiny2des: dict[str, str] = {}
    with open(CDE_FLAT) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            des = (row.get("designation") or "").strip()
            tiny2des[(row.get("tinyId") or "").strip()] = des
            rich.setdefault(
                des,
                " ".join(
                    x
                    for x in [
                        row.get("designation", ""),
                        row.get("question_text", ""),
                        row.get("definition", ""),
                        row.get("permissible_values", ""),
                    ]
                    if x
                ),
            )
    rich_corpus = [rich.get(c, c) for c in cde_ids]
    return cde_ids, cde_vecs, rich_corpus, tiny2des


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows (so a dot product is cosine)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def coclustering_inputs() -> tuple[dict, dict, dict, dict]:
    """Parse the PhenX crosswalk -> (var_text, var_study, by_pxvar, by_proto).

    var_text: dbGaP variable id -> description; var_study: dbGaP var -> phs study (versionless);
    by_pxvar/by_proto: PhenX variable / protocol -> set(dbGaP var ids).
    """
    import openpyxl

    path = ensure_phenx_crosswalk()
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["With mappings"]
    it = ws.iter_rows(values_only=True)
    hdr = next(it)
    ix = {h: i for i, h in enumerate(hdr) if h}
    var_text, var_study = {}, {}
    by_pxvar, by_proto = defaultdict(set), defaultdict(set)
    for r in it:
        px, proto = r[ix["VARIABLE_ID"]], r[ix["PROTOCOL_NAME"]]
        dvar, study, desc = r[ix["dbGaP VARIABLE_ID"]], r[ix["dbGaP STUDY_ID"]], r[ix["dbGaP VARIABLE_DESCRIPTION"]]
        if not (px and dvar and study and desc):
            continue
        var_text.setdefault(dvar, str(desc).strip())
        var_study[dvar] = str(study).split(".")[0]
        by_pxvar[px].add(dvar)
        by_proto[proto].add(dvar)
    return var_text, var_study, dict(by_pxvar), dict(by_proto)
