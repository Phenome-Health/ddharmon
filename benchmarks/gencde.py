"""Benchmark E — GenCDE synthesis quality.

Scores the GenCDEs ddharmon synthesizes for ``novel`` concept groups against DataTecnica/FAIRkit's work
(Long et al., npj Digit Med 2026), on two axes:

1. **Reproducibility (FAIRkit's own protocol).** Regenerate the same novels K times and measure component-
   wise semantic equivalence (variable name / title / definition / permissible values). FAIRkit published
   25 CDEs regenerated 3x: name 76.0% / title 69.3% / permissible-values 65.3% / description 46.7%
   (:data:`PUBLISHED_FAIRKIT`). This is the apples-to-apples comparison — no gold alignment needed.

2. **Gold comparison (RoP expert-curated anchors).** DataTecnica open-sourced ~224 EXPERT-CURATED CDEs at
   github.com/datatecnica/RoP_biomedical (``data/anchors/*.json``, CC-BY-NC — internal eval + attribution
   only, never bundled/shipped). Where a synthesized GenCDE aligns to an anchor, score the same components
   against a human-validated target.

The scorer is embedder-injectable: pass the pipeline's BioLORD provider for semantic (cosine) component
similarity; omit it for a deterministic lexical fallback (used by the $0 tests). No HARD gate floor is set
yet — the first metered synthesis run establishes the baseline; a floor is added to ``gate.py`` after.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from ddharmon.harmonization.models import GenCDE
from ddharmon.models.data_dictionary import ResponseOption

# FAIRkit reproducibility study (Long et al. 2026, revision) — 25 CDEs regenerated 3x, component-wise
# semantic-equivalence means. Our numbers are reported next to these; we do NOT claim to beat them.
PUBLISHED_FAIRKIT = {
    "variable_name": 0.760,
    "title": 0.693,
    "permissible_values": 0.653,
    "definition": 0.467,
}

_ROP_ANCHORS_API = "https://api.github.com/repos/datatecnica/RoP_biomedical/contents/data/anchors"
_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "benchmarks" / "rop_anchors"

# RoP item_type -> our data_type vocabulary
_ITEM_TYPE = {"enum": "categorical", "binary": "binary", "numeric": "numeric", "string": "text", "date": "date"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def rop_anchor_to_gencde(anchor: dict) -> GenCDE:
    """Map one RoP anchor row (SPEC.md schema) into our :class:`GenCDE` for comparison.

    RoP ``item`` -> preferred_name/title, ``description`` -> definition, ``item_type`` -> data_type,
    ``values`` (pipe-delimited labels, or a ``min-max`` range for numeric) -> permissible_values or bounds,
    ``alternate_names`` -> aliases, ``unit_of_measure`` -> units.
    """
    item = str(anchor.get("item", "")).strip()
    item_type = _norm(str(anchor.get("item_type", "")))
    data_type = _ITEM_TYPE.get(item_type, item_type)
    raw_values = str(anchor.get("values", "")).strip()
    pv: list[ResponseOption] = []
    minimum = maximum = None
    if data_type in ("numeric", "date"):
        m = re.match(r"^\(?\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*\)?$", raw_values)
        if m:
            minimum, maximum = float(m.group(1)), float(m.group(2))
    elif raw_values:
        for label in raw_values.split("|"):
            label = label.strip()
            if label:
                pv.append(ResponseOption(code=label, label=label))
    aliases = [a.strip() for a in str(anchor.get("alternate_names", "")).split("|") if a.strip()]
    uom = str(anchor.get("unit_of_measure", "")).strip()
    return GenCDE(
        gencde_id=str(anchor.get("rop_accession") or anchor.get("id") or f"ROP:{item}"),
        preferred_name=item,
        title=item,
        definition=str(anchor.get("description", "")).strip(),
        question_text="",
        data_type=data_type,
        permissible_values=pv,
        units=uom or None,
        minimum_value=minimum,
        maximum_value=maximum,
        aliases=aliases,
        generated_by="rule",
    )


def load_rop_anchors(anchors_dir: Path) -> list[GenCDE]:
    """Load expert-curated RoP anchors from a local directory of ``*.json`` files -> GenCDEs.

    Each file may hold a single anchor object or a list of anchor objects.
    """
    out: list[GenCDE] = []
    for path in sorted(Path(anchors_dir).glob("*.json")):
        data = json.loads(path.read_text())
        for anchor in data if isinstance(data, list) else [data]:
            if isinstance(anchor, dict) and anchor.get("item"):
                out.append(rop_anchor_to_gencde(anchor))
    return out


def ensure_rop_anchors() -> Path:
    """Fetch the RoP expert-curated anchors into the gitignored cache (network; best-effort). Returns the dir.

    CC-BY-NC: for internal benchmarking with attribution only — never bundled or shipped.
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    if list(_CACHE.glob("*.json")):
        return _CACHE
    req = urllib.request.Request(_ROP_ANCHORS_API, headers={"User-Agent": "ddharmon-benchmarks"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - trusted GitHub API host
        listing = json.load(resp)
    for entry in listing:
        if entry.get("name", "").endswith(".json") and entry.get("download_url"):
            dest = _CACHE / entry["name"]
            dreq = urllib.request.Request(entry["download_url"], headers={"User-Agent": "ddharmon-benchmarks"})
            with urllib.request.urlopen(dreq, timeout=120) as dr, open(dest, "wb") as f:  # noqa: S310
                f.write(dr.read())
    return _CACHE


def _text_sim(a: str, b: str, embed: Callable[[list[str]], np.ndarray] | None) -> float:
    """Semantic similarity of two short strings in [0,1]. Cosine if an embedder is given, else lexical."""
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _norm(a) == _norm(b):
        return 1.0
    if embed is not None:
        va, vb = embed([a])[0], embed([b])[0]
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return max(0.0, min(1.0, float(np.dot(va, vb)) / denom)) if denom else 0.0
    return fuzz.token_sort_ratio(_norm(a), _norm(b)) / 100.0


def _pv_jaccard(a: list[ResponseOption], b: list[ResponseOption]) -> float:
    """Jaccard overlap of two permissible-value sets by normalized label. Both empty -> 1.0."""
    sa = {_norm(ro.label) for ro in a if _norm(ro.label)}
    sb = {_norm(ro.label) for ro in b if _norm(ro.label)}
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def component_scores(a: GenCDE, b: GenCDE, embed: Callable[[list[str]], np.ndarray] | None = None) -> dict[str, float]:
    """Per-component semantic equivalence between two GenCDEs, in [0,1] (FAIRkit's four components)."""
    return {
        "variable_name": _text_sim(a.preferred_name, b.preferred_name, embed),
        "title": _text_sim(a.title, b.title, embed),
        "definition": _text_sim(a.definition, b.definition, embed),
        "permissible_values": _pv_jaccard(a.permissible_values, b.permissible_values),
    }


def _mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=float)
    return (float(arr.mean()), float(arr.std())) if arr.size else (0.0, 0.0)


def reproducibility_report(
    runs: Sequence[Sequence[GenCDE]],
    embed: Callable[[list[str]], np.ndarray] | None = None,
) -> dict:
    """Component-wise semantic-equivalence across K regeneration runs of the same novels (FAIRkit protocol).

    Matches GenCDEs across runs by ``gencde_id`` and averages :func:`component_scores` over every run-pair
    for each shared id. Returns ``{component: {mean, std}, n_concepts, n_runs, published_fairkit}``.
    """
    if len(runs) < 2:
        raise ValueError("reproducibility needs >= 2 regeneration runs")
    by_run = [{g.gencde_id: g for g in run} for run in runs]
    shared = set.intersection(*(set(d) for d in by_run)) if by_run else set()
    per_component: dict[str, list[float]] = {
        k: [] for k in ("variable_name", "title", "definition", "permissible_values")
    }
    for gid in shared:
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                s = component_scores(by_run[i][gid], by_run[j][gid], embed)
                for k, v in s.items():
                    per_component[k].append(v)
    report = {"n_concepts": len(shared), "n_runs": len(runs), "published_fairkit": PUBLISHED_FAIRKIT}
    for k, xs in per_component.items():
        mean, std = _mean_std(xs)
        report[k] = {"mean": round(mean, 3), "std": round(std, 3)}
    return report
