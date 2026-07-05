"""Expert-in-the-loop (EITL) review-campaign export for the v2 split-aware pipeline.

Turns a :class:`~ddharmon.harmonization.leanb.LeanBResult` into reviewer-ready CSV
campaigns:

- ``<stem>_match_review.csv`` — adopt/refine groups, one row per distinct source
  *question* paired with the matched CDE (the LLM's actual pick, not a re-derived
  top-1). Binary review: does the source question mean the same as the CDE?
- ``<stem>_outlier_check.csv`` — (only when embeddings are supplied) the
  centroid-furthest member of each large group vs. its concept, to catch strays.
- ``<stem>_freetext_review.csv`` — open-text / comment fields routed *out* of the
  match campaign (they have no value-codable answer set, so CDE matching does not
  apply); concept-only review.

Two things are load-bearing and were learned the hard way:

1. **The A→B import contract.** Review-UI cells must contain NO raw CR/LF. Multi-line
   content uses U+2028 (LINE SEPARATOR) written as the escape — a *pasted* raw U+2028
   gets normalized to a space by editors, silently producing run-on text. Blank lines
   between sections use ``U+2028 + U+200B + U+2028`` (a bare double-U+2028 collapses).
   CSVs are ``QUOTE_ALL``. Split a campaign with the ``csv`` module, never
   ``str.splitlines()`` (it also splits on U+2028).
2. **Reviewer-pass refinements** (validated against real EITL campaigns): free-text
   routing, templated-family collapse, catch-all magnet flagging, a conservative
   qualifier-divergence granularity flag, honest ``match_cosine`` (no mislabeled
   "confidence"), and target-card framing that doesn't mislabel a CDE designation as a
   variable name. All are cohort-agnostic, env-overridable diagnostics — never hard gates.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ddharmon.models.data_dictionary import DataDictionary, Field

if TYPE_CHECKING:
    # Import-time-only: a runtime import would close a cycle (harmonization.transform imports this module),
    # so importing ``ddharmon.export.eitl`` before ``ddharmon.harmonization`` would fail. LeanBResult is
    # used only in annotations (strings under ``from __future__ import annotations``).
    from ddharmon.harmonization.leanb import LeanBResult

# --- A→B import contract --------------------------------------------------------
LS = "\u2028"  # LINE SEPARATOR — survives CSV import, renders as a line break in the review UI.
_ZWSP = "\u200b"  # zero-width space; gives the blank-line segment non-empty content that survives a trim.
_GAP = LS + _ZWSP + LS  # an EITL-safe blank line between labeled sections.

# Per-cohort source documentation, so a reviewer can check a variable against its origin.
# Cohort-agnostic: an unknown cohort just gets no link. Override via export(..., source_docs=...).
DEFAULT_SOURCE_DOCS = {
    "CLSA": "https://www.clsa-elcv.ca/data-collection",
    "UKBB": "https://biobank.ndph.ox.ac.uk/showcase/search.cgi",
    "AllOfUs": "https://databrowser.researchallofus.org/survey",
}

# --- reviewer-pass diagnostics (env-overridable; cohort-agnostic; never hard gates) ---
FAMILY_MIN = int(os.environ.get("EITL_FAMILY_MIN", "4"))  # collapse >=N same-target rows sharing a name stem
FAMILY_PREFIX = int(os.environ.get("EITL_FAMILY_PREFIX", "6"))  # min shared name-prefix chars
MAGNET_MIN = int(os.environ.get("EITL_MAGNET_MIN", "8"))  # flag a CDE absorbing >=N distinct (post-collapse) sources
OUTLIER_MIN_N = int(os.environ.get("EITL_OUTLIER_MIN_N", "5"))  # only probe groups with >=N members

# Generic open-text signals (English conventions, NOT cohort-specific identifiers).
_OPEN_TOKEN = re.compile(
    r"(specify|describe|free[\s_-]?text|please list|elaborat|in your own words|"
    r"other,?\s*(please\s*)?(specify|describe|list))",
    re.I,
)
_TEXT_DT = {"text", "string", "char", "character", "free text"}

# Domain-general qualifier-axis words (English concepts, NOT cohort identifiers).
_QUALIFIER_WORDS = [
    "home",
    "work",
    "mailing",
    "employment",
    "residence",
    "residential",
    "business",
    "office",
    "spouse",
    "partner",
    "mother",
    "father",
    "parent",
    "parental",
    "maternal",
    "paternal",
    "child",
    "children",
    "son",
    "daughter",
    "sibling",
    "brother",
    "sister",
    "grandparent",
    "grandmother",
    "grandfather",
    "contact",
    "proxy",
    "informant",
    "caregiver",
    "left",
    "right",
    "bilateral",
    "ipsilateral",
    "contralateral",
]


# --- contract helpers -----------------------------------------------------------
def clean(s: object) -> str:
    """Collapse ALL whitespace (incl. newlines/tabs) to single spaces — the newline-guard."""
    return " ".join(str(s).split())


def pack(lines: list[str]) -> str:
    """Join cleaned, non-empty lines with U+2028 (an EITL-safe break). Asserts no raw CR/LF."""
    out = LS.join(clean(x) for x in lines if clean(x))
    assert "\n" not in out and "\r" not in out, "raw newline leaked into a field"
    return out


def labeled(pairs: Sequence[tuple[str, object]]) -> str:
    """Render ``(label, value)`` pairs as ``Label: value`` sections separated by blank lines."""
    out = _GAP.join(f"{lbl}: {clean(val)}" for lbl, val in pairs if clean(val))
    assert "\n" not in out and "\r" not in out, "raw newline leaked into a field"
    return out


def cde_url(tiny_id: str) -> str:
    """Link to the NIH CDE repository entry (the target side)."""
    return f"https://cde.nlm.nih.gov/deView?tinyId={tiny_id}" if tiny_id else ""


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as a QUOTE_ALL CSV after asserting no cell contains a raw CR/LF."""
    if not rows:
        return
    cols = list(rows[0].keys())
    for r in rows:
        for c in cols:
            assert "\n" not in str(r.get(c, "")) and "\r" not in str(r.get(c, "")), f"newline in {path.name}:{c}"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


def qtext(f: Field | None) -> str:
    """The human-readable QUESTION TEXT for semantic matching — never the variable name or values."""
    if f is None:
        return ""
    return clean(f.question_text or f.description or f.short_label or "")


def build_cde_lookup(cde_dict: DataDictionary) -> dict[str, dict]:
    """designation -> {tinyId, question_text, definition} from a loaded NIH_CDE dictionary.

    The CDE catalog is loaded with ``field_id="tinyId"``, ``question_text="question_text"``,
    ``description="definition"`` (see the public pipeline loader), so we read those back here.
    """
    out: dict[str, dict] = {}
    for designation, fld in cde_dict.fields.items():
        if designation not in out:
            out[designation] = {
                "tinyId": fld.field_id or "",
                "question_text": fld.question_text or "",
                "definition": fld.description or "",
            }
    return out


def target_card(designation: str, cde: dict) -> str:
    """Render the target CDE without mislabeling its designation as a 'Variable name'.

    Many CDE designations are phrased AS questions; labeling them 'Variable name' read as a
    confusing duplicate of the question. Show the designation under 'CDE' only when it adds
    something beyond the question text; if there is no separate question text, the designation
    IS the question.
    """
    des_c, qt_c = clean(designation), clean(cde.get("question_text", ""))
    desc = clean(cde.get("definition", ""))[:280]
    if qt_c:
        cde_name = "" if des_c.lower() == qt_c.lower() else des_c
        return labeled([("CDE", cde_name), ("Question text", qt_c), ("Description", desc)])
    return labeled([("CDE (question)", des_c), ("Description", desc)])


def _var(fid: str) -> str:
    """The bare variable name from a ``cohort:var`` member id."""
    return fid.split(":", 1)[-1].split("#", 1)[0]


def _quals(name: str) -> frozenset[str]:
    h = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").lower()
    toks = set(h.split())
    # whole-token match (any length) OR substring match for long, low-false-positive words only
    # (avoids 'son' inside 'person', 'work' inside 'network', etc.)
    return frozenset(q for q in _QUALIFIER_WORDS if q in toks or (len(q) >= 6 and q in h))


def looks_freetext(f: Field | None, fid: str) -> bool:
    """A field is open-text (not value-harmonizable to a coded CDE) on GENERIC signals: an open-text
    token in the question/name, or a text data type with no response options. Conservative — errs
    toward keeping a field in the main campaign rather than wrongly routing a real concept out.
    """
    if f is None:
        return False
    name = _var(fid)
    q = (getattr(f, "question_text", "") or "") + " " + (getattr(f, "short_label", "") or "")
    if _OPEN_TOKEN.search(q) or _OPEN_TOKEN.search(name):
        return True
    dt = (getattr(f, "data_type", "") or "").strip().lower()
    return dt in _TEXT_DT and not getattr(f, "response_options", None)


def qualifier_divergence(fids: list[str]) -> str:
    """Granularity-loss flag — CONSERVATIVE. Fires only when >=2 fields sharing a question text carry
    DIFFERENT domain-general qualifier-axis words in their names (work vs home, mother vs father, left
    vs right) — distinct QUALIFIED concepts collapsed onto one CDE because the discriminating qualifier
    lives only in the name. Does NOT fire on benign naming differences (sex / biological_sex)."""
    if len(fids) < 2:
        return ""
    names = [_var(f) for f in fids]
    per = [_quals(n) for n in names]
    distinct_quals = set().union(*per) if per else set()
    if len(distinct_quals) < 2 or len({q for q in per if q}) < 2:
        return ""  # no qualifier words, or all carry the same single qualifier
    shown = ", ".join(names[:4]) + ("…" if len(names) > 4 else "")
    return (
        f"Granularity check: {len(fids)} variables share this question text but their names carry different "
        f"qualifiers ({', '.join(sorted(distinct_quals))}; e.g. {shown}) — likely distinct qualified concepts "
        "(address type / subject / laterality). Confirm one shared element (record the qualifier as context) "
        "vs. split into distinct elements."
    )


def collapse_families(rows: list[dict]) -> tuple[list[dict], int]:
    """Collapse templated families: >=FAMILY_MIN rows pointing at the SAME target CDE whose source
    variable names share a >=FAMILY_PREFIX-char prefix (relatives / measurements / numbered slots)
    -> ONE representative row (highest cosine). Cohort-agnostic. Returns ``(kept_rows, n_collapsed)``.
    """
    by_target: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_target[r["target_id"]].append(r)
    kept: list[dict] = []
    collapsed = 0
    for group in by_target.values():
        if len(group) < FAMILY_MIN:
            kept.extend(group)
            continue
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in group:
            buckets[_var(r["source_id"]).lower()[:FAMILY_PREFIX]].append(r)
        for pref, fam in buckets.items():
            if len(fam) < FAMILY_MIN or len(pref) < FAMILY_PREFIX:
                kept.extend(fam)
                continue
            fam.sort(key=lambda r: -r["match_cosine"])
            rep = dict(fam[0])
            others = [_var(r["source_id"]) for r in fam[1:]]
            note = (
                f"Templated family: {len(fam)} variables point at this same CDE, differing only by a name "
                f"token (e.g. {', '.join(others[:4])}{'…' if len(others) > 4 else ''}). Review once — the "
                "differing token is a slot/qualifier (relative, measurement, condition), not a separate concept."
            )
            rep["source_text"] = rep["source_text"] + _GAP + clean(note)
            assert "\n" not in rep["source_text"] and "\r" not in rep["source_text"]
            rep["n_vars_sharing"] = sum(r["n_vars_sharing"] for r in fam)
            rep["review_unit"] = "templated_family"
            rep["source_id"] = f"{rep['leaf_uid']}#family:{pref}"
            kept.append(rep)
            collapsed += len(fam) - 1
    return kept, collapsed


def _field_lookup(source_dicts: Mapping[str, DataDictionary]) -> dict[str, Field]:
    """Build ``"cohort:var" -> Field`` across all source cohort dictionaries."""
    out: dict[str, Field] = {}
    for cohort, dd in source_dicts.items():
        for vn, f in dd.fields.items():
            out[f"{cohort}:{vn}"] = f
    return out


def export_split_eitl_campaign(
    result: LeanBResult,
    source_dicts: Mapping[str, DataDictionary],
    cde_lookup: dict[str, dict],
    out_dir: str | Path,
    *,
    stem: str = "eitl",
    embedded: Mapping[str, object] | None = None,
    source_docs: Mapping[str, str] | None = None,
) -> dict[str, int]:
    """Write the EITL review campaign CSVs for a split-aware ``LeanBResult``.

    Args:
        result: the v2 harmonization result (one record per concept-group).
        source_dicts: cohort name -> source DataDictionary (for member question text / data type).
        cde_lookup: designation -> {tinyId, question_text, definition} (see :func:`build_cde_lookup`).
        out_dir: directory to write the CSV(s) into.
        stem: filename stem; files are ``<stem>_match_review.csv`` etc.
        embedded: optional cohort -> EmbeddedDictionary; when given, an outlier-check campaign is
            produced (centroid-furthest member of each large group).
        source_docs: optional cohort -> documentation URL override (defaults to DEFAULT_SOURCE_DOCS).

    Returns:
        dict of campaign name -> row count written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = _field_lookup(source_dicts)
    docs = dict(DEFAULT_SOURCE_DOCS, **(source_docs or {}))

    def src_url(fid: str) -> str:
        return docs.get(fid.split(":", 1)[0], "")

    match_rows: list[dict] = []
    n_no_qtext = 0

    for rec in result.records:
        if rec.verdict not in ("adopt", "refine") or not rec.cde_id:
            continue  # novel routes to GenCDE residual — no existing-CDE claim to review
        des = rec.cde_id
        cde = cde_lookup.get(des, {})
        tiny = rec.cde_external_id or cde.get("tinyId", "")
        tgt_text = target_card(des, cde)
        match_cos = round(float(rec.chosen_cos if rec.chosen_cos is not None else (rec.top1_cos or 0.0)), 4)
        ideal = clean(rec.ideal_cde)

        # one row per distinct source QUESTION within the group
        by_q: dict[str, list[str]] = defaultdict(list)
        for fid in rec.member_variable_names:
            q = qtext(fields.get(fid))
            if not q:
                n_no_qtext += 1
                continue
            by_q[q].append(fid)

        for fids in by_q.values():
            f0 = fields.get(fids[0])
            vname = _var(fids[0])
            cohorts = sorted({f.split(":", 1)[0] for f in fids})
            qt = clean(getattr(f0, "question_text", "") or "") or clean(getattr(f0, "short_label", "") or "")
            desc = clean(getattr(f0, "description", "") or "")
            more = f" (+{len(fids) - 1} more variable(s) share this question)" if len(fids) > 1 else ""
            show_desc = desc if desc and desc != qt and desc != vname else ""
            match_rows.append(
                {
                    "source_text": labeled(
                        [("Variable name", vname + more), ("Question text", qt), ("Description", show_desc)]
                    ),
                    "source_id": fids[0],
                    "source_dataset": "+".join(cohorts),
                    "source_url": src_url(fids[0]),
                    "target_text": tgt_text,
                    "target_id": tiny,
                    "target_dataset": "NIH_CDE",
                    "target_url": cde_url(tiny),
                    "pair_type": rec.verdict,
                    "match_cosine": match_cos,
                    "llm_reasoning": pack(
                        [
                            rec.rationale or "(no rationale recorded for this match)",
                            f"Ideal CDE (LLM-generated coverage anchor): {ideal}" if ideal else "",
                            qualifier_divergence(fids),
                            "Semantic match only: response values / data types are checked in the transformation review.",
                        ]
                    ),
                    "leaf_uid": rec.cluster_id,
                    "cross_cohort": rec.cross_cohort,
                    "n_vars_sharing": len(fids),
                    "review_unit": "question",
                }
            )

    # ── reviewer-pass post-process: free-text routing, family collapse, magnet flag ──
    def _is_ft(r: dict) -> bool:
        return looks_freetext(fields.get(r["source_id"]), r["source_id"])

    freetext_rows = [r for r in match_rows if _is_ft(r)]
    keep_rows = [r for r in match_rows if not _is_ft(r)]
    keep_rows, n_collapsed = collapse_families(keep_rows)

    per_tgt: dict[str, list[dict]] = defaultdict(list)
    for r in keep_rows:
        if r["target_id"]:
            per_tgt[r["target_id"]].append(r)
    n_magnets = 0
    for rs in per_tgt.values():
        if len(rs) < MAGNET_MIN:
            continue
        n_magnets += 1
        note = (
            f"Catch-all check: {len(rs)} distinct source concepts were matched to this one CDE. Generic CDEs "
            "act as magnets — verify THIS field truly IS this CDE, not a coverage gap (novel)."
        )
        for r in rs:
            r["llm_reasoning"] = r["llm_reasoning"] + _GAP + clean(note)

    keep_rows.sort(key=lambda r: -r["match_cosine"])
    freetext_rows.sort(key=lambda r: -r["match_cosine"])

    counts: dict[str, int] = {}
    write_csv(out_dir / f"{stem}_match_review.csv", keep_rows)
    counts["match_review"] = len(keep_rows)
    if freetext_rows:
        write_csv(out_dir / f"{stem}_freetext_review.csv", freetext_rows)
    counts["freetext_review"] = len(freetext_rows)

    outlier_rows = _build_outlier_rows(result, fields, embedded, src_url) if embedded else []
    if outlier_rows:
        write_csv(out_dir / f"{stem}_outlier_check.csv", outlier_rows)
    counts["outlier_check"] = len(outlier_rows)
    counts["skipped_no_question"] = n_no_qtext
    counts["collapsed_families"] = n_collapsed
    counts["magnet_cdes"] = n_magnets
    return counts


def _build_outlier_rows(
    result: LeanBResult,
    fields: dict[str, Field],
    embedded: Mapping[str, object],
    src_url,
) -> list[dict]:
    """Centroid-furthest member of each large group vs. its concept (needs embeddings)."""
    import numpy as np

    def vec(fid: str):
        cohort, _, var = fid.partition(":")
        emb = embedded.get(cohort)
        return None if emb is None else emb.embeddings.get(var)  # type: ignore[attr-defined]

    rows: list[dict] = []
    for rec in result.records:
        members = rec.member_variable_names
        if len(members) < OUTLIER_MIN_N:
            continue
        vecs = {m: vec(m) for m in members}
        usable = {m: v for m, v in vecs.items() if v is not None}
        if len(usable) < OUTLIER_MIN_N:
            continue
        mat = np.stack(list(usable.values())).astype(np.float32)
        cen = mat.mean(axis=0)
        cen = cen / (np.linalg.norm(cen) or 1.0)
        cos = {m: float((v / (np.linalg.norm(v) or 1.0)) @ cen) for m, v in usable.items()}
        order = sorted(cos, key=lambda m: cos[m])
        far = order[0]
        close = order[-3:][::-1]
        ff = fields.get(far)
        fq = qtext(ff)
        if not fq:
            continue
        rows.append(
            {
                "source_text": labeled(
                    [
                        ("Variable name", _var(far)),
                        ("Question text", fq),
                        ("Description", clean(getattr(ff, "description", "") or "")),
                    ]
                ),
                "source_id": far,
                "source_dataset": far.split(":", 1)[0],
                "source_url": src_url(far),
                "target_text": labeled(
                    [("Cluster concept", rec.concept)] + [("Other member", qtext(fields.get(c)) or "?") for c in close]
                ),
                "target_id": rec.cluster_id,
                "target_dataset": f"leaf:{rec.cluster_id}",
                "target_url": "",
                "pair_type": "outlier_check",
                "llm_reasoning": pack(
                    [
                        f"This is the group's centroid-FURTHEST member (cos = {cos[far]:.3f}).",
                        "Does its question share the group's concept, or is it a stray?",
                    ]
                ),
                "leaf_uid": rec.cluster_id,
                "member_centroid_cos": round(cos[far], 3),
            }
        )
    rows.sort(key=lambda r: r["member_centroid_cos"])
    return rows
