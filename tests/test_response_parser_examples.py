"""Regression gate for the comma fix: it changes NO cohort parse and only the broken catalog parses.

``parse_value_encoding`` reads every cohort dictionary, so a parser fix is only safe if it is proven not to move
the parses that were already right. This module runs the parser over every value encoding in the shipped example
dictionaries and compares it with the parser as it stood before the fix, frozen below as ``_legacy_parse``:

* All of Us, CLSA, UK Biobank, MESA — every parse is IDENTICAL (measured when the fix landed: 0 of 11,358
  encoded fields changed).
* NIH CDE catalog (``all_cdes_flat.tsv``) — a parse may change only by undoing a comma split made at a non-code
  word ("Autoimmune condition (e.g." | "rheumatoid arthritis, …)") or by splitting a raw ``value=meaning`` item
  the old parser left whole. Nothing else, and the option count never changes (measured: 1,328 of 14,842 fields,
  1,040 unique encodings).
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pytest

from ddharmon.values.response_parser import _is_code_token, parse_value_encoding

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"

#: (file, delimiter, value-encoding column) — the columns ``data/examples/harmonize_example.json`` maps.
COHORTS = {
    "AllOfUs": ("all_of_us_surveys.csv", ",", "Choices, Calculations, OR Slider Labels"),
    "CLSA": ("clsa_baseline.csv", ",", "value_encoding"),
    "UKBB": ("ukbb_showcase.csv", ",", "value_encoding"),
    "MESA": ("mesa_dbgap.csv", ",", "value_encoding"),
}
CDE_CATALOG = ("all_cdes_flat.tsv", "\t", "permissible_values")


def _encodings(name: str, sep: str, col: str) -> list[str]:
    path = EXAMPLES / name
    if not path.exists():
        pytest.skip(f"{name} is not in data/examples")
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as fh:
        return [v.strip() for row in csv.DictReader(fh, delimiter=sep) if (v := (row.get(col) or "").strip())]


# ── the parser before the comma fix, frozen (the oracle; never edit it to make a test pass) ──────────────


def _legacy_parse(raw: str) -> list[tuple[str, str, int]]:
    raw = raw.strip()
    if not raw:
        return []
    for fn in (_legacy_parenthesized, _legacy_code_equals, _legacy_code_comma, _legacy_slash):
        out = fn(raw)
        if out:
            return out
    return []


def _legacy_parenthesized(raw: str) -> list[tuple[str, str, int]] | None:
    if not re.match(r"\s*\(", raw):
        return None
    matches = re.compile(r"\(([^)]+)\)\s*([^|]*)").findall(raw)
    if len(matches) < 2:
        return None
    return [(c.strip(), lab.strip() or c.strip(), i) for i, (c, lab) in enumerate(matches)]


def _legacy_code_equals(raw: str) -> list[tuple[str, str, int]] | None:
    if "=" not in raw:
        return None
    parts = re.split(r"\s*\|\s*", raw)
    if len(parts) < 2:
        return None
    out = []
    for i, part in enumerate(parts):
        part = part.strip()
        if "=" not in part:
            return None
        code, label = (s.strip() for s in part.split("=", 1))
        if code and label:
            out.append((code, label, i))
    return out if len(out) >= 2 else None


def _legacy_code_comma(raw: str) -> list[tuple[str, str, int]] | None:
    if "|" not in raw:
        return None
    parts = re.split(r"\s*\|\s*", raw)
    if len(parts) < 2:
        return None
    out = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if "," not in part:
            out.append((part, part, i))
            continue
        code, label = (s.strip() for s in part.split(",", 1))
        if code:
            out.append((code, label or code, i))
    return out if len(out) >= 2 else None


def _legacy_slash(raw: str) -> list[tuple[str, str, int]] | None:
    if "|" in raw or "=" in raw or "(" in raw:
        return None
    parts = raw.split("/")
    if len(parts) < 2 or len(parts) > 3:
        return None
    out = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part or len(part) > 30:
            return None
        out.append((str(i), part, i))
    return out


def _now(raw: str) -> list[tuple[str, str, int]]:
    return [(o.code, o.label, o.order) for o in parse_value_encoding(raw)]


# ── the gate ──────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cohort", sorted(COHORTS))
def test_cohort_parses_are_unchanged(cohort):
    encodings = _encodings(*COHORTS[cohort])
    assert len(encodings) > 1000  # the dictionary really was read — not a vacuous pass
    changed = [raw for raw in encodings if _now(raw) != _legacy_parse(raw)]
    assert changed == [], f"{cohort}: {len(changed)} parses moved, e.g. {changed[:2]}"


def test_catalog_parses_change_only_where_the_old_parse_was_broken():
    encodings = _encodings(*CDE_CATALOG)
    changed = 0
    for raw in encodings:
        old, new = _legacy_parse(raw), _now(raw)
        if old == new:
            continue
        changed += 1
        assert len(new) == len(old), raw  # one option per item, before and after
        parts = re.split(r"\s*\|\s*", raw)  # an option's `order` is its index here
        items = [p.strip() for p in parts if p.strip()]
        # a list whose every item opens with a code and a comma is a CODED list — its parse must not move
        assert not all("," in p and _is_code_token(p.partition(",")[0].strip()) for p in items), raw
        for (oc, ol, oi), (nc, nl, ni) in zip(old, new, strict=True):
            if (oc, ol, oi) == (nc, nl, ni):
                continue
            assert oi == ni, raw
            item = parts[oi].strip()
            # before: the item was cut at its first comma, or a `value=meaning` item was left whole …
            old_broken = ("," in item and oc == item.split(",", 1)[0].strip()) or (oc == ol == item and "=" in item)
            # … after: the item is whole — its own code and label, or split only at its `value=meaning`
            new_whole = (nc, nl) == (item, item) or bool(re.fullmatch(re.escape(nc) + r"\s*=\s*" + re.escape(nl), item))
            assert old_broken and new_whole, (raw, (oc, ol), (nc, nl))
    assert changed > 0  # the fix is exercised on the real catalog
