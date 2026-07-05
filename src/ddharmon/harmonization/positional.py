"""Detect POSITIONAL-ENUMERATION concepts ($0, no LLM) — wide-format repeating measures.

A positional enumeration is a repeating measure encoded as numbered columns (CLSA ``Prescribed -
Medication 1..40``, UKBB ``FI1..FI13: duration viewed``, …). The integer is an OCCURRENCE INDEX (which
slot in a list), NOT a semantic qualifier — so the columns are ONE repeating concept (index -> array
position), and harmonizing/reviewing each numbered column separately is wrong and wasteful.

Detection is cohort-agnostic and rule-based: strip every digit-run in each member label to ``#`` to get a
SIGNATURE. If one signature dominates the group AND the integers it varies over are several & contiguous,
it's a positional enumeration. This is exactly what distinguishes it from a genuine qualifier matrix (whose
members keep DISTINCT signatures after stripping digits — each item asks something different).

Thresholds are parameters (not hard-coded gates) so the behavior stays tunable and cohort-agnostic. Ported
from the nb05 research sandbox (``enrich_positional.py``; Run 016c/d).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

DOMINANT_SHARE = 0.70  # one stripped signature must cover >= this fraction of members
MIN_DISTINCT_INTS = 4  # the varying integer must take >= this many distinct values
MIN_DENSITY = 0.50  # distinct_ints / (max-min+1): the integer range is reasonably contiguous (not sparse)

_DIGITS = re.compile(r"\d+")
_NON_ALPHA_HASH = re.compile(r"[^a-z#]+")

# Enumerated-entity family (M2 Phase 2b): same-template/different-ENTITY members (a food-frequency battery,
# a medication/condition checklist). Distinct from a positional enumeration (the varying slot is a WORD, not
# a digit). Per the split rule these are ONE rollup concept — recognizing the family lets the pipeline skip
# chunk-splitting it. Conservative by construction (high precision): a genuine heterogeneous pool fails the
# high per-member template fraction + short-slot checks.
FAMILY_MIN_MEMBERS = 8  # a family is MANY entities; small sets aren't worth collapsing (and are riskier)
FAMILY_DOMINANT_SHARE = 0.70  # >= this fraction of members must fit the shared template
FAMILY_MIN_TEMPLATE_TOKENS = 3  # the shared stem must be a real question, not a couple of filler words
FAMILY_MAX_SLOT_TOKENS = 3  # the varying entity slot must be SHORT (an entity, not a distinct question)
FAMILY_MIN_TEMPLATE_FRAC = 0.60  # each fitting member must be MOSTLY template (little varies)

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class PositionalEnumeration:
    """Evidence that a group of member labels is a wide-format repeating measure."""

    signature: str  # the dominant digit-stripped label, e.g. "medication #"
    dominant_share: float  # fraction of members matching the signature
    n_occurrences: int  # count of distinct integers the signature varies over
    int_range: tuple[int, int]  # (min, max) of those integers
    density: float  # n_occurrences / (max - min + 1)


def signature(text: str) -> str:
    """Digit-stripped signature of a label: every digit-run -> ``#``, punctuation collapsed.

    ``"Prescribed - Medication 12"`` and ``"Prescribed - Medication 3"`` -> ``"prescribed medication #"``.
    """
    stripped = _DIGITS.sub("#", text.lower())
    return " ".join(_NON_ALPHA_HASH.sub(" ", stripped).split())


def detect_positional_enumeration(
    labels: list[str],
    *,
    dominant_share: float = DOMINANT_SHARE,
    min_distinct_ints: int = MIN_DISTINCT_INTS,
    min_density: float = MIN_DENSITY,
) -> PositionalEnumeration | None:
    """Return the enumeration evidence if ``labels`` are a numbered repeating measure, else ``None``.

    ``labels`` are the member labels of one concept group (empty labels are ignored). A group qualifies
    when: one digit-stripped signature (that actually contains a digit) covers ``>= dominant_share`` of the
    labels, and the integers that signature varies over are ``>= min_distinct_ints`` distinct AND span a
    range that is ``>= min_density`` contiguous. A qualifier matrix (distinct signatures per item) or a
    small/sparse set of numbers is correctly rejected.
    """
    labels = [t for t in (s.strip() for s in labels) if t]
    if len(labels) < min_distinct_ints:
        return None
    sigs = Counter(signature(t) for t in labels)
    dom_sig, dom_n = sigs.most_common(1)[0]
    share = dom_n / len(labels)
    if "#" not in dom_sig or share < dominant_share:
        return None
    ints: set[int] = set()
    for t in labels:
        if signature(t) == dom_sig:
            ints.update(int(x) for x in _DIGITS.findall(t))
    if len(ints) < min_distinct_ints:
        return None
    density = len(ints) / (max(ints) - min(ints) + 1)
    if density < min_density:
        return None
    return PositionalEnumeration(
        signature=dom_sig,
        dominant_share=round(share, 3),
        n_occurrences=len(ints),
        int_range=(min(ints), max(ints)),
        density=round(density, 3),
    )


@dataclass(frozen=True)
class EnumeratedFamily:
    """Evidence that member labels are a same-template / different-ENTITY family (one rollup concept)."""

    template: str  # the shared template tokens (sorted), e.g. "do eat how often you"
    dominant_share: float  # fraction of members that fit the template
    n_entities: int  # count of distinct varying entity slots
    template_tokens: int  # number of shared template tokens
    slot_tokens: float  # median varying-slot length in tokens


def detect_enumerated_family(
    labels: list[str],
    *,
    min_members: int = FAMILY_MIN_MEMBERS,
    dominant_share: float = FAMILY_DOMINANT_SHARE,
    min_template_tokens: int = FAMILY_MIN_TEMPLATE_TOKENS,
    max_slot_tokens: int = FAMILY_MAX_SLOT_TOKENS,
    min_template_frac: float = FAMILY_MIN_TEMPLATE_FRAC,
) -> EnumeratedFamily | None:
    """Return family evidence if ``labels`` are a same-template/different-entity battery, else ``None``.

    A family qualifies when: there are ``>= min_members`` labels; a shared TEMPLATE of ``>=
    min_template_tokens`` tokens (each present in ``>= dominant_share`` of members) exists; ``>=
    dominant_share`` of members FIT it (``>= min_template_frac`` of the member's tokens are template tokens
    AND the varying remainder is ``<= max_slot_tokens``); and the fitting members leave ``>= min_members``
    DISTINCT entity slots. Conservative: a heterogeneous pool shares only filler words (few template tokens)
    or leaves long distinct remainders (low fit), and a positional/numbered family has a 1-token template
    (its entity is a digit) — all rejected here. Cohort-agnostic and rule-based, like
    :func:`detect_positional_enumeration`.
    """
    labels = [t for t in (s.strip() for s in labels) if t]
    toks = [_WORD.findall(t.lower()) for t in labels]
    toks = [t for t in toks if t]
    if len(toks) < min_members:
        return None
    present: Counter[str] = Counter()
    for t in toks:
        present.update(set(t))
    n = len(toks)
    template = {w for w, c in present.items() if c / n >= dominant_share}
    if len(template) < min_template_tokens:
        return None
    residuals: list[tuple[str, ...]] = []
    for t in toks:
        frac = sum(1 for w in t if w in template) / len(t)
        residual = tuple(w for w in t if w not in template)
        if frac >= min_template_frac and len(residual) <= max_slot_tokens:
            residuals.append(residual)
    share = len(residuals) / n
    if share < dominant_share:
        return None
    distinct = {r for r in residuals if r}
    if len(distinct) < min_members:  # many DISTINCT entities — not duplicates / a near-empty slot
        return None
    return EnumeratedFamily(
        template=" ".join(sorted(template)),
        dominant_share=round(share, 3),
        n_entities=len(distinct),
        template_tokens=len(template),
        slot_tokens=round(median(len(r) for r in residuals), 1),
    )
