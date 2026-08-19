"""Cohort-agnostic text hygiene shared by ingestion, the harmonization prompts, and the
calibration tooling.

This is the single, lightweight (``re`` + ``html`` only — no numpy/torch) home for three
classes of *source-data artifact* that pollute what the LLM (judge / assign / split / spec-gen)
and the embeddings see. Each was discovered on real dictionary data while rebuilding an
expert-review campaign for the coherence judge:

1. **Administrative / data-collection text** (:func:`clean_field_text`) — instrument-administration
   preambles (``ACE touchscreen question "…"``), help-feature message tails, HTML tags/entities, and
   generic survey/CDE instruction boilerplate. Noise that dilutes concept signal in both the
   embedding text and the prompts.
2. **Missing / refused / don't-know sentinel codes** (:func:`is_sentinel_label`,
   :func:`strip_sentinel_encodings`) — non-substantive response codes (``-9=MISSING``,
   ``-3=Prefer not to answer``) encoded as if they were real answer options. For a numeric field
   whose only encoding is ``-9=MISSING`` the model would read ``values: -9=MISSING`` and mistake it
   for a single-option categorical variable.

All rules are **cohort-agnostic and discovered-not-hardcoded**: patterns are abstracted to the
generic *class* of artifact (an administration wrapper, a sentinel LABEL) rather than any
cohort-specific string, per the preprocessing-design + no-overfit conventions. Sentinel detection
is LABEL-based (not code-based) so it works across cohorts with different missing-code conventions.

Previously these patterns were mirrored ad-hoc in ``scripts/coherence_calibration.py`` (display side)
and partially in ``harmonization/leanb.py`` (``CDE_TEXT_BOILERPLATE``); this module unifies them.
"""

from __future__ import annotations

import html
import re

# ---------------------------------------------------------------------------
# 1. Administrative / data-collection text
# ---------------------------------------------------------------------------

# Generic survey/CDE instruction boilerplate that carries no concept signal but pollutes BM25
# (spurious lexical hits) and clutters the prompt/candidate blocks. UNIVERSAL data-collection
# artifacts (interviewer / skip-logic / multi-select instructions), NOT cohort-specific — the M5
# audit's worst case was ethnicity matching a CDE whose only distinctive text was "READ IF
# NECESSARY" @0.45. Matched case-insensitively as whole phrases; extend generically, never add a
# cohort-specific term. (Moved here from harmonization/leanb.py so ingestion + prompts + display
# share one list.)
CDE_TEXT_BOILERPLATE: tuple[str, ...] = (
    "read if necessary",
    "do not read",
    "read out",
    "read all that apply",
    "select all that apply",
    "check all that apply",
    "mark all that apply",
    "choose all that apply",
    "select one",
    "if necessary",
    "for office use only",
    "office use only",
    "see instructions",
    "please specify",
)
_BOILERPLATE_RE = re.compile("|".join(re.escape(p) for p in CDE_TEXT_BOILERPLATE), re.IGNORECASE)

# An instrument-administration preamble ("ACE touchscreen question", "Sleep online question:", …)
# IMMEDIATELY FOLLOWED by the real question in quotes — unwrap to the quoted question and drop the
# trailing help/HTML tail. Anchored: at most 80 chars of preamble, no quote inside the preamble (so
# we never cut INTO a quoted question that itself mentions "question"), and the closing quote must be
# followed by an HTML tag or end-of-string. Requiring the following quote is what makes this safe to
# run over EVERY field at ingest — a benign "Questions about diet" (no quote) is left untouched.
_PREAMBLE_QUOTED_RE = re.compile(
    r'^[^"“”]{0,80}?\bquestions?\b\s*:?\s*["“”\'](.*?)["“”\']\s*(?=<|$)',
    re.IGNORECASE | re.DOTALL,
)
# A leading quoted question with NO preamble whose closing quote is followed by an HTML tag — the
# trailing markup (a help-message table/break) is the administration signature that makes unwrapping
# safe. Requiring the tag (not end-of-string) means a lone quoted string — e.g. a quoted option-label
# echo like `"Do not know"` — is NOT unwrapped here (that is the option-echo step's job).
_QUOTED_RE = re.compile(r'^\s*["“”\'](.*?)["“”\']\s*(?=<)', re.DOTALL)
# Match only RECOGNISED HTML tags (a whitelist of common elements) rather than any `<…>` — so a
# domain angle-bracket token like MESA's `<OR>` disjunction separator (or `<50%`, `<=3`) is preserved
# while `<p>`, `<br>`, `<table …>`, `</td>` are stripped.
_HTML_TAG_NAMES = (
    "p|br|div|span|table|thead|tbody|tfoot|tr|td|th|ul|ol|li|dl|dt|dd|b|i|u|em|strong|a|h[1-6]|hr|"
    "img|sup|sub|blockquote|pre|code|small|font|center|caption|colgroup|col"
)
_TAG_RE = re.compile(rf"</?(?:{_HTML_TAG_NAMES})\b[^>]*>", re.IGNORECASE)


def clean_field_text(s: object, *, strip_boilerplate: bool = True) -> str:
    """Strip administrative / data-collection wrappers from a field's text, UNTRUNCATED.

    Pipeline: unescape HTML entities → unwrap a leading quoted question out of an instrument
    preamble (dropping the trailing help HTML) → strip recognised HTML tags → (optionally) strip
    CDE/survey boilerplate phrases → collapse whitespace.

    ``strip_boilerplate`` (default True) removes short generic phrases like "select all that apply"
    / "please specify" (:data:`CDE_TEXT_BOILERPLATE`). This is on for the display/CDE-candidate paths
    but the ingestion preprocessor calls it with ``strip_boilerplate=False``: mid-sentence phrase
    removal leaves degenerate residue ("Please specify." → ".", "(check all that apply)?" → "( )?")
    and nickel-and-dimes cohorts that have no *structural* artifacts — the preprocess step targets
    the structural wrappers (HTML / instrument preamble / help tail) only.

    Returns ``""`` for empty/whitespace-only input. May return ``""`` when the input was *pure*
    markup (e.g. ``"<p></p>"``); callers that must never blank a field (ingestion) apply the result
    only when it is a non-empty change (see ``preprocessor._strip_administrative_text``).
    """
    if not s:
        return ""
    text: str = html.unescape(str(s))
    m = _PREAMBLE_QUOTED_RE.match(text) or _QUOTED_RE.match(text)
    if m:
        text = m.group(1)
    text = _TAG_RE.sub(" ", text)
    if strip_boilerplate:
        text = _BOILERPLATE_RE.sub(" ", text)
    return " ".join(text.split()).strip()


# ---------------------------------------------------------------------------
# 2. Missing / refused / don't-know sentinel codes
# ---------------------------------------------------------------------------

# Non-substantive response codes encoded as if they were real answer options (MESA ``-9=MISSING``,
# UKBB ``-3=Prefer not to answer``). Matched on the LABEL (cohort-agnostic — codes differ per
# cohort). SUBSTR = matched anywhere in the label; EXACT = whole-label only (short tokens like "na"
# / "dk" that would over-match as substrings of real words e.g. "banana", "dka").
_SENTINEL_SUBSTR: tuple[str, ...] = (
    "missing",
    "prefer not",
    "not applicable",
    "do not know",
    "don't know",
    "dont know",
    "refused",
    "not asked",
    "declined",
)
_SENTINEL_EXACT: frozenset[str] = frozenset(
    {"unknown", "n/a", "na", "no answer", "dk", "nk", "not sure", "refuse", "none available"}
)


def is_sentinel_label(lbl: object) -> bool:
    """True if ``lbl`` is a missing/refused/don't-know/not-applicable sentinel (or empty).

    Label-based and cohort-agnostic. Treats empty/whitespace as a sentinel so callers filtering an
    option list drop blanks too.
    """
    s = str(lbl or "").strip().lower()
    if not s:
        return True
    return any(sub in s for sub in _SENTINEL_SUBSTR) or s in _SENTINEL_EXACT


def _encoding_label(part: str) -> str:
    """Extract the LABEL from one ``code=label`` (or ``code, label``) encoding fragment."""
    part = part.strip()
    if "=" in part:
        return part.split("=", 1)[1].strip()
    if "," in part:
        return part.split(",", 1)[1].strip()
    return part


def filter_sentinel_labels(labels: list[str]) -> list[str]:
    """Drop sentinel labels (and blanks) from an already-parsed option-label list, order-preserving."""
    return [x for x in labels if not is_sentinel_label(x)]


def strip_sentinel_encodings(raw: object) -> str:
    """Remove sentinel entries from a ``code=label|code=label`` value-encoding string.

    Splits on ``|``, drops any fragment whose LABEL is a sentinel (keeping the ``code=label`` shape
    of the survivors), and rejoins. A field whose encoding is *only* sentinels (e.g. a numeric field
    encoded ``-9=MISSING``) collapses to ``""`` → the caller renders no option tail, so the model
    reads it as numeric rather than a single-option categorical.

    Cosmetic separators (surrounding whitespace) are preserved on kept fragments so the rendered
    value set matches the source style.
    """
    if not raw:
        return ""
    kept = [frag for frag in str(raw).split("|") if not is_sentinel_label(_encoding_label(frag))]
    return "|".join(kept)
