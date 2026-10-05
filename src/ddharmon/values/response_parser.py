"""Parse value_encoding_raw strings into ResponseOption lists.

Handles multiple formats found across cohort data dictionaries:

    Arivale:   (1) Less than once per month|(2) 1-3 times per month|...
    All of Us: Birthplace_USA, USA | PMI_Other, Other
    Simple:    Yes/No, Male/Female, 1=Yes|2=No
    NIH CDE:   Asthma | Autoimmune condition (e.g., lupus, vasculitis) | Thrombotic disorders=Thrombotic Disorders

A COMMA IS A CODE/LABEL SEPARATOR ONLY AFTER A CODE. Labels carry commas — "(e.g., a, b, c)",
"Less than $10,000", "Other, Specify" — so splitting a list item on its first comma cut every such label in
two and made the text before the comma a "code" that exists nowhere in the source. The comma form is taken
only when EVERY item opens with a code token followed by a comma; otherwise each item is a bare label (its
own code), or ``value=meaning`` where the catalog wrote one — the NIH CDE flattener emits ``value=meaning``
only where the two differ, so a catalog list is routinely MIXED.
"""

from __future__ import annotations

import logging
import re

from ddharmon.models.data_dictionary import ResponseOption

logger = logging.getLogger(__name__)

#: An item separator: a pipe with any surrounding whitespace.
_ITEM_SEP = re.compile(r"\s*\|\s*")

#: The characters a code token is made of — letters and digits joined by ``_ . : -``.
_CODE_CHARS = re.compile(r"[A-Za-z0-9_.:\-]+")


def parse_value_encoding(raw: str) -> list[ResponseOption]:
    """Parse a value_encoding_raw string into ResponseOption objects.

    Tries formats in order of specificity:
    1. Parenthesized code: (1) Label|(2) Label  (Arivale)
    2. Code-equals-label:  1=Yes|2=No  (every item carries a code)
    3. Code-comma-label:   Code, Label | Code, Label  (REDCap/All of Us; every item opens with a code token)
    4. Pipe list:          Label | Label | value=meaning  (NIH CDE; bare labels, optionally mixed)
    5. Slash-delimited:    Yes/No, Male/Female (2-3 options only)

    Args:
        raw: The raw value encoding string.

    Returns:
        List of ResponseOption objects. Empty list if unparseable.
    """
    raw = raw.strip()
    if not raw:
        return []

    # Try each format
    result = _parse_parenthesized(raw)
    if result:
        return result

    result = _parse_code_equals_label(raw)
    if result:
        return result

    result = _parse_code_comma_label(raw)
    if result:
        return result

    result = _parse_pipe_list(raw)
    if result:
        return result

    result = _parse_slash_delimited(raw)
    if result:
        return result

    return []


def _parse_parenthesized(raw: str) -> list[ResponseOption] | None:
    """Parse (code) label | (code) label format.

    Examples:
        (1) Less than once per month|(2) 1-3 times per month
        (0) No|(1) Yes

    Only matches when options START with (code), not parentheticals mid-label.
    """
    # Must start with (code) pattern or have |(code) after pipe
    if not re.match(r"\s*\(", raw):
        return None

    pattern = re.compile(r"\(([^)]+)\)\s*([^|]*)")
    matches = pattern.findall(raw)

    if len(matches) < 2:
        return None

    options = []
    for i, (code, label) in enumerate(matches):
        label = label.strip()
        if not label:
            label = code.strip()
        options.append(ResponseOption(code=code.strip(), label=label, order=i))

    return options


def _parse_code_equals_label(raw: str) -> list[ResponseOption] | None:
    """Parse code=label|code=label format — every item carries a code.

    Examples:
        1=Yes|2=No
        1=Male|2=Female|3=Other

    A list where only SOME items carry ``=`` is not this format; it is a pipe list (:func:`_parse_pipe_list`).
    """
    if "=" not in raw:
        return None

    parts = _ITEM_SEP.split(raw)
    if len(parts) < 2:
        return None

    options = []
    for i, part in enumerate(parts):
        part = part.strip()
        if "=" not in part:
            return None  # mixed format -> the pipe list handles it
        code, label = part.split("=", 1)
        code = code.strip()
        label = label.strip()
        if not code or not label:
            continue
        options.append(ResponseOption(code=code, label=label, order=i))

    return options if len(options) >= 2 else None


def _is_code_token(token: str) -> bool:
    """Whether ``token`` reads as a CODE rather than the first word of a label.

    A code is an identifier (``[A-Za-z0-9_.:-]``, no spaces) carrying a digit or an underscore — the shape of
    every REDCap / All of Us code ("PMI_Other", "WhatRaceEthnicity_AIAN", "1"). A bare word such as "Other",
    "Yes" or "No" is a label that happens to precede a comma ("Other, Specify", "No, neuronal loss").
    A code token holds no parenthesis, so a comma right after one is always at parenthesis depth 0.
    """
    return bool(_CODE_CHARS.fullmatch(token)) and bool(re.search(r"[\d_]", token))


def _parse_code_comma_label(raw: str) -> list[ResponseOption] | None:
    """Parse Code, Label | Code, Label format (REDCap/All of Us).

    Examples:
        Birthplace_USA, USA | PMI_Other, Other
        WhatRaceEthnicity_AIAN, American Indian or Alaska Native | WhatRaceEthnicity_Asian, Asian

    EVERY item must open with a code token and a comma; the label is everything after that first comma, so
    commas inside a label ("Hispanic, Latino, or Spanish (For example: Cuban, …)") are kept. One item that is
    not coded means the list is not coded — it is a pipe list of labels.
    """
    if "|" not in raw:
        return None

    parts = _ITEM_SEP.split(raw)
    if len(parts) < 2:
        return None

    options = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        code, sep, label = part.partition(",")
        code = code.strip()
        label = label.strip()
        if not sep or not _is_code_token(code):
            return None
        options.append(ResponseOption(code=code, label=label or code, order=i))

    return options if len(options) >= 2 else None


def _value_meaning_split(item: str) -> tuple[str, str]:
    """``(code, label)`` for one pipe-list item: split at ``value=meaning``, else the item is its own code.

    The separator is the first ``=`` at parenthesis depth 0 that is not part of a comparison operator
    (``<=``, ``>=``, ``!=``, ``==``), with text on both sides — so "BMI >= 30" and "Score (0=none)" stay whole.
    """
    depth = 0
    for i, ch in enumerate(item):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            before, after = item[i - 1 : i], item[i + 1 : i + 2]
            if before in ("<", ">", "!", "=") or after == "=":
                continue
            code, label = item[:i].strip(), item[i + 1 :].strip()
            if code and label:
                return code, label
    return item, item


def _parse_pipe_list(raw: str) -> list[ResponseOption] | None:
    """Parse a pipe-separated list of labels, optionally mixed with ``value=meaning`` items (NIH CDE).

    Examples:
        Years | Months
        Asthma | Autoimmune condition (e.g., lupus, vasculitis) | Thrombotic disorders=Thrombotic Disorders

    A bare item is both code and label, verbatim — commas included. A ``value=meaning`` item is split there.
    """
    if "|" not in raw:
        return None

    parts = _ITEM_SEP.split(raw)
    if len(parts) < 2:
        return None

    options = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        code, label = _value_meaning_split(part)
        options.append(ResponseOption(code=code, label=label, order=i))

    return options if len(options) >= 2 else None


def _parse_slash_delimited(raw: str) -> list[ResponseOption] | None:
    """Parse simple slash-delimited options (2-3 only).

    Examples:
        Yes/No
        Male/Female
        Yes/No/Unknown
    """
    if "|" in raw or "=" in raw or "(" in raw:
        return None

    parts = raw.split("/")
    if len(parts) < 2 or len(parts) > 3:
        return None

    # Each part should be a single short word/phrase
    options = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part or len(part) > 30:
            return None  # Too long to be a simple option label
        options.append(ResponseOption(code=str(i), label=part, order=i))

    return options
