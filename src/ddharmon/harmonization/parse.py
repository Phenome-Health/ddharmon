"""Tolerant JSON parsing of LLM responses.

Robust JSON extraction lifted from nb 05 (``_extract_json`` /
``_payload_from_response``): models occasionally narrate before the JSON, wrap
it in ``` fences, or append trailing commentary. We never drop a sub-cluster
over our own parse strictness.

``extract_json`` / ``payload_from_response`` / ``parse_verdict_payload`` serve the classify-only
adopt/refine/novel pass; ``salvage_objects`` is the shared long-array rescue used wherever a response
carries a list long enough to hit the token cap mid-way.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull a JSON object out of an LLM response that may wrap it in prose/fences.

    Strategy: strip a fenced block if present, then ``raw_decode`` at each ``{``
    and keep the widest dict that decodes. ``raw_decode`` consumes exactly one
    JSON value and ignores trailing data, so prose preambles, nested objects, and
    trailing commentary are all handled.
    """
    t = text.strip()
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()

    decoder = json.JSONDecoder()
    best: dict | None = None
    best_span = -1
    i = t.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(t, i)
        except json.JSONDecodeError:
            obj, end = None, i
        if isinstance(obj, dict) and end - i > best_span:
            best, best_span = obj, end - i
        i = t.find("{", i + 1)

    if best is not None:
        return best
    return json.loads(t)  # nothing decoded — raise a clear JSONDecodeError


def salvage_objects(text: str, key: str | None = None) -> list[dict]:
    """Recover the complete objects from a JSON array the model cut off mid-way (token cap).

    Locates the ``"<key>": [`` array (or the first bare ``[`` when ``key`` is None), then walks brace depth
    collecting each balanced ``{…}`` and parsing it on its own — so an incomplete trailing object is dropped
    instead of failing the whole parse. Used where a response carries a long list (a score's components, a
    batch of ideas) and truncation is a real risk.
    """
    t = text.strip()
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()

    start = -1
    if key:
        km = re.search(rf'"{re.escape(key)}"\s*:\s*\[', t)
        if km:
            start = km.end()
    if start < 0:
        start = t.find("[") + 1 if "[" in t else -1
    if start <= 0:
        return []

    objs: list[dict] = []
    depth = 0
    obj_start = -1
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    try:
                        obj = json.loads(t[obj_start : i + 1])
                    except (ValueError, TypeError):
                        obj = None
                    if isinstance(obj, dict):
                        objs.append(obj)
                    obj_start = -1
        elif ch == "]" and depth == 0:
            break
    return objs


def payload_from_response(resp: object) -> dict:
    """Extract a JSON payload dict from a runner/batch response record.

    The runner stores valid-JSON output inline as a parsed object, so ``resp`` is
    usually already a dict. Anthropic-style ``{"content": "<json string>"}``
    wrappers and raw string fallbacks (prose + fenced JSON) are also supported.
    """
    if isinstance(resp, dict):
        if "content" in resp and isinstance(resp["content"], str):
            return extract_json(resp["content"])
        return resp
    return extract_json(str(resp))


def parse_verdict_payload(resp: object) -> dict | None:
    """Parse one A/R/N response into its payload dict, or None on parse failure.

    Returns the dict with at least ``verdict``; callers attach cluster context.
    """
    try:
        payload = payload_from_response(resp)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or "verdict" not in payload:
        return None
    return payload
