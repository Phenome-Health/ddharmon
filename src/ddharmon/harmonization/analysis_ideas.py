"""Post-run "analysis ideas" — suggest (never run) downstream analyses a harmonization unlocks.

``harmonize_leanb`` shows WHAT is harmonizable across cohorts; this makes visible what that harmonization
*enables*. One LLM pass reads the run's harmonized concepts + which cohorts contribute to each, and proposes
concrete, grounded cross-cohort analyses (association tests, replication / meta-analysis, pooled prevalence,
subgroup / mediation, …).

HARD SCOPE (inherited from the metadata-only invariant): ddharmon ingests data dictionaries only, never
participant-level data. This therefore **proposes** analyses — it does not run them. The only signal it has
is which cohorts share a concept. Output is explicitly "hypotheses to explore", not results, and every idea
is grounded in concepts ACTUALLY present in the run (hallucinated concepts are dropped).

Library use::

    from ddharmon.harmonization import generate_analysis_ideas
    from ddharmon.llm.anthropic_client import AnthropicClient

    result = harmonize_leanb(embedded, generate=…, split=…, classify=…)   # a LeanBResult
    client = AnthropicClient(model_name="claude-sonnet-4-6")
    ideas = generate_analysis_ideas(result.records, client.complete).ideas
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ddharmon.harmonization.models import LeanBRecord

_MAX_CONCEPTS = 40  # bound the prompt to the most-connected concepts
_DEFAULT_MAX_IDEAS = 8

# The LLM call: ``complete(prompt, *, system, max_tokens) -> str`` — matches
# ``ddharmon.llm.anthropic_client.AnthropicClient.complete`` (and LiteLLMClient.complete).
CompleteFn = Callable[..., str]


@dataclass
class AnalysisIdea:
    """One suggested downstream cross-cohort analysis the harmonization makes newly possible."""

    title: str
    hypothesis: str
    concepts: list[str]  # grounded in the run's own concepts (a subset of the digest)
    cohorts: list[str]
    method: str
    why_newly_possible: str
    category: str


@dataclass
class AnalysisIdeasResult:
    """Return of :func:`generate_analysis_ideas`: the ideas + how many cross-cohort concepts were fed in."""

    ideas: list[AnalysisIdea] = field(default_factory=list)
    n_concepts: int = 0


@dataclass
class ConceptDigestEntry:
    """One cross-cohort concept summarized for the prompt (the enabling signal for a pooled analysis)."""

    concept: str
    cohorts: list[str]
    verdict: str
    cde: str | None
    n_members: int


def build_concept_digest(records: Sequence[LeanBRecord]) -> list[ConceptDigestEntry]:
    """The enabling signal: concepts present in ≥2 cohorts (a cross-cohort overlap is what makes a pooled
    analysis newly possible), sorted by cohort breadth then group size and capped to keep the prompt bounded.
    """
    digest: list[ConceptDigestEntry] = []
    for r in records:
        cohorts = sorted({c for c in (r.cohorts or []) if c})
        if len(cohorts) < 2:
            continue
        digest.append(
            ConceptDigestEntry(
                concept=(r.concept or "").strip() or "(unlabeled concept)",
                cohorts=cohorts,
                verdict=r.verdict or "",
                cde=r.cde_id,
                n_members=int(r.n_members or len(r.member_variable_names)),
            )
        )
    digest.sort(key=lambda d: (len(d.cohorts), d.n_members), reverse=True)
    return digest[:_MAX_CONCEPTS]


def _build_prompt(digest: list[ConceptDigestEntry], max_ideas: int) -> tuple[str, str]:
    concept_lines = "\n".join(
        f"- {d.concept} — cohorts: {', '.join(d.cohorts)}" + (f"; CDE {d.cde}" if d.cde else "") for d in digest
    )
    allowed = [d.concept for d in digest]
    system = (
        "You are a biomedical research methodologist. You are given CONCEPTS that have been harmonized "
        "across multiple cohorts; each lists which cohorts contain it. Propose concrete, scientifically "
        "plausible DOWNSTREAM ANALYSES that are newly possible now that these concepts align across cohorts "
        "— e.g. cross-cohort association tests, replication / meta-analysis, pooled prevalence, subgroup or "
        "mediation analyses.\n\n"
        "STRICT RULES:\n"
        "- Ground every idea ONLY in the concepts listed by the user. NEVER invent a variable or concept.\n"
        "- Every idea must use concepts present in ≥2 cohorts — that cross-cohort overlap is the whole point.\n"
        "- These are HYPOTHESES TO EXPLORE, not findings. Never claim a result or causal effect; name a method.\n"
        "- You have ONLY metadata (which cohorts share a concept). You have NO participant-level data.\n\n"
        "Respond with ONLY valid JSON (no markdown fences) matching this schema:\n"
        '{"ideas": [{"title": string, "hypothesis": string, "concepts": [string], "cohorts": [string], '
        '"method": string, "whyNewlyPossible": string, "category": string}]}'
    )
    user = (
        f"Harmonized cross-cohort concepts (concept — cohorts[; CDE]):\n{concept_lines}\n\n"
        f"Propose up to {max_ideas} analysis ideas, most impactful first. Each idea's `concepts` MUST be a "
        f"subset of exactly these labels (copy them verbatim): {allowed}"
    )
    return system, user


def _salvage_truncated_ideas(text: str) -> list[dict[str, Any]]:
    """Recover complete idea objects from a JSON array the model cut off mid-way (token cap).

    Locates the ``ideas`` array (or a bare ``[`` list), then walks brace depth collecting each balanced
    ``{…}`` object and parsing it on its own — so the incomplete trailing object is dropped instead of
    failing the whole parse. Returns the list of complete idea dicts (possibly empty).
    """
    m = re.search(r'"ideas"\s*:\s*\[', text)
    start_idx = m.end() if m else (text.find("[") + 1 if "[" in text else -1)
    if start_idx <= 0:
        return []
    objs: list[dict[str, Any]] = []
    depth = 0
    obj_start = -1
    in_str = False
    esc = False
    for i in range(start_idx, len(text)):
        ch = text[i]
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
                        obj = json.loads(text[obj_start : i + 1])
                        if isinstance(obj, dict):
                            objs.append(obj)
                    except (ValueError, TypeError):
                        pass
                    obj_start = -1
        elif ch == "]" and depth == 0:
            break
    return objs


def _parse_ideas(raw: str, allowed: set[str]) -> list[AnalysisIdea]:
    """Tolerantly parse the model's JSON into grounded :class:`AnalysisIdea`\\ s.

    Strips markdown fences, accepts either ``{"ideas": […]}`` or a bare list, and — if the JSON is truncated
    (the model hit the token cap mid-array) — salvages the complete idea objects. Then intersects each idea's
    ``concepts`` with ``allowed`` (dropping hallucinated ones) and drops any idea with none grounded.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        text = text.removeprefix("json").strip()
    try:
        data = json.loads(text)
        items = data.get("ideas") if isinstance(data, dict) else data
    except (ValueError, TypeError):
        items = _salvage_truncated_ideas(text)  # likely truncated at the token cap — keep complete ideas
    if not isinstance(items, list):
        return []

    out: list[AnalysisIdea] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        grounded = [c for c in (it.get("concepts") or []) if c in allowed]
        if not grounded:
            continue  # every concept was hallucinated → drop the idea
        out.append(
            AnalysisIdea(
                title=str(it.get("title", "")).strip(),
                hypothesis=str(it.get("hypothesis", "")).strip(),
                concepts=grounded,
                cohorts=[str(c) for c in (it.get("cohorts") or [])],
                method=str(it.get("method", "")).strip(),
                why_newly_possible=str(it.get("whyNewlyPossible", "")).strip(),
                category=str(it.get("category", "")).strip(),
            )
        )
    return out


def generate_analysis_ideas(
    records: Sequence[LeanBRecord], complete: CompleteFn, *, max_ideas: int = _DEFAULT_MAX_IDEAS
) -> AnalysisIdeasResult:
    """Generate grounded analysis ideas from a run's records via one LLM call.

    ``records`` are the ``LeanBRecord``\\ s of a :class:`~ddharmon.harmonization.leanb.LeanBResult`.
    ``complete`` is an ``AnthropicClient.complete``-style callable ``(prompt, *, system, max_tokens) -> str``.
    Returns an :class:`AnalysisIdeasResult`; ``ideas`` is empty when the run has no cross-cohort concept
    (nothing a pooled analysis could newly enable).
    """
    digest = build_concept_digest(records)
    if not digest:
        return AnalysisIdeasResult(ideas=[], n_concepts=0)
    system, user = _build_prompt(digest, max_ideas)
    # Headroom for several detailed ideas; the parser also salvages a truncated array as a backstop so a
    # verbose response never collapses to zero ideas.
    raw = complete(user, system=system, max_tokens=4096)
    ideas = _parse_ideas(raw, allowed={d.concept for d in digest})
    return AnalysisIdeasResult(ideas=ideas[:max_ideas], n_concepts=len(digest))
