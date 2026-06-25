"""Prompts for the v2 lean head/tail harmonization pipeline (3 LLM stages).

Validated in the research harness before productionization. A semantic cluster is grouped by an
embedding that ignores the variable name, so one cluster can pool MORE THAN ONE distinct concept that
merely shares surface wording. The 3 stages are:

1. **generate-ideal** — describe the *ideal* CDE for the concept with NO candidates shown. An
   independent target: judging coverage against a self-specified ideal avoids the failure mode of
   judging it *from* the retrieved candidates (which biases toward a match). Qualifier-faithful: the
   ideal must preserve any source qualifier (address type, subject, laterality, body site, condition,
   time window) and enumerate distinct concepts when members carry different qualifiers.
2. **split-assign** — given the ideal(s) *and* the cluster-level candidate CDEs, PARTITION the members
   into distinct-concept groups (split on the object/referent axis, not the value axis), and for each
   group rank the candidates + commit a verdict (adopt / refine / novel) + chosen candidate.
3. **per-group re-assign** — each split-out group is re-retrieved (its own member centroid + BM25) and
   re-assigned as a single concept against candidates retrieved specifically for THAT group, so a group
   is never judged against the blended-cluster pool.

adopt/refine route to a CDE assignment; novel routes to the GenCDE / clustering residual (the tail).
"""

from __future__ import annotations

import json

# ── stage 1: generate the ideal CDE (no candidates shown), qualifier-faithful ──
SYS_GENERATE_IDEAL = (
    "You are a biomedical Common Data Element (CDE) expert. Given a source concept (a cluster of harmonized "
    "data-dictionary fields that measure the same thing), describe the IDEAL CDE that would capture it: its "
    "concept/measurement, the question it answers, and the expected answer type or units. Output a concise, "
    "catalog-style CDE description. Describe what the ideal CDE WOULD be from first principles — do NOT "
    "reference, assume, or defer to any existing catalog or candidate list.\n"
    "PRESERVE the source qualifier, never invent one: if the fields specify a qualifier (address type "
    "[home/work/mailing], subject/person [self/spouse/contact], laterality [left/right], body site, "
    "condition, or time window) — often carried in the variable NAME when the question text is generic — "
    "reflect it in the ideal. Do NOT assume a default context (e.g. do not call a ZIP 'residential' unless "
    "the source says so). If the member fields carry DIFFERENT qualifiers (e.g. work vs home vs another "
    "person's address), say so explicitly rather than picking one. Return JSON only."
)
IDEAL_SCHEMA = json.dumps({"ideal_cde": "<concise CDE description: concept + question + answer type/units>"})

# ── stage 2: split-assign — partition into distinct-concept groups, decide each ──
SYS_SPLIT = (
    "You assign data-dictionary fields to Common Data Elements (CDEs). You are given (1) the IDEAL CDE(s) for "
    "the source concept; (2) the concept's MEMBER fields, each prefixed with an id like [m1]; (3) a NUMBERED "
    "list of candidate CDEs from a real catalog.\n"
    "The members were grouped by an embedding that IGNORES the variable name, so one cluster can pool MORE "
    "THAN ONE distinct concept that merely shares surface wording. FIRST decide whether the members are ONE "
    "concept or SEVERAL, then PARTITION them into distinct-concept groups.\n"
    "SPLIT RULE — split on the OBJECT/REFERENT axis, NOT the value axis:\n"
    "  • SPLIT when the WHO/WHAT being measured changes — a person's HOME address vs their EMPLOYER's address "
    "vs a CONTACT's address are DISTINCT concepts (different object) and MUST be separate groups; a postal "
    "CODE vs a GEOSPATIAL measure derived from it (road length / distance near a postal code) is a different "
    "measurement entirely → separate.\n"
    "  • DO NOT SPLIT on value-specificity of the SAME object: members naming different VALUES of one property "
    "of the same object (e.g. different specific cancer types the SUBJECT may have, different specific "
    "medications) are a coarser-grained rollup of ONE concept — keep them in ONE group.\n"
    "  • DO NOT SPLIT cosmetic synonyms ('zip code' vs 'postal code', same referent) or a REPEATING MEASURE "
    "(same concept across numbered slots/occurrences — 'Medication 1..N', 'visit 1..N'; the slot number is an "
    "occurrence index, NEVER a qualifier; keep as ONE group).\n"
    "Default to ONE group; split ONLY for a clear object/referent change. Do not over-split on wording.\n"
    "For EACH group, rank the candidates by how well each realizes that group's concept, then decide a "
    "verdict:\n"
    "  adopt  — a candidate IS the concept exactly (same precise question + compatible answer set).\n"
    "  refine — the right concept is reachable from a candidate but needs specialization to fit this group. "
    "This INCLUDES binding a GENERIC building-block candidate (e.g. a bare 'Zip Code' element with no precise "
    "question) to this group's specific object/qualifier. Set cde_id = the building-block candidate and put "
    "the PRECISE specialization in 'concept' (e.g. 'Employer address ZIP code'). PREFER refining an existing "
    "building block over minting new.\n"
    "  novel  — NO candidate (not even a generic building block) realizes the concept; a new CDE (GenCDE) is "
    "needed.\n"
    "PRESERVE THE DISTINGUISHING AXIS: if a group's concept names a specific condition, disease, body site, "
    "time window, or other qualifier, a candidate that names a DIFFERENT specific value of that axis is NOT a "
    "match — the qualifier is part of the concept, not a refinement (e.g. 'age told you had kidney STONES' does "
    "not match a CDE for 'age told you had kidney FAILURE'). Match only a candidate with the SAME qualifier, or "
    "one explicitly generic/templated over it; otherwise choose novel.\n"
    "Distinct concepts get distinct decisions — never a silent shared adopt. Judge on meaning, not wording. "
    "Return JSON only."
)
SPLIT_SCHEMA = json.dumps(
    {
        "groups": [
            {
                "member_ids": ["<ids like m1, m2 that belong to this concept>"],
                "concept": "<short concept label>",
                "ranking": "[<candidate numbers, best first>]",
                "verdict": "adopt|refine|novel",
                "cde_id": "<chosen candidate number, or null for novel>",
                "rationale": "<one sentence>",
            }
        ]
    }
)

# ── stage 3: per-group single-concept re-assign (over per-group re-retrieved candidates) ──
SYS_GROUP_REASSIGN = (
    "You assign ONE source concept to a Common Data Element (CDE). You are given (1) the concept label; (2) a "
    "sample of its member fields; (3) a NUMBERED list of candidate CDEs retrieved specifically for THIS "
    "concept. Rank the candidates by how well each realizes the concept, then decide one verdict:\n"
    "  adopt  — a candidate IS the concept exactly (same precise question + compatible answer set).\n"
    "  refine — the concept is reachable from a candidate but needs specialization. This INCLUDES binding a "
    "GENERIC building-block candidate (e.g. a bare 'Zip Code' element with no precise question) to this "
    "concept's specific object/qualifier — prefer refining an existing building block over minting new.\n"
    "  novel  — NO candidate (not even a generic building block) realizes the concept; a new CDE is needed.\n"
    "Be STRICT and PRESERVE THE DISTINGUISHING AXIS: a candidate of a different qualifier, granularity, or "
    "axis is NOT a match. If the concept names a specific condition, disease, body site, time window, or other "
    "qualifier, a candidate naming a DIFFERENT specific value of that axis is NOT a match — the qualifier is "
    "part of the concept, not a refinement (e.g. 'age told you had kidney STONES' does not match a CDE for "
    "'age told you had kidney FAILURE'). Match only a candidate with the SAME qualifier, or one explicitly "
    "generic/templated over it; otherwise choose novel. Judge on meaning, not wording. Return JSON only."
)

# Reused by stage 3 (per-group single-concept assign): ranking + verdict + chosen candidate number.
ASSIGN_SCHEMA = json.dumps(
    {
        "ranking": "[<candidate numbers, best realizing the concept first>]",
        "verdict": "adopt|refine|novel",
        "cde_id": "<chosen candidate number, or null for novel>",
        "rationale": "<one sentence>",
    }
)


def build_ideal_user_prompt(member_lines: list[str]) -> str:
    """Stage-1 user prompt: the member sample only (no candidates)."""
    sample = "; ".join(member_lines)
    return f"Source concept — member fields (sample): {sample}\n\nDescribe the ideal CDE. Return JSON."


def build_split_user_prompt(ideal_cde: str, numbered_members: list[tuple[str, str]], candidate_block: str) -> str:
    """Stage-2 user prompt: the ideal(s) + the [mK]-prefixed members + the numbered candidate block.

    ``numbered_members`` is a list of ``(member_id, text)`` pairs (e.g. ``("m1", "Age in years ...")``).
    """
    mlines = "\n".join(f"  [{mid}] {txt}" for mid, txt in numbered_members)
    return (
        f"IDEAL CDE(s) for the source concept (may name >1 if heterogeneous):\n{ideal_cde[:500]}\n\n"
        f"Member fields:\n{mlines}\n\n"
        f"Candidate CDEs:\n{candidate_block}\n\n"
        f"Partition the members into distinct-concept groups (default ONE unless a qualifier changes the "
        f"referent), then for each group return ranking + verdict + chosen candidate number as JSON."
    )


def build_group_assign_user_prompt(concept: str, member_lines: list[str], candidates: list[str]) -> str:
    """Stage-3 user prompt: a single group's concept label + member sample + per-group numbered candidates."""
    sample = "; ".join(member_lines)
    cand_block = "\n".join(f"  [{i + 1}] {text}" for i, text in enumerate(candidates))
    return (
        f"Concept: {concept[:120]}\n\n"
        f"Member fields (sample): {sample}\n\n"
        f"Candidate CDEs (retrieved for THIS concept):\n{cand_block}\n\n"
        f"Rank, then return verdict + chosen candidate number as JSON."
    )
