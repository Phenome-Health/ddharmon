"""Prompts for the lean head/tail harmonization pipeline (3 LLM stages).

Validated through benchmark experiments before productionization. A semantic cluster is grouped by an
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

# M15 — ENFORCED split output (opt-in). The soft TEXT schema above is dropped by the model on ~35% of
# clusters (bare single-group, no `groups` wrapper) → the residual members route novel. This is a real
# JSON Schema for a FORCED tool call: the wrapper +
# per-group required fields are structurally guaranteed, so the model cannot return a bare object. It does
# NOT force semantic completeness (the model still chooses member_ids) — the "every member in exactly one
# group" completeness is instructed here + repaired downstream by the residual group. Kills the FORMAT drop.
SPLIT_TOOL_NAME = "emit_groups"
SPLIT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "minItems": 1,
            "description": "The distinct-concept partition of the members. Every member id (m1, m2, …) "
            "must appear in exactly ONE group; do not omit any member.",
            "items": {
                "type": "object",
                "properties": {
                    "member_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ids like m1, m2 that belong to this concept",
                    },
                    "concept": {"type": "string", "description": "short concept label"},
                    "ranking": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "candidate numbers, best realizing this concept first",
                    },
                    "verdict": {"type": "string", "enum": ["adopt", "refine", "novel"]},
                    "cde_id": {"type": ["string", "null"], "description": "chosen candidate number, null for novel"},
                    "rationale": {"type": "string", "description": "one sentence"},
                },
                "required": ["member_ids", "concept", "verdict"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}

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

# ── M4: representation-mismatch → refine, not novel (opt-in) ──
# Appended to the split (stage 2) and per-group re-assign (stage 3) system prompts when enabled. The audit of
# the full-5 run found ~372 vars wrongly routed NOVEL against a rich CDE at cosine up to 0.805 because the
# model treated a difference in the ANSWER encoding (banding, flag, composite, unit) as a concept mismatch.
# Representation is bridgeable by a transform spec — so it is refine, not novel. Orthogonal to the qualifier
# rule the base prompts already carry (a different OBJECT/referent, or a different specific value of a
# distinguishing axis, is still novel). Judge the CONCEPT, not the encoding.
REPRESENTATION_REFINE_CLAUSE = (
    "REPRESENTATION MISMATCH IS REFINE, NOT NOVEL — judge the underlying CONCEPT, not the answer encoding. "
    "If a candidate measures the SAME concept of the SAME object but in a different REPRESENTATION, that is "
    "refine (a value/derivation transform bridges the two encodings) — set cde_id = that candidate and put "
    "the specialization in 'concept'. This covers: (a) a categorical BANDING/bucketing of a continuous "
    "candidate (age BANDS vs age in years; a BMI/income/education CATEGORY vs its underlying value); (b) a "
    "YES/NO or present/absent FLAG of a candidate's scaled, count, or continuous measure; (c) a "
    "COMPOSITE / sum-score / index built from the candidate's components, or one component of the "
    "candidate's composite; (d) a different UNIT, SCALE, or code set for the same quantity. Choose novel "
    "ONLY when the underlying concept or the object/referent genuinely differs — NEVER merely because the "
    "answer TYPE, granularity, or code set differs. This does NOT relax the qualifier rule above: a "
    "different object/referent, or a different specific value of a distinguishing axis (condition, body "
    "site, laterality, time window), is still novel."
)


# ── M11: measurand-axis split (opt-in) ──
# The split rule partitions on the OBJECT/REFERENT axis only, so distinct MEASURANDS sharing one object at one
# encounter (systolic vs diastolic BP vs pulse) were left fused as ONE group — a demo-visible over-merge
# (BP+pulse lumped, 24 vars, one mislabeled concept; .planning todo 2026-07-14-investigate-…-lumped-in). This
# clause adds a MEASURAND axis to the split rule and, on generate-ideal, tells the seed to enumerate distinct
# measurands instead of bundling them (which primed the fusion). Guards both reinforcers found in the trace: a
# bundling candidate CDE (paired systolic/diastolic) does NOT license merging, and it does not override the
# repeating-measure rule. Opt-in until an A/B on the offending cluster validates it (split-hardening regressed
# once before — see project_split_wrapper_drop_bug); appended to the stage-2 split + stage-1 ideal prompts.
MEASURAND_SPLIT_CLAUSE = (
    "MEASURAND AXIS — also SPLIT when the distinct QUANTITY being measured changes, even for the SAME subject "
    "at the SAME encounter. Systolic blood pressure, diastolic blood pressure, and pulse/heart rate are "
    "DIFFERENT measurements (different physical quantity, unit, and normal range) and MUST be separate groups — "
    "as must, e.g., height vs weight, or temperature vs respiratory rate. Measurement METADATA (device/method, "
    "timing, posture, cuff size) is NOT the measured quantity: put it in its own group, never fold it into a "
    "measurement concept. An existing candidate CDE that BUNDLES several measurands (e.g. one 'blood pressure' "
    "element pairing systolic and diastolic) does NOT license merging them here — split first, then each group "
    "may adopt/refine the part that fits. This does NOT override the repeating-measure rule: the SAME measurand "
    "across numbered readings/occurrences (bp1, bp2; reading 1..N) stays ONE group."
)
MEASURAND_IDEAL_CLAUSE = (
    "If the member fields measure DIFFERENT quantities/MEASURANDS (e.g. systolic vs diastolic blood pressure vs "
    "pulse/heart rate; height vs weight), enumerate them as DISTINCT concepts rather than bundling them into "
    "one CDE — even when they co-occur at the same encounter. Measurement metadata (device/method, timing) is "
    "not the measured quantity."
)


# ── (b) coherence levers — OPT-IN, default OFF; validate via frozen-substrate A/B before default-on ──
# Root cause of an observed blood-pressure over-merge: the
# generate-ideal SEED framed distinct measurands as ONE CDE ("systolic and diastolic ... recorded as two
# separate fields"), and a BUNDLING candidate CDE then licensed a single `adopt` over 24 heterogeneous
# members. The split CRITERION was NOT at fault — it faithfully followed a bundling seed + a bundling
# candidate. So these levers target the SEED and the ADOPT-decision, not the split criterion (broadening
# that regressed via the split wrapper-drop and the measurand_split A/B). Kept as
# TWO separate flags so an A/B attributes each independently (the refuted measurand_split arm bundled its
# split-clause and its ideal-clause together and could not separate them).

# M13 — de-bias generate-ideal (full variant, NOT an append): the base prompt opens with "fields that
# measure the same thing", presuming one concept, and only handles different QUALIFIERS. This variant drops
# that presumption and instructs enumeration of distinct MEASURANDS/CONCEPTS as distinct ideals. General
# (measurand + semantic-category), no cohort- or example-specific rule.
SYS_GENERATE_IDEAL_DEBIASED = (
    "You are a biomedical Common Data Element (CDE) expert. Given a source cluster of data-dictionary fields "
    "grouped by an embedding that IGNORES the variable name — so the cluster MAY pool more than one distinct "
    "concept — describe the IDEAL CDE(s) that would capture it: concept/measurement, the question answered, "
    "and the expected answer type or units. Output concise, catalog-style CDE description(s). Describe what "
    "the ideal WOULD be from first principles — do NOT reference, assume, or defer to any existing catalog or "
    "candidate list.\n"
    "ONE CONCEPT PER IDEAL — DO NOT BUNDLE. If the members measure DIFFERENT quantities/measurands, or "
    "represent DIFFERENT concepts (e.g. a diagnosis vs a family history vs a medication; two distinct physical "
    "measurements taken at the same encounter), enumerate EACH as its OWN ideal CDE rather than merging them "
    "into one composite — even when they co-occur. Members that are only different VALUES of ONE property of "
    "the same object (different specific conditions, different medications) are ONE concept — keep them "
    "together. Measurement metadata (device/method, timing, posture) is NOT the measured quantity.\n"
    "PRESERVE the source qualifier, never invent one: if the fields specify a qualifier (address type "
    "[home/work/mailing], subject/person [self/spouse/contact], laterality [left/right], body site, "
    "condition, or time window) — often carried in the variable NAME when the question text is generic — "
    "reflect it in the ideal. Do NOT assume a default context (e.g. do not call a ZIP 'residential' unless "
    "the source says so). If members carry DIFFERENT qualifiers (e.g. work vs home vs another person's "
    "address), enumerate them rather than picking one. Return JSON only."
)

# M14 — bundling-candidate→adopt guard: ONE short clause (kept minimal — lengthening the split prompt
# regressed output-format compliance / wrapper-drop). Appended to the stage-2 split + stage-3 re-assign
# prompts. General: any candidate that bundles multiple measured quantities/concepts, not just paired BP.
BUNDLING_GUARD_CLAUSE = (
    "BUNDLING GUARD: a candidate that BUNDLES several distinct measured quantities or concepts (e.g. one "
    "element pairing two different measurements) does NOT license a shared adopt. If the members span more "
    "than one measured quantity or concept, SPLIT them into separate groups FIRST, then each group may "
    "adopt/refine the part that fits — never adopt a bundling candidate to keep distinct measures merged."
)


def split_system_prompt(
    representation_refine: bool = False, measurand_split: bool = False, bundle_guard: bool = False
) -> str:
    """Stage-2 split system prompt, optionally with the M4 representation, M11 measurand, and/or M14 guard."""
    prompt = SYS_SPLIT
    if representation_refine:
        prompt = f"{prompt}\n{REPRESENTATION_REFINE_CLAUSE}"
    if measurand_split:
        prompt = f"{prompt}\n{MEASURAND_SPLIT_CLAUSE}"
    if bundle_guard:
        prompt = f"{prompt}\n{BUNDLING_GUARD_CLAUSE}"
    return prompt


def generate_ideal_system_prompt(measurand_split: bool = False, debias_ideal: bool = False) -> str:
    """Stage-1 generate-ideal system prompt.

    ``debias_ideal`` (M13) SELECTS the de-biased variant (no presumption of one concept; enumerate distinct
    measurands/concepts). ``measurand_split`` (M11, refuted) appends the narrow measurand clause. M13 is a
    full-variant swap and takes precedence over the M11 append when both are set (they target the same bias).
    """
    if debias_ideal:
        return SYS_GENERATE_IDEAL_DEBIASED
    return f"{SYS_GENERATE_IDEAL}\n{MEASURAND_IDEAL_CLAUSE}" if measurand_split else SYS_GENERATE_IDEAL


def group_reassign_system_prompt(representation_refine: bool = False, bundle_guard: bool = False) -> str:
    """Stage-3 per-group re-assign system prompt, optionally with the M4 representation and/or M14 guard."""
    prompt = SYS_GROUP_REASSIGN
    if representation_refine:
        prompt = f"{prompt}\n{REPRESENTATION_REFINE_CLAUSE}"
    if bundle_guard:
        prompt = f"{prompt}\n{BUNDLING_GUARD_CLAUSE}"
    return prompt


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


# ── step 2: dual-sample coherence judge (post-assign, read-only) ──────────────────────────────────
# A proposed concept-GROUP is verified by summarizing its k1 centroid-CLOSEST members (the core theme)
# and checking that its k2 centroid-FURTHEST members share that theme. Our improvement over Islam 2026: Islam
# uses the SAME centroid-closest sample to generate AND verify (self-fulfilling — a tight core with an
# off-concept boundary always passes); the DISJOINT k1/k2 sampling catches exactly the tight-core /
# heterogeneous-boundary failure the BP-measurand and arthritis over-merges are. The granularity verdict names
# whether the group is ONE concept (single), a matrix collapsible only with a varying slot (qualify), or too
# varied for even that (split). This is a FLAG, never an auto-split — the cure is human re-adjudication.
SYS_COHERENCE = (
    "You are a domain expert in biomedical data dictionary harmonization. You are evaluating whether a "
    "proposed GROUP of cohort field descriptions forms a coherent semantic concept — the same underlying "
    "measurement/concept, regardless of phrasing — or whether it over-merges distinct concepts that one "
    "CDE would silently collapse.\n"
    "Cohort attribution rule: refer to cohort source ONLY as given in each field's `cohort:variable_name` "
    "prefix in the prompt. Do NOT infer cohort identity from variable-code patterns (e.g. 3-letter prefixes, "
    "`dsN_M` codes). If variable codes are opaque, say so without guessing which cohort or instrument they "
    "belong to — the prompt's explicit cohort tag is authoritative.\n"
    "Granularity check: a group can be coherent yet still pool many distinct measurements. Judge it from your "
    "own summary. If the summary stays faithful to ALL members with NO placeholder, set "
    'granularity.verdict = "single". If it is faithful only as a TEMPLATE needing one varying slot (e.g. '
    '"...for [condition]" or "various conditions"), set "qualify" and record that slot as `axis` with its '
    '`distinct_values`. If members are too varied for even a one-slot summary, set "split". A "qualify"/'
    '"split" group is a matrix of distinct things one CDE would silently collapse.\n'
    "Return JSON only, no commentary outside the JSON object. Schema:\n"
    "{\n"
    '  "summary": "<one-sentence theme of the group core>",\n'
    '  "coherent": <true if the peripheral items share the core theme; false otherwise>,\n'
    '  "outliers": [<periphery position of items that do not fit, 1-indexed>],\n'
    '  "granularity": {\n'
    '    "verdict": "single | qualify | split",\n'
    '    "axis": "<the slot your summary needed to stay faithful, e.g. mental-health condition> | null",\n'
    '    "distinct_values": ["<the distinct fillers across members, e.g. ADHD, depression, ...>"]\n'
    "  }\n"
    "}"
)
COHERENCE_SCHEMA = json.dumps(
    {
        "summary": "<one-sentence theme of the group core>",
        "coherent": "<true|false — do the periphery members share the core theme>",
        "outliers": ["<1-indexed periphery positions that do not fit>"],
        "granularity": {
            "verdict": "single|qualify|split",
            "axis": "<varying slot, or null>",
            "distinct_values": ["<distinct fillers across members>"],
        },
    }
)


def build_coherence_user_prompt(n_members: int, core_texts: list[str], periphery_texts: list[str]) -> str:
    """Step-2 user prompt: the k1 CORE members (centroid-closest) + the k2 PERIPHERY members (furthest).

    Two-step framing (summarize the core theme, then verify the periphery against it) so the periphery is
    judged against a summary built from the core — never from itself (the self-fulfilling failure we fix).
    """
    core = "\n".join(f"  core {i + 1}: {t}" for i, t in enumerate(core_texts))
    periphery = "\n".join(f"  periphery {i + 1}: {t}" for i, t in enumerate(periphery_texts))
    return (
        f"This proposed harmonization group has {n_members} members. Below are the {len(core_texts)} CORE "
        f"members (closest to the group centroid) and {len(periphery_texts)} PERIPHERY members (furthest "
        f"from the centroid).\n\n"
        f"Step 1 — Summarize the THEME shared by the core members:\n{core}\n\n"
        f"Step 2 — Decide whether each PERIPHERY member shares that theme. The group is coherent only if the "
        f"periphery clearly belongs to the same concept as the core.\n\n"
        f"{periphery}\n\n"
        f"Return the JSON object specified in the system prompt. List the 1-indexed periphery positions of "
        f"any items that don't fit the core theme. Then set granularity: does your Step 1 summary hold for "
        f"ALL members with no placeholder (single), only as a one-slot template like 'X for [condition]' "
        f"(qualify — give axis and distinct_values), or not even then (split)?"
    )


# ── distinct-KINDS discriminator (R2): the second read on a `qualify` group ───────────────────────
# The coherence judge calls a group `qualify` when its one-sentence summary holds only as a one-slot
# TEMPLATE (an axis + distinct values). That is ambiguous: the axis may be a qualifier value-set of ONE
# concept (milk *by fat content*, PHQ items — coherent, don't flag) OR a label papering over genuinely
# DIFFERENT measurands (a length + a mass + a rate — an over-merge, flag). This cheap second call resolves
# that split. It reads ONLY the judge's own outputs (summary / axis / distinct_values), so it runs after
# the coherence stage on qualify groups. R2 flags a qualify group iff this returns `distinct_kinds`.
# Validated on returned human pairwise gold: the distinct-kinds rule reached 100% recall on the
# human-confirmed over-merges, recovering the qualify-fusions the split-only rule misses.
SYS_KINDS = (
    "A prior step grouped several survey/clinical variables and found they all vary along ONE named axis. "
    "Your job: decide whether this group is ONE harmonizable concept (a single attribute recorded with a "
    "value-set / qualifier) or an OVER-MERGE of genuinely distinct concepts that only share an axis label.\n\n"
    "Given the group SUMMARY, the AXIS name, and the DISTINCT VALUES observed along that axis, classify:\n\n"
    "• values_of_one_property — the values are alternative VALUES of a SINGLE attribute:\n"
    "    - a which-X rollup (which country, which language, which brand);\n"
    "    - a by-WHEN / by-WHERE / by-WHICH-VISIT rollup (morning vs evening; visit 1 vs 2);\n"
    "    - the same measurement repeated or indexed (trial 1, 2, 3).\n"
    "  One concept plus a qualifier value-set. → coherent, do NOT flag.\n\n"
    "• distinct_kinds — the values name genuinely DIFFERENT attributes or referents, not values of one:\n"
    "    - different physical quantities of the same object (a length vs a mass vs a rate);\n"
    "    - different referent populations (the participant vs a relative);\n"
    "    - an unrelated mix of dimensions bundled together (a status + an exposure + a history item).\n"
    "  An over-merge that should be re-adjudicated. → flag.\n\n"
    "Decide by asking: are these ONE thing recorded under different settings/instances, or SEVERAL different "
    "things? If a single well-formed CDE with a value-set could faithfully capture EVERY member, it is "
    "values_of_one_property; if faithful capture needs SEPARATE CDEs, it is distinct_kinds. Return only the "
    "tool call."
)
KINDS_TOOL_NAME = "classify_group"
KINDS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["values_of_one_property", "distinct_kinds"]},
        "rationale": {"type": "string", "description": "one line"},
    },
    "required": ["kind", "rationale"],
}


def build_kinds_user_prompt(summary: str, axis: str, distinct_values: list[str]) -> str:
    """Discriminator user prompt from the coherence judge's own outputs (summary / axis / distinct_values)."""
    dv = distinct_values or []
    return (
        f"SUMMARY: {summary or ''}\n"
        f"AXIS: {axis or ''}\n"
        f"DISTINCT VALUES ({len(dv)}): {json.dumps(dv, ensure_ascii=False)}"
    )


# ── re-adjudication: human-triggered re-split of an over-merged group ─────────────────────────────
# When a group has been FLAGGED as over-merged (by the coherence judge's `split` verdict or a human
# reviewer marking it incoherent), this prompt re-partitions it. It differs from SYS_SPLIT in its PRIOR:
# the split stage defaults to ONE group and splits only on a clear change; here the group is ALREADY known
# to fuse ≥2 concepts, so the default is INVERTED — find the distinct concepts. It shares SYS_SPLIT's axes
# and guards (so it does not over-split values of one property / repeating measures / cosmetic synonyms) and
# emits the SAME {groups:[…]} output contract (SPLIT_SCHEMA / SPLIT_TOOL_SCHEMA) so downstream reuse is exact.
SYS_READJUDICATE = (
    "You assign data-dictionary fields to Common Data Elements (CDEs). This group of MEMBER fields (each "
    "prefixed with an id like [m1]) was already FLAGGED as OVER-MERGED — a prior judgment found it fuses "
    "TWO OR MORE distinct concepts into one. Your job is to RE-PARTITION it into distinct-concept groups. "
    "Do NOT return it as one group — it is known to be multiple; find the genuine concept boundaries.\n"
    "PARTITION AXES — split on any axis where the members are genuinely different concepts:\n"
    "  • OBJECT/REFERENT — the who/what being measured changes (home vs employer address; self vs spouse).\n"
    "  • MEASURAND/QUANTITY — distinct physical quantities sharing one object/encounter (systolic vs "
    "diastolic blood pressure vs pulse; height vs weight); measurement metadata (device/method, timing, "
    "posture) is its OWN group, not the measured quantity.\n"
    "  • SEMANTIC CATEGORY — a diagnosis vs a family history vs a medication vs a symptom/flare of the SAME "
    "condition are DISTINCT concepts and MUST be separate groups.\n"
    "  • ENTITY SLOT — a shared question template over many distinct entities (a provider seen for "
    "[condition]; liking for [food]) collapses distinct measurements; separate the distinct entities.\n"
    "GUARDS — do NOT manufacture splits: members that are only different VALUES of ONE property of the same "
    "object (different specific cancer types, different medications the subject takes) are ONE concept; do "
    "NOT split a REPEATING MEASURE (same concept across numbered occurrences — reading 1..N; the slot number "
    "is an occurrence index, never a qualifier); do NOT split cosmetic synonyms ('zip' vs 'postal code').\n"
    "For EACH resulting group: rank the candidate CDEs by how well each realizes that group's concept, then "
    "commit a verdict — adopt (a candidate IS the concept exactly), refine (reachable from a candidate but "
    "needs specialization; set cde_id to it and put the precise concept in 'concept'), or novel (no candidate "
    "realizes it). Judge on meaning, not wording. Return JSON only."
)


def build_readjudicate_user_prompt(
    numbered_members: list[tuple[str, str]],
    candidate_block: str,
    *,
    axis: str = "",
    distinct_values: list[str] | None = None,
    desired_n: int | None = None,
) -> str:
    """Re-adjudication user prompt: the flagged members + candidates + the incoherence hint from the judge.

    ``axis`` / ``distinct_values`` are the coherence judge's granularity signal (the axis on which the group
    was found to fuse distinct concepts, and the observed fillers). ``desired_n`` is an optional
    human-supplied target sub-concept count ("split this into ~N").
    """
    mlines = "\n".join(f"  [{mid}] {txt}" for mid, txt in numbered_members)
    hint = ""
    if axis:
        hint = f"\nPrior judgment: this group fuses distinct concepts along the axis '{axis[:120]}'."
        if distinct_values:
            hint += f" Observed distinct values: {', '.join(str(v) for v in distinct_values[:15])}."
    if desired_n and desired_n > 0:
        hint += f"\nThe reviewer expects approximately {desired_n} distinct concept(s)."
    return (
        f"This group was flagged as over-merged and must be re-partitioned into distinct concepts.{hint}\n\n"
        f"Member fields:\n{mlines}\n\n"
        f"Candidate CDEs:\n{candidate_block}\n\n"
        f"Partition the members into distinct-concept groups (find the genuine boundaries — do NOT return one "
        f"group), then for each group return ranking + verdict + chosen candidate number as JSON."
    )
