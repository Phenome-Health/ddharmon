"""Reviewer group overrides: apply a human's regrouping to the post-split groups BEFORE the paid assign.

A staged review lets a reviewer reshape the concept groups the split produced: move a variable from one group
to another (or out of every group, or in from the clustering's leftovers), and form a NEW group of their own,
which may pull members from several clusters. Until this module those decisions were recorded and never
consumed. :func:`~ddharmon.harmonization.leanb.harmonize_leanb` now takes them as ``group_overrides`` and
applies them after the split replay and before assign:

1. **Resolve the membership** (:func:`resolve_group_membership`, pure). Every moved member leaves the group
   it was in and joins its destination; an existing group that loses every member is DROPPED; a New group
   holds exactly the members moved into it. Nothing else moves.
2. **Generate one ideal per New or CHANGED group** (:func:`prepare_group_ideals`) — the only new paid calls. The
   ideal anchors a group's novel decision (and seeds its GenCDE), so it must describe the members the group is
   assigned WITH. A New group was never a cluster and has none; a group whose membership the reviewer changed
   has one written for members it no longer has. Each gets a fresh one from its
   final members, written exactly as a cluster's ideal is (members only, no candidates). An UNCHANGED group
   keeps the ideal it was split with and buys nothing, and a group that will not be assigned (out of scope, or
   emptied) buys nothing either — the ideal is consumed only by the assign.
3. **Rebuild the assign prompts** (:func:`apply_group_overrides`). An untouched group's prompt is kept
   byte-for-byte, so its cached answer still hits; a changed or New group is RE-RETRIEVED on its own members
   (local, free) and prompted exactly like a split group, against its regenerated ideal.

WHAT STAYS FIXED, deliberately. The clusters, their pre-split ideals, the split and the coherence judge are what
the reviewer looked at when they decided; they are not re-run. ``LeanBResult.concept_groups`` keeps showing the
groups as the split produced them — overrides change what gets ASSIGNED, not what the earlier gate displayed
(the same rule the Gate-1 scope follows). A changed group keeps its ``group_id``, so every decision keyed on
it (scope, rename, a later pick) still applies to it.

AN ACCEPTED DIVISION IS ONE OF THESE TOO. Accepting the judge's proposed division of an over-merged group at Gate 1
buys a re-split (:func:`~ddharmon.harmonization.leanb.readjudicate_split_only`) whose children exist on that result
only — a later leg re-runs the split and keeps the group whole. :func:`division_overrides` turns the children into
New groups with their members moved in, so the division is applied on every later leg like any other regrouping.

PROMPT IDS ARE CONTENT-ADDRESSED on the resolved membership, so a replay with the same overrides rebuilds the
same ids (a resumed leg answers them from its checkpoint for $0) and a different membership can never be
answered with a response written for other members:

* a New or changed group's ideal — ``leanb:groupideal:<group_id>@<member content id>``, a namespace of its own
  that never collides with a cluster ideal (``leanb:ideal:<cluster content id>``). A caller that keeps
  ``generate`` closed to new prompts on a replay (a partition-drift guard) routes these through
  ``group_generate``;
* a changed group's assign — ``<its original assign id>@<member content id>``;
* a New group's assign — ``leanb:groupassign:<group_id>@<member content id>``.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ddharmon.embedding.service import EmbeddedDictionary
from ddharmon.harmonization.anchor import CDE_COHORT, build_field_lookup
from ddharmon.harmonization.leanb_prompts import (
    IDEAL_SCHEMA,
    build_ideal_user_prompt,
    generate_ideal_system_prompt,
    group_reassign_system_prompt,
)
from ddharmon.harmonization.models import ConceptGroup
from ddharmon.harmonization.pipeline import PromptRecord
from ddharmon.harmonization.substrate import cluster_content_id
from ddharmon.models.cluster import FieldReference

logger = logging.getLogger(__name__)

#: The prompt-id namespace of a New group's generated ideal (see the module docstring).
GROUP_IDEAL_PREFIX = "leanb:groupideal:"


@dataclass(frozen=True)
class ReviewerGroup:
    """A group a REVIEWER formed. ``group_id`` is theirs to mint and must stay stable across every later run
    of the same decisions (the staged UI mints ``rev:<uuid>``); ``name`` becomes the group's concept label.

    ``split_from`` is set on a PART of an accepted division (:func:`division_overrides`): the id of the group it
    was divided out of. It is provenance only — the record the part becomes carries it as ``readjudicated_from``.
    """

    group_id: str
    name: str = ""
    split_from: str = ""


@dataclass(frozen=True)
class GroupOverrides:
    """A reviewer's regrouping, applied by ``harmonize_leanb(group_overrides=...)`` before assign.

    ``moves`` maps a member (``"cohort:variable"``, the form of ``member_variable_names``) to its destination:
    an existing post-split group id, one of ``new_groups``' ids, or ``None`` for "in no group". A member may
    start in no group (a clustering leftover). ``new_groups`` are the reviewer's own groups, in the order they
    were made; one that ends up with no member forms nothing and costs nothing.
    """

    moves: Mapping[str, str | None] = field(default_factory=dict)
    new_groups: Sequence[ReviewerGroup] = ()

    def is_empty(self) -> bool:
        return not self.moves and not self.new_groups


def division_overrides(
    parts: Sequence[ConceptGroup],
    group_ids: Sequence[str],
    *,
    base: GroupOverrides | None = None,
) -> GroupOverrides:
    """An ACCEPTED division as the regrouping that reproduces it on every later leg.

    ``parts`` are the child groups :func:`~ddharmon.harmonization.leanb.readjudicate_split_only` carved out of a
    flagged group, and ``group_ids`` the reviewer ids to give them, one per part, in order (the staged UI mints
    ``rev:<uuid>``). Those children exist on the Gate-1 result only: the next leg re-runs the split, which keeps
    the group whole. So the division is carried the one way a reviewer's regrouping survives a replay — each part
    a New group (named for the concept the re-split gave it, ``split_from`` the divided group) and each of its
    members MOVED into it. The divided group is left with no member, so it is dropped and never assigned; each
    part is re-retrieved and gets an ideal of its own, like any New group.

    ``base`` is the reviewer's other decisions, kept: the division's moves win for the members it names.
    """
    if len(group_ids) != len(parts):
        raise ValueError(f"a division needs one group id per part: {len(parts)} parts, {len(group_ids)} ids")
    moves: dict[str, str | None] = dict(base.moves) if base is not None else {}
    new_groups = list(base.new_groups) if base is not None else []
    for i, (part, gid) in enumerate(zip(parts, group_ids, strict=True)):
        new_groups.append(
            ReviewerGroup(group_id=gid, name=part.concept or f"Part {i + 1}", split_from=part.readjudicated_from)
        )
        for member in part.member_variable_names:
            moves[member] = gid
    return GroupOverrides(moves=moves, new_groups=tuple(new_groups))


def _member_index(field_refs: list[FieldReference], cde_cohort: str) -> dict[str, int]:
    """``"cohort:var" -> embedding row`` for every SOURCE field (catalog rows are never group members)."""
    return {
        f"{r.dictionary_name}:{r.variable_name}": i for i, r in enumerate(field_refs) if r.dictionary_name != cde_cohort
    }


def resolve_group_membership(
    group_assign_prompts: list[PromptRecord],
    overrides: GroupOverrides,
    field_refs: list[FieldReference],
    *,
    cde_cohort: str = CDE_COHORT,
) -> tuple[dict[str, list[str]], set[str]]:
    """The groups' members AFTER the overrides, and the ids of every group whose membership changed.

    Returns ``(members, touched)``: ``members`` maps each existing group id (in prompt order) and then each
    New group id (in ``new_groups`` order) to its members; a group left empty maps to ``[]``. An existing
    group keeps its surviving members in their original order and gains its moved-in members in embedding-row
    order, so the result is a pure function of the overrides' CONTENT, not of the order a mapping was built in.

    Raises ``ValueError`` — before anything is paid for — on an override that names nothing real: a destination
    that is neither an existing group nor a declared New group, a member that is not a source field of this
    run, or a New group id that is declared twice or collides with an existing group.
    """
    row_of = _member_index(field_refs, cde_cohort)
    existing = [str(p.context.get("group_id", "")) for p in group_assign_prompts]
    new_ids = [g.group_id for g in overrides.new_groups]
    if len(set(new_ids)) != len(new_ids):
        raise ValueError(f"a New group id is declared more than once: {sorted(new_ids)}")
    clash = sorted(set(new_ids) & set(existing))
    if clash:
        raise ValueError(f"New group id(s) {clash} collide with existing concept groups")
    destinations = set(existing) | set(new_ids)
    for member, dest in overrides.moves.items():
        if member not in row_of:
            raise ValueError(f"cannot move {member!r}: it is not a source variable of this run")
        if dest is not None and dest not in destinations:
            raise ValueError(f"cannot move {member!r} to {dest!r}: no such concept group or New group")

    current: dict[str, str] = {}
    for gid, prompt in zip(existing, group_assign_prompts, strict=True):
        for m in prompt.context.get("member_variable_names", []):
            current.setdefault(str(m), gid)
    moved_in: dict[str, list[str]] = {gid: [] for gid in [*existing, *new_ids]}
    for member, dest in overrides.moves.items():
        if dest is not None and current.get(member) != dest:
            moved_in[dest].append(member)

    members: dict[str, list[str]] = {}
    touched: set[str] = set()
    for gid, prompt in zip(existing, group_assign_prompts, strict=True):
        before = [str(m) for m in prompt.context.get("member_variable_names", [])]
        kept = [m for m in before if overrides.moves.get(m, gid) == gid]
        after = kept + sorted(moved_in[gid], key=row_of.__getitem__)
        members[gid] = after
        if after != before:
            touched.add(gid)
    for gid in new_ids:
        members[gid] = sorted(moved_in[gid], key=row_of.__getitem__)
        if members[gid]:
            touched.add(gid)
    return members, touched


def _member_dicts(
    keys: list[str], embedded_dicts: list[EmbeddedDictionary], field_refs: list[FieldReference], cde_cohort: str
) -> list[dict]:
    """The member dicts a group prompt is built from — the same texts :func:`prepare_leanb` carries."""
    from ddharmon.harmonization.leanb import _MEMBER_PROMPT_TRUNC, _MEMBER_TRUNC, _member_prompt_text, _member_text

    lookup = build_field_lookup(embedded_dicts)
    row_of = _member_index(field_refs, cde_cohort)
    out = []
    for key in keys:
        row = row_of[key]
        ref = field_refs[row]
        fld = lookup.get((ref.dictionary_name, ref.variable_name))
        out.append(
            {
                "dictionary_name": ref.dictionary_name,
                "variable_name": ref.variable_name,
                "text": _member_text(fld, ref)[:_MEMBER_TRUNC],
                "prompt_text": _member_prompt_text(fld, ref)[:_MEMBER_PROMPT_TRUNC],
                "row": row,
            }
        )
    return out


def _content_id(keys: list[str]) -> str:
    """The order-independent content id of a member set (the same hash a cluster's id is)."""
    return cluster_content_id([(cohort, var) for cohort, _, var in (k.partition(":") for k in keys)])


def prepare_group_ideals(
    group_assign_prompts: list[PromptRecord],
    overrides: GroupOverrides,
    embedded_dicts: list[EmbeddedDictionary],
    field_refs: list[FieldReference],
    *,
    cde_cohort: str = CDE_COHORT,
    model_tag: str,
    measurand_split: bool = False,
    only_group_ids: Collection[str] | None = None,
) -> list[PromptRecord]:
    """One generate-ideal prompt per New or CHANGED group that has members — the overrides' only paid calls.

    A group qualifies when its membership differs from the split's (every filled New group does, by definition)
    and it still has a member. An unchanged group keeps the ideal it was split with. ``only_group_ids`` (the
    run's assign scope; ``None`` = every group) drops the rest: an ideal is consumed only by the assign, so a
    group that will not be assigned buys none.

    Written exactly like a cluster's ideal (members only, no candidates: it describes what the ideal CDE
    WOULD be, so it anchors the novel decision rather than rationalizing whatever was retrieved). Id =
    ``leanb:groupideal:<group_id>@<member content id>``. Order: existing groups in prompt order, then New groups
    in declared order — a pure function of the overrides' content, so a replay rebuilds the same list.
    """
    from ddharmon.harmonization.leanb import _SAMPLE_MEMBERS

    members, touched = resolve_group_membership(group_assign_prompts, overrides, field_refs, cde_cohort=cde_cohort)
    scope = None if only_group_ids is None else set(only_group_ids)
    candidates = [
        (str(p.context.get("group_id", "")), str(p.context.get("concept", "")), False) for p in group_assign_prompts
    ] + [(g.group_id, g.name, True) for g in overrides.new_groups]
    prompts: list[PromptRecord] = []
    for gid, name, is_new in candidates:
        keys = members.get(gid) or []
        if gid not in touched or not keys or (scope is not None and gid not in scope):
            continue
        mems = _member_dicts(keys, embedded_dicts, field_refs, cde_cohort)
        prompts.append(
            PromptRecord(
                id=f"{GROUP_IDEAL_PREFIX}{gid}@{_content_id(keys)}",
                system_prompt=generate_ideal_system_prompt(measurand_split),
                user_prompt=build_ideal_user_prompt([m["prompt_text"] for m in mems][:_SAMPLE_MEMBERS]),
                schema=IDEAL_SCHEMA,
                model_tag=model_tag,
                context={
                    "group_id": gid,
                    "name": name,
                    "member_variable_names": list(keys),
                    "reviewer_group": is_new,
                },
            )
        )
    return prompts


def apply_group_overrides(
    group_assign_prompts: list[PromptRecord],
    overrides: GroupOverrides,
    ideal_prompts: list[PromptRecord],
    ideal_responses: dict[str, object],
    embedded_dicts: list[EmbeddedDictionary],
    embeddings: NDArray[np.float32],
    field_refs: list[FieldReference],
    *,
    cde_cohort: str = CDE_COHORT,
    cde_dict: EmbeddedDictionary | None = None,
    top_k: int,
    model_tag: str,
    clean_cde_text: bool = False,
    representation_refine: bool = False,
) -> tuple[list[PromptRecord], set[str]]:
    """The per-group assign prompts after the overrides, and the ids of the groups the reviewer shaped.

    Untouched groups keep their prompt byte-for-byte; an emptied group is dropped; a changed group is rebuilt
    under ``<original id>@<member content id>`` with its group id and concept unchanged and its ideal the one
    REGENERATED for its final members (``ideal_regenerated`` on its context) — or, when no regenerated answer
    came back (or it was out of scope and never asked), the ideal it was split with, so it is never left with no
    anchor; each New group with members gets ``leanb:groupassign:<group_id>@<member content id>``, the reviewer's
    name as its concept, its own generated ideal (``ideal_responses`` for ``ideal_prompts``) and
    ``reviewer_group`` on its context.
    """
    from ddharmon.harmonization.leanb import _build_backbone, _parse_ideal, group_assign_prompt

    members, touched = resolve_group_membership(group_assign_prompts, overrides, field_refs, cde_cohort=cde_cohort)
    if not touched:
        return list(group_assign_prompts), touched
    backbone = _build_backbone(
        embedded_dicts, build_field_lookup(embedded_dicts), cde_cohort, cde_dict, clean_text=clean_cde_text
    )
    row_of = {(r.dictionary_name, r.variable_name): i for i, r in enumerate(field_refs)}

    def build(prompt_id: str, keys: list[str], identity: dict, system_prompt: str) -> PromptRecord:
        return group_assign_prompt(
            f"{prompt_id}@{_content_id(keys)}",
            _member_dicts(keys, embedded_dicts, field_refs, cde_cohort),
            identity,
            embeddings=embeddings,
            backbone=backbone,
            row_of=row_of,
            top_k=top_k,
            system_prompt=system_prompt,
            model_tag=model_tag,
        )

    ideal_of = {str(p.context["group_id"]): _parse_ideal(ideal_responses.get(p.id)) for p in ideal_prompts}
    out: list[PromptRecord] = []
    for prompt in group_assign_prompts:
        gid = str(prompt.context.get("group_id", ""))
        if gid not in touched:
            out.append(prompt)
            continue
        keys = members[gid]
        if not keys:  # every member moved away: nothing left to assign
            continue
        ctx = prompt.context
        regenerated = ideal_of.get(gid, "")
        identity = {
            "cluster_id": ctx.get("cluster_id", ""),
            "group_idx": ctx.get("group_idx", 0),
            "group_id": gid,
            "concept": ctx.get("concept", ""),
            "ideal_cde": regenerated or ctx.get("ideal_cde", ""),
            "split_raw": ctx.get("split_raw", {}),
            "regrouped": True,
            "ideal_regenerated": bool(regenerated),
        }
        out.append(build(prompt.id, keys, identity, prompt.system_prompt))

    reassign_system = group_reassign_system_prompt(representation_refine)
    for group in overrides.new_groups:
        keys = members.get(group.group_id) or []
        if not keys:
            continue
        identity = {
            "cluster_id": group.group_id,  # a reviewer group spans clusters: it is its own unit
            "group_idx": 0,
            "group_id": group.group_id,
            "concept": group.name,
            "ideal_cde": ideal_of.get(group.group_id, ""),
            "split_raw": {},
            "reviewer_group": True,
            **({"readjudicated_from": group.split_from} if group.split_from else {}),
        }
        out.append(build(f"leanb:groupassign:{group.group_id}", keys, identity, reassign_system))
    logger.info(
        "apply_group_overrides: %d moves, %d new groups -> %d groups reshaped, %d assign prompts",
        len(overrides.moves),
        len(overrides.new_groups),
        len(touched),
        len(out),
    )
    return out, touched
