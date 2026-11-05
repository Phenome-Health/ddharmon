"""Gate 1 score SUGGESTIONS — the free, retrieval-only half of the score builder's match.

The paid match (hybrid retrieval -> one LLM judge) runs on Gate 4, after Gate 1 has been continued, so on a live
run Gate 1 has no matches to seed its scope from. :func:`suggest_groups` runs ONLY the retrieval half against Gate
1's own groups: per declared component, the group-diverse shortlist over the run's variables, each group scored by
the DENSE COSINE of its best-matching member. That score is absolute (BioLORD vectors are L2-normalised), so one
calibrated cut-off means the same thing for every component — unlike the RRF fusion the shortlist ranks by, which
is rank-based and not comparable across components. With no dense encoder there is no comparable score, so no
suggestions at all (and the result says why) rather than a threshold over BM25.

Pure and deterministic: a hashing bag-of-words embedder stands in for BioLORD; there is no ``complete`` anywhere.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re

import numpy as np
import pytest

from ddharmon.harmonization import (
    GATE1_SUGGEST_MIN_COSINE,
    ScoreComponent,
    suggest_groups,
    suggestions_to_dict,
)
from ddharmon.harmonization.composite import _OPTION_ID_SEP, build_variable_index

_DIM = 64


def _vector(text: str) -> np.ndarray:
    """A deterministic bag-of-words vector (NOT normalised — :func:`suggest_groups` must normalise)."""
    v = np.zeros(_DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % _DIM] += 1.0
    return v


def _embed(texts: list[str]) -> np.ndarray:
    return np.stack([_vector(t) for t in texts]) * 3.0  # deliberately off-unit length


def _cos(a: str, b: str) -> float:
    va, vb = _vector(a), _vector(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def _components(*names: str) -> list[ScoreComponent]:
    # Names only, as a Gate 1 declaration states them (no definition is invented — ``definition_for``).
    return [ScoreComponent(name=n, definition="") for n in names]


@pytest.fixture
def field_index() -> dict[str, dict]:
    return {
        "UKBB:grip_l": {"name": "Hand grip strength (left)"},
        "UKBB:grip_r": {"name": "Hand grip strength (right)"},
        "CLSA:grip": {"name": "Grip strength dynamometer"},
        "UKBB:walk": {"name": "Usual walking pace"},
        "CLSA:gait": {"name": "Gait speed timed walk"},
        "UKBB:sleep": {"name": "Sleep duration hours"},
        "UKBB:stray": {"name": "Grip strength stray variable in no group"},
    }


@pytest.fixture
def groups() -> dict[str, list[str]]:
    return {
        "g_grip#g0": ["UKBB:grip_l", "UKBB:grip_r", "CLSA:grip"],
        "g_walk#g0": ["UKBB:walk", "CLSA:gait"],
        "g_sleep#g0": ["UKBB:sleep"],
    }


def _by_group(result, component: str) -> dict[str, dict]:
    comp = next(c for c in result.components if c.component == component)
    return {g.group_id: {"score": g.score, "best": g.best_member, "option": g.best_option} for g in comp.groups}


def test_each_group_is_scored_by_the_cosine_of_its_best_member(field_index, groups):
    result = suggest_groups(_components("Grip strength"), field_index, groups, embed=_embed, top_k=8)
    assert result.scored is True
    got = _by_group(result, "Grip strength")
    query = "Grip strength."
    expected_best = max(groups["g_grip#g0"], key=lambda m: _cos(query, field_index[m]["name"]))
    assert got["g_grip#g0"]["best"] == expected_best
    assert got["g_grip#g0"]["score"] == pytest.approx(_cos(query, field_index[expected_best]["name"]), abs=1e-4)
    # An absolute cosine in [-1, 1], not an RRF fusion score (which is ~1/(60+rank)).
    assert 0.5 < got["g_grip#g0"]["score"] <= 1.0


def test_groups_come_back_best_first(field_index, groups):
    result = suggest_groups(_components("Grip strength"), field_index, groups, embed=_embed, top_k=8)
    scores = [g.score for g in result.components[0].groups]
    assert scores == sorted(scores, reverse=True)
    assert result.components[0].groups[0].group_id == "g_grip#g0"


def test_top_k_counts_distinct_groups_not_near_duplicate_variables(field_index, groups):
    # Three grip variables in one group spend ONE slot: with top_k=2 a second, different group still appears.
    result = suggest_groups(_components("Grip strength walking"), field_index, groups, embed=_embed, top_k=2)
    ids = [g.group_id for g in result.components[0].groups]
    assert len(ids) == len(set(ids)) == 2
    assert set(ids) == {"g_grip#g0", "g_walk#g0"}


def test_a_variable_in_no_group_never_surfaces(field_index, groups):
    result = suggest_groups(_components("Grip strength"), field_index, groups, embed=_embed, top_k=8)
    members = [g.best_member for c in result.components for g in c.groups]
    assert "UKBB:stray" not in members


def test_membership_is_whatever_the_caller_passes__a_moved_variable_scores_for_its_new_group(field_index):
    # Gate 1's EFFECTIVE membership: the reviewer moved CLSA:grip into a New group of their own.
    effective = {
        "g_grip#g0": ["UKBB:grip_l", "UKBB:grip_r"],
        "rg-mine": ["CLSA:grip"],
        "g_walk#g0": ["UKBB:walk", "CLSA:gait"],
    }
    result = suggest_groups(_components("Grip strength dynamometer"), field_index, effective, embed=_embed)
    got = _by_group(result, "Grip strength dynamometer")
    assert got["rg-mine"]["best"] == "CLSA:grip"
    assert got["g_grip#g0"]["best"] in {"UKBB:grip_l", "UKBB:grip_r"}
    assert got["rg-mine"]["score"] > got["g_grip#g0"]["score"]


def test_an_emptied_group_is_never_suggested(field_index):
    effective = {"g_grip#g0": [], "g_walk#g0": ["UKBB:walk", "CLSA:gait", "UKBB:grip_l"]}
    result = suggest_groups(_components("Grip strength"), field_index, effective, embed=_embed)
    assert "g_grip#g0" not in _by_group(result, "Grip strength")


def test_no_dense_encoder_means_no_suggestions_and_says_why(field_index, groups):
    result = suggest_groups(_components("Grip strength", "Gait"), field_index, groups, embed=None)
    assert result.scored is False
    assert [c.component for c in result.components] == ["Grip strength", "Gait"]
    assert all(c.groups == [] for c in result.components)
    # The reason names the actual cause: lexical/RRF scores are not comparable across components.
    assert "dense" in result.reason.lower() and "comparable" in result.reason.lower()


def test_the_function_takes_no_model_call():
    params = inspect.signature(suggest_groups).parameters
    assert "complete" not in params


def test_deterministic(field_index, groups):
    a = suggestions_to_dict(
        suggest_groups(_components("Grip strength", "Walking pace"), field_index, groups, embed=_embed)
    )
    b = suggestions_to_dict(
        suggest_groups(_components("Grip strength", "Walking pace"), field_index, groups, embed=_embed)
    )
    assert a == b


def test_a_checklist_option_is_named_as_the_best_member_with_its_option_label():
    fi = {
        "UKBB:eye": {
            "name": "Eye problems disorders",
            "dataType": "Categorical (multiple)",
            "valueEncoding": "1=Glaucoma|2=Cataract|3=Diabetes related eye disease|4=Injury or trauma",
        },
        "UKBB:sleep": {"name": "Sleep duration"},
    }
    index = build_variable_index(fi, checklist_members={"UKBB:eye"})
    assert any(_OPTION_ID_SEP in e.concept_id for e in index)  # the option units exist
    result = suggest_groups(
        _components("Glaucoma"), fi, {"g_eye#g0": ["UKBB:eye"], "g_sleep#g0": ["UKBB:sleep"]}, embed=_embed
    )
    got = _by_group(result, "Glaucoma")
    assert got["g_eye#g0"]["best"] == "UKBB:eye"  # the parent VARIABLE id, never the option-suffixed unit id
    assert got["g_eye#g0"]["option"] == "Glaucoma"


def test_serialised_shape_is_camel_cased_and_carries_the_threshold(field_index, groups):
    result = suggest_groups(_components("Grip strength"), field_index, groups, embed=_embed)
    payload = suggestions_to_dict(result)
    json.dumps(payload)  # JSON-ready
    assert payload["scored"] is True
    assert payload["threshold"] == GATE1_SUGGEST_MIN_COSINE
    assert payload["scoreKind"] == "dense_cosine"
    comp = payload["components"][0]
    assert comp["component"] == "Grip strength"
    first = comp["groups"][0]
    assert set(first) >= {"groupId", "score", "bestMember"}
    assert "bestOption" not in first  # only present when the best unit is a checklist option


def test_the_calibrated_threshold_is_a_cosine():
    assert 0.0 < GATE1_SUGGEST_MIN_COSINE < 1.0


def test_empty_inputs_are_not_errors(field_index, groups):
    assert suggest_groups([], field_index, groups, embed=_embed).components == []
    empty = suggest_groups(_components("Grip strength"), {}, groups, embed=_embed)
    assert empty.scored is True and empty.components[0].groups == []
