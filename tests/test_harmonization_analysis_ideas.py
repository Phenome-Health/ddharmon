"""Post-run "analysis ideas" — the core library capability.

Covers the concept digest (cross-cohort filter + ordering), grounding (hallucinated concepts dropped),
truncated-JSON salvage, and the empty (no cross-cohort concept) short-circuit. No real LLM call — a fake
``complete`` returns canned JSON.
"""

from __future__ import annotations

import json

from ddharmon.harmonization import AnalysisIdea, build_concept_digest, generate_analysis_ideas
from ddharmon.harmonization.analysis_ideas import _parse_ideas
from ddharmon.harmonization.models import LeanBRecord


def _rec(concept: str, cohorts: list[str], *, verdict: str = "adopt", cde_id: str | None = None, n_members: int = 0):
    return LeanBRecord(
        cluster_id="c",
        verdict=verdict,
        route="assigned",
        group_id=concept,
        concept=concept,
        cde_id=cde_id,
        cohorts=cohorts,
        cross_cohort=len(cohorts) >= 2,
        n_members=n_members or len(cohorts),
    )


def test_build_concept_digest_keeps_only_cross_cohort_sorted():
    """Digest keeps concepts in ≥2 cohorts (the pooling signal), most cohorts first, carrying the CDE id."""
    records = [
        _rec("Smoking status", ["A", "B"], cde_id="SmokeCDE", n_members=4),
        _rec("CVD", ["A", "B", "C"], n_members=6),
        _rec("Local-only", ["A"], n_members=1),  # single-cohort → dropped
    ]
    digest = build_concept_digest(records)
    assert [d.concept for d in digest] == ["CVD", "Smoking status"]  # most cohorts first
    assert digest[0].cohorts == ["A", "B", "C"] and digest[0].cde is None
    assert digest[1].cde == "SmokeCDE"


def test_generate_grounds_ideas_and_drops_hallucinations():
    """Generation returns AnalysisIdea dataclasses grounded in the run's concepts; ungrounded ideas drop."""
    records = [_rec("Smoking status", ["A", "B"]), _rec("CVD", ["A", "B", "C"])]

    def fake_complete(prompt, *, system=None, max_tokens=512):
        return json.dumps(
            {
                "ideas": [
                    {"title": "Pooled smoking→CVD", "hypothesis": "h", "concepts": ["Smoking status", "CVD"],
                     "cohorts": ["A", "B"], "method": "logistic regression", "whyNewlyPossible": "w", "category": "assoc"},
                    {"title": "Hallucinated", "hypothesis": "h", "concepts": ["Made-up concept"], "cohorts": ["A"],
                     "method": "m", "whyNewlyPossible": "w", "category": "x"},
                ]
            }
        )  # fmt: skip

    out = generate_analysis_ideas(records, fake_complete)
    assert out.n_concepts == 2
    assert all(isinstance(i, AnalysisIdea) for i in out.ideas)
    titles = [i.title for i in out.ideas]
    assert "Pooled smoking→CVD" in titles and "Hallucinated" not in titles  # ungrounded idea dropped
    idea = out.ideas[0]
    assert idea.concepts == ["Smoking status", "CVD"] and idea.why_newly_possible == "w"


def test_no_cross_cohort_returns_empty_without_calling_llm():
    """A run with no concept shared by ≥2 cohorts short-circuits to empty — no LLM call."""
    called: list[int] = []

    def fake_complete(prompt, *, system=None, max_tokens=512):
        called.append(1)
        return "{}"

    out = generate_analysis_ideas([_rec("Solo", ["A"])], fake_complete)
    assert out.ideas == [] and out.n_concepts == 0
    assert not called  # never hit the model


def test_parse_ideas_salvages_truncated_json():
    """A response cut off mid-array (invalid JSON) still yields the complete idea; the partial one drops."""
    truncated = (
        '{"ideas":[{"title":"One","hypothesis":"h","concepts":["A"],"cohorts":["X"],"method":"m",'
        '"whyNewlyPossible":"w","category":"c"},{"title":"Two","hypothesis":"cut off and never clo'
    )
    ideas = _parse_ideas(truncated, allowed={"A", "B"})
    assert len(ideas) == 1 and ideas[0].title == "One" and ideas[0].concepts == ["A"]
