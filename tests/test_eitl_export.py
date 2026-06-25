"""Tests for the v2 split-aware EITL campaign export (ddharmon.export.eitl).

Covers the A→B import contract (no raw newlines, U+2028 breaks, tinyId links, QUOTE_ALL)
and the reviewer-pass refinements (free-text routing, templated-family collapse, magnet
flag, qualifier-divergence flag, honest match_cosine, target-card framing).
"""

from __future__ import annotations

import csv

import pytest

from ddharmon.export import eitl
from ddharmon.export.eitl import (
    LS,
    build_cde_lookup,
    clean,
    export_split_eitl_campaign,
    labeled,
    looks_freetext,
    pack,
    qualifier_divergence,
    target_card,
)
from ddharmon.harmonization import LeanBRecord, LeanBResult
from ddharmon.models.data_dictionary import DataDictionary, Field

# --- contract helpers ----------------------------------------------------------


def test_clean_collapses_all_whitespace():
    assert clean("a\nb\tc\r\n  d") == "a b c d"


def test_pack_and_labeled_never_leak_raw_newlines():
    out = pack(["line one\nwith newline", "line two", "", "  "])
    assert "\n" not in out and "\r" not in out
    assert LS in out  # the two non-empty lines are joined by the line separator

    lab = labeled([("Variable name", "age"), ("Question text", "What is your age?\n"), ("Description", "")])
    assert "\n" not in lab and "\r" not in lab
    assert "Description" not in lab  # empty section dropped
    assert "Variable name: age" in lab


def test_target_card_does_not_mislabel_designation():
    # designation == question text -> 'CDE' name is suppressed (no confusing duplicate)
    card = target_card("What is your age?", {"question_text": "What is your age?", "definition": "Age in years."})
    assert "Question text: What is your age?" in card
    assert "CDE: What is your age?" not in card
    # no separate question text -> the designation IS the question
    card2 = target_card("Participant age", {"question_text": "", "definition": "Age."})
    assert "CDE (question): Participant age" in card2


def test_qualifier_divergence_fires_on_axis_words_not_benign_synonyms():
    assert qualifier_divergence(["C:home_zip", "C:work_zip"]) != ""
    assert qualifier_divergence(["C:address_mother", "C:address_father"]) != ""
    # benign cross-cohort naming difference must NOT fire
    assert qualifier_divergence(["C:sex", "D:biological_sex"]) == ""
    # 'son' inside 'person' must not false-fire
    assert qualifier_divergence(["C:person_id", "D:respondent"]) == ""
    assert qualifier_divergence(["C:only_one"]) == ""  # <2 fields


def test_looks_freetext_generic_signals():
    assert looks_freetext(_field("other_specify", q="Please specify the other condition"), "C:other_specify")
    assert looks_freetext(_field("notes", q="General notes", dt="text"), "C:notes")  # text dtype, no options
    assert not looks_freetext(_field("age", q="What is your age?", dt="integer"), "C:age")
    assert not looks_freetext(None, "C:x")


# --- build_cde_lookup ----------------------------------------------------------


def test_build_cde_lookup_reads_tinyid_question_definition():
    cde_dd = DataDictionary(
        name="NIH_CDE",
        cohort_name="NIH_CDE",
        fields={
            "Age Value": Field(
                variable_name="Age Value",
                description="The age of the participant.",
                field_id="ABC123",
                question_text="What is the participant's age?",
            )
        },
    )
    lk = build_cde_lookup(cde_dd)
    assert lk["Age Value"] == {
        "tinyId": "ABC123",
        "question_text": "What is the participant's age?",
        "definition": "The age of the participant.",
    }


# --- fixtures / helpers --------------------------------------------------------


def _field(var: str, *, q: str = "", desc: str = "", dt: str | None = None, short: str = "") -> Field:
    return Field(
        variable_name=var,
        description=desc or q or var,
        question_text=q or None,
        short_label=short or None,
        data_type=dt,
    )


def _record(cluster_id, verdict, cde_id, members, *, concept="concept", chosen_cos=0.8, cross=False) -> LeanBRecord:
    return LeanBRecord(
        cluster_id=cluster_id,
        verdict=verdict,
        route="assigned" if verdict in ("adopt", "refine") else "gencde_residual",
        group_id=f"{cluster_id}#g0",
        concept=concept,
        cde_id=cde_id,
        cde_external_id="" if cde_id is None else f"TINY_{cde_id[:4]}",
        ideal_cde=f"ideal for {concept}",
        rationale="matched on meaning",
        chosen_cos=chosen_cos,
        top1_cos=chosen_cos,
        member_variable_names=members,
        cohorts=sorted({m.split(":", 1)[0] for m in members}),
        cross_cohort=cross,
        n_members=len(members),
    )


@pytest.fixture
def campaign(tmp_path):
    """A small but representative split-aware result + its exported campaign."""
    cohort_a = DataDictionary(
        name="CohortA",
        cohort_name="CohortA",
        fields={
            "age": _field("age", q="What is your age?", dt="integer"),
            "home_zip": _field("home_zip", q="What is your postal code?", desc="Home postal code"),
            "work_zip": _field("work_zip", q="What is your postal code?", desc="Work postal code"),
            "comments": _field("comments", q="Please specify any other details", dt="text"),
            **{f"medication_{i}": _field(f"medication_{i}", q=f"Name of medication {i}") for i in range(1, 6)},
        },
    )
    source_dicts = {"CohortA": cohort_a}
    cde_lookup = {
        "Age Value": {"tinyId": "AGE001", "question_text": "What is your age in years?", "definition": "Age."},
        "Postal Code": {"tinyId": "ZIP001", "question_text": "Postal/ZIP code", "definition": "A postal code."},
        "Medication Name": {"tinyId": "MED001", "question_text": "Medication name", "definition": "A drug name."},
    }
    result = LeanBResult(
        records=[
            _record("c1", "adopt", "Age Value", ["CohortA:age"], concept="age", chosen_cos=0.91),
            # two zip vars share a question but carry different qualifiers -> divergence flag
            _record(
                "c2", "refine", "Postal Code", ["CohortA:home_zip", "CohortA:work_zip"], concept="zip", chosen_cos=0.7
            ),
            # free-text field -> routed out of match_review
            _record("c3", "adopt", "Age Value", ["CohortA:comments"], concept="comments", chosen_cos=0.5),
            # templated family: 5 distinct-question meds at one CDE -> collapse to 1 row
            _record(
                "c4",
                "refine",
                "Medication Name",
                [f"CohortA:medication_{i}" for i in range(1, 6)],
                concept="med",
                chosen_cos=0.6,
            ),
            # novel -> must NOT appear in match_review
            _record("c5", "novel", None, ["CohortA:age"], concept="novel thing", chosen_cos=0.2),
        ]
    )
    counts = export_split_eitl_campaign(result, source_dicts, cde_lookup, tmp_path, stem="t")
    return tmp_path, counts


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --- campaign behavior ---------------------------------------------------------


def test_match_review_written_with_contract(campaign):
    tmp_path, counts = campaign
    rows = _read(tmp_path / "t_match_review.csv")
    assert rows, "match_review should have rows"
    # CONTRACT: no raw CR/LF in any cell; multi-section cells use U+2028
    for r in rows:
        for v in r.values():
            assert "\n" not in v and "\r" not in v
    assert any(LS in r["source_text"] for r in rows)


def test_tinyid_links_present(campaign):
    tmp_path, _ = campaign
    rows = _read(tmp_path / "t_match_review.csv")
    assert all(r["target_url"].startswith("https://cde.nlm.nih.gov/deView?tinyId=") for r in rows if r["target_id"])
    assert any(r["target_id"] for r in rows)


def test_no_mislabeled_confidence_column(campaign):
    tmp_path, _ = campaign
    rows = _read(tmp_path / "t_match_review.csv")
    assert "llm_confidence" not in rows[0]
    assert "match_cosine" in rows[0]  # honest cosine stays


def test_novel_records_absent_from_match_review(campaign):
    tmp_path, _ = campaign
    rows = _read(tmp_path / "t_match_review.csv")
    assert all(r["leaf_uid"] != "c5" for r in rows)


def test_freetext_routed_out(campaign):
    tmp_path, counts = campaign
    assert counts["freetext_review"] >= 1
    ft = _read(tmp_path / "t_freetext_review.csv")
    assert any("comments" in r["source_id"] for r in ft)
    # the free-text field must NOT also be in match_review
    mr = _read(tmp_path / "t_match_review.csv")
    assert all("comments" not in r["source_id"] for r in mr)


def test_templated_family_collapsed(campaign):
    tmp_path, counts = campaign
    assert counts["collapsed_families"] >= 4  # 5 meds -> 1 row, 4 collapsed
    rows = _read(tmp_path / "t_match_review.csv")
    fam = [r for r in rows if r.get("review_unit") == "templated_family"]
    assert len(fam) == 1
    assert "Templated family" in fam[0]["source_text"]
    assert int(fam[0]["n_vars_sharing"]) == 5


def test_qualifier_divergence_note_in_reasoning(campaign):
    tmp_path, _ = campaign
    rows = _read(tmp_path / "t_match_review.csv")
    zip_rows = [r for r in rows if r["leaf_uid"] == "c2"]
    assert zip_rows and "Granularity check" in zip_rows[0]["llm_reasoning"]


def test_magnet_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(eitl, "MAGNET_MIN", 3)
    monkeypatch.setattr(eitl, "FAMILY_MIN", 99)  # disable collapse so distinct sources remain
    dd = DataDictionary(
        name="C",
        cohort_name="C",
        fields={f"v{i}": _field(f"v{i}", q=f"distinct question number {i}") for i in range(5)},
    )
    result = LeanBResult(
        records=[_record("m", "refine", "Catch All CDE", [f"C:v{i}" for i in range(5)], chosen_cos=0.5)]
    )
    cde_lookup = {"Catch All CDE": {"tinyId": "CA1", "question_text": "Other condition", "definition": "x"}}
    export_split_eitl_campaign(result, {"C": dd}, cde_lookup, tmp_path, stem="m")
    rows = _read(tmp_path / "m_match_review.csv")
    assert rows and all("Catch-all check" in r["llm_reasoning"] for r in rows)
