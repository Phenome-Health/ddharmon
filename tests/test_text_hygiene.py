"""Tests for ddharmon.text_hygiene — the shared, cohort-agnostic cleaning primitives.

Covers the three prompt-hygiene defects found on real cohort data (2026-07-30): administrative /
data-collection wrappers, and missing/refused/don't-know sentinel codes.
"""

from __future__ import annotations

from ddharmon.text_hygiene import (
    clean_field_text,
    filter_sentinel_labels,
    is_sentinel_label,
    strip_sentinel_encodings,
)


class TestCleanFieldText:
    """Administrative / data-collection text stripping."""

    def test_empty_inputs(self) -> None:
        assert clean_field_text("") == ""
        assert clean_field_text(None) == ""
        assert clean_field_text("   ") == ""

    def test_unwraps_instrument_preamble_and_drops_help_html(self) -> None:
        # The canonical UKBB ACE pattern: administration preamble + quoted question + trailing help HTML.
        raw = 'ACE touchscreen question "Do you smoke tobacco now?" <table><tr><td>Help: ...</td></tr></table>'
        assert clean_field_text(raw) == "Do you smoke tobacco now?"

    def test_unwraps_bare_quoted_question_with_trailing_help(self) -> None:
        raw = '"How many cups of coffee do you drink?" <br>If the participant activated Help they saw: ...'
        assert clean_field_text(raw) == "How many cups of coffee do you drink?"

    def test_strips_html_tags_and_entities(self) -> None:
        assert clean_field_text("<p>Weight&nbsp;in&nbsp;kg</p>") == "Weight in kg"

    def test_strips_survey_boilerplate(self) -> None:
        out = clean_field_text("Ethnic background (select all that apply)")
        assert "select all that apply" not in out.lower()
        assert "Ethnic background" in out

    def test_boilerplate_flag_off_preserves_phrase(self) -> None:
        # The ingestion path passes strip_boilerplate=False (structural-only) — no mid-sentence residue.
        assert clean_field_text("Please specify.", strip_boilerplate=False) == "Please specify."
        assert clean_field_text("Race (select all that apply)", strip_boilerplate=False) == (
            "Race (select all that apply)"
        )

    def test_preserves_non_html_angle_brackets(self) -> None:
        # Domain angle-bracket tokens (MESA `<OR>`, comparisons) are NOT HTML tags — keep them.
        assert clean_field_text("VF <OR> ASYSTOLE") == "VF <OR> ASYSTOLE"
        assert clean_field_text("systolic <50 mmHg") == "systolic <50 mmHg"

    def test_strips_recognised_html_only(self) -> None:
        assert clean_field_text("Weight <b>in</b> kg <OR> lbs") == "Weight in kg <OR> lbs"

    def test_benign_text_with_question_word_is_untouched(self) -> None:
        # No quoted question follows "Questions" -> the preamble unwrap must NOT fire (no over-strip).
        assert clean_field_text("Questions about diet and nutrition") == "Questions about diet and nutrition"

    def test_pure_markup_returns_empty(self) -> None:
        # The ingestion caller guards against blanking; the primitive itself may legitimately return "".
        assert clean_field_text("<p></p>") == ""

    def test_collapses_whitespace(self) -> None:
        assert clean_field_text("blood   pressure\n\tsystolic") == "blood pressure systolic"

    def test_does_not_touch_interior_quotes(self) -> None:
        # A quoted phrase not anchored at the start (no trailing tag/EOS boundary) is left as-is.
        text = 'The variable measures the so-called "resting" heart rate at baseline'
        assert clean_field_text(text) == text


class TestIsSentinelLabel:
    """Label-based (cohort-agnostic) missing/refused/DK detection."""

    def test_substring_sentinels(self) -> None:
        for lbl in ("MISSING", "Missing value", "Prefer not to answer", "Do not know", "Refused", "Not applicable"):
            assert is_sentinel_label(lbl), lbl

    def test_exact_sentinels(self) -> None:
        for lbl in ("Unknown", "N/A", "na", "DK", "No answer", "Refuse"):
            assert is_sentinel_label(lbl), lbl

    def test_empty_is_sentinel(self) -> None:
        assert is_sentinel_label("")
        assert is_sentinel_label(None)
        assert is_sentinel_label("   ")

    def test_real_options_are_not_sentinels(self) -> None:
        for lbl in ("Yes", "No", "Male", "Female", "Every day", "Banana", "years", "Current smoker"):
            assert not is_sentinel_label(lbl), lbl

    def test_short_exact_tokens_do_not_overmatch_as_substrings(self) -> None:
        # "na"/"dk" are EXACT-only so they must not fire inside real words.
        assert not is_sentinel_label("banana")
        assert not is_sentinel_label("dka")  # diabetic ketoacidosis


class TestStripSentinelEncodings:
    """Route-side value-encoding filtering (the concept-identity route drops sentinels)."""

    def test_sentinel_only_collapses_to_empty(self) -> None:
        # A numeric field encoded only as a missing sentinel -> no option tail (reads as numeric).
        assert strip_sentinel_encodings("-9=MISSING") == ""

    def test_mixed_keeps_real_options(self) -> None:
        assert strip_sentinel_encodings("1=Yes|0=No|-9=Missing") == "1=Yes|0=No"

    def test_multiple_sentinels_dropped(self) -> None:
        raw = "-9=MISSING|0=NO|1=YES|9=DO NOT KNOW"
        assert strip_sentinel_encodings(raw) == "0=NO|1=YES"

    def test_ukbb_prefer_not_and_dk(self) -> None:
        raw = "1=Low|2=High|-1=Do not know|-3=Prefer not to answer"
        assert strip_sentinel_encodings(raw) == "1=Low|2=High"

    def test_comma_separated_labels(self) -> None:
        assert strip_sentinel_encodings("1, Yes|0, No|-9, Missing") == "1, Yes|0, No"

    def test_all_real_unchanged(self) -> None:
        assert strip_sentinel_encodings("1=Yes|0=No") == "1=Yes|0=No"

    def test_empty(self) -> None:
        assert strip_sentinel_encodings("") == ""
        assert strip_sentinel_encodings(None) == ""


class TestFilterSentinelLabels:
    def test_drops_sentinels_order_preserving(self) -> None:
        assert filter_sentinel_labels(["Yes", "Missing", "No", "Prefer not to answer"]) == ["Yes", "No"]
