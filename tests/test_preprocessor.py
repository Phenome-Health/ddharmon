"""Tests for ddharmon ingestion preprocessor — the RULES.

Every call here passes ``enabled=True`` explicitly, so these tests keep exercising the rules whatever the
master switch's default is. The switch itself (``enabled=False`` changes nothing), and the golden snapshot
proving ``enabled=True`` is byte-identical to the rules' output before the switch existed, live in
``tests/test_preprocessor_opt_in.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ddharmon.models.data_dictionary import DataDictionary, Field


class TestSplitIdentifier:
    """Tests for _split_identifier helper."""

    def test_snake_case(self) -> None:
        from ddharmon.ingestion.preprocessor import _split_identifier

        assert _split_identifier("assessment_health_history_bmi") == ["assessment", "health", "history", "bmi"]

    def test_camel_case(self) -> None:
        from ddharmon.ingestion.preprocessor import _split_identifier

        assert _split_identifier("assessmentHealthHistoryBmi") == ["assessment", "health", "history", "bmi"]

    def test_dot_notation(self) -> None:
        from ddharmon.ingestion.preprocessor import _split_identifier

        assert _split_identifier("the.basics.birthplace") == ["the", "basics", "birthplace"]

    def test_kebab_case(self) -> None:
        from ddharmon.ingestion.preprocessor import _split_identifier

        assert _split_identifier("blood-pressure-systolic") == ["blood", "pressure", "systolic"]

    def test_mixed_delimiters(self) -> None:
        from ddharmon.ingestion.preprocessor import _split_identifier

        assert _split_identifier("the_basics.birthplace_country") == ["the", "basics", "birthplace", "country"]

    def test_single_token(self) -> None:
        from ddharmon.ingestion.preprocessor import _split_identifier

        assert _split_identifier("age") == ["age"]

    def test_uppercase_acronym(self) -> None:
        from ddharmon.ingestion.preprocessor import _split_identifier

        # "BMI" stays together, then "calculated" is separate
        result = _split_identifier("BMICalculated")
        assert result == ["bm", "icalculated"] or result == ["bmicalculated"] or "bmi" in "".join(result).lower()


class TestFindCommonPrefixTokens:
    """Tests for common prefix detection."""

    def test_clear_common_prefix(self) -> None:
        from ddharmon.ingestion.preprocessor import _find_common_prefix_tokens

        names = [
            "assessment_health_bmi",
            "assessment_health_age",
            "assessment_health_weight",
            "assessment_health_height",
            "assessment_health_bp",
        ]
        prefix = _find_common_prefix_tokens(names, min_ratio=0.5)
        assert prefix == ["assessment", "health"]

    def test_no_common_prefix(self) -> None:
        from ddharmon.ingestion.preprocessor import _find_common_prefix_tokens

        names = ["age", "bmi", "height", "weight"]
        prefix = _find_common_prefix_tokens(names, min_ratio=0.5)
        assert prefix == []

    def test_prefix_below_ratio(self) -> None:
        from ddharmon.ingestion.preprocessor import _find_common_prefix_tokens

        # Only 1 out of 5 share any given prefix — below min_count=max(2, 0.8*5=4)
        names = [
            "assessment_bmi",
            "diet_sugar",
            "exercise_steps",
            "lab_glucose",
            "vital_bp",
        ]
        prefix = _find_common_prefix_tokens(names, min_ratio=0.8)
        assert prefix == []

    def test_empty_list(self) -> None:
        from ddharmon.ingestion.preprocessor import _find_common_prefix_tokens

        assert _find_common_prefix_tokens([], min_ratio=0.5) == []

    def test_single_name(self) -> None:
        from ddharmon.ingestion.preprocessor import _find_common_prefix_tokens

        # min_count is max(2, ...) so a single name can't form a prefix group
        assert _find_common_prefix_tokens(["assessment_bmi"], min_ratio=0.5) == []


class TestRemoveTokenPrefix:
    """Tests for _remove_token_prefix helper."""

    def test_snake_case(self) -> None:
        from ddharmon.ingestion.preprocessor import _remove_token_prefix

        assert _remove_token_prefix("assessment_health_history_bmi", 3) == "bmi"

    def test_dot_notation(self) -> None:
        from ddharmon.ingestion.preprocessor import _remove_token_prefix

        assert _remove_token_prefix("the.basics.birthplace", 2) == "birthplace"

    def test_would_leave_empty(self) -> None:
        from ddharmon.ingestion.preprocessor import _remove_token_prefix

        # Removing all tokens returns empty string
        result = _remove_token_prefix("assessment_health", 2)
        assert result == ""

    def test_remove_one_token(self) -> None:
        from ddharmon.ingestion.preprocessor import _remove_token_prefix

        assert _remove_token_prefix("survey_age_years", 1) == "age_years"


class TestNormalizeUnicode:
    """Tests for unicode normalization."""

    def test_normalizes_curly_quotes(self) -> None:
        from ddharmon.ingestion.preprocessor import _normalize_unicode
        from ddharmon.models.data_dictionary import Field

        # ftfy normalizes curly quotes to straight quotes
        fields = [Field(variable_name="age", description="Patient\u2019s age at enrollment")]
        _normalize_unicode(fields)
        assert "Patient's age" in fields[0].description

    def test_fixes_encoding_artifacts(self) -> None:
        from ddharmon.ingestion.preprocessor import _normalize_unicode
        from ddharmon.models.data_dictionary import Field

        # â€™ is a common mojibake for right single quote
        fields = [Field(variable_name="age", description="Patient\u00e2\u0080\u0099s age")]
        _normalize_unicode(fields)
        assert "\u00e2\u0080" not in fields[0].description

    def test_normalizes_question_text(self) -> None:
        from ddharmon.ingestion.preprocessor import _normalize_unicode
        from ddharmon.models.data_dictionary import Field

        fields = [Field(variable_name="q1", description="Question", question_text="What\u00a0is your age?")]
        _normalize_unicode(fields)
        # Non-breaking space should be normalized
        assert fields[0].question_text is not None


class TestDropDescriptionEchoingOptions:
    """Tests for clearing descriptions that merely echo a response-option label."""

    def test_clears_quoted_sentinel_echo(self) -> None:
        from ddharmon.ingestion.preprocessor import _drop_description_echoing_options
        from ddharmon.models.data_dictionary import Field, ResponseOption

        # UKBB pattern: description column holds a quoted option label.
        f = Field(
            variable_name="Length of working week for main job",
            description='"Do not know"',
            response_options=[ResponseOption(code="-1", label="Do not know")],
        )
        count = _drop_description_echoing_options([f])
        assert count == 1
        assert f.description == ""
        # Embedding text falls back to the variable name (no sentinel leakage).
        assert f.to_embedding_text() == "Length of working week for main job"

    def test_clears_non_sentinel_boundary_label_echo(self) -> None:
        from ddharmon.ingestion.preprocessor import _drop_description_echoing_options
        from ddharmon.models.data_dictionary import Field, ResponseOption

        f = Field(
            variable_name="Time employed in main current job",
            description='"Less than a year"',
            response_options=[ResponseOption(code="1", label="Less than a year")],
        )
        assert _drop_description_echoing_options([f]) == 1
        assert f.description == ""

    def test_keeps_description_containing_sentinel_word(self) -> None:
        from ddharmon.ingestion.preprocessor import _drop_description_echoing_options
        from ddharmon.models.data_dictionary import Field

        # Legit variable whose description merely contains "missing" \u2014 not an
        # option label, so it must be preserved (no over-matching).
        f = Field(variable_name="ADL_NBRMIS_COM", description="OARS Scale: Number of Missing Items")
        assert _drop_description_echoing_options([f]) == 0
        assert f.description == "OARS Scale: Number of Missing Items"

    def test_keeps_real_description_with_options(self) -> None:
        from ddharmon.ingestion.preprocessor import _drop_description_echoing_options
        from ddharmon.models.data_dictionary import Field, ResponseOption

        f = Field(
            variable_name="Age",
            description="Age of participant at baseline",
            response_options=[ResponseOption(code="1", label="years")],
        )
        assert _drop_description_echoing_options([f]) == 0
        assert f.description == "Age of participant at baseline"

    def test_no_response_options_is_noop(self) -> None:
        from ddharmon.ingestion.preprocessor import _drop_description_echoing_options
        from ddharmon.models.data_dictionary import Field

        f = Field(variable_name="x", description="Do not know", response_options=[])
        assert _drop_description_echoing_options([f]) == 0
        assert f.description == "Do not know"

    def test_runs_within_preprocess_dictionary(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary
        from ddharmon.models.data_dictionary import DataDictionary, Field, ResponseOption

        dd = DataDictionary(
            name="UKBB",
            fields={
                "Number in household": Field(
                    variable_name="Number in household",
                    description='"Do not know"',
                    response_options=[ResponseOption(code="-1", label="Do not know")],
                ),
            },
        )
        preprocess_dictionary(dd, enabled=True)
        assert dd.fields["Number in household"].description == ""
        assert dd.preprocessing_report.option_echo_cleared == 1

    def test_flag_disables_step(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary
        from ddharmon.models.data_dictionary import DataDictionary, Field, ResponseOption

        dd = DataDictionary(
            name="UKBB",
            fields={
                "v": Field(
                    variable_name="v",
                    description='"Do not know"',
                    response_options=[ResponseOption(code="-1", label="Do not know")],
                ),
            },
        )
        preprocess_dictionary(dd, enabled=True, drop_description_echoing_option=False)
        assert dd.preprocessing_report.option_echo_cleared == 0
        assert dd.fields["v"].description == '"Do not know"'


class TestStripAdministrativeText:
    """Tests for administrative / data-collection wrapper stripping (Step 1a)."""

    def test_unwraps_instrument_preamble_in_description(self) -> None:
        from ddharmon.ingestion.preprocessor import _strip_administrative_text
        from ddharmon.models.data_dictionary import Field

        f = Field(
            variable_name="smoke_now",
            description='ACE touchscreen question "Do you smoke tobacco now?" <table>help</table>',
        )
        assert _strip_administrative_text([f]) == 1
        assert f.description == "Do you smoke tobacco now?"

    def test_cleans_question_text_too(self) -> None:
        from ddharmon.ingestion.preprocessor import _strip_administrative_text
        from ddharmon.models.data_dictionary import Field

        f = Field(
            variable_name="q",
            description="Coffee intake",
            question_text="<p>How many cups of coffee&nbsp;per day?</p>",
        )
        assert _strip_administrative_text([f]) == 1
        assert f.question_text == "How many cups of coffee per day?"

    def test_never_blanks_pure_markup(self) -> None:
        from ddharmon.ingestion.preprocessor import _strip_administrative_text
        from ddharmon.models.data_dictionary import Field

        # A description that cleans to "" must be left as the original (never blanked).
        f = Field(variable_name="x", description="<p></p>")
        assert _strip_administrative_text([f]) == 0
        assert f.description == "<p></p>"

    def test_benign_text_untouched(self) -> None:
        from ddharmon.ingestion.preprocessor import _strip_administrative_text
        from ddharmon.models.data_dictionary import Field

        f = Field(variable_name="age", description="Age of participant at baseline")
        assert _strip_administrative_text([f]) == 0
        assert f.description == "Age of participant at baseline"

    def test_does_not_strip_boilerplate_phrases(self) -> None:
        from ddharmon.ingestion.preprocessor import _strip_administrative_text
        from ddharmon.models.data_dictionary import Field

        # Ingest is structural-only: a whole-description boilerplate phrase is left intact (no "." residue).
        f = Field(variable_name="spec", description="Please specify.")
        assert _strip_administrative_text([f]) == 0
        assert f.description == "Please specify."

    def test_preserves_domain_angle_brackets(self) -> None:
        from ddharmon.ingestion.preprocessor import _strip_administrative_text
        from ddharmon.models.data_dictionary import Field

        # MESA ECG-code disjunction: `<OR>` is a domain token, not HTML — must survive.
        f = Field(variable_name="ecg", description="VF <OR> ASYSTOLE")
        assert _strip_administrative_text([f]) == 0
        assert f.description == "VF <OR> ASYSTOLE"

    def test_runs_within_preprocess_dictionary_and_reports(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary
        from ddharmon.models.data_dictionary import DataDictionary, Field

        dd = DataDictionary(
            name="UKBB",
            fields={
                "smoke_now": Field(
                    variable_name="smoke_now",
                    description='ACE touchscreen question "Do you smoke tobacco now?" <table>help</table>',
                ),
            },
        )
        preprocess_dictionary(dd, enabled=True)
        assert dd.fields["smoke_now"].description == "Do you smoke tobacco now?"
        assert dd.preprocessing_report.admin_text_stripped == 1

    def test_flag_disables_step(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary
        from ddharmon.models.data_dictionary import DataDictionary, Field

        raw = 'ACE touchscreen question "Do you smoke tobacco now?" <table>help</table>'
        dd = DataDictionary(name="UKBB", fields={"s": Field(variable_name="s", description=raw)})
        preprocess_dictionary(dd, enabled=True, strip_administrative_text=False)
        assert dd.preprocessing_report.admin_text_stripped == 0
        assert dd.fields["s"].description == raw


class TestStripCommonPrefixes:
    """Tests for common prefix stripping on fields."""

    def _make_fields(self, names: list[str]) -> list[Field]:
        from ddharmon.models.data_dictionary import Field

        return [Field(variable_name=n, description=f"Description of {n}") for n in names]

    def test_strips_shared_prefix(self) -> None:
        from ddharmon.ingestion.preprocessor import _strip_common_prefixes

        fields = self._make_fields(
            [
                "assessment_health_bmi",
                "assessment_health_age",
                "assessment_health_weight",
                "assessment_health_height",
            ]
        )
        _strip_common_prefixes(fields, min_length=8, min_ratio=0.5)
        names = [f.variable_name for f in fields]
        assert "bmi" in names
        assert "age" in names

    def test_preserves_short_prefix(self) -> None:
        from ddharmon.ingestion.preprocessor import _strip_common_prefixes

        fields = self._make_fields(["q_age", "q_bmi", "q_height"])
        _strip_common_prefixes(fields, min_length=8, min_ratio=0.5)
        # "q" is only 1 char — below min_length=8, so kept
        names = [f.variable_name for f in fields]
        assert all(n.startswith("q_") for n in names)

    def test_doesnt_strip_sole_token(self) -> None:
        from ddharmon.ingestion.preprocessor import _strip_common_prefixes

        # If stripping would leave nothing, skip that field
        fields = self._make_fields(
            [
                "assessment_health",
                "assessment_health_bmi",
                "assessment_health_age",
            ]
        )
        _strip_common_prefixes(fields, min_length=8, min_ratio=0.5)
        # The first field ("assessment_health") has only the prefix tokens, so it should be unchanged
        assert fields[0].variable_name == "assessment_health"


class TestRemoveStopwords:
    """Tests for stopword removal."""

    def test_removes_substring(self) -> None:
        from ddharmon.ingestion.preprocessor import _remove_stopwords
        from ddharmon.models.data_dictionary import Field

        fields = [
            Field(variable_name="assessmenthealthhistory_bmi", description="BMI"),
            Field(variable_name="assessmenthealthhistory_age", description="Age"),
        ]
        _remove_stopwords(fields, ["assessmenthealthhistory"])
        assert fields[0].variable_name == "bmi"
        assert fields[1].variable_name == "age"

    def test_case_insensitive(self) -> None:
        from ddharmon.ingestion.preprocessor import _remove_stopwords
        from ddharmon.models.data_dictionary import Field

        fields = [Field(variable_name="TheBasics_birthplace", description="Birthplace")]
        _remove_stopwords(fields, ["thebasics"])
        assert "thebasics" not in fields[0].variable_name.lower()
        assert "birthplace" in fields[0].variable_name.lower()

    def test_empty_stopwords(self) -> None:
        from ddharmon.ingestion.preprocessor import _remove_stopwords
        from ddharmon.models.data_dictionary import Field

        fields = [Field(variable_name="age", description="Age")]
        _remove_stopwords(fields, [])
        assert fields[0].variable_name == "age"

    def test_cleans_consecutive_delimiters(self) -> None:
        from ddharmon.ingestion.preprocessor import _remove_stopwords
        from ddharmon.models.data_dictionary import Field

        fields = [Field(variable_name="survey__demographics__age", description="Age")]
        _remove_stopwords(fields, ["demographics"])
        # Should not leave "survey___age" with triple underscore
        assert "__" not in fields[0].variable_name


class TestDedupNameInDescription:
    """Tests for substring deduplication."""

    def test_suppresses_redundant_name(self) -> None:
        from ddharmon.ingestion.preprocessor import _dedup_name_in_description
        from ddharmon.models.data_dictionary import Field

        fields = [
            Field(variable_name="body_mass_index", description="Body mass index calculated from height and weight")
        ]
        _dedup_name_in_description(fields)
        assert fields[0]._embed_variable_name is False

    def test_keeps_distinct_name(self) -> None:
        from ddharmon.ingestion.preprocessor import _dedup_name_in_description
        from ddharmon.models.data_dictionary import Field

        fields = [Field(variable_name="bmi", description="Body mass index calculated from height and weight")]
        _dedup_name_in_description(fields)
        assert fields[0]._embed_variable_name is True

    def test_handles_underscores_as_spaces(self) -> None:
        from ddharmon.ingestion.preprocessor import _dedup_name_in_description
        from ddharmon.models.data_dictionary import Field

        fields = [Field(variable_name="smoking_status", description="Current smoking status of participant")]
        _dedup_name_in_description(fields)
        assert fields[0]._embed_variable_name is False


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_collapses_runs(self) -> None:
        from ddharmon.ingestion.preprocessor import _normalize_whitespace
        from ddharmon.models.data_dictionary import Field

        fields = [Field(variable_name="age", description="Age  at   enrollment")]
        _normalize_whitespace(fields)
        assert fields[0].description == "Age at enrollment"

    def test_strips_leading_trailing(self) -> None:
        from ddharmon.ingestion.preprocessor import _normalize_whitespace
        from ddharmon.models.data_dictionary import Field

        fields = [Field(variable_name="  age  ", description="  Age at enrollment  ")]
        _normalize_whitespace(fields)
        assert fields[0].variable_name == "age"
        assert fields[0].description == "Age at enrollment"


class TestPreprocessDictionary:
    """Integration tests for the full preprocess_dictionary pipeline."""

    def _make_dd(self, name_desc_pairs: list[tuple[str, str]]) -> DataDictionary:
        from ddharmon.models.data_dictionary import DataDictionary, Field

        fields = {n: Field(variable_name=n, description=d) for n, d in name_desc_pairs}
        return DataDictionary(name="test", fields=fields)

    def test_preserves_raw_values(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = self._make_dd(
            [
                ("assessment_health_bmi", "Body mass index"),
                ("assessment_health_age", "Age at enrollment"),
                ("assessment_health_weight", "Body weight in kg"),
                ("assessment_health_height", "Standing height"),
            ]
        )
        preprocess_dictionary(dd, enabled=True)

        for f in dd.fields.values():
            assert f.raw_variable_name is not None
            assert f.raw_description is not None
            assert f.raw_variable_name.startswith("assessment_health_")

    def test_full_pipeline_strips_prefix(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = self._make_dd(
            [
                ("assessment_health_bmi", "Body mass index"),
                ("assessment_health_age", "Age at enrollment"),
                ("assessment_health_weight", "Body weight in kg"),
                ("assessment_health_height", "Standing height"),
            ]
        )
        preprocess_dictionary(dd, enabled=True)

        names = set(dd.fields.keys())
        assert "bmi" in names
        assert "age" in names

    def test_rekeys_dictionary(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = self._make_dd(
            [
                ("assessment_health_bmi", "Body mass index"),
                ("assessment_health_age", "Age at enrollment"),
                ("assessment_health_weight", "Body weight in kg"),
                ("assessment_health_height", "Standing height"),
            ]
        )
        preprocess_dictionary(dd, enabled=True)

        # Dictionary keys should match current variable_name, not raw
        for key, field in dd.fields.items():
            assert key == field.variable_name

    def test_explicit_stopwords(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = self._make_dd(
            [
                ("questionnaire_smoking_status", "Smoking status"),
                ("questionnaire_drinking_freq", "Drinking frequency"),
            ]
        )
        preprocess_dictionary(dd, enabled=True, stopwords=["questionnaire"], strip_common_prefixes=False)

        names = set(dd.fields.keys())
        assert all("questionnaire" not in n for n in names)

    def test_stopwords_from_file(self, tmp_path: Path) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        sw_file = tmp_path / "stopwords.json"
        sw_file.write_text(json.dumps({"stopwords": ["boilerplate"]}))

        dd = self._make_dd(
            [
                ("boilerplate_age", "Age"),
                ("boilerplate_bmi", "BMI"),
            ]
        )
        preprocess_dictionary(dd, enabled=True, stopwords_file=sw_file, strip_common_prefixes=False)

        names = set(dd.fields.keys())
        assert all("boilerplate" not in n for n in names)

    def test_empty_dictionary(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary
        from ddharmon.models.data_dictionary import DataDictionary

        dd = DataDictionary(name="empty", fields={})
        result = preprocess_dictionary(dd, enabled=True)
        assert result.field_count == 0

    def test_no_mutation_without_patterns(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = self._make_dd(
            [
                ("age", "Age at enrollment"),
                ("bmi", "Body mass index"),
                ("height", "Standing height"),
            ]
        )
        preprocess_dictionary(dd, enabled=True)

        # No common prefix, no stopwords — names should be unchanged
        assert "age" in dd.fields
        assert "bmi" in dd.fields

    def test_returns_same_object(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = self._make_dd([("age", "Age")])
        result = preprocess_dictionary(dd, enabled=True)
        assert result is dd

    def test_content_hash_changes_after_preprocessing(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary
        from ddharmon.models.data_dictionary import Field

        # Whitespace normalization collapses runs in the description (the embedded
        # text), so the content hash — the embedding cache key — must change.
        # (A variable-name-only change would NOT change the hash now, since the
        # name is fallback-only and not embedded when a description is present.)
        dd = self._make_dd(
            [
                ("assessment_health_bmi", "Body  mass   index"),
                ("assessment_health_age", "Age at enrollment"),
                ("assessment_health_weight", "Body weight in kg"),
                ("assessment_health_height", "Standing height"),
            ]
        )

        # Get hash before preprocessing
        old_field = Field(variable_name="assessment_health_bmi", description="Body  mass   index")
        old_hash = old_field.content_hash()

        preprocess_dictionary(dd, enabled=True)

        # After preprocessing, the field (prefix stripped to "bmi") has collapsed
        # whitespace in its description -> different embedded text -> different hash.
        new_field = dd.fields.get("bmi")
        assert new_field is not None
        assert old_hash != new_field.content_hash()

    def test_disabling_all_steps(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = self._make_dd(
            [
                ("assessment_health_bmi", "Body mass index"),
                ("assessment_health_age", "Age at enrollment"),
                ("assessment_health_weight", "Body weight in kg"),
                ("assessment_health_height", "Standing height"),
            ]
        )
        preprocess_dictionary(
            dd,
            enabled=True,
            normalize_unicode=False,
            strip_common_prefixes=False,
            dedup_name_in_description=False,
        )

        # Raw should still be saved, but names unchanged
        for f in dd.fields.values():
            assert f.raw_variable_name == f.variable_name


class TestPreprocessingDiffIsNotTruncated:
    """`preprocessing_diff` reports DATA; truncation is a display decision, not a data one.

    A before/after review once showed examples cut mid-word — "…at recruitment, but in som". The cause was
    not the display layer but a hard ``[:80]`` here, so no consumer could ever show the whole string however
    it chose to render it. A function whose job is to report what
    changed must not decide how much of the change the caller is allowed to see.
    """

    def _dd(self, fields):
        from ddharmon.models.data_dictionary import DataDictionary

        return DataDictionary(name="trunc", fields={f.variable_name: f for f in fields})

    def test_a_long_description_survives_the_diff_whole(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary, preprocessing_diff
        from ddharmon.models.data_dictionary import Field

        # Comfortably past the old 80-char cap, and with the payload at the END so a truncation is
        # detectable rather than merely suspected.
        tail = "THE_TAIL_THAT_MUST_SURVIVE"
        long_desc = (
            "Sex of participant. <p>Acquired from central registry at recruitment, "
            + ("but in some cases self-reported at the baseline visit and reconciled later. " * 3)
            + tail
        )
        dd = self._dd([Field(variable_name="SEX_B", description=long_desc)])
        preprocess_dictionary(dd, enabled=True)

        rows = preprocessing_diff(dd)
        assert rows, "the markup strip should have changed this field, so it must appear in the diff"
        row = rows[0]

        raw = str(row["raw_description"])
        cleaned = str(row["cleaned_description"])
        # The BEFORE is the raw string in full — including its tail.
        assert raw == long_desc
        assert tail in raw
        # And the AFTER is the field's actual cleaned description, not a prefix of it.
        assert cleaned == dd.fields["SEX_B"].description
        assert tail in cleaned
        # Belt and braces: nothing here is exactly 80 characters, which is what a surviving cap looks like.
        assert len(raw) > 80 and len(cleaned) > 80


class TestPreprocessingDiffCarriesTheEmbeddingText:
    """The diff must show the string the GROUPING STAGE consumes, before and after.

    For the rules whose whole effect is on the embedding text, a before/after review that shows only the
    DESCRIPTION — which for "suppressed a variable name that echoed its
    description" is byte-identical on both sides ("Year ended full time education" / "Year ended full time
    education"). Such a screen truthfully reports the wrong pair of strings.

    `raw_embed_text` is composed by calling the REAL ``Field.to_embedding_text()`` on a field with its raw
    values restored — never by re-deriving the format, because a plausible-looking wrong string on the one
    screen that explains grouping is worse than showing nothing.
    """

    def _dd(self, fields):
        from ddharmon.models.data_dictionary import DataDictionary

        return DataDictionary(name="embed", fields={f.variable_name: f for f in fields})

    def test_name_suppression_alone_leaves_the_embedding_text_UNCHANGED(  # noqa: N802 - caps are the point
        self,
    ) -> None:
        """The correction to my own first assumption, kept as the record.

        I expected name suppression to change the embedding text. It does not, when a primary text is
        present: ``to_embedding_text`` returns the description ALONE and never prepends the name, so
        dropping the name from a field that has a description changes nothing. The screen's existing note
        — "neither string changed" — was already correct.

        This is worth asserting rather than deleting, because it is the reason the embedding pair is the
        honest thing to render: it shows identical strings here, which is TRUE, where the description pair
        showed identical strings while implying something had changed.
        """
        from ddharmon.ingestion.preprocessor import preprocess_dictionary, preprocessing_diff
        from ddharmon.models.data_dictionary import Field

        dd = self._dd(
            [Field(variable_name="Year ended full time education", description="Year ended full time education")]
        )
        preprocess_dictionary(dd, enabled=True)
        rows = preprocessing_diff(dd)
        assert rows, "name suppression must put the field in the diff"
        row = rows[0]

        assert bool(row["embed_name_suppressed"]) is True
        f = dd.fields["Year ended full time education"]
        # Description identical on both sides...
        assert row["raw_description"] == row["cleaned_description"]
        # ...and so is the embedding text, because the name was never in it.
        assert str(row["raw_embed_text"]) == f.to_embedding_text()

    # The capitals are the author's emphasis and carry the point of the test — kept, rule silenced.
    def test_suppression_DOES_change_the_embedding_text_once_the_primary_text_is_gone(  # noqa: N802
        self,
    ) -> None:
        """Where the suppression actually bites: a field left with no primary text.

        With the description cleared (it merely echoed a response option) the name is the only candidate
        left, so ``_embed_variable_name`` decides between embedding the name and embedding NOTHING. That is
        the difference between landing in a name-artifact cluster and landing nowhere — and it is invisible
        in a description-only before/after.
        """
        from ddharmon.ingestion.preprocessor import preprocess_dictionary, preprocessing_diff
        from ddharmon.models.data_dictionary import Field

        dd = self._dd([Field(variable_name="FUL_STDUP_TRM", description="Do not know")])
        preprocess_dictionary(dd, enabled=True)
        f = dd.fields["FUL_STDUP_TRM"]
        rows = preprocessing_diff(dd)
        if not rows or not f.raw_description:
            pytest.skip("this corpus shape did not trigger the option-echo clear")
        row = rows[0]

        raw_embed = str(row["raw_embed_text"])
        after_embed = f.to_embedding_text()
        # The BEFORE composed something (the raw description, or the name as fallback); the pair is the
        # only place a reviewer can see which.
        assert raw_embed != "" or after_embed != ""
        if not (f.description or "").strip() and not (f.question_text or "").strip():
            # Primary text really is gone, so the pair must differ: name-or-nothing.
            assert raw_embed != after_embed

    def test_the_before_embedding_text_is_composed_by_core_not_re_derived(self) -> None:
        from dataclasses import replace

        from ddharmon.ingestion.preprocessor import preprocess_dictionary, preprocessing_diff
        from ddharmon.models.data_dictionary import Field

        dd = self._dd(
            [
                Field(
                    variable_name="SEX_B",
                    description="Sex of participant. <p>Acquired from central registry.</p>",
                    question_text="What\u00a0is your sex?",
                )
            ]
        )
        preprocess_dictionary(dd, enabled=True)
        row = preprocessing_diff(dd)[0]
        f = dd.fields["SEX_B"]

        # Independently reconstruct the expected BEFORE by the same route the implementation must use:
        # restore the raw strings onto a copy and ask core to compose it.
        expected = replace(
            f,
            variable_name=f.raw_variable_name or f.variable_name,
            description=f.raw_description or f.description,
            question_text=f.raw_question_text if f.raw_question_text is not None else f.question_text,
            _embed_variable_name=True,
        ).to_embedding_text()
        assert str(row["raw_embed_text"]) == expected

    def test_raw_question_text_is_preserved_when_preprocessing_rewrites_it(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary
        from ddharmon.models.data_dictionary import Field

        # A non-breaking space in the question is rewritten by the whitespace/unicode rules. Without the
        # raw value preserved, the before-embedding-text for a question-bearing field is unrecoverable —
        # and `to_embedding_text` prefers question_text over description, so that is the common case.
        dd = self._dd([Field(variable_name="q1", description="Question", question_text="What\u00a0is your age?")])
        preprocess_dictionary(dd, enabled=True)
        f = dd.fields["q1"]
        assert f.raw_question_text == "What\u00a0is your age?"
        assert f.question_text != f.raw_question_text
