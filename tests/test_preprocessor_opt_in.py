"""Dictionary preparation has a master switch: ``preprocess_dictionary(dd, enabled=...)``.

A caller that must not rewrite an uploaded dictionary's text unasked passes ``enabled=False``. The default,
``enabled=True``, runs every rule exactly as before the switch existed.

This module carries the tests that make the switch safe:

1. :class:`TestEnabledTrueIsByteIdenticalToTheOldDefault` — the frozen golden snapshot. It was captured
   from the code as it stood immediately BEFORE the switch was added, and both ``enabled=True`` and the bare
   default call must reproduce it byte for byte.
2. :class:`TestSkippedPathLeavesNoNone` — the trap. ``raw_variable_name`` / ``raw_description`` were
   populated as a SIDE EFFECT of mutation; when nothing mutates they must still hold strings, or a caller
   that builds a lookup key out of ``raw_variable_name`` silently keys on ``"cohort:None"``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ddharmon.models.data_dictionary import DataDictionary

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "preprocessing_golden_default.json"


def _golden_dictionary() -> DataDictionary:
    """Rebuild the exact input the golden snapshot was captured from.

    Exercises all six rules at once: unicode repair, administrative-text stripping, option-echo
    clearing, placeholder replacement, common-prefix stripping and name-in-description dedup.
    """
    from ddharmon.models.data_dictionary import DataDictionary, Field, ResponseOption

    rows: list[tuple[str, str, str]] = [
        (
            "study_alpha_smoking_status",
            'ACE touchscreen question "Do you smoke?"<p>The following checks were performed:'
            "<ul><li>If answer < 1 then rejected</li></ul>",
            'ACE touchscreen question "Do you smoke?"',
        ),
        ("study_alpha_bmi_calculated", "Body mass  index   calculated — from height and weight", ""),
        ("study_alpha_curly_quote", "The participant’s “age” at visit", ""),
        ("study_alpha_echo_option", "Do not know", ""),
        ("study_alpha_dedup_name", "study_alpha_dedup_name is the field measuring grip strength", ""),
    ]
    # >= placeholder_min_count (10) identical descriptions, so the placeholder rule fires.
    rows += [
        (f"study_alpha_placeholder_{i:02d}", "Field description available on the study website", "") for i in range(12)
    ]

    fields = {}
    for var, desc, question in rows:
        f = Field(variable_name=var, description=desc, question_text=question or None)
        if "echo_option" in var:
            f.response_options = [ResponseOption(code="1", label="Do not know"), ResponseOption(code="2", label="Yes")]
        fields[var] = f
    return DataDictionary(name="GOLDEN", fields=fields)


def _snapshot(dd: DataDictionary) -> dict:
    """The same shape the golden fixture was captured in."""
    snap: dict = {}
    for name, f in sorted(dd.fields.items()):
        snap[name] = {
            "variable_name": f.variable_name,
            "description": f.description,
            "question_text": f.question_text,
            "embed_variable_name": f._embed_variable_name,
            "embedding_text": f.to_embedding_text(),
        }
    rep = dd.preprocessing_report  # type: ignore[attr-defined]
    snap["__report__"] = {
        "total_fields": rep.total_fields,
        "unicode_fixed": rep.unicode_fixed,
        "admin_text_stripped": rep.admin_text_stripped,
        "placeholders_replaced": rep.placeholders_replaced,
        "prefix_stripped": rep.prefix_stripped,
        "prefix_value": rep.prefix_value,
        "name_deduped": rep.name_deduped,
        "option_echo_cleared": rep.option_echo_cleared,
        "whitespace_fixed": rep.whitespace_fixed,
        "names_changed": rep.names_changed,
        "descriptions_changed": rep.descriptions_changed,
    }
    return snap


class TestEnabledTrueIsByteIdenticalToTheOldDefault:
    """``enabled=True`` — and the bare default call — reproduce the rules' output from before the switch.

    If this fails, the rules have drifted: adding a master switch must not have changed what they do.
    """

    def test_matches_the_frozen_golden_snapshot(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = preprocess_dictionary(_golden_dictionary(), enabled=True)
        assert _snapshot(dd) == json.loads(GOLDEN_PATH.read_text())

    def test_the_default_call_still_runs_the_rules(self) -> None:
        """Existing callers pass no ``enabled``: their output must not change."""
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = preprocess_dictionary(_golden_dictionary())
        assert dd.preprocessing_report.enabled is True  # type: ignore[attr-defined]
        assert _snapshot(dd) == json.loads(GOLDEN_PATH.read_text())

    def test_the_golden_input_really_does_exercise_every_rule(self) -> None:
        """Guards the guard: a golden that changed nothing would pass vacuously forever."""
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        rep = preprocess_dictionary(_golden_dictionary(), enabled=True).preprocessing_report  # type: ignore[attr-defined]
        assert rep.unicode_fixed > 0
        assert rep.admin_text_stripped > 0
        assert rep.option_echo_cleared > 0
        assert rep.placeholders_replaced > 0
        assert rep.prefix_stripped > 0
        assert rep.name_deduped > 0


class TestSwitchedOff:
    """``enabled=False`` → no text is rewritten."""

    def test_switching_off_changes_no_text(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        before = {
            n: (f.variable_name, f.description, f.question_text, f._embed_variable_name)
            for n, f in _golden_dictionary().fields.items()
        }
        dd = preprocess_dictionary(_golden_dictionary(), enabled=False)
        after = {
            n: (f.variable_name, f.description, f.question_text, f._embed_variable_name) for n, f in dd.fields.items()
        }
        assert after == before

    def test_switching_off_is_not_merely_a_quiet_on(self) -> None:
        """The golden input DOES change under the rules — so an unchanged result proves the skip."""
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        skipped = preprocess_dictionary(_golden_dictionary(), enabled=False)
        prepared = preprocess_dictionary(_golden_dictionary(), enabled=True)
        assert _snapshot(skipped) != _snapshot(prepared)

    def test_the_keys_are_not_rekeyed(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = preprocess_dictionary(_golden_dictionary(), enabled=False)
        assert "study_alpha_placeholder_00" in dd.fields
        assert "00" not in dd.fields

    def test_the_report_is_attached_and_says_it_was_skipped(self) -> None:
        """A consumer reads ``preprocessing_report.total_fields``; it must not vanish."""
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        rep = preprocess_dictionary(_golden_dictionary(), enabled=False).preprocessing_report  # type: ignore[attr-defined]
        assert rep.enabled is False
        assert rep.total_fields == 17
        assert rep.names_changed == 0
        assert rep.descriptions_changed == 0
        assert "not run" in str(rep).lower() or "skipped" in str(rep).lower()

    def test_it_logs_that_preparation_was_skipped(self, caplog) -> None:
        """A silent no-op in a stage that used to rewrite 46% of a dictionary is an hour of debugging."""
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        with caplog.at_level(logging.INFO, logger="ddharmon.ingestion.preprocessor"):
            preprocess_dictionary(_golden_dictionary(), enabled=False)
        assert any("skipped" in r.getMessage().lower() for r in caplog.records)

    def test_an_empty_dictionary_is_returned_unchanged(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary
        from ddharmon.models.data_dictionary import DataDictionary

        dd = DataDictionary(name="empty", fields={})
        assert preprocess_dictionary(dd, enabled=False) is dd


class TestSkippedPathLeavesNoNone:
    """The trap: ``raw_*`` used to be populated as a side effect of mutation."""

    def test_raw_fields_hold_the_unchanged_strings(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        for f in preprocess_dictionary(_golden_dictionary(), enabled=False).fields.values():
            assert isinstance(f.raw_variable_name, str)
            assert isinstance(f.raw_description, str)
            assert f.raw_variable_name == f.variable_name
            assert f.raw_description == f.description
            # question_text is legitimately absent on some fields; raw mirrors it either way.
            assert f.raw_question_text == f.question_text

    def test_a_raw_name_lookup_key_is_never_cohort_none(self) -> None:
        """A caller doing ``lookup[f"{cohort}:{f.raw_variable_name}"]`` must never key on ``"cohort:None"``."""
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = preprocess_dictionary(_golden_dictionary(), enabled=False)
        keys = {f"COHORT:{f.raw_variable_name}" for f in dd.fields.values()}
        assert "COHORT:None" not in keys

    def test_the_raw_embedding_text_still_composes(self) -> None:
        from ddharmon.ingestion.preprocessor import _raw_embed_text, preprocess_dictionary

        for f in preprocess_dictionary(_golden_dictionary(), enabled=False).fields.values():
            assert _raw_embed_text(f) == f.to_embedding_text()


class TestSkippedDiffIsEmpty:
    """``preprocessing_diff`` reports that nothing changed — correct, not broken."""

    def test_the_diff_is_empty_when_preparation_did_not_run(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary, preprocessing_diff

        assert preprocessing_diff(preprocess_dictionary(_golden_dictionary(), enabled=False)) == []

    def test_the_loaders_embed_variable_name_false_is_not_reported_as_a_change(self) -> None:
        """The regression this guards against.

        ``embed_variable_name=False`` is a LOADER choice (opaque-code dictionaries — the CDEMapper and
        AI-READI benchmark gold both use it). ``preprocessing_diff`` infers name-suppression from that
        same flag, so with preparation skipped every one of those fields would otherwise be reported as
        "preparation suppressed the name" when preparation never ran at all.
        """
        from ddharmon.ingestion.preprocessor import preprocess_dictionary, preprocessing_diff

        dd = _golden_dictionary()
        for f in dd.fields.values():
            f._embed_variable_name = False

        assert preprocessing_diff(preprocess_dictionary(dd, enabled=False)) == []

    def test_the_diff_still_reports_changes_when_preparation_did_run(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary, preprocessing_diff

        assert preprocessing_diff(preprocess_dictionary(_golden_dictionary(), enabled=True))


class TestPerRuleFlagsSurvive:
    """The per-rule ablation flags must still work alongside the master switch."""

    def test_each_rule_is_independently_switchable_when_enabled(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        rep = preprocess_dictionary(
            _golden_dictionary(), enabled=True, strip_administrative_text=False
        ).preprocessing_report  # type: ignore[attr-defined]
        assert rep.admin_text_stripped == 0
        assert rep.placeholders_replaced > 0  # the other rules still ran

    def test_the_six_per_rule_defaults_are_still_true(self) -> None:
        """The switch is ONE master flag; it must not have flipped the per-rule defaults underneath."""
        import inspect

        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        params = inspect.signature(preprocess_dictionary).parameters
        for rule in (
            "normalize_unicode",
            "strip_administrative_text",
            "drop_description_echoing_option",
            "strip_common_prefixes",
            "dedup_name_in_description",
            "replace_placeholder_descriptions",
        ):
            assert params[rule].default is True, f"{rule} default must stay True"
        assert params["enabled"].default is True

    def test_a_per_rule_flag_cannot_switch_preparation_back_on(self) -> None:
        """The master switch short-circuits BEFORE any rule — it is not one vote among seven."""
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        dd = preprocess_dictionary(
            _golden_dictionary(), enabled=False, normalize_unicode=True, strip_common_prefixes=True
        )
        assert dd.fields["study_alpha_curly_quote"].description == "The participant’s “age” at visit"


class TestToEmbeddingTextIsIndifferent:
    """Nothing downstream needs to know whether preparation ran."""

    def test_it_composes_from_whatever_text_is_present(self) -> None:
        from ddharmon.ingestion.preprocessor import preprocess_dictionary

        skipped = preprocess_dictionary(_golden_dictionary(), enabled=False)
        prepared = preprocess_dictionary(_golden_dictionary(), enabled=True)
        for dd in (skipped, prepared):
            for f in dd.fields.values():
                assert isinstance(f.to_embedding_text(), str)
                assert isinstance(f.content_hash(), str)
