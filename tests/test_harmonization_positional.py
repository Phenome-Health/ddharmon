"""Tests for the positional-enumeration (repeating-measure) detector."""

from __future__ import annotations

from ddharmon.harmonization.positional import (
    EnumeratedFamily,
    PositionalEnumeration,
    detect_enumerated_family,
    detect_positional_enumeration,
    signature,
)


class TestSignature:
    def test_strips_digit_runs(self):
        assert signature("Prescribed - Medication 12") == "prescribed medication #"
        assert signature("Prescribed - Medication 3") == "prescribed medication #"

    def test_collapses_punctuation_and_case(self):
        assert signature("FI13: duration viewed") == "fi# duration viewed"
        assert signature("Visit_2_date") == "visit # date"


class TestDetect:
    def test_detects_numbered_repeating_measure(self):
        labels = [f"Prescribed - Medication {i}" for i in range(1, 13)]  # 1..12
        pe = detect_positional_enumeration(labels)
        assert isinstance(pe, PositionalEnumeration)
        assert pe.signature == "prescribed medication #"
        assert pe.n_occurrences == 12 and pe.int_range == (1, 12)
        assert pe.dominant_share == 1.0 and pe.density == 1.0

    def test_rejects_qualifier_matrix(self):
        # distinct concepts (distinct signatures after digit-stripping) — NOT a repeating measure
        labels = [
            "Systolic blood pressure",
            "Diastolic blood pressure",
            "Heart rate",
            "Body temperature",
            "Respiratory rate",
        ]
        assert detect_positional_enumeration(labels) is None

    def test_rejects_too_few_occurrences(self):
        assert detect_positional_enumeration(["Medication 1", "Medication 2", "Medication 3"]) is None

    def test_rejects_sparse_integer_range(self):
        # only 4 distinct ints but scattered over a huge range -> low density -> not a contiguous enumeration
        labels = [f"Medication {i}" for i in (1, 2, 3, 400)]
        assert detect_positional_enumeration(labels) is None

    def test_rejects_labels_without_digits(self):
        labels = ["income", "income level", "income bracket", "income category", "income group"]
        assert detect_positional_enumeration(labels) is None

    def test_tolerates_minority_off_signature_members(self):
        # 10 numbered + 2 stray: dominant share 10/12 = 0.83 >= 0.70 -> still detected
        labels = [f"Contact {i} phone" for i in range(1, 11)] + ["Primary email", "Notes"]
        pe = detect_positional_enumeration(labels)
        assert pe is not None and pe.n_occurrences == 10

    def test_dominant_share_threshold_is_respected(self):
        # 5 numbered + 5 distinct -> dominant share 0.5 < 0.70 -> rejected
        labels = [f"Med {i}" for i in range(1, 6)] + ["Age", "Sex", "Height", "Weight", "Region"]
        assert detect_positional_enumeration(labels) is None
        # loosening the threshold flips it (thresholds are tunable, not hard gates)
        assert detect_positional_enumeration(labels, dominant_share=0.4) is not None

    def test_ignores_blank_labels(self):
        labels = [f"Visit {i} date" for i in range(1, 6)] + ["", "   "]
        pe = detect_positional_enumeration(labels)
        assert pe is not None and pe.n_occurrences == 5


class TestEnumeratedFamily:
    _FOODS = [
        "apples",
        "bananas",
        "oranges",
        "grapes",
        "carrots",
        "broccoli",
        "spinach",
        "potatoes",
        "chicken",
        "beef",
    ]

    def test_detects_food_frequency_battery(self):
        labels = [f"How often do you eat {food}" for food in self._FOODS]  # 10 same-template members
        fam = detect_enumerated_family(labels)
        assert isinstance(fam, EnumeratedFamily)
        assert fam.n_entities == 10 and fam.template_tokens >= 4
        assert {"how", "often", "do", "you", "eat"} <= set(fam.template.split())

    def test_rejects_heterogeneous_pool(self):
        # distinct questions sharing only a couple of filler words -> long distinct remainders -> not a family
        labels = [
            "What is your age in years",
            "How many children do you have",
            "Do you currently smoke cigarettes",
            "Rate your overall health today",
            "What is your annual household income",
            "How tall are you without shoes",
            "When were you last hospitalized",
            "Which region of the country do you live in",
        ]
        assert detect_enumerated_family(labels) is None

    def test_rejects_too_few_members(self):
        labels = [f"How often do you eat {food}" for food in self._FOODS[:5]]  # 5 < min_members
        assert detect_enumerated_family(labels) is None

    def test_does_not_fire_on_numbered_family(self):
        # a positional/numbered family has a 1-token template (its varying slot is a digit) -> not an
        # entity family (that's detect_positional_enumeration's job)
        labels = [f"Medication {i}" for i in range(1, 21)]
        assert detect_enumerated_family(labels) is None

    def test_rejects_identical_labels(self):
        # no DISTINCT entities (all the same) -> not an enumeration
        labels = ["How often do you eat fruit"] * 10
        assert detect_enumerated_family(labels) is None

    def test_long_slot_is_not_a_family(self):
        # members share a stem but each has a LONG distinct remainder -> a heterogeneous battery, not entities
        labels = [
            f"In the past week how often did you {act}"
            for act in [
                "feel nervous or on edge for no reason",
                "have trouble falling or staying asleep at night",
                "experience a loss of interest in daily activities",
                "notice your heart racing during rest",
                "avoid social situations because of worry",
                "feel unable to control your worrying thoughts",
                "have difficulty concentrating on tasks",
                "feel restless or unable to sit still",
            ]
        ]
        assert detect_enumerated_family(labels) is None
