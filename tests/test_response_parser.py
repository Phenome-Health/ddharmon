"""Tests for value_encoding_raw -> ResponseOption parsing."""

from ddharmon.values.response_parser import parse_value_encoding


class TestParenthesizedFormat:
    """Arivale format: (code) label|(code) label"""

    def test_ordinal_frequency(self):
        raw = "(1) Less than once per month|(2) 1-3 times per month|(3) Once per week|(4) 2-4 times per week"
        opts = parse_value_encoding(raw)
        assert len(opts) == 4
        assert opts[0].code == "1"
        assert opts[0].label == "Less than once per month"
        assert opts[0].order == 0
        assert opts[3].code == "4"
        assert opts[3].label == "2-4 times per week"

    def test_binary_yes_no(self):
        raw = "(0) No|(1) Yes"
        opts = parse_value_encoding(raw)
        assert len(opts) == 2
        assert opts[0].code == "0"
        assert opts[0].label == "No"
        assert opts[1].code == "1"
        assert opts[1].label == "Yes"


class TestCodeEqualsLabelFormat:
    """Simple format: 1=Yes|2=No"""

    def test_binary(self):
        raw = "1=Yes|2=No"
        opts = parse_value_encoding(raw)
        assert len(opts) == 2
        assert opts[0].code == "1"
        assert opts[0].label == "Yes"

    def test_three_options(self):
        raw = "1=Male|2=Female|3=Other"
        opts = parse_value_encoding(raw)
        assert len(opts) == 3
        assert opts[2].label == "Other"


class TestCodeCommaLabelFormat:
    """REDCap/All of Us format: Code, Label | Code, Label"""

    def test_simple(self):
        raw = "Birthplace_USA, USA | PMI_Other, Other"
        opts = parse_value_encoding(raw)
        assert len(opts) == 2
        assert opts[0].code == "Birthplace_USA"
        assert opts[0].label == "USA"
        assert opts[1].code == "PMI_Other"
        assert opts[1].label == "Other"

    def test_with_parenthetical_in_label(self):
        raw = (
            "WhatRaceEthnicity_AIAN, American Indian or Alaska Native "
            "(For example: Aztec, Navajo) | "
            "WhatRaceEthnicity_Asian, Asian (For example: Chinese, Japanese)"
        )
        opts = parse_value_encoding(raw)
        assert len(opts) == 2
        assert opts[0].code == "WhatRaceEthnicity_AIAN"
        assert "American Indian" in opts[0].label
        assert "Aztec" in opts[0].label  # parenthetical preserved

    def test_many_options(self):
        raw = "A, Option A | B, Option B | C, Option C | D, Option D"
        opts = parse_value_encoding(raw)
        assert len(opts) == 4


class TestSlashDelimited:
    """Simple slash format: Yes/No"""

    def test_yes_no(self):
        opts = parse_value_encoding("Yes/No")
        assert len(opts) == 2
        assert opts[0].label == "Yes"
        assert opts[1].label == "No"

    def test_three_options(self):
        opts = parse_value_encoding("Male/Female/Other")
        assert len(opts) == 3

    def test_too_many_rejected(self):
        opts = parse_value_encoding("A/B/C/D")
        assert len(opts) == 0  # >3 options rejected for slash format


class TestEdgeCases:
    def test_empty_string(self):
        assert parse_value_encoding("") == []

    def test_whitespace_only(self):
        assert parse_value_encoding("   ") == []

    def test_single_value_no_parse(self):
        assert parse_value_encoding("Continuous") == []

    def test_coding_reference_no_parse(self):
        """UKBB 'Coding 6332' should not parse into options."""
        assert parse_value_encoding("Coding 6332") == []


# ── a comma splits a code from its label only at parenthesis depth 0, after a code-like token ──

#: The NIH-endorsed "Health Conditions - Disease Disorders" CDE (tinyId hbgYw7HPpXW), verbatim from
#: ``nih_endorsed_flat.tsv``. The catalog flattener writes ``value=meaning`` only where the two differ and the
#: bare value otherwise, so this list is MIXED: 34 bare labels, several carrying "(e.g., a, b, c)", plus one
#: ``Thrombotic disorders=Thrombotic Disorders``. Formerly the first comma inside "(e.g., …" was read as a
#: code/label separator, so Gates 2 and 3 showed "rheumatoid arthritis, systemic lupus erythematosus,
#: vasculitis)" and the ``=`` item arrived raw.
HEALTH_CONDITIONS_PV = (
    "Alzheimer’s Disease | Asthma | Autoimmune condition (e.g., rheumatoid arthritis, systemic lupus "
    "erythematosus, vasculitis) | Cancer | Chronic fatigue | Chronic kidney disease | Chronic musculoskeletal "
    "condition (e.g., back pain, osteoarthritis, osteoporosis) | Coronary heart disease | Dementia | Dental "
    "diseases and conditions (e.g., caries, periodontal disease, oral and pharyngeal cancer) | Diabetes (Type I) "
    "| Diabetes (Type II) | Epilepsy | Eye disorders and/or diabetic eye diseases (e.g., cataract, glaucoma, "
    "amblyopia, myopia and other refractive errors, age-related macular degeneration, diabetic retinopathy, "
    "ocular trauma, uveitis, keratoconus) | Heart failure | Hepatitis | High blood pressure/hypertension | High "
    "cholesterol | HIV/AIDS | Immunodeficiency | Multiple Sclerosis | Obesity | Other chronic liver disease | "
    "Other chronic neurological condition (e.g., Parkinson’s disease, migraine) | Other chronic respiratory "
    "disease (e.g., COPD, emphysema) | Other substance use disorder (e.g., drugs and/or alcohol dependence) | "
    "Psychological and/or psychiatric disease or disorder (e.g., anxiety, depression, bipolar disorder) | Sickle "
    "Cell Disease | Sleep disorder (e.g., insomnia, sleep apnea, narcolepsy) | Smoking | Solid Organ Transplant | "
    "Stroke | Thrombotic disorders=Thrombotic Disorders | Other chronic diseases/specify | None of the above"
)


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


class TestCatalogBareAndMixedLists:
    """NIH CDE permissible values: bare labels, ``value=meaning`` where they differ, commas inside labels."""

    def test_health_conditions_every_label_whole(self):
        opts = parse_value_encoding(HEALTH_CONDITIONS_PV)
        assert len(opts) == 35  # one option per "|" item — nothing split, nothing merged
        assert all(_balanced(o.label) and _balanced(o.code) for o in opts)
        auto = opts[2]
        assert (
            auto.code
            == auto.label
            == ("Autoimmune condition (e.g., rheumatoid arthritis, systemic lupus erythematosus, vasculitis)")
        )
        assert opts[23].label == "Other chronic neurological condition (e.g., Parkinson’s disease, migraine)"
        assert opts[24].label == "Other chronic respiratory disease (e.g., COPD, emphysema)"
        assert [o.order for o in opts] == list(range(35))

    def test_mixed_value_equals_meaning_item_is_split(self):
        opts = parse_value_encoding(HEALTH_CONDITIONS_PV)
        thrombotic = next(o for o in opts if o.code.startswith("Thrombotic"))
        assert (thrombotic.code, thrombotic.label) == ("Thrombotic disorders", "Thrombotic Disorders")
        assert not any("=" in o.label for o in opts)

    def test_depth_zero_comma_after_a_word_is_not_a_code(self):
        # "Other, Specify" is a catalog LABEL; the other items carry no comma at all.
        opts = parse_value_encoding("Yes | No | Other, Specify")
        assert [(o.code, o.label) for o in opts] == [("Yes", "Yes"), ("No", "No"), ("Other, Specify", "Other, Specify")]

    def test_thousands_separators_stay_in_the_label(self):
        opts = parse_value_encoding("Less than $10,000 | $10,000-$24,999 | $200,000 or more")
        assert [o.label for o in opts] == ["Less than $10,000", "$10,000-$24,999", "$200,000 or more"]

    def test_every_item_comma_bearing_but_no_code_token(self):
        # all items have a depth-0 comma, but "No"/"Yes" are words, not codes (and would collide as codes).
        opts = parse_value_encoding("No, neuronal loss | Yes, neuronal loss")
        assert [o.code for o in opts] == ["No, neuronal loss", "Yes, neuronal loss"]

    def test_comparison_operator_is_not_a_code_separator(self):
        opts = parse_value_encoding("BMI >= 30 | BMI < 30 | Unknown")
        assert [o.label for o in opts] == ["BMI >= 30", "BMI < 30", "Unknown"]

    def test_mixed_equals_list_keeps_bare_items(self):
        opts = parse_value_encoding("Days | Don't Know | wk=Weeks")
        assert [(o.code, o.label) for o in opts] == [("Days", "Days"), ("Don't Know", "Don't Know"), ("wk", "Weeks")]


class TestCodeCommaLabelUnchanged:
    """The REDCap / All of Us shape still splits at the code — including labels with depth-0 commas."""

    def test_label_with_depth_zero_commas_after_the_code(self):
        raw = (
            "WhatRaceEthnicity_Hispanic, Hispanic, Latino, or Spanish (For example: Columbian, Cuban) | "
            "PMI_PreferNotToAnswer, I prefer not to answer"
        )
        opts = parse_value_encoding(raw)
        assert [(o.code, o.label) for o in opts] == [
            ("WhatRaceEthnicity_Hispanic", "Hispanic, Latino, or Spanish (For example: Columbian, Cuban)"),
            ("PMI_PreferNotToAnswer", "I prefer not to answer"),
        ]

    def test_numeric_codes(self):
        opts = parse_value_encoding("1, Yes | 2, No | 99, Don't know, refused")
        assert [(o.code, o.label) for o in opts] == [("1", "Yes"), ("2", "No"), ("99", "Don't know, refused")]
