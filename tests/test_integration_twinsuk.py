"""Integration tests for TwinsUK CSV end-to-end ingestion.

Tests load_dictionary() against the real TwinsUK_phenotypes.csv file,
verifying SNOMED extraction, hierarchy detection, and embedding text
generation all work together.

Skips automatically if TwinsUK CSV is not available (CI-friendly).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ddharmon.ingestion import load_dictionary
from ddharmon.models import DataDictionary
from ddharmon.models.enums import FieldRole

TWINSUK_PATH = Path(__file__).parent.parent / "data" / "examples" / "TwinsUK_phenotypes.csv"

TWINSUK_MAP = {
    "Historical_ID": FieldRole.VARIABLE_NAME,
    "Phenotype_Description": FieldRole.DESCRIPTION,
    "Data_Type": FieldRole.CATEGORY,
    "snomed_term_1": FieldRole.STANDARD_CODE,
    "snomed_term_2": FieldRole.STANDARD_CODE,
    "snomed_term_3": FieldRole.STANDARD_CODE,
    "snomed_term_4": FieldRole.STANDARD_CODE,
}


@pytest.fixture
def twinsuk_dict() -> DataDictionary:
    """Load TwinsUK data dictionary -- skips if file not available."""
    if not TWINSUK_PATH.exists():
        pytest.skip("TwinsUK CSV not available")
    return load_dictionary(
        TWINSUK_PATH,
        cohort_name="TwinsUK",
        variable_name="Historical_ID",
        description="Phenotype_Description",
        category="Data_Type",
        standard_code="snomed_term_1",
    )


def test_load_twinsuk_basic(twinsuk_dict: DataDictionary) -> None:
    """Verify basic DataDictionary properties after loading TwinsUK CSV."""
    assert twinsuk_dict.field_count >= 9000, f"Expected >= 9000 fields, got {twinsuk_dict.field_count}"
    assert twinsuk_dict.source_path is not None
    assert twinsuk_dict.source_path.name == "TwinsUK_phenotypes.csv"
    assert twinsuk_dict.cohort_name == "TwinsUK"


def test_twinsuk_schema_detection(twinsuk_dict: DataDictionary) -> None:
    """Verify category mapping from Data_Type column."""
    sections = {f.category for f in twinsuk_dict.fields.values() if f.category is not None}

    expected_sections = {"Measurements", "Self-reported", "Other dataset"}
    assert expected_sections.issubset(sections), f"Missing sections: {expected_sections - sections}"

    category_counts: dict[str, int] = {}
    for f in twinsuk_dict.fields.values():
        if f.category is not None and not f._synthetic:
            category_counts[f.category] = category_counts.get(f.category, 0) + 1

    assert (
        category_counts.get("Measurements", 0) > 1000
    ), f"Expected > 1000 Measurements fields, got {category_counts.get('Measurements', 0)}"
    assert (
        category_counts.get("Self-reported", 0) > 7000
    ), f"Expected > 7000 Self-reported fields, got {category_counts.get('Self-reported', 0)}"


def test_twinsuk_snomed_extraction(twinsuk_dict: DataDictionary) -> None:
    """Verify SNOMED codes are extracted from snomed_term columns."""
    fields_with_snomed = [
        f for f in twinsuk_dict.fields.values() if not f._synthetic and f.standard_codes.get("SNOMED")
    ]

    assert len(fields_with_snomed) > 5000, f"Expected > 5000 fields with SNOMED codes, got {len(fields_with_snomed)}"

    cystatin_c_found = False
    for f in fields_with_snomed:
        if "1002561000000109" in f.standard_codes.get("SNOMED", []):
            cystatin_c_found = True
            break
    assert cystatin_c_found, "Expected SNOMED code 1002561000000109 (Cystatin C) not found"


def test_twinsuk_hierarchy(twinsuk_dict: DataDictionary) -> None:
    """Verify hierarchy detection creates synthetic parents from backslash prefixes."""
    assert (
        twinsuk_dict.synthetic_field_count > 400
    ), f"Expected > 400 synthetic parents, got {twinsuk_dict.synthetic_field_count}"

    children_count = sum(1 for f in twinsuk_dict.fields.values() if f.parent_field_id is not None and not f._synthetic)
    assert children_count > 1000, f"Expected > 1000 child fields, got {children_count}"

    synthetic_parents = [f for f in twinsuk_dict.fields.values() if f._synthetic and f.children]
    assert len(synthetic_parents) > 0, "No synthetic parents with children found"

    parent = synthetic_parents[0]
    for child_name in parent.children:
        child = twinsuk_dict.fields.get(child_name)
        assert child is not None, f"Child {child_name} not found in fields"
        assert (
            child.parent_field_id == parent.variable_name
        ), f"Child {child_name} parent_field_id={child.parent_field_id}, expected {parent.variable_name}"


def test_twinsuk_embedding_text(twinsuk_dict: DataDictionary) -> None:
    """Verify embedding text composition and content hashing work correctly."""
    cystatin_fields = [f for f in twinsuk_dict.fields.values() if "cystatin" in f.description.lower()]
    assert cystatin_fields, "No field with 'cystatin' in description found"

    field = cystatin_fields[0]
    embedding_text = field.to_embedding_text()

    assert field.variable_name in embedding_text
    assert field.description in embedding_text

    if field.category:
        assert field.category in embedding_text

    content_hash = field.content_hash()
    assert len(content_hash) == 16, f"Expected 16-char hash, got {len(content_hash)} chars"
    assert all(c in "0123456789abcdef" for c in content_hash), f"Hash contains non-hex chars: {content_hash}"


def test_twinsuk_na_handling(twinsuk_dict: DataDictionary) -> None:
    """Verify NA values are handled correctly in standard codes and variable names."""
    na_code_fields = []
    for f in twinsuk_dict.fields.values():
        snomed_codes = f.standard_codes.get("SNOMED", [])
        if "NA" in snomed_codes or "na" in snomed_codes or "N/A" in snomed_codes:
            na_code_fields.append(f.variable_name)

    assert not na_code_fields, f"Fields with NA as SNOMED code: {na_code_fields[:5]}"

    empty_names = [
        f.variable_name for f in twinsuk_dict.fields.values() if not f.variable_name or not f.variable_name.strip()
    ]
    assert not empty_names, f"Fields with empty variable names found: {len(empty_names)}"
