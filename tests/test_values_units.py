"""Tests for the C2 N1 curated unit canonicalizer + linear conversion."""

from __future__ import annotations

import math

import pytest

from ddharmon.values import (
    UnitCanonicalizer,
    canonical_unit,
    convert_units,
    is_identity_conversion,
)

APPROX = pytest.approx


class TestCanonical:
    def test_recognises_spellings(self):
        assert canonical_unit("kg") == ("mass", "kg")
        assert canonical_unit("Kilograms") == ("mass", "kg")
        assert canonical_unit("lbs") == ("mass", "lb")
        assert canonical_unit("cm") == ("length", "cm")
        assert canonical_unit("inches") == ("length", "in")
        assert canonical_unit("°C") == ("temperature", "degC")
        assert canonical_unit("Fahrenheit") == ("temperature", "degF")
        assert canonical_unit("years") == ("time", "year")
        assert canonical_unit("months") == ("time", "month")
        assert canonical_unit("%") == ("proportion", "percent")

    def test_recognises_new_physical_spellings(self):
        # temperature spellings observed in the NIH CDE uom column
        assert canonical_unit("Cel") == ("temperature", "degC")
        assert canonical_unit("degree Celsius") == ("temperature", "degC")
        assert canonical_unit("degrees celsius") == ("temperature", "degC")
        # micrometre / micron
        assert canonical_unit("micrometer") == ("length", "um")
        assert canonical_unit("micron") == ("length", "um")
        # angle family (goniometry / range-of-motion)
        assert canonical_unit("degrees of arc") == ("angle", "degree")
        assert canonical_unit("Degrees") == ("angle", "degree")
        assert canonical_unit("radians") == ("angle", "radian")
        assert canonical_unit("arcmin") == ("angle", "arcminute")
        # cmH2O + bracketed mmHg spelling
        assert canonical_unit("cmH2O") == ("pressure", "cmH2O")
        assert canonical_unit("centimeter of water") == ("pressure", "cmH2O")
        assert canonical_unit("mm[Hg]") == ("pressure", "mmHg")

    def test_degree_symbol_disambiguation(self):
        # a bare degree symbol is an angle; "°C" must stay temperature
        assert canonical_unit("°") == ("angle", "degree")
        assert canonical_unit("°C") == ("temperature", "degC")

    def test_unrecognised_returns_none(self):
        assert canonical_unit("widgets") is None
        assert canonical_unit("") is None
        assert canonical_unit(None) is None
        # molar concentration intentionally excluded (analyte-specific factor)
        assert canonical_unit("mmol/L") is None
        assert canonical_unit("mg/dL") is None

    def test_same_family(self):
        c = UnitCanonicalizer()
        assert c.same_family("kg", "lb")
        assert not c.same_family("kg", "cm")
        assert not c.same_family("kg", "widgets")


class TestConvert:
    def test_mass_kg_to_lb(self):
        factor, offset = convert_units("kg", "lb")
        assert factor == APPROX(1 / 0.45359237)  # ~2.2046
        assert offset == APPROX(0.0)
        assert 80 * factor + offset == APPROX(176.37, abs=0.01)

    def test_length_in_to_cm(self):
        factor, offset = convert_units("in", "cm")
        assert factor == APPROX(2.54) and offset == APPROX(0.0)

    def test_temperature_f_to_c_has_offset(self):
        factor, offset = convert_units("degF", "degC")
        assert factor == APPROX(5 / 9)
        assert offset == APPROX(-160 / 9)
        assert 98.6 * factor + offset == APPROX(37.0, abs=0.01)  # body temp

    def test_temperature_c_to_f(self):
        factor, offset = convert_units("C", "F")
        assert 37.0 * factor + offset == APPROX(98.6, abs=0.01)

    def test_time_months_to_years(self):
        factor, offset = convert_units("months", "years")
        assert factor == APPROX(1 / 12) and offset == APPROX(0.0)
        assert 24 * factor + offset == APPROX(2.0)

    def test_proportion_percent_to_fraction(self):
        factor, offset = convert_units("%", "fraction")
        assert factor == APPROX(0.01) and offset == APPROX(0.0)

    def test_time_subunits(self):
        f, o = convert_units("minutes", "seconds")
        assert 2 * f + o == APPROX(120.0)
        f, o = convert_units("ms", "s")
        assert 500 * f + o == APPROX(0.5)
        f, o = convert_units("s", "min")
        assert 90 * f + o == APPROX(1.5)

    def test_volume_family(self):
        f, o = convert_units("L", "mL")
        assert 2 * f + o == APPROX(2000.0)
        f, o = convert_units("mL", "dL")
        assert 250 * f + o == APPROX(2.5)
        assert canonical_unit("milliliter") == ("volume", "mL")
        assert canonical_unit("cc") == ("volume", "mL")

    def test_frequency_family(self):
        f, o = convert_units("kHz", "Hz")
        assert 8 * f + o == APPROX(8000.0)
        assert canonical_unit("hertz") == ("frequency", "Hz")

    def test_mmhg_spelled_out(self):
        assert canonical_unit("millimeter of mercury") == ("pressure", "mmHg")
        f, o = convert_units("kPa", "millimeters of mercury")
        assert 1 * f + o == APPROX(7.50061683)

    def test_angle_degree_radian(self):
        f, o = convert_units("degree", "radian")
        assert 180.0 * f + o == APPROX(math.pi)
        f, o = convert_units("radian", "degree")
        assert math.pi * f + o == APPROX(180.0)
        f, o = convert_units("arcminute", "degree")
        assert 90 * f + o == APPROX(1.5)

    def test_length_micrometre(self):
        f, o = convert_units("um", "mm")
        assert 2500 * f + o == APPROX(2.5)
        f, o = convert_units("micron", "cm")
        assert 10000 * f + o == APPROX(1.0)

    def test_pressure_cmh2o(self):
        f, o = convert_units("cmH2O", "mmHg")
        assert 10 * f + o == APPROX(7.35559, abs=1e-4)
        assert convert_units("cmH2O", "kg") is None  # cross-family still blocked

    def test_identical_unit_is_identity(self):
        factor, offset = convert_units("kg", "kg")
        assert is_identity_conversion(factor, offset)

    def test_round_trip_inverse(self):
        f1, o1 = convert_units("kg", "lb")
        f2, o2 = convert_units("lb", "kg")
        # apply forward then back -> original (within fp tolerance)
        x = 73.5
        assert (x * f1 + o1) * f2 + o2 == APPROX(x)

    def test_cross_family_is_none(self):
        assert convert_units("kg", "cm") is None

    def test_unrecognised_is_none(self):
        assert convert_units("widgets", "kg") is None
        assert convert_units("kg", None) is None


class TestIdentityHelper:
    def test_detects_identity(self):
        assert is_identity_conversion(1.0, 0.0)
        assert is_identity_conversion(1.0 + 1e-12, -1e-12)

    def test_rejects_nonidentity(self):
        assert not is_identity_conversion(2.2046, 0.0)
        assert not is_identity_conversion(1.0, -17.78)
