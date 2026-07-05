"""C2 N1 — curated, dependency-free unit canonicalization + linear conversion.

The N1 transform class is a **linear unit/scale conversion**: ``target = source * factor + offset``
(kg↔lb, cm↔in, °C↔°F, months↔years, % ↔ fraction). It is deterministic and data-free, so its spec can
be **authored** (not just detected) and graded against canonical factors.

**Units-dependency fork — RESOLVED (C2, 2026-06-29): a curated no-dep table**, not ``pint``/UCUM. A small
table of the unit families that actually recur in cross-cohort survey/clinical harmonization
(mass, length, temperature, time, volume, frequency, pressure, proportion, angle) keeps the package
dependency-free (a 1.0 API-surface commitment) and keeps every conversion auditable. Each family maps each
known unit spelling to a linear ``(factor, offset)`` to the family's canonical unit
(``x_canonical = x * factor + offset``); a conversion between two same-family units composes those linears,
so the result is always ``target = source * factor + offset`` — exactly the N1 :class:`TransformSpec` shape.

**Intentionally excluded** (left to flag ``needs_review`` rather than authored with a false factor):
molar/mass-concentration conversions (mg/dL ↔ mmol/L) — analyte-specific factor (glucose 18.0,
cholesterol 38.7, …); and non-linear / rate units (beats-per-minute, mg/kg/day, °/s) that the simple
``factor·x + offset`` model cannot express.
"""

from __future__ import annotations

import math
import re

# Each family: canonical-unit token -> {unit_token: (factor, offset) to the canonical unit}.
# x_canonical = x * factor + offset. Pure physical / dimensionless families only (see module docstring).
_FAMILIES: dict[str, dict[str, tuple[float, float]]] = {
    "mass": {  # canonical: kg
        "kg": (1.0, 0.0),
        "g": (1e-3, 0.0),
        "mg": (1e-6, 0.0),
        "lb": (0.45359237, 0.0),
        "oz": (0.028349523125, 0.0),
        "st": (6.35029318, 0.0),  # stone
    },
    "length": {  # canonical: cm
        "cm": (1.0, 0.0),
        "m": (100.0, 0.0),
        "mm": (0.1, 0.0),
        "um": (1e-4, 0.0),  # micrometre / micron
        "in": (2.54, 0.0),
        "ft": (30.48, 0.0),
    },
    "temperature": {  # canonical: degC  (x_C = F*5/9 - 160/9 ; x_C = K - 273.15)
        "degC": (1.0, 0.0),
        "degF": (5.0 / 9.0, -160.0 / 9.0),
        "K": (1.0, -273.15),
    },
    "time": {  # canonical: year
        "year": (1.0, 0.0),
        "month": (1.0 / 12.0, 0.0),
        "week": (7.0 / 365.25, 0.0),
        "day": (1.0 / 365.25, 0.0),
        "hour": (1.0 / (365.25 * 24.0), 0.0),
        "minute": (1.0 / (365.25 * 24.0 * 60.0), 0.0),
        "second": (1.0 / (365.25 * 24.0 * 3600.0), 0.0),
        "millisecond": (1.0 / (365.25 * 24.0 * 3600.0 * 1000.0), 0.0),
    },
    "volume": {  # canonical: mL
        "mL": (1.0, 0.0),
        "L": (1000.0, 0.0),
        "dL": (100.0, 0.0),
        "uL": (1e-3, 0.0),
    },
    "frequency": {  # canonical: Hz
        "Hz": (1.0, 0.0),
        "kHz": (1000.0, 0.0),
        "MHz": (1e6, 0.0),
    },
    "pressure": {  # canonical: mmHg
        "mmHg": (1.0, 0.0),
        "kPa": (7.50061683, 0.0),
        "cmH2O": (0.735559, 0.0),  # 1 cmH2O (4 °C) = 98.0638 Pa; 1 mmHg = 133.322 Pa
    },
    "proportion": {  # canonical: fraction [0,1]
        "fraction": (1.0, 0.0),
        "percent": (0.01, 0.0),
    },
    "angle": {  # canonical: degree (goniometry / range-of-motion; deg↔rad is a pure linear factor)
        "degree": (1.0, 0.0),
        "radian": (180.0 / math.pi, 0.0),
        "arcminute": (1.0 / 60.0, 0.0),
        "arcsecond": (1.0 / 3600.0, 0.0),
    },
}

# Raw spelling (after _norm) -> canonical unit token. Drives recognition; many spellings per unit.
_ALIASES: dict[str, str] = {
    # mass
    "kg": "kg",
    "kgs": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "kilo": "kg",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "gm": "g",
    "mg": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "st": "st",
    "stone": "st",
    "stones": "st",
    # length
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "in": "in",
    "inch": "in",
    "inches": "in",
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
    "um": "um",
    "micrometer": "um",
    "micrometers": "um",
    "micrometre": "um",
    "micrometres": "um",
    "micron": "um",
    "microns": "um",
    # temperature
    "degc": "degC",
    "c": "degC",
    "celsius": "degC",
    "centigrade": "degC",
    "cel": "degC",
    "degreecelsius": "degC",
    "degreescelsius": "degC",
    "degf": "degF",
    "f": "degF",
    "fahrenheit": "degF",
    "degreefahrenheit": "degF",
    "degreesfahrenheit": "degF",
    "k": "K",
    "kelvin": "K",
    "degreekelvin": "K",
    "degreeskelvin": "K",
    # time / age
    "year": "year",
    "years": "year",
    "yr": "year",
    "yrs": "year",
    "y": "year",
    "a": "year",
    "month": "month",
    "months": "month",
    "mo": "month",
    "mon": "month",
    "week": "week",
    "weeks": "week",
    "wk": "week",
    "wks": "week",
    "day": "day",
    "days": "day",
    "d": "day",
    "hour": "hour",
    "hours": "hour",
    "hr": "hour",
    "hrs": "hour",
    "h": "hour",
    "minute": "minute",
    "minutes": "minute",
    "min": "minute",
    "second": "second",
    "seconds": "second",
    "sec": "second",
    "s": "second",
    "millisecond": "millisecond",
    "milliseconds": "millisecond",
    "ms": "millisecond",
    "msec": "millisecond",
    # volume
    "ml": "mL",
    "milliliter": "mL",
    "milliliters": "mL",
    "millilitre": "mL",
    "millilitres": "mL",
    "cc": "mL",
    "cubiccentimeter": "mL",
    "cubiccentimeters": "mL",
    "cm3": "mL",
    "l": "L",
    "liter": "L",
    "liters": "L",
    "litre": "L",
    "litres": "L",
    "dl": "dL",
    "deciliter": "dL",
    "deciliters": "dL",
    "ul": "uL",
    "microliter": "uL",
    "microliters": "uL",
    "microlitre": "uL",
    "microlitres": "uL",
    # frequency
    "hz": "Hz",
    "hertz": "Hz",
    "khz": "kHz",
    "kilohertz": "kHz",
    "mhz": "MHz",
    "megahertz": "MHz",
    # pressure
    "mmhg": "mmHg",
    "kpa": "kPa",
    "millimeterhg": "mmHg",
    "millimetershg": "mmHg",
    "millimeterofmercury": "mmHg",
    "millimetersofmercury": "mmHg",
    "millimetreofmercury": "mmHg",
    "millimetresofmercury": "mmHg",
    "mm[hg]": "mmHg",
    "cmh2o": "cmH2O",
    "centimeterofwater": "cmH2O",
    "centimetersofwater": "cmH2O",
    "centimetreofwater": "cmH2O",
    "centimetresofwater": "cmH2O",
    "centimeterh2o": "cmH2O",
    "centimetersh2o": "cmH2O",
    "cmwater": "cmH2O",
    # proportion
    "fraction": "fraction",
    "frac": "fraction",
    "proportion": "fraction",
    "ratio": "fraction",
    "percent": "percent",
    "pct": "percent",
    "percentage": "percent",
    # angle (bare "°" normalises to "deg" via _SYMBOL_MAP; "°C" stays "degc" → temperature)
    "deg": "degree",
    "degree": "degree",
    "degrees": "degree",
    "degreeofarc": "degree",
    "degreesofarc": "degree",
    "arcdegree": "degree",
    "arcdegrees": "degree",
    "radian": "radian",
    "radians": "radian",
    "rad": "radian",
    "arcminute": "arcminute",
    "arcminutes": "arcminute",
    "arcmin": "arcminute",
    "arcsecond": "arcsecond",
    "arcseconds": "arcsecond",
    "arcsec": "arcsecond",
}

# canonical token -> family, built once from _FAMILIES.
_UNIT_FAMILY: dict[str, str] = {u: fam for fam, units in _FAMILIES.items() for u in units}

# degree-symbol / micro-sign normalisations applied before alias lookup.
_SYMBOL_MAP = {"°": "deg", "µ": "u", "μ": "u", "%": "percent"}


def _norm(unit: str) -> str:
    """Normalise a raw unit string to an alias-lookup token (lowercase, symbol-folded, depluralised tail)."""
    s = unit.strip().lower()
    for sym, rep in _SYMBOL_MAP.items():
        s = s.replace(sym, rep)
    s = re.sub(r"[\s.]+", "", s)  # drop spaces/dots: "deg c" -> "degc", "kg." -> "kg"
    return s


class UnitCanonicalizer:
    """Recognise a unit spelling and convert linearly between two same-family units.

    Stateless and dependency-free; methods are instance methods so a caller can hold one and (later)
    extend the alias map without touching module globals.
    """

    def canonical(self, unit: str | None) -> tuple[str, str] | None:
        """Return ``(family, canonical_unit_token)`` for a unit spelling, or ``None`` if unrecognised."""
        if not unit or not unit.strip():
            return None
        tok = _ALIASES.get(_norm(unit))
        if tok is None:
            return None
        return _UNIT_FAMILY[tok], tok

    def same_family(self, a: str | None, b: str | None) -> bool:
        ca, cb = self.canonical(a), self.canonical(b)
        return ca is not None and cb is not None and ca[0] == cb[0]

    def convert(self, source_unit: str | None, target_unit: str | None) -> tuple[float, float] | None:
        """Linear conversion ``(factor, offset)`` s.t. ``target_value = source_value * factor + offset``.

        Returns ``None`` when either unit is unrecognised or they belong to different families.
        ``(1.0, 0.0)`` when the units are identical (an identity / no-op conversion).
        """
        cs, ct = self.canonical(source_unit), self.canonical(target_unit)
        if cs is None or ct is None or cs[0] != ct[0]:
            return None
        (fs, os_), (ft, ot) = _FAMILIES[cs[0]][cs[1]], _FAMILIES[ct[0]][ct[1]]
        # x_canon = src*fs + os_ ; target = (x_canon - ot)/ft = src*(fs/ft) + (os_-ot)/ft
        factor = fs / ft
        offset = (os_ - ot) / ft
        return round(factor, 10), round(offset, 10)


# module-level singleton for the common stateless case
_CANON = UnitCanonicalizer()


def canonical_unit(unit: str | None) -> tuple[str, str] | None:
    """Module-level convenience over :meth:`UnitCanonicalizer.canonical`."""
    return _CANON.canonical(unit)


def convert_units(source_unit: str | None, target_unit: str | None) -> tuple[float, float] | None:
    """Module-level convenience over :meth:`UnitCanonicalizer.convert`."""
    return _CANON.convert(source_unit, target_unit)


def is_identity_conversion(factor: float, offset: float, *, tol: float = 1e-9) -> bool:
    """True when ``(factor, offset)`` is a no-op (factor≈1, offset≈0) — i.e. the units already match."""
    return abs(factor - 1.0) <= tol and abs(offset) <= tol
