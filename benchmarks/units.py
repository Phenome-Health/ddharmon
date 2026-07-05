"""Benchmark C2 — numeric transform specs: unit conversion (N1) + arithmetic-verify (N2).

Two $0, deterministic, dependency-free checks over the C2 numeric transform engine:

  * **N1** — apply the curated unit canonicalizer (``ddharmon.values.convert_units``) to a hand-verified
    gold of ``(source_unit, target_unit, sample_value -> expected_value)`` cases drawn from AUTHORITATIVE
    conversion factors (kg/lb, cm/in, °F/°C, months/years, % / fraction, g/kg, m/cm). The expected values
    are written independently of the implementation's factor table, so a perfect score is real external
    agreement, not a tautology. Scores the fraction of sample values converted within tolerance.

  * **N2** — exercise the deterministic formula verifier (``ddharmon.harmonization.verify_formula``) on a
    curated set of canonical derivations (BMI, months->years, % ->fraction), each with test-input cases.
    The CORRECT formula must verify at 1.0 (oracle self-check) and a deliberately WRONG formula must score
    low — i.e. the harness distinguishes a right spec from a wrong one. (The LLM formula-GENERATOR arm vs
    the ATHLOS-55 arithmetic gold needs the ATHLOS arithmetic parser + API keys, so — like the categorical
    LLM recode arm — it stays OUT of the $0 gate. See benchmarks/README.md.)

  PYTHONHASHSEED=0 python -m benchmarks.units
"""

from __future__ import annotations

import json
import math

from benchmarks import _common as common
from ddharmon.harmonization import verify_formula
from ddharmon.values import convert_units

# N1 gold — (source_unit, target_unit, sample_value, expected_value). Expected values from authoritative
# factors (1 kg = 2.2046226 lb; 1 in = 2.54 cm; °C = (°F-32)*5/9; …), independent of the impl table.
N1_GOLD: list[tuple[str, str, float, float]] = [
    ("kg", "lb", 70.0, 154.32358),
    ("lb", "kg", 154.0, 69.853225),
    ("cm", "in", 180.0, 70.866142),
    ("in", "cm", 72.0, 182.88),
    ("degF", "degC", 98.6, 37.0),
    ("degC", "degF", 37.0, 98.6),
    ("months", "years", 30.0, 2.5),
    ("years", "months", 3.0, 36.0),
    ("%", "fraction", 50.0, 0.5),
    ("fraction", "%", 0.25, 25.0),
    ("g", "kg", 2500.0, 2.5),
    ("m", "cm", 1.75, 175.0),
    ("um", "mm", 2500.0, 2.5),  # 1 µm = 1e-3 mm
    ("mm", "um", 2.5, 2500.0),
    ("degree", "radian", 180.0, math.pi),  # 180° = π rad (exact external constant)
    ("radian", "degree", math.pi, 180.0),
    ("arcminute", "degree", 90.0, 1.5),  # 60 arcmin = 1°
    ("cmH2O", "mmHg", 10.0, 7.35559),  # 1 cmH2O = 0.735559 mmHg (98.0638 Pa / 133.322 Pa)
    ("mmHg", "cmH2O", 14.71118, 20.0),
]

# N2 gold — canonical deterministic derivations, each with a correct formula, a wrong foil, and cases.
N2_GOLD: list[dict] = [
    {
        "name": "bmi",
        "formula": "weight / (height ** 2)",
        "wrong": "weight * height",
        "cases": [{"weight": 80, "height": 2, "expected": 20.0}, {"weight": 50, "height": 2, "expected": 12.5}],
    },
    {
        "name": "months_to_years",
        "formula": "source / 12",
        "wrong": "source * 12",
        "cases": [{"source": 24, "expected": 2.0}, {"source": 6, "expected": 0.5}],
    },
    {
        "name": "percent_to_fraction",
        "formula": "source / 100",
        "wrong": "source",
        "cases": [{"source": 50, "expected": 0.5}, {"source": 25, "expected": 0.25}],
    },
]


def score_n1(rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> dict:
    """Fraction of N1 gold cases the canonicalizer converts within tolerance (+ unit-recognition rate)."""
    n = correct = recognized = 0
    for src, tgt, val, exp in N1_GOLD:
        n += 1
        conv = convert_units(src, tgt)
        if conv is None:
            continue
        recognized += 1
        factor, offset = conv
        if math.isclose(val * factor + offset, exp, rel_tol=rel_tol, abs_tol=abs_tol):
            correct += 1
    return {
        "n": n,
        "recognized": recognized,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "recognition_rate": round(recognized / n, 4) if n else 0.0,
    }


def score_n2() -> dict:
    """Oracle (correct-formula) vs wrong-formula verify accuracy over the N2 derivations."""
    oracle, wrong, per = [], [], []
    for g in N2_GOLD:
        ok = verify_formula(g["formula"], g["cases"])["accuracy"]
        bad = verify_formula(g["wrong"], g["cases"])["accuracy"]
        oracle.append(ok)
        wrong.append(bad)
        per.append({"name": g["name"], "oracle": ok, "wrong": bad})
    return {
        "cases": len(N2_GOLD),
        "oracle_accuracy": round(sum(oracle) / len(oracle), 4) if oracle else 0.0,
        "wrong_accuracy": round(sum(wrong) / len(wrong), 4) if wrong else 0.0,
        "per_derivation": per,
    }


def main() -> None:
    n1 = score_n1()
    n2 = score_n2()
    result = {"benchmark": "units_numeric_transform", "n1": n1, "n2": n2}

    print(f"\n{'=' * 64}\nC2 numeric transform-spec benchmark\n{'=' * 64}")
    print(
        f"  N1 unit conversion : {n1['correct']}/{n1['n']} correct (acc {n1['accuracy']:.3f}), "
        f"recognition {n1['recognition_rate']:.0%}"
    )
    print(
        f"  N2 formula verify  : oracle {n2['oracle_accuracy']:.3f} (correct formulas) vs "
        f"wrong {n2['wrong_accuracy']:.3f} (foils)"
    )
    for d in n2["per_derivation"]:
        print(f"      {d['name']:22s} oracle {d['oracle']:.2f} / wrong {d['wrong']:.2f}")

    out = common.CACHE_DIR / "units_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\n  wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
