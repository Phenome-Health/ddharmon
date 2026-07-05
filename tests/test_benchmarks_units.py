"""Fast-suite guard for the C2 numeric benchmark (benchmarks/units.py).

The N1 unit gold and the N2 formula-verify oracle are $0 and deterministic, so they run inline here as a
regression guard on the units canonicalizer + the formula verifier — the same numbers the WS4 gate floors.
"""

from __future__ import annotations

from benchmarks.units import score_n1, score_n2


class TestN1UnitGold:
    def test_all_conversions_correct(self):
        m = score_n1()
        assert m["recognition_rate"] == 1.0  # every gold unit is recognized
        assert m["accuracy"] == 1.0  # and converts to the authoritative expected value


class TestN2FormulaVerify:
    def test_oracle_self_check_is_perfect(self):
        m = score_n2()
        assert m["oracle_accuracy"] == 1.0  # correct formulas verify at 1.0

    def test_wrong_formulas_are_rejected(self):
        m = score_n2()
        assert m["wrong_accuracy"] < 0.5  # deliberate foils do not pass verification
        # per-derivation: every correct formula beats its wrong foil
        for d in m["per_derivation"]:
            assert d["oracle"] > d["wrong"]
