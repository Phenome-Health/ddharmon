"""WS4 — standing-benchmark release gate: run the $0 benchmarks and assert regression floors.

Runs Benchmark A (CDEMapper retrieval recall), Benchmark B (PhenX cross-cohort co-clustering) and
Benchmark D (AI-READI var->concept retrieval) as isolated subprocesses (reproducible under
``PYTHONHASHSEED=0``), then checks their result JSONs against floors. Exits non-zero if any HARD floor is
breached.

HARD gates are DETERMINISTIC signals only: CDEMapper hybrid recall, AI-READI dense recall (both dense
cosine / BM25 / RRF, no randomness) and the PhenX cut-INDEPENDENT embedding separability Δ (computed on
the raw normalized embeddings with seeded sampling). The PhenX co-clustering macro/micro use UMAP+HDBSCAN,
which is not bit-reproducible across processes, so they are ADVISORY — reported, never gating.

Floors are set with margin below the committed BioLORD-2023 baselines (see benchmarks/README.md); they are
REGRESSION guards, not targets. Lowering one requires a re-baseline + a written reason.

  PYTHONHASHSEED=0 python -m benchmarks.gate
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from benchmarks import _common as common

ENCODER = "FremyCompany/BioLORD-2023"  # the committed default these floors are baselined against

# HARD floors — deterministic signals; a breach exits non-zero. (bench, json-path, floor, label)
HARD: list[tuple[str, tuple[str, ...], float, str]] = [
    ("cdemapper", ("hybrid", "@5"), 0.63, "CDEMapper hybrid recall@5"),
    ("cdemapper", ("hybrid", "@100"), 0.90, "CDEMapper hybrid recall@100"),
    ("phenx", ("embedding_signal", "separation"), 0.55, "PhenX separability Δ (cut-independent)"),
    # AI-READI dense beats hybrid here (BM25 hurts on short standardized concept names) → gate dense.
    ("aireadi", ("dense", "@5"), 0.58, "AI-READI var→concept dense recall@5 (held-out)"),
    ("aireadi", ("dense", "@100"), 0.85, "AI-READI var→concept dense recall@100 (held-out)"),
    # C2 numeric transform specs — deterministic, no encoder/keys. N1 = curated unit-conversion gold;
    # N2 = the formula-verifier's oracle self-check (correct formulas must verify at 1.0).
    ("units", ("n1", "accuracy"), 0.95, "C2 N1 unit-conversion accuracy (curated gold)"),
    ("units", ("n2", "oracle_accuracy"), 0.99, "C2 N2 formula-verify oracle self-check"),
]
# ADVISORY — UMAP/HDBSCAN-cut-dependent (run-to-run variable); reported, never gating.
ADVISORY: list[tuple[str, tuple[str, ...], float, str]] = [
    ("phenx", ("pxvar", "macro_recall"), 0.38, "PhenX VAR macro recall (UMAP cut → advisory)"),
]


def _run(*args: str) -> None:
    """Run a benchmark module as a reproducible subprocess; abort the gate if it errors."""
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run([sys.executable, "-m", *args], cwd=str(common.REPO_ROOT), env=env)  # noqa: S603
    if proc.returncode != 0:
        print(f"!! benchmark subprocess failed: {' '.join(args)} (exit {proc.returncode}) — gate cannot run")
        sys.exit(2)


def _dig(d: dict, path: tuple[str, ...]) -> float | None:
    cur: object = d
    for k in path:
        cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur if isinstance(cur, (int, float)) else None


def main() -> None:
    print(f"WS4 benchmark gate — running standing benchmarks (encoder default: {ENCODER})…\n")
    _run("benchmarks.cdemapper")
    _run("benchmarks.phenx", "--level", "pxvar")
    _run("benchmarks.aireadi")
    _run("benchmarks.units")
    results = {
        "cdemapper": json.loads((common.CACHE_DIR / "cdemapper_result.json").read_text()),
        "phenx": json.loads((common.CACHE_DIR / "phenx_result.json").read_text()),
        "aireadi": json.loads((common.CACHE_DIR / "aireadi_result.json").read_text()),
        "units": json.loads((common.CACHE_DIR / "units_result.json").read_text()),
    }

    print(f"\n{'=' * 66}\nWS4 BENCHMARK GATE\n{'=' * 66}")
    failed: list[str] = []
    print("  HARD floors (deterministic — gating):")
    for bench, path, floor, label in HARD:
        val = _dig(results[bench], path)
        ok = val is not None and val >= floor
        print(f"    {'PASS' if ok else 'FAIL'}  {label:<44} {val}  (floor {floor})")
        if not ok:
            failed.append(label)
    print("  Advisory (UMAP-cut-dependent — reported, non-gating):")
    for bench, path, floor, label in ADVISORY:
        val = _dig(results[bench], path)
        flag = "ok" if (val is not None and val >= floor) else "below"
        print(f"    {flag:>5} {label:<44} {val}  (ref {floor})")

    if failed:
        print(f"\n  ✗ GATE FAILED — {len(failed)} hard floor(s) breached: {failed}")
        sys.exit(1)
    print("\n  ✓ GATE PASSED — all hard floors cleared.")


if __name__ == "__main__":
    main()
