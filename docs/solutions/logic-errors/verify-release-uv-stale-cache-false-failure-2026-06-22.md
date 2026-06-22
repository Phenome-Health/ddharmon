---
title: verify_release.py false VERIFICATION FAILED from uv's stale negative index cache
date: 2026-06-22
category: logic-errors/
module: scripts/verify_release.py
problem_type: logic_error
component: tooling
symptoms:
  - "verify_release.py reports VERIFICATION FAILED on a healthy, already-published release"
  - "Stage 1 sees X.Y.Z live on the PyPI JSON API but Stage 2 'uv pip install' cannot resolve it"
  - "uv: No solution found when resolving dependencies: Because there is no version of ddharmon==X.Y.Z"
  - "The 3x15s install retry loop never recovers, and re-running the gate fails identically"
root_cause: logic_error
resolution_type: code_fix
severity: high
related_components:
  - development_workflow
  - documentation
tags:
  - uv
  - pypi
  - release-verification
  - cache-invalidation
  - retry-logic
  - trusted-publishing
  - propagation-lag
  - false-negative
---

# verify_release.py false VERIFICATION FAILED from uv's stale negative index cache

## Problem

`scripts/verify_release.py` (the post-publish PyPI verification gate) printed `VERIFICATION
FAILED` on a perfectly healthy **v0.6.1** release: PyPI's JSON API showed the version live,
but `uv pip install ddharmon==0.6.1` couldn't resolve it. A false-negative on an immutable,
already-published artifact wastes release-day triage and can trigger an unnecessary yank +
needless patch release.

## Symptoms

- Stage 1 (JSON API poll) **passes** — `ddharmon 0.6.1 is live` — while Stage 2 (install)
  **fails**. This split is the diagnostic signature: JSON says yes, install says no.
- All three install retries fail identically (`3×15s` wasted), ending in:
  ```
  install attempt 1/3 failed (propagation lag?); retrying in 15s...
  install attempt 2/3 failed (propagation lag?); retrying in 15s...
  install attempt 3/3 failed (propagation lag?); retrying in 15s...
  × No solution found when resolving dependencies:
  ╰─▶ Because there is no version of ddharmon==0.6.1 and you require
      ddharmon==0.6.1, we can conclude that your requirements are unsatisfiable.
  VERIFICATION FAILED for ddharmon 0.6.1.
  ```
- Re-running `verify_release.py 0.6.1` from scratch **still** fails — the failure is sticky
  across process invocations (the cache outlives the run).
- Meanwhile `curl -s https://pypi.org/simple/ddharmon/ | grep 0.6.1` shows the wheel and sdist
  (with provenance attestations) already present — the artifact is real and healthy.

## What Didn't Work

- **Re-running `verify_release.py 0.6.1` again.** Failed identically. The retry loop (`3×15s`)
  and a second process both reuse uv's persistent on-disk cache, which now holds a *negative*
  simple-index entry. More attempts against the same stale cache can't change the answer.
- **Waiting for PyPI propagation.** A red herring. The artifact was already fully live on the
  simple/install index — `pypi.org/simple/ddharmon/` listed both files. The blocker was the
  local uv cache, not PyPI. Time alone never clears it, because nothing re-queries the index.

## Solution

**The fix** (commit `c68eb65`, PR #16) — in `install_into_venv()`, force a fresh index query
on every retry:

```python
# BEFORE
for attempt in range(1, retries + 1):
    result = _run(["uv", "pip", "install", "--python", str(py), spec])
    if result.returncode == 0:
        return py
    ...

# AFTER
for attempt in range(1, retries + 1):
    cmd = ["uv", "pip", "install", "--python", str(py)]
    # On retry, bypass uv's index cache. The first attempt can cache a negative
    # simple-index response (the just-published version not visible yet); without
    # --refresh every retry reuses that stale "not found" and keeps failing even
    # after the index propagates, defeating the whole point of the retry loop.
    if attempt > 1:
        cmd.append("--refresh")
    cmd.append(spec)
    result = _run(cmd)
    if result.returncode == 0:
        return py
    ...
```

The first attempt stays cache-friendly (fast when the version is already propagated); every
retry appends `--refresh` to bypass the stale index cache. After the fix, both
`verify_release.py 0.6.1` (core) and `--full` (installs `ddharmon[all]`, exercising the
sentence-transformers embedding stack) passed: VERIFIED.

**Immediate operator workarounds** (no code change needed):

```bash
# Option A: force a fresh index read on a one-off install
uv pip install --refresh ddharmon==0.6.1

# Option B: evict the package's cache entries, then re-run the gate
uv cache clean ddharmon
python scripts/verify_release.py 0.6.1
```

**Confirm the artifact is genuinely fine** before assuming the release is broken — check the
*simple* index (the one installers actually read), not just the JSON API:

```bash
curl -s https://pypi.org/simple/ddharmon/ | grep -F 'ddharmon-0.6.1'
# wheel + sdist present -> release is healthy; the failure is local cache, not PyPI
# (use the precise '<pkg>-<version>' form, not a bare 'grep 0.6.1' that also matches 0.6.10)
```

## Why This Works

PyPI exposes a package through two surfaces with **independent propagation**: the JSON API
(`/pypi/<pkg>/<version>/json`, what Stage 1 polls) and the simple/install index
(`/simple/<pkg>/`, what `uv`/`pip` resolve against). The JSON API frequently updates **ahead**
of the simple index, so there is a window where Stage 1 sees the version but an installer does
not.

`uv` caches simple-index responses on disk, including **negative** results. If the very first
`uv pip install` lands inside that window, uv records "ddharmon 0.6.1 not found" and reuses it.
Without an invalidation signal, subsequent attempts — even in a brand-new process — read that
cached "not found" and fail, long after the index has actually propagated. That is why both the
in-loop retries and a fresh re-run stayed red.

`--refresh` tells uv to ignore cached index responses and re-fetch from PyPI, so the retry
resolves against the now-propagated simple index. Putting it **only on retries** (`attempt > 1`)
keeps the common case — version already propagated — on the fast cached path, and pays the
network round-trip cost only when the first attempt has actually failed. The retry loop regains
its purpose: each retry asks PyPI again instead of re-reading a stale local "no."

## Prevention

- **Always bust the cache in post-publish install-verification retry loops.** A retry that
  re-reads a cached negative is a no-op. Add `--refresh` (or `--no-cache`) on retries:
  ```python
  cmd = ["uv", "pip", "install", "--python", str(py)]
  if attempt > 1:
      cmd.append("--refresh")   # or "--no-cache"
  cmd.append(spec)
  ```
  Prefer `--refresh` over `--no-cache` so the first attempt still benefits from caching; reserve
  `--no-cache` for fully hermetic one-shot checks.
- **When a just-published version "isn't found," check the simple index first.**
  `curl -s https://pypi.org/simple/<pkg>/ | grep <version>` distinguishes a real publish failure
  from a local-cache / propagation-window artifact before anyone reaches for a yank.
- **Poll the surface installers actually use in Stage 1.** Stage 1 polling only the JSON API is
  what lets Stage 1 pass while Stage 2 fails. Polling (or additionally checking) `/simple/<pkg>/`
  would gate on the same surface the install resolves against and close the
  JSON-ahead-of-simple gap:
  ```bash
  curl -s https://pypi.org/simple/<pkg>/ | grep -q "<pkg>-<version>" && echo live
  ```
- **Treat false negatives as high-stakes.** PyPI versions are immutable — a release can never be
  re-uploaded. A verification gate that fails on a healthy artifact must not drive a reflexive
  yank + patch release; require a simple-index check (artifact actually missing/corrupt) before
  any yank decision.

## Related Issues

- **PR #16** — `fix(verify): bust uv index cache on install retries` (commit `c68eb65`), the fix
  documented here. Merged to `Phenome-Health/ddharmon` `main` on 2026-06-22.
- **PR #15** — `release: v0.6.1`, the release whose verification surfaced this false negative.
- `scripts/verify_release.py` — the gate; the fix lives in `install_into_venv()`.
- `scripts/smoke_test.py` — Stage 3 of the gate (run against the installed wheel).
- `.claude/skills/publish-to-pypi/SKILL.md` — the release workflow that drives the gate; the
  first place a future releaser lands.
