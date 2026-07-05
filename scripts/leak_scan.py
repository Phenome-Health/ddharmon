#!/usr/bin/env python3
"""Leak scan — fail if internal or local-machine artifacts would ship in a public repo.

Phenome Health ships public repos (``Phenome-Health/ddharmon``, ``ddharmon-ui``) as curated
forward-ports of an internal research monorepo. This scanner is the "ship clean" gate: it greps
tracked (or staged) files for strings that must never reach a public repo — local filesystem
paths, personal handles, and internal-only terms — and exits non-zero on any finding.

Profiles (a research repo legitimately contains internal names; a public repo must not):
  * ``public``   — local paths + internal names/handles/terms (the CI gate on public repos)
  * ``internal`` — local paths only (a light check for the internal research repo)

Usage::

    python scripts/leak_scan.py                      # all tracked files, public profile
    python scripts/leak_scan.py --staged             # only staged files (pre-commit)
    python scripts/leak_scan.py --profile internal   # research-repo profile (paths only)

Pure standard library — no dependencies, safe to run in CI without an install step.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# (regex, human description, {profiles it applies to})
RULES: list[tuple[str, str, set[str]]] = [
    (r"/Users/[^/\s\"'`]+", "local macOS home path", {"public", "internal"}),
    # /home/ubuntu is a conventional deploy placeholder — allow it, flag other user homes.
    (r"/home/(?!ubuntu\b)[A-Za-z0-9_.-]+", "local Linux home path", {"public", "internal"}),
    (r"\bai-coding\b", "local workspace directory name", {"public", "internal"}),
    (r"\bInsync\b", "local workspace directory name", {"public", "internal"}),
    (r"trentleslie", "personal GitHub handle / content mirror", {"public"}),
    (r"[Gg]reptile", "internal code-review mirror process", {"public"}),
    (r"secure[ -]?server", "internal-infrastructure term", {"public"}),
    (r"competitive_landscape", "internal strategy document", {"public"}),
    (r"ph-arpa-data-harmonization", "internal dev-repo name", {"public"}),
]

# A line containing this marker is skipped — for guard scripts that legitimately define
# the patterns above (this scanner, build_demo_bundle.py's privacy gate, etc.).
IGNORE_MARKER = "leak-scan-ignore"

# Files never scanned (this scanner defines the patterns; binaries can't be grepped).
SKIP_NAMES = {"leak_scan.py"}
SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot", ".zip", ".gz", ".parquet", ".npy",
)


def _tracked_files(staged: bool) -> list[str]:
    cmd = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
        if staged
        else ["git", "ls-files"]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f]


def scan(files: list[str], profile: str) -> list[tuple[str, int, str, str]]:
    rules = [(re.compile(p), desc) for p, desc, profs in RULES if profile in profs]
    findings: list[tuple[str, int, str, str]] = []
    for f in files:
        path = Path(f)
        if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if IGNORE_MARKER in line:
                continue
            for rx, desc in rules:
                m = rx.search(line)
                if m:
                    findings.append((f, lineno, desc, m.group(0)))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail if internal/local artifacts would ship publicly.")
    ap.add_argument("--staged", action="store_true", help="scan only staged files (pre-commit)")
    ap.add_argument("--profile", choices=["public", "internal"], default="public")
    args = ap.parse_args()

    files = _tracked_files(args.staged)
    findings = scan(files, args.profile)

    if findings:
        print(f"leak-scan: {len(findings)} finding(s) [profile={args.profile}]", file=sys.stderr)
        for f, lineno, desc, frag in findings:
            print(f"  {f}:{lineno}: {desc} -> {frag!r}", file=sys.stderr)
        print(
            "\nRemove these before committing/shipping. "
            "If a match is a genuine false positive, refine the rule in scripts/leak_scan.py.",
            file=sys.stderr,
        )
        return 1

    print(f"leak-scan: clean ({len(files)} files scanned, profile={args.profile})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
