#!/usr/bin/env python3
"""Weekly GitHub releases/tags/commits watcher for competitive intel.

Pinned list of ~20 repos drawn from `docs/competitive_landscape.md`. For
each, queries the GitHub REST API for releases, tags, and (optionally)
recent commits to the default branch since a cutoff date. Writes a
markdown section to stdout (or an `--out` file), suitable for appending
to ``docs/competitive_watch.md``.

Auth: uses the ``gh`` CLI (``gh auth status`` must be green). Falls back to
unauthenticated requests with their tighter rate limit if ``gh`` is missing.

Usage::

    python scripts/github_releases.py --since 2026-05-01
    python scripts/github_releases.py --since 2026-05-01 --out /tmp/gh.md
    python scripts/github_releases.py --repos extra.json --since ...

Tested against repos listed in TRACKED_REPOS — adjust there to add/remove.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

logger = logging.getLogger("github_releases")

# Pinned tracked repos. Each entry is (org/name, why_tracked).
#
# Add/remove freely. Keep the "why" short — appears in the report.
TRACKED_REPOS: list[tuple[str, str]] = [
    # DataTecnica (Forge, RoP, FAIRkit/GenCDE)
    ("datatecnica/RoP_biomedical", "RoP — Forge's harmonized CDE catalog (1.33M)"),
    ("datatecnica/Forge-Documentation", "Forge — schema registry + governance product docs"),
    ("datatecnica/GenCDE", "FAIRkit / GenCDE — generate-then-match LLM pipeline"),
    # Monarch Initiative
    ("monarch-initiative/cde-harmonization", "Monarch CDE clustering / Krishnamurthy pipeline"),
    ("monarch-initiative/ontogpt", "SPIRES / OntoGPT — LLM grounding to ontologies"),
    # Yale Clinical NLP Lab (CDEMapper + TopicForest)
    ("BIDS-Xu-Lab/CDE-Mapping-Tool", "CDEMapper — Yale Wang lab"),
    # VIDA-NYU (BDI-Kit + Harmonia)
    ("VIDA-NYU/bdi-kit", "BDI-Kit — schema/value matching primitives"),
    ("VIDA-NYU/harmonia", "Harmonia — LLM agent over BDI-Kit"),
    ("VIDA-NYU/magneto-matcher", "Magneto — GDC benchmark harness"),
    # SCAI-BIO Fraunhofer (ADHTEB + Datastew)
    ("SCAI-BIO/ADHTEB", "ADHTEB — AD harmonization eval benchmark"),
    ("SCAI-BIO/datastew", "Datastew — deployable arm of SCAI vertical"),
    # Ulster/UCL Harmony
    ("harmonydata/harmony", "Harmony — questionnaire item matching"),
    # Maelstrom Research
    ("maelstrom-research/Rmonize", "Rmonize — gold-standard retrospective harmonization"),
    # Reference / linkage projects
    ("XubingHao/BMC2021_DE", "Hao 2024 — NACC↔ADNI↔NIH-CDE triangle"),
    ("KRR-Oxford/OAEI-Bio-ML", "OAEI Bio-ML — ontology alignment benchmark"),
    ("jchen-BUlab/LLM_NLP_dataharmonization", "Li 2025 — EU↔JP AD LLM harmonization"),
    # Internal stack consumers (so we catch upstream changes too)
    ("arpanauts/biomapper", "biomapper — identifier harmonization layer"),
]


# ----- HTTP -----


def gh_available() -> bool:
    return shutil.which("gh") is not None


def gh_api(path: str) -> Any:
    """Invoke ``gh api`` for an authenticated call; raise on non-zero exit."""
    proc = subprocess.run(
        ["gh", "api", "--paginate", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # 404 is common for "no releases" or private repos — surface as empty.
        if "Not Found" in proc.stderr or proc.returncode == 1:
            logger.debug("gh api %s: %s", path, proc.stderr.strip())
            return []
        raise RuntimeError(f"gh api {path} failed: {proc.stderr.strip()}")
    body = proc.stdout.strip()
    if not body:
        return []
    # `gh api --paginate` returns concatenated JSON arrays; safest parse:
    # split on `][` boundary, then re-wrap. For typed endpoints (list-of-objects)
    # this works; for single-object endpoints we'd never paginate.
    if body.startswith("["):
        # Concatenate paginated arrays.
        chunks = body.replace("][", ",").strip()
        return json.loads(chunks)
    return json.loads(body)


def unauth_get(path: str) -> Any:
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "competitive-watch"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        logger.warning("GitHub HTTP %s for %s", e.code, url)
        return []


def github_get(path: str, use_gh: bool) -> Any:
    return gh_api(path) if use_gh else unauth_get(path)


# ----- Per-repo fetchers -----


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # GitHub returns trailing 'Z'; strptime is awkward, fromisoformat handles it on 3.11+
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_repo_activity(
    repo: str, since: datetime, use_gh: bool, max_commits: int = 5
) -> dict[str, Any]:
    """Return releases + tags + recent commits to default branch since cutoff."""
    out: dict[str, Any] = {"repo": repo, "releases": [], "tags": [], "commits": []}

    releases = github_get(f"/repos/{repo}/releases?per_page=20", use_gh)
    if isinstance(releases, list):
        for r in releases:
            published = parse_iso(r.get("published_at") or r.get("created_at"))
            if published and published >= since:
                out["releases"].append(
                    {
                        "name": r.get("name") or r.get("tag_name"),
                        "tag": r.get("tag_name"),
                        "published_at": (r.get("published_at") or r.get("created_at")),
                        "url": r.get("html_url"),
                        "body": (r.get("body") or "").strip(),
                    }
                )

    # Tags don't carry a date directly; we read the underlying commit each one
    # points at, but that's an extra call per tag. Skip date filtering on tags
    # for v1 — just surface all (typically <20) and let the user eyeball.
    tags = github_get(f"/repos/{repo}/tags?per_page=10", use_gh)
    if isinstance(tags, list):
        existing_tags = {r["tag"] for r in out["releases"]}
        for t in tags:
            name = t.get("name")
            if name and name not in existing_tags:
                out["tags"].append({"name": name, "sha": (t.get("commit") or {}).get("sha")})

    # Commits to default branch since cutoff — capped at max_commits to keep
    # the report scannable; quiet repos still emit a line, busy ones show
    # latest few + a "(+ N more)" marker.
    since_iso = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commits = github_get(
        f"/repos/{repo}/commits?since={quote(since_iso)}&per_page=20", use_gh
    )
    if isinstance(commits, list):
        for c in commits[:max_commits]:
            commit = c.get("commit") or {}
            out["commits"].append(
                {
                    "sha": (c.get("sha") or "")[:7],
                    "message": (commit.get("message") or "").splitlines()[0][:120],
                    "date": (commit.get("author") or {}).get("date"),
                    "url": c.get("html_url"),
                }
            )
        out["n_commits_total"] = len(commits)
    return out


# ----- Rendering -----


def render_repo(activity: dict[str, Any], why: str) -> list[str]:
    repo = activity["repo"]
    rel, tags, commits = activity["releases"], activity["tags"], activity["commits"]
    total_commits = activity.get("n_commits_total", len(commits))
    if not (rel or tags or commits):
        return []  # silent on no-op repos in per-run output
    lines = [f"### [{repo}](https://github.com/{repo}) — {why}", ""]
    if rel:
        for r in rel:
            body_preview = (r["body"][:200] + "…") if len(r["body"]) > 200 else r["body"]
            lines.append(
                f"- 🏷️ **Release `{r['tag']}`** ({r['published_at'][:10]}): "
                f"[{r['name']}]({r['url']})"
            )
            if body_preview:
                lines.append(f"  > {body_preview}")
    if tags:
        tag_str = ", ".join(f"`{t['name']}`" for t in tags[:5])
        more = f" (+ {len(tags) - 5} more)" if len(tags) > 5 else ""
        lines.append(f"- 🔖 Recent tags: {tag_str}{more}")
    if commits:
        lines.append(f"- 📝 **{total_commits} commit(s)** since cutoff (top {len(commits)}):")
        for c in commits:
            lines.append(f"  - [`{c['sha']}`]({c['url']}) {c['message']}")
        if total_commits > len(commits):
            lines.append(f"  - … ({total_commits - len(commits)} more)")
    lines.append("")
    return lines


def render(report: list[dict[str, Any]], since: datetime, quiet_repos: list[str]) -> str:
    today = date.today().isoformat()
    lines = [
        f"## GitHub releases watch — {today}",
        "",
        f"Since `{since.date().isoformat()}` across **{len(report) + len(quiet_repos)}** "
        f"tracked repos. **{len(report)}** had activity; "
        f"**{len(quiet_repos)}** were quiet.",
        "",
    ]
    if report:
        lines.append("### Activity")
        lines.append("")
        for r in report:
            why = next((w for repo, w in TRACKED_REPOS if repo == r["repo"]), "")
            lines.extend(render_repo(r, why))
    if quiet_repos:
        lines.append(f"<details><summary>Quiet repos ({len(quiet_repos)})</summary>")
        lines.append("")
        for q in quiet_repos:
            lines.append(f"- {q}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines) + "\n"


# ----- Main -----


def load_repos(extra_json: str | None) -> list[tuple[str, str]]:
    repos = list(TRACKED_REPOS)
    if extra_json:
        extra = json.loads(Path(extra_json).read_text())
        for item in extra:
            if isinstance(item, list) and len(item) == 2:
                repos.append((item[0], item[1]))
            elif isinstance(item, dict) and "repo" in item:
                repos.append((item["repo"], item.get("why", "")))
    return repos


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--since",
        required=True,
        help="ISO date or ISO datetime cutoff for releases / commits.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Append output to this file. Defaults to stdout.",
    )
    p.add_argument(
        "--repos",
        default=None,
        help="Optional JSON file of extra repos to track (list of [repo, why] pairs).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Accept both YYYY-MM-DD and full ISO datetime.
    since_dt = parse_iso(args.since) or parse_iso(args.since + "T00:00:00Z")
    if since_dt is None:
        logger.error("Could not parse --since %r", args.since)
        return 2
    if since_dt.tzinfo is None:
        since_dt = since_dt.replace(tzinfo=timezone.utc)

    use_gh = gh_available()
    if not use_gh:
        logger.warning("gh CLI not available; falling back to unauth GitHub API (rate-limited).")

    repos = load_repos(args.repos)
    logger.info("Tracking %d repos since %s (gh=%s)", len(repos), since_dt.date(), use_gh)

    report: list[dict[str, Any]] = []
    quiet: list[str] = []
    for repo, _why in repos:
        try:
            activity = fetch_repo_activity(repo, since_dt, use_gh)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", repo, e)
            continue
        if activity["releases"] or activity["tags"] or activity["commits"]:
            report.append(activity)
            logger.info(
                "%s: %d releases, %d tags, %d commits",
                repo,
                len(activity["releases"]),
                len(activity["tags"]),
                len(activity["commits"]),
            )
        else:
            quiet.append(repo)
        time.sleep(0.2)  # be polite even with auth

    md = render(report, since_dt, quiet)
    if args.out:
        # Append so the orchestrator can stack sections in one rolling log.
        with open(args.out, "a") as f:
            f.write(md)
        logger.info("Appended GitHub section to %s", args.out)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
