#!/usr/bin/env python3
"""One-time citation-neighborhood backfill over competitive_landscape.md.

For each DOI / arXiv ID / PMCID / PMID extracted from
``docs/competitive_landscape.md`` (and any ``--extra`` markdown files),
fetch the 1-hop citation neighborhood via OpenAlex — works *cited by* the
seed and works that *cite* the seed — then diff against the already-tracked
set and write a triage markdown listing what's new.

Use case: catch the historical-paper miss class (e.g., Pan et al. 2022 sat
4 years in the citation neighborhood of CDEMapper / Krishnamurthy / PhenX
without us indexing it). The forward-time release watcher won't catch
these; this script does, in one pass.

Stdlib only. OpenAlex is a free public API; no auth required. The
``--mailto`` argument routes us to OpenAlex's polite pool (faster).

Usage::

    python scripts/citation_backfill.py
    python scripts/citation_backfill.py --extra .planning/todos/pending/*.md
    python scripts/citation_backfill.py --limit-seeds 5  # smoke test
    python scripts/citation_backfill.py --dry-run -v     # log only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

logger = logging.getLogger("citation_backfill")

OPENALEX = "https://api.openalex.org"
DEFAULT_MAILTO = "noreply@phenomehealth.org"

# Identifier extractors. Trailing punctuation is stripped per-match.
RE_DOI = re.compile(r"\b10\.\d{4,9}/[^\s)\],]+", re.IGNORECASE)
RE_ARXIV = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arXiv[:\s]+)(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)
RE_PMC = re.compile(r"\bPMC(\d{4,})\b")
RE_PMID = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


@dataclass(frozen=True)
class Seed:
    """A paper identifier we can hand to OpenAlex's /works/<id> endpoint."""

    kind: str  # "doi" | "arxiv" | "pmcid" | "pmid"
    value: str

    @property
    def openalex_lookup(self) -> str:
        if self.kind == "doi":
            return f"doi:{self.value}"
        if self.kind == "arxiv":
            # OpenAlex indexes arXiv preprints under their DOI shadow.
            return f"doi:10.48550/arXiv.{self.value.split('v')[0]}"
        if self.kind == "pmcid":
            return f"pmcid:{self.value}"
        if self.kind == "pmid":
            return f"pmid:{self.value}"
        raise ValueError(self.kind)


def extract_seeds(text: str) -> set[Seed]:
    """Pull every paper identifier we can find from a blob of markdown."""
    out: set[Seed] = set()
    for m in RE_DOI.finditer(text):
        doi = m.group(0).rstrip(".,;:)]")
        # arXiv DOIs are captured separately so they share canonical lookup.
        if doi.lower().startswith("10.48550/arxiv."):
            continue
        # medRxiv / bioRxiv DOIs carry `v1` / `v2` revision suffixes that
        # aren't part of the canonical form OpenAlex indexes — strip them.
        # Match optional preceding `/` to handle Research Square's
        # `<doi>/v1` form without leaving a trailing slash that breaks lookup.
        doi = re.sub(r"/?v\d+$", "", doi)
        out.add(Seed("doi", doi.lower()))
    for m in RE_ARXIV.finditer(text):
        out.add(Seed("arxiv", m.group(1)))
    for m in RE_PMC.finditer(text):
        out.add(Seed("pmcid", f"PMC{m.group(1)}"))
    for m in RE_PMID.finditer(text):
        out.add(Seed("pmid", m.group(1)))
    return out


def openalex_get(path: str, mailto: str) -> dict[str, Any] | None:
    sep = "&" if "?" in path else "?"
    url = f"{OPENALEX}{path}{sep}mailto={quote(mailto)}"
    req = urllib.request.Request(
        url, headers={"User-Agent": f"citation-backfill ({mailto})"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code != 404:
            logger.warning("OpenAlex HTTP %s for %s", e.code, url)
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning("OpenAlex %s: %s", url, e)
        return None


def resolve(seed: Seed, mailto: str) -> dict[str, Any] | None:
    return openalex_get(f"/works/{seed.openalex_lookup}", mailto)


def short_id(openalex_url: str) -> str:
    return openalex_url.rsplit("/", 1)[-1]


def fetch_refs(work: dict[str, Any], mailto: str) -> list[dict[str, Any]]:
    """Works that this seed cites (its references)."""
    refs = work.get("referenced_works") or []
    if not refs:
        return []
    out: list[dict[str, Any]] = []
    for i in range(0, len(refs), 50):
        ids = "|".join(short_id(r) for r in refs[i : i + 50])
        page = openalex_get(f"/works?filter=ids.openalex:{ids}&per-page=50", mailto)
        if page:
            out.extend(page.get("results", []))
        time.sleep(0.15)
    return out


def fetch_citers(work_id: str, mailto: str, max_results: int) -> list[dict[str, Any]]:
    """Works that cite this seed (forward citations)."""
    wid = short_id(work_id)
    out: list[dict[str, Any]] = []
    cursor: str | None = "*"
    while cursor and len(out) < max_results:
        page = openalex_get(
            f"/works?filter=cites:{wid}&per-page=50&cursor={quote(cursor)}", mailto
        )
        if not page:
            break
        out.extend(page.get("results", []))
        cursor = (page.get("meta") or {}).get("next_cursor")
        time.sleep(0.15)
    return out[:max_results]


def first_author(work: dict[str, Any]) -> str:
    auths = work.get("authorships") or []
    if not auths:
        return "?"
    name = (auths[0].get("author") or {}).get("display_name") or "?"
    return name + (" et al." if len(auths) > 1 else "")


def venue(work: dict[str, Any]) -> str:
    src = (work.get("primary_location") or {}).get("source") or {}
    return src.get("display_name") or "?"


# v1.1 noise filter -----
#
# The v1.0 report was 95% field-staple infrastructure papers (NAR database
# issues, foundational embedding/ontology refs) that get cited by everything.
# Two cheap filters cut the noise without dropping real signal:
#
#   1. **Venue blocklist** — papers in venues that are *exclusively*
#      database/infrastructure refs (Nucleic Acids Research) are dropped.
#      Default is conservative; user extends via repeated --blocklist-venue.
#
#   2. **Topic overlap** — OpenAlex tags each work with up to 3 topics. We
#      drop findings whose top-3 topics don't overlap the *union* of top-3
#      topics across seed papers. Same-field staples stay; off-field
#      neighbors (database curation, protein function annotation, drug
#      discovery) drop out.
#
# Both can be disabled with --no-filter (smoke testing / sanity check that
# we're not over-filtering).
VENUE_BLOCKLIST_DEFAULT = ("nucleic acids research",)


def normalize_venue(work: dict[str, Any]) -> str:
    src = (work.get("primary_location") or {}).get("source") or {}
    return (src.get("display_name") or "").strip().lower()


def topics_of(work: dict[str, Any], k: int = 3) -> set[str]:
    topics = work.get("topics") or []
    return {
        (t.get("display_name") or "").strip().lower()
        for t in topics[:k]
        if t.get("display_name")
    }


def primary_topic_of(work: dict[str, Any]) -> str:
    """The most specific topic OpenAlex assigns to this work."""
    pt = work.get("primary_topic") or {}
    return (pt.get("display_name") or "").strip().lower()


def publication_date(work: dict[str, Any]) -> str:
    """ISO date string; falls back to publication_year + '-01-01' when only year is set."""
    pd = work.get("publication_date")
    if pd:
        return str(pd)
    py = work.get("publication_year")
    return f"{py}-01-01" if py else ""


def filter_reason(
    work: dict[str, Any],
    seed_topics_top: set[str],
    seed_topics_primary: set[str],
    venue_blocklist: set[str],
    mode: str,
) -> str | None:
    """Return a short reason string if the work should be filtered out, else None.

    ``mode='strict'`` (used for delta runs): require the finding's *primary*
    topic to be in the seed *primary*-topic union — drops same-field-but-off-
    target papers (SBERT, DrugBank) that share a broad parent topic with our
    seeds but aren't actually doing harmonization.

    ``mode='loose'`` (used for one-time backfill): keep findings whose top-3
    topics intersect the seed top-3 topic union — more recall, more noise.
    """
    if normalize_venue(work) in venue_blocklist:
        return "venue blocklist"
    if mode == "strict":
        if seed_topics_primary:
            pt = primary_topic_of(work)
            if pt and pt not in seed_topics_primary:
                return "primary-topic mismatch"
    else:
        if seed_topics_top and not (topics_of(work) & seed_topics_top):
            return "topic mismatch"
    return None


def render(
    seeds: dict[str, dict[str, Any]],
    findings: dict[str, list[tuple[str, str]]],
    filtered: dict[str, tuple[list[tuple[str, str]], str]],
    works: dict[str, dict[str, Any]],
    seed_topics: set[str],
    venue_blocklist: set[str],
    out_path: Path,
) -> str:
    today = date.today().isoformat()
    lines: list[str] = [
        f"# Competitive citation-neighborhood backfill — {today}",
        "",
        "Generated by `scripts/citation_backfill.py`. One-time pass over "
        "identifiers extracted from `docs/competitive_landscape.md` "
        "(+ any `--extra` files). Each finding lists the seed papers — "
        "already-tracked entries — that cite it or are cited by it.",
        "",
        "## Summary",
        "",
        f"- Seed papers resolved to OpenAlex: **{len(seeds)}**",
        f"- Findings passing noise filter: **{len(findings)}**",
        f"- Findings filtered as noise (see appendix): **{len(filtered)}**",
        f"- Venue blocklist: {sorted(venue_blocklist) or '(none)'}",
        f"- Seed topic set ({len(seed_topics)}): {sorted(seed_topics)[:6]}"
        + (" …" if len(seed_topics) > 6 else ""),
        "",
        "Triage tip: items with many **seed connections** are cross-cited "
        "by multiple known competitors — strongest signal we should track "
        "them. High **cited-by** counts mark influential work in the "
        "broader field; eyeball whether they belong in "
        "`competitive_landscape.md`.",
        "",
        "## Findings (sorted by seed-connections desc, then cited-by desc)",
        "",
    ]

    if not findings:
        lines.append("_No findings passed the filter. "
                     "(Likely OpenAlex citation-graph lag for recent papers — "
                     "papers published after the cutoff haven't yet been "
                     "indexed as citing our seeds. Common on the first few "
                     "weekly runs.)_")
        lines.append("")

    def sort_key(item: tuple[str, list[tuple[str, str]]]) -> tuple[int, int]:
        wid, conns = item
        return (-len(conns), -(works.get(wid, {}).get("cited_by_count") or 0))

    for wid, conns in sorted(findings.items(), key=sort_key):
        w = works.get(wid, {})
        title = w.get("title") or "(untitled)"
        year = w.get("publication_year") or "?"
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        link = w.get("doi") or wid
        cby = w.get("cited_by_count")
        lines.append(f"### {title}")
        lines.append("")
        cite_link = f"· DOI [{doi}]({link})" if doi else f"· [{link}]({link})"
        lines.append(f"- **{first_author(w)}**, {year}, *{venue(w)}*  {cite_link}")
        lines.append(
            f"- Cited by: {cby if cby is not None else '?'} · Seed connections: {len(conns)}"
        )
        if conns:
            seed_strs = ", ".join(f"{short_id(sid)} ({rel})" for sid, rel in conns)
            lines.append(f"- Surfaced via: {seed_strs}")
        lines.append("")

    if filtered:
        lines.append("---")
        lines.append("")
        lines.append("## Appendix — filtered as noise")
        lines.append("")
        lines.append(
            "Collapsed list. Most are field-staple databases / ontologies / "
            "method primitives that are universal background cites — useful "
            "awareness, not actionable competitive signal. Re-run with "
            "`--no-filter` to expand."
        )
        lines.append("")

        def filt_key(item: tuple[str, tuple[list[tuple[str, str]], str]]) -> tuple[str, int, int]:
            wid, (conns, reason) = item
            return (reason, -len(conns), -(works.get(wid, {}).get("cited_by_count") or 0))

        last_reason: str | None = None
        for wid, (conns, reason) in sorted(filtered.items(), key=filt_key):
            if reason != last_reason:
                lines.append(f"### Filtered: {reason}")
                lines.append("")
                last_reason = reason
            w = works.get(wid, {})
            title = w.get("title") or "(untitled)"
            year = w.get("publication_year") or "?"
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            link = w.get("doi") or wid
            lines.append(
                f"- *{title}* — {first_author(w)}, {year}, {venue(w)} "
                f"({len(conns)} seed conn) "
                + (f"[{doi}]({link})" if doi else f"[link]({link})")
            )
        lines.append("")

    content = "\n".join(lines) + "\n"
    out_path.write_text(content)
    return content


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--landscape", default="docs/competitive_landscape.md")
    p.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra markdown files to also scan for seed identifiers.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output markdown. Defaults to docs/competitive_watch_backfill_<today>.md.",
    )
    p.add_argument("--mailto", default=DEFAULT_MAILTO)
    p.add_argument(
        "--max-citing",
        type=int,
        default=200,
        help="Cap forward-citation results per seed (some papers have thousands).",
    )
    p.add_argument(
        "--limit-seeds",
        type=int,
        default=0,
        help="Smoke-test mode: process only the first N seeds.",
    )
    p.add_argument(
        "--blocklist-venue",
        action="append",
        default=None,
        help=(
            "Drop findings published in this venue (case-insensitive). "
            "Repeatable. Default: %s." % (list(VENUE_BLOCKLIST_DEFAULT),)
        ),
    )
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip the noise filter (venue blocklist + topic overlap).",
    )
    p.add_argument(
        "--filter-mode",
        choices=("strict", "loose"),
        default="loose",
        help=(
            "strict = primary-topic-only overlap (tighter, for delta/weekly runs); "
            "loose = top-3 topic overlap (default, for one-time backfill)."
        ),
    )
    p.add_argument(
        "--since",
        default=None,
        help=(
            "ISO date (YYYY-MM-DD). Delta mode: only emit *forward* citations "
            "(papers that newly cite seeds) published on/after this date; drop "
            "seed→reference back-citations entirely. Use for weekly runs."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--dry-run", action="store_true", help="Log progress; do not write output."
    )
    args = p.parse_args()
    venue_blocklist: set[str] = set(
        v.strip().lower()
        for v in (args.blocklist_venue if args.blocklist_venue is not None else VENUE_BLOCKLIST_DEFAULT)
    )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    paths = [Path(args.landscape), *(Path(x) for x in args.extra)]
    # Fix C (2026-05-28): the silent `if p.exists()` skip + zero-seeds-but-exit-0
    # behavior caused two remote runs to produce empty watch sections with no
    # diagnostic signal in the committed markdown. Now: log loud AND fail with
    # exit code 3 if zero seeds extracted, so the orchestrator surfaces the
    # diagnosis (stderr → "Watcher failed: ..." in the markdown).
    missing: list[Path] = []
    for p in paths:
        if not p.exists():
            missing.append(p)
            logger.warning(
                "Seed file does not exist (silently skipped): %s "
                "(cwd=%s, resolved=%s)",
                p, Path.cwd(), p.resolve(),
            )
    text = "\n".join(p.read_text(errors="ignore") for p in paths if p.exists())
    seeds = sorted(extract_seeds(text), key=lambda s: (s.kind, s.value))
    if not seeds:
        # Fatal diagnostic: dump everything that might explain a zero-seed run.
        # All emitted to stderr so the orchestrator's subprocess-failure path
        # writes them into the markdown for post-hoc inspection.
        diag = [
            "ERROR: zero seed identifiers extracted from input file(s).",
            f"  Arguments: landscape={args.landscape!r} extra={args.extra!r}",
            f"  CWD: {Path.cwd()}",
            f"  Resolved paths:",
        ]
        for p in paths:
            diag.append(
                f"    - {p} -> {p.resolve()} (exists={p.exists()}, "
                f"size={p.stat().st_size if p.exists() else '?'})"
            )
        if missing:
            diag.append(f"  Missing: {[str(m) for m in missing]}")
        if text:
            sample = text[:300].replace("\n", " ⏎ ")
            diag.append(
                f"  Read {len(text)} chars total; first 300: {sample!r}"
            )
        else:
            diag.append("  Read 0 chars total (no files existed or all were empty).")
        for line in diag:
            print(line, file=sys.stderr)
        return 3
    if args.limit_seeds:
        seeds = seeds[: args.limit_seeds]
    by_kind = {k: sum(1 for s in seeds if s.kind == k) for k in ("doi", "arxiv", "pmcid", "pmid")}
    logger.info("Extracted %d seed identifiers (%s)", len(seeds), by_kind)

    resolved: dict[str, dict[str, Any]] = {}
    for s in seeds:
        w = resolve(s, args.mailto)
        if w and w.get("id"):
            resolved[w["id"]] = w
            logger.debug("Resolved %s:%s -> %s", s.kind, s.value, short_id(w["id"]))
        else:
            logger.info("Unresolved seed %s:%s", s.kind, s.value)
        time.sleep(0.1)
    logger.info("Resolved %d / %d seeds", len(resolved), len(seeds))

    # Compute seed-topic sets for filtering. We track both top-3 and primary
    # so --filter-mode can pick the right tightness.
    seed_topics_top: set[str] = set()
    seed_topics_primary: set[str] = set()
    if not args.no_filter:
        for w in resolved.values():
            seed_topics_top.update(topics_of(w))
            pt = primary_topic_of(w)
            if pt:
                seed_topics_primary.add(pt)
        logger.info(
            "Seed primary topics (%d): %s",
            len(seed_topics_primary),
            sorted(seed_topics_primary),
        )

    # In --since mode, drop seed→reference back-citations entirely (old by
    # construction) and only keep forward citations newer than the cutoff.
    delta_mode = args.since is not None
    if delta_mode:
        logger.info("Delta mode: --since=%s (back-citations dropped)", args.since)

    findings: dict[str, list[tuple[str, str]]] = {}
    filtered: dict[str, tuple[list[tuple[str, str]], str]] = {}
    all_works: dict[str, dict[str, Any]] = dict(resolved)

    def add_finding(wid: str, work: dict[str, Any], seed_id: str, rel: str) -> None:
        if wid in resolved:
            return
        all_works[wid] = work

        # --since handling: drop back-citations; gate forward citations by date.
        if delta_mode:
            if rel == "cited by":
                return  # seed→ref is by definition old; not new signal
            pd = publication_date(work)
            if pd and pd < args.since:
                conns, _ = filtered.get(wid, ([], "older than --since"))
                conns.append((seed_id, rel))
                filtered[wid] = (conns, "older than --since")
                return

        if args.no_filter:
            findings.setdefault(wid, []).append((seed_id, rel))
            return

        reason = filter_reason(
            work, seed_topics_top, seed_topics_primary, venue_blocklist, args.filter_mode
        )
        if reason is None:
            findings.setdefault(wid, []).append((seed_id, rel))
        else:
            conns, _ = filtered.get(wid, ([], reason))
            conns.append((seed_id, rel))
            filtered[wid] = (conns, reason)

    for i, (seed_id, seed) in enumerate(resolved.items(), 1):
        # In delta mode we skip the references API call entirely — saves time
        # and bandwidth since we'd drop them all anyway.
        refs = [] if delta_mode else fetch_refs(seed, args.mailto)
        citers = fetch_citers(seed_id, args.mailto, args.max_citing)
        for r in refs:
            rid = r.get("id")
            if rid:
                add_finding(rid, r, seed_id, "cited by")
        for c in citers:
            cid = c.get("id")
            if cid:
                add_finding(cid, c, seed_id, "cites")
        logger.info(
            "[%d/%d] %s: %d refs, %d citers (passed: %d, filtered: %d)",
            i,
            len(resolved),
            short_id(seed_id),
            len(refs),
            len(citers),
            len(findings),
            len(filtered),
        )

    out_path = Path(args.out) if args.out else Path(
        f"docs/competitive_watch_backfill_{date.today().isoformat()}.md"
    )
    if args.dry_run:
        logger.info(
            "--dry-run: would write %s (passed: %d, filtered: %d)",
            out_path, len(findings), len(filtered),
        )
        return 0
    # Render uses the topic set actually active for filtering (so the report
    # accurately reflects what the filter saw).
    active_seed_topics = seed_topics_primary if args.filter_mode == "strict" else seed_topics_top
    render(resolved, findings, filtered, all_works, active_seed_topics, venue_blocklist, out_path)
    logger.info(
        "Wrote %s (passed: %d, filtered: %d)",
        out_path, len(findings), len(filtered),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
