"""Scrape PhenX -> dbGaP variable linkages into the ddharmon benchmark schema.

SPIKE (2026-05-27). The PhenX Toolkit publishes its dbGaP CDE linkages only behind
a CakePHP search portal (https://www.phenxtoolkit.org/vsearch) -- no bulk CSV. This
script does the CSRF handshake, runs a keyword/PhenX-ID/dbGaP-ID search, and parses
the server-rendered results table into clusters:

    PhenX variable (CDE anchor)  <--  N dbGaP variables (cohort members), each tagged
    with a mapping level: identical | comparable | related.

That maps onto our sub-cluster -> CDE-anchor architecture: group rows by `target_id`
(the PhenX variable) to recover a human-curated, multi-cohort equivalence cluster.

Two-hop reality (see todo 2026-04-10):
  * Hop 1 (this script): cluster membership + match level + cohort/study. DONE.
  * Hop 2 (TODO): question text + response options. The CDE-anchor side lives on the
    PhenX protocol page (/protocols/view/<id>, server-rendered, ~270KB); the member
    side lives on the dbGaP variable page (NCBI). Stubbed columns below.

Stdlib only (urllib + re + csv) so it runs without `uv sync`.

Run:  python scripts/scrape_phenx_dbgap.py tobacco "physical activity" alcohol
Out:  data/benchmarks/raw/phenx_dbgap/<slug>.csv   (shared benchmark COLUMNS)
"""

from __future__ import annotations

import csv
import html
import io
import re
import sys
import time
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

BASE = "https://www.phenxtoolkit.org"
UA = "Mozilla/5.0 (ddharmon benchmark spike; contact bhargav.vemuri@phenomehealth.org)"
OUT_DIR = Path("data/benchmarks/raw/phenx_dbgap")

# Aligns with scripts/build_benchmark.py COLUMNS so output slots straight into the
# unified benchmark. source = dbGaP variable (cohort), target = PhenX variable (CDE).
COLUMNS = [
    "source_id",            # phvNNNNNNNN  (dbGaP variable accession)
    "source_label",         # cohort/study display name from vsearch
    "source_description",   # TODO hop-2: dbGaP variable description + value list
    "source_cohort",        # phsNNNNNN    (dbGaP study accession)
    "target_id",            # PhenX variable ID (the CDE anchor / cluster key)
    "target_label",         # PhenX variable name
    "target_description",   # TODO hop-2: PhenX protocol question text + response options
    "target_cohort",        # "PhenX" (the CDE standard)
    "relation",             # identical | comparable | related
    "confidence",           # left blank (human-curated, no score published)
    "mapping_source",       # "PhenX-dbGaP"
    "domain",               # search term used
    "notes",                # protocol name + protocol_url for hop-2 enrichment
]

# Within the "dbGaP With Similar Variables" cell, members are grouped under a bold
# study header (1 study -> many phv variables):
#   <a ...study_id=phs000287>Cardiovascular Health Study (CHS) Cohort</a>
#     <a ...term=phv00104879>phv00104879</a> (comparable)
#     <a ...term=phv00106207>phv00106207</a> (comparable)
# So we tokenize study-headers and member-vars in document order and bind each phv to
# the most recent preceding study.
_TOKEN_RE = re.compile(
    r"study_id=(?P<phs>phs\d+)\"[^>]*>(?P<cohort>[^<]+)</a>"
    r"|term=(?P<phv>phv\d+)\"[^>]*>[^<]*</a>\s*\((?P<level>identical|comparable|related)\)",
    re.I,
)


def _open(opener: urllib.request.OpenerDirector, url: str, data: bytes | None = None) -> str:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    with opener.open(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def _session() -> tuple[urllib.request.OpenerDirector, str]:
    """GET /vsearch to seed the CSRF cookie and return (opener, form token)."""
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    html = _open(opener, f"{BASE}/vsearch")
    m = re.search(r'name="_csrfToken"[^>]*?value="([^"]+)"', html)
    if not m:
        raise RuntimeError("could not locate _csrfToken on /vsearch")
    return opener, m.group(1)


def search(opener: urllib.request.OpenerDirector, token: str, term: str) -> str:
    payload = urllib.parse.urlencode({"_csrfToken": token, "searchTerm": term}).encode()
    return _open(opener, f"{BASE}/vsearch/results", data=payload)


def _fetch_csv(opener: urllib.request.OpenerDirector, path: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(_open(opener, BASE + path))))


def protocol_questions(opener: urllib.request.OpenerDirector, pid: str) -> dict[str, str]:
    """/protocols/export/<pid> -> {PhenX Variable ID: question text}. The CDE-anchor text."""
    return {
        r["Variable ID"]: r["Variable Description"]
        for r in _fetch_csv(opener, f"/protocols/export/{pid}")
        if r.get("Variable ID")
    }


def internal_id_map(opener: urllib.request.OpenerDirector, pid: str) -> dict[str, str]:
    """Parse the protocol page once: {PhenX Variable ID: internal id} for export-mapping."""
    page = _open(opener, f"{BASE}/protocols/view/{pid}")
    return dict(re.findall(r"modal-(PX\d+).*?export-mapping/(\d+)", page, re.S))


def member_descriptions(opener: urllib.request.OpenerDirector, internal_id: str) -> dict[str, str]:
    """/protocols/export-mapping/<internal_id> -> {phv: dbGaP variable description}."""
    return {
        r["dbGAP Variable"]: r["dbGAP Variable Description"]
        for r in _fetch_csv(opener, f"/protocols/export-mapping/{internal_id}")
        if r.get("dbGAP Variable")
    }


def enrich(opener: urllib.request.OpenerDirector, rows: list[dict]) -> None:
    """Hop 2: fill target_description (anchor question text) and source_description
    (dbGaP member description) in place, from PhenX's CSV export endpoints. Caches per
    protocol / per anchor so nothing is fetched twice. Response options are NOT in these
    exports -- that's hop 3 (dbGaP var_report)."""
    q_cache: dict[str, dict[str, str]] = {}
    iid_cache: dict[str, dict[str, str]] = {}
    member_cache: dict[str, dict[str, str]] = {}
    for r in rows:
        pm = re.search(r"/protocols/view/(\d+)", r["notes"])
        if not pm:
            continue
        pid = pm.group(1)
        if pid not in q_cache:
            q_cache[pid] = protocol_questions(opener, pid)
            iid_cache[pid] = internal_id_map(opener, pid)
            time.sleep(1.0)
        r["target_description"] = q_cache[pid].get(r["target_id"], "")
        iid = iid_cache[pid].get(r["target_id"])
        if iid and iid not in member_cache:
            member_cache[iid] = member_descriptions(opener, iid)
            time.sleep(1.0)
        if iid:
            r["source_description"] = member_cache[iid].get(r["source_id"], "")


def _strip(markup: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", markup)).strip())


def parse(results_html: str, domain: str) -> list[dict]:
    """Parse the vsearch results table into one row per (dbGaP member -> PhenX anchor)."""
    i = results_html.find('id="table-vsearch-results"')
    if i < 0:
        return []
    table = results_html[i : results_html.find("</table>", i)]
    rows: list[dict] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
        cells = re.findall(r"<td.*?>(.*?)</td>", tr, re.S)
        if len(cells) < 4:
            continue  # header / spacer
        phenx_id = _strip(cells[0])
        phenx_name = _strip(cells[1])
        protocol_name = _strip(cells[2])
        pm = re.search(r'href="(/protocols/view/[^"#]+)', cells[2])
        protocol_url = BASE + pm.group(1) if pm else ""
        member_blob = cells[3]
        cur_phs, cur_cohort = "", ""
        for m in _TOKEN_RE.finditer(member_blob):
            if m.group("phs"):
                cur_phs, cur_cohort = m.group("phs"), _strip(m.group("cohort"))
                continue
            rows.append(
                {
                    "source_id": m.group("phv"),
                    "source_label": cur_cohort,
                    "source_description": "",  # hop-2
                    "source_cohort": cur_phs,
                    "target_id": phenx_id,
                    "target_label": phenx_name,
                    "target_description": "",  # hop-2
                    "target_cohort": "PhenX",
                    "relation": m.group("level").lower(),
                    "confidence": "",
                    "mapping_source": "PhenX-dbGaP",
                    "domain": domain,
                    "notes": f"protocol={protocol_name}; protocol_url={protocol_url}",
                }
            )
    return rows


def main(terms: list[str], do_enrich: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    opener, token = _session()
    for term in terms:
        rows = parse(search(opener, token, term), term)
        if do_enrich:
            enrich(opener, rows)
        slug = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
        out = OUT_DIR / f"{slug}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        clusters = len({r["target_id"] for r in rows})
        cohorts = len({r["source_cohort"] for r in rows})
        multi = sum(
            1
            for t in {r["target_id"] for r in rows}
            if len({r["source_cohort"] for r in rows if r["target_id"] == t}) >= 3
        )
        print(
            f"{term:>20}: {len(rows):4d} pairs | {clusters:3d} PhenX anchors "
            f"| {cohorts:3d} studies | {multi:3d} anchors w/ >=3 cohorts -> {out}"
        )
        time.sleep(1.0)  # be polite to RTI's server


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-enrich"]
    main(args or ["tobacco"], do_enrich="--no-enrich" not in sys.argv)
