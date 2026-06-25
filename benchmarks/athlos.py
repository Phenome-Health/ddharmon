"""Benchmark C — ATHLOS gold: value-recode / transform-spec correctness.

"Are the value-recode specs generated correctly?" — mapping a source variable's coded response options
into a canonical target variable's value set (the transform layer), NOT variable/CDE matching (that's
Benchmarks A/B). Ground truth: the ATHLOS ageing-project harmonisation scripts (Maelstrom-coordinated;
github.com/athlosproject/athlos-project.github.io, AGPL-3) — ~1,900 per-variable `.Rmd` scripts, each
carrying the SOURCE coded value set + TARGET harmonised value set + the exact `car::recode` algorithm.
No turnkey value-recode benchmark exists publicly; ATHLOS is the open exception (survey/epi domain).

This is the $0, portable, reproducible gold builder + scorer. It parses the scripts into a
source-code -> target-code gold and scores predicted recode mappings (oracle self-check + identity and
label-similarity baselines). The LLM recode-generator arm (give the model source+target value sets ->
predicted mapping, the real number) lives in the sandbox (`bench_athlos.py --llm`) — it needs API keys,
so it is not part of the $0 gate. Scope = clean CATEGORICAL recodes; continuous / quantile / derived /
multi-response transforms are detected and skipped.

  PYTHONHASHSEED=0 python -m benchmarks.athlos
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from benchmarks import _common as common

# --- regexes over the .Rmd text -------------------------------------------------

# Harmonisation transform line, tolerant of trailing args + computed source exprs:
#   t..$marital_status <- car::recode(t..$PMARRY, "3=4; 4=3; NA=999")
RE_RECODE = re.compile(r'\$(?P<tgt>\w+)\s*<-\s*car::recode\(\s*(?P<srcexpr>[^,]+?)\s*,\s*"(?P<rec>[^"]*)"')
# Target value set:  ..$var <- labelled(..$var, labels = c("single"=1, "widow"=4, ...))
RE_LABELLED = re.compile(r"\$(?P<tgt>\w+)\s*<-\s*labelled\([^,]+,\s*labels\s*=\s*c\((?P<labs>[^)]*)\)")
RE_LABEL_PAIR = re.compile(r'"([^"]*)"\s*=\s*(-?\d+)')
RE_CAT_ROW = re.compile(r"\|\s*\*\*Categories\*\*\s*\|(?P<cell>.*)")  # table-format Categories header row
RE_CAT_PAIR = re.compile(r"(-?\d+)\s*=\s*([^;]+)")  # code=label pair (after normalising separators to ;)
RE_CAT_BULLET_HDR = re.compile(r"^\s*\*\s*Categories\s*:")
RE_CAT_BULLET_ITEM = re.compile(r"^\s*\+\s*`\s*(-?\d+)\s*=\s*([^`]*)`")
RE_TABLE_CONT = re.compile(r"^\|\s*\|")  # multi-row table continuation (empty first cell)
RE_TABLE_FIELD = re.compile(r"\|\s*\*\*")  # a |**Field**| row ends a Categories block


# --- car::recode mini-language --------------------------------------------------


@dataclass
class RecodeSpec:
    pairs: list = field(default_factory=list)  # (matcher_fn, target_code)
    na_target: object = None
    else_target: object = None
    has_na: bool = False
    has_else: bool = False
    literal: bool = True  # False if any lhs/rhs references runtime vars / arithmetic / indexing


def _parse_rhs(rhs: str):
    rhs = rhs.strip()
    if rhs.upper() == "NA":
        return "NA"
    m = re.fullmatch(r"['\"]?(-?\d+)['\"]?", rhs)
    return int(m.group(1)) if m else rhs.strip("'\"")


def _make_matcher(lhs: str):
    """Return (predicate code->bool, is_literal) for one car::recode lhs token (categorical scope)."""
    lhs = lhs.strip()
    m = re.fullmatch(r"(-?\d+)\s*:\s*(-?\d+)", lhs)  # integer range
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lambda c, lo=lo, hi=hi: isinstance(c, int) and lo <= c <= hi), True
    m = re.fullmatch(r"c\((.*)\)", lhs)  # c(a,b,c)
    if m:
        toks = [t.strip().strip("'\"") for t in m.group(1).split(",")]
        if all(re.fullmatch(r"-?\d+", t) for t in toks if t):
            vals = {int(t) for t in toks if t}
            return (lambda c, vals=vals: c in vals), True
        vals = set(toks)
        return (lambda c, vals=vals: str(c) in vals), True
    if re.fullmatch(r"-?\d+", lhs):  # bare int
        v = int(lhs)
        return (lambda c, v=v: c == v), True
    if re.fullmatch(r"""['"][^'"]*['"]""", lhs):  # quoted string literal
        s = lhs.strip("'\"")
        return (lambda c, s=s: str(c) == s), True
    return (lambda c: False), False  # non-literal (qq[..], arithmetic, lo:/:hi, ...)


def _rhs_literal(rhs: str) -> bool:
    rhs = rhs.strip()
    return rhs.upper() == "NA" or bool(re.fullmatch(r"""['"]?-?\d+['"]?|['"][^'"]*['"]""", rhs))


def parse_recode(spec_str: str) -> RecodeSpec:
    out = RecodeSpec()
    for chunk in spec_str.split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        lhs, rhs = (s.strip() for s in chunk.split("=", 1))
        if not _rhs_literal(rhs):
            out.literal = False
        tgt = _parse_rhs(rhs)
        if lhs.upper() == "NA":
            out.na_target, out.has_na = tgt, True
        elif lhs.lower() == "else":
            out.else_target, out.has_else = tgt, True
        else:
            matcher, lit = _make_matcher(lhs)
            out.literal = out.literal and lit
            out.pairs.append((matcher, tgt))
    return out


def apply_recode(spec: RecodeSpec, code):
    """Map one source code (int, or the string 'NA') to its target per car::recode semantics."""
    if code == "NA":
        if spec.has_na:
            return spec.na_target
        return spec.else_target if spec.has_else else "NA"
    for matcher, tgt in spec.pairs:
        if matcher(code):
            return tgt
    return spec.else_target if spec.has_else else code  # unmatched pass-through


# --- parse one .Rmd into recode entries ----------------------------------------


@dataclass
class GoldEntry:
    target_var: str
    source_var: str
    source_values: dict  # {code(str): label}
    target_values: dict  # {code(str): label}
    recode: str
    gold_map: dict  # {source_code(str): target_code(str)}
    nonidentity: bool
    uid: str = ""
    instances: int = 1
    cohorts: list = field(default_factory=list)


def _parse_cat_cell(cell: str) -> dict:
    cell = cell.replace("<br/>", ";").replace("<br>", ";").replace("`", ";")
    return {c: lbl.strip() for c, lbl in RE_CAT_PAIR.findall(cell)}


def _nearest_preceding_categories(lines, idx):
    """Source value set from the nearest preceding Categories block (table row, multi-row
    continuation, or `* Categories:` bullets). Guard: stop if we cross a previous car::recode
    (this section has no Categories of its own -> don't borrow the previous section's)."""
    for j in range(idx - 1, -1, -1):
        if RE_RECODE.search(lines[j]):
            return None
        m = RE_CAT_ROW.search(lines[j])
        if m:
            cats = _parse_cat_cell(m.group("cell"))
            for k in range(j + 1, min(j + 60, len(lines))):
                lk = lines[k]
                if RE_TABLE_CONT.match(lk) and not RE_TABLE_FIELD.search(lk):
                    cats.update(_parse_cat_cell(lk))
                elif lk.strip() == "":
                    continue
                else:
                    break
            return cats or None
        if RE_CAT_BULLET_HDR.match(lines[j]):
            cats = {}
            for k in range(j + 1, min(j + 80, len(lines))):
                mb = RE_CAT_BULLET_ITEM.match(lines[k])
                if mb:
                    cats[mb.group(1)] = mb.group(2).strip()
                elif lines[k].strip() == "" or lines[k].lstrip().startswith("+"):
                    continue
                else:
                    break
            return cats or None
    return None


def _labelled_near(lines, idx, target_var):
    for j in range(idx, min(idx + 10, len(lines))):
        m = RE_LABELLED.search(lines[j])
        if m and m.group("tgt") == target_var:
            return {code: lbl for lbl, code in RE_LABEL_PAIR.findall(m.group("labs"))}
    return None


def parse_rmd(path: Path, root: Path):
    """Yield (kind, payload) per car::recode section in a file; kind is 'ok' or 'skip'."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cohort = path.relative_to(root).parts[0] if path.is_relative_to(root) else path.parent.name
    out = []
    for i, line in enumerate(lines):
        rm = RE_RECODE.search(line)
        if not rm:
            continue
        tgt, rec = rm.group("tgt"), rm.group("rec")
        sm = re.search(r"\$(\w+)", rm.group("srcexpr"))
        src = sm.group(1) if sm else rm.group("srcexpr").strip()
        spec = parse_recode(rec)
        if not spec.literal:
            out.append(("skip", {"reason": "nonliteral_recode"}))
            continue
        src_cats = _nearest_preceding_categories(lines, i)
        tgt_vals = _labelled_near(lines, i, tgt)
        if not src_cats or not tgt_vals:
            out.append(("skip", {"reason": "missing_src_or_tgt_valueset"}))
            continue
        orphan = [
            c
            for c, t in re.findall(r"(?<![\w.:])(-?\d+)\s*=\s*'?(-?\d+)'?", rec)
            if int(t) < 900 and -10 <= int(c) <= 90 and c not in src_cats
        ]
        if orphan:
            out.append(("skip", {"reason": "incomplete_source_valueset"}))
            continue
        source_codes = [int(c) for c in src_cats] + (["NA"] if spec.has_na else [])
        gold_map, real_oor = {}, False
        for c in source_codes:
            t = apply_recode(spec, c)
            if t == "NA" or not isinstance(t, int):
                continue
            if t < 900 and str(t) not in tgt_vals:
                real_oor = True
            gold_map[str(c)] = str(t)
        if not gold_map:
            out.append(("skip", {"reason": "empty_gold_map"}))
            continue
        if real_oor:
            out.append(("skip", {"reason": "target_out_of_range"}))  # multi-response/derivation source
            continue
        nonident = any(sc != "NA" and gold_map[sc] != sc for sc in gold_map)
        out.append(
            (
                "ok",
                {
                    "target_var": tgt,
                    "source_var": src,
                    "source_values": src_cats,
                    "target_values": tgt_vals,
                    "recode": rec.strip(),
                    "gold_map": gold_map,
                    "nonidentity": nonident,
                    "cohort": cohort,
                },
            )
        )
    return out


def build_gold(root: Path) -> tuple[list, Counter]:
    raw_ok, skips = [], Counter()
    for p in sorted(root.rglob("*.Rmd")):
        for kind, payload in parse_rmd(p, root):
            if kind == "ok":
                raw_ok.append(payload)
            else:
                skips[payload["reason"]] += 1

    dedup: dict = {}
    for r in raw_ok:
        sig = (r["target_var"], tuple(sorted(r["source_values"].items())), re.sub(r"\s+", "", r["recode"]))
        if sig in dedup:
            dedup[sig].instances += 1
            if r["cohort"] not in dedup[sig].cohorts:
                dedup[sig].cohorts.append(r["cohort"])
        else:
            uid = hashlib.sha1(
                json.dumps(
                    [r["target_var"], sorted(r["source_values"].items()), sorted(r["target_values"].items())],
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:12]
            dedup[sig] = GoldEntry(
                target_var=r["target_var"],
                source_var=r["source_var"],
                source_values=r["source_values"],
                target_values=r["target_values"],
                recode=r["recode"],
                gold_map=r["gold_map"],
                nonidentity=r["nonidentity"],
                uid=uid,
                cohorts=[r["cohort"]],
            )
    return list(dedup.values()), skips


# --- $0 baselines + scorer ------------------------------------------------------


def _ratio(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def baseline_label_sim(e: GoldEntry) -> dict:
    """Naive recode: map each source code to the target code with the most similar label."""
    tgt_items = list(e.target_values.items())
    pred = {sc: max(tgt_items, key=lambda kv: _ratio(slab, kv[1]))[0] for sc, slab in e.source_values.items()}
    if "NA" in e.gold_map:
        miss = [c for c, lbl in e.target_values.items() if "missing" in lbl.lower()]
        pred["NA"] = miss[0] if miss else max(tgt_items, key=lambda kv: _ratio("missing", kv[1]))[0]
    return pred


def baseline_identity(e: GoldEntry) -> dict:
    """Identity recode: keep the same code if it exists in the target, else map to missing."""
    miss = next((c for c, lbl in e.target_values.items() if "missing" in lbl.lower()), None)
    pred = {sc: (sc if sc in e.target_values else (miss or sc)) for sc in e.source_values}
    if "NA" in e.gold_map:
        pred["NA"] = miss or "NA"
    return pred


def score(entries, predictor) -> dict:
    """entry_exact / pair_acc incl-NA / pair_acc excl-NA over gold entries. predictor(entry)->{sc:tc}."""
    n = exact = pt = po = ptx = pox = 0
    for e in entries:
        pred = predictor(e)
        all_ok = True
        for sc, gt in e.gold_map.items():
            hit = str(pred.get(sc)) == str(gt)
            pt += 1
            po += hit
            if sc != "NA":
                ptx += 1
                pox += hit
            all_ok = all_ok and hit
        n += 1
        exact += all_ok
    return {
        "entries": n,
        "entry_exact_acc": round(exact / n, 4) if n else 0.0,
        "pair_acc_incl_na": round(po / pt, 4) if pt else 0.0,
        "pair_acc_excl_na": round(pox / ptx, 4) if ptx else 0.0,
    }


def main() -> None:
    root = common.ensure_athlos_repo()
    entries, skips = build_gold(root)
    nonident = [e for e in entries if e.nonidentity]

    baselines = {
        "oracle_selfcheck": score(entries, lambda e: dict(e.gold_map)),
        "identity": score(entries, baseline_identity),
        "label_similarity": score(entries, baseline_label_sim),
        "identity_nonidentity_only": score(nonident, baseline_identity),
        "label_similarity_nonidentity_only": score(nonident, baseline_label_sim),
    }
    result = {
        "benchmark": "athlos_value_recode",
        "unique_gold_entries": len(entries),
        "cohort_wave_instances": sum(e.instances for e in entries),
        "distinct_target_vars": len({e.target_var for e in entries}),
        "nonidentity_entries": len(nonident),
        "skipped": dict(skips),
        "baselines": baselines,
    }

    print(f"\n{'='*64}\nATHLOS value-recode gold (Benchmark C)\n{'='*64}")
    print(f"  unique gold entries : {len(entries)}  (from {result['cohort_wave_instances']} cohort-wave instances)")
    print(f"  distinct target vars: {result['distinct_target_vars']}")
    print(f"  non-identity (teeth): {len(nonident)} ({len(nonident)/len(entries):.0%})")
    print(f"  skipped             : {dict(skips)}")
    print("  baselines (entry_exact / pair_incl_na / pair_excl_na):")
    for name, m in baselines.items():
        print(f"    {name:34s} {m['entry_exact_acc']:.3f} / {m['pair_acc_incl_na']:.3f} / {m['pair_acc_excl_na']:.3f}")

    gold_out = common.CACHE_DIR / "athlos" / "athlos_gold.json"
    gold_out.write_text(json.dumps([asdict(e) for e in entries], indent=1))
    res_out = common.CACHE_DIR / "athlos_result.json"
    res_out.write_text(json.dumps(result, indent=1))
    print(
        f"\n  wrote {gold_out.relative_to(common.REPO_ROOT)} ({len(entries)} entries) + "
        f"{res_out.relative_to(common.REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
