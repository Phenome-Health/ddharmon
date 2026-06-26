"""Command-line entry point for ddharmon.

Wired to the ``ddharmon`` console script via ``[project.scripts]`` in
``pyproject.toml`` (``ddharmon = "ddharmon.cli:main"``).

Subcommands:

* ``ddharmon harmonize`` — the v2 split-aware pipeline: load dictionaries → embed →
  cluster → retrieve candidate CDEs → generate-ideal → split into concept-groups →
  assign (adopt/refine/novel) → route → write records + an expert-review (EITL) campaign.
* ``ddharmon cluster`` — cluster equivalent variables across dictionaries ($0; no LLM).

Inputs are given either inline (``NAME=path.csv`` plus ``--cde path.tsv``) or via a JSON
``--config`` file (which can pin exact column mappings per input). Columns are auto-detected
from the header when not given explicitly; explicit mappings always win.

Heavy imports (sentence-transformers, clustering, the LLM stack) are deferred into the command
bodies so ``ddharmon --help`` / ``--version`` work in a thin environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click

from ddharmon import __version__

# Column-mapping keys accepted by load_dictionary (used for auto-detect + config validation).
_LOAD_COLUMN_KEYS = frozenset(
    {
        "variable_name",
        "field_id",
        "description",
        "short_label",
        "data_type",
        "units",
        "category",
        "coding_id",
        "value_encoding",
        "standard_code",
        "question_text",
        "validation",
        "parent_id",
    }
)

_DEFAULT_MODEL = "claude-sonnet-4-6"


# ── input resolution ────────────────────────────────────────────────


def _read_header(path: str | Path) -> list[str]:
    """Read the column header of a CSV/TSV (delimiter inferred from extension)."""
    import csv

    p = Path(path)
    delimiter = "\t" if p.suffix.lower() in {".tsv", ".tab"} else ","
    with open(p, newline="", encoding="utf-8") as f:
        return next(csv.reader(f, delimiter=delimiter), [])


def _autodetect_columns(path: str | Path) -> dict[str, str]:
    """Guess column → load_dictionary-role mapping from the header via the schema registry."""
    from ddharmon.ingestion.schema_registry import SchemaRegistry

    mapping = SchemaRegistry().detect_roles(_read_header(path))
    best: dict[str, tuple[str, float]] = {}
    for col, match in mapping.role_map.items():
        key = match.role.name.lower()
        if key in _LOAD_COLUMN_KEYS and (key not in best or match.confidence > best[key][1]):
            best[key] = (col, match.confidence)
    return {key: col for key, (col, _) in best.items()}


def _parse_inline_inputs(inputs: tuple[str, ...]) -> list[dict[str, Any]]:
    """Parse ``NAME=path`` / ``path`` tokens into input specs (columns auto-detected)."""
    specs: list[dict[str, Any]] = []
    for token in inputs:
        if "=" in token:
            name, path = token.split("=", 1)
        else:
            path = token
            name = Path(token).stem
        specs.append({"name": name, "path": path, "columns": {}})
    return specs


def _resolve_config(config_path: str | None, inputs: tuple[str, ...], cde: str | None) -> dict[str, Any]:
    """Build the run config from a JSON file or inline args."""
    if config_path:
        cfg: dict[str, Any] = json.loads(Path(config_path).read_text())
        base = Path(config_path).resolve().parent  # resolve relative paths against the config's dir

        def _abs(p: str) -> str:
            return p if Path(p).is_absolute() else str(base / p)

        cfg["cde"]["path"] = _abs(cfg["cde"]["path"])
        for spec in cfg["inputs"]:
            spec["path"] = _abs(spec["path"])
        cfg.setdefault("options", {})
        return cfg
    if not inputs or not cde:
        raise click.UsageError("Provide either --config, or one or more NAME=path inputs together with --cde.")
    return {
        "cde": {"name": "NIH_CDE", "path": cde, "columns": {}},
        "inputs": _parse_inline_inputs(inputs),
        "options": {},
    }


def _columns_for(spec: dict[str, Any]) -> dict[str, str]:
    """Auto-detected columns overlaid with the spec's explicit mappings (explicit wins)."""
    columns = dict(_autodetect_columns(spec["path"]))
    columns.update(spec.get("columns") or {})
    return {k: v for k, v in columns.items() if k in _LOAD_COLUMN_KEYS}


def _load_dict(spec: dict[str, Any]) -> Any:
    """Load + preprocess one dictionary."""
    from ddharmon.ingestion import load_dictionary, preprocess_dictionary

    return preprocess_dictionary(
        load_dictionary(
            spec["path"],
            cohort_name=spec["name"],
            embed_variable_name=spec.get("embed_variable_name", True),
            **_columns_for(spec),
        )
    )


def _load_and_embed(spec: dict[str, Any], provider: Any) -> Any:
    """Load, preprocess, and embed one dictionary; echo a one-line summary."""
    from ddharmon.embedding import embed_dictionary

    dd = _load_dict(spec)
    click.echo(f"  {spec['name']}: {dd.field_count} fields")
    return embed_dictionary(dd, provider=provider)


def _make_batch_runner(work_dir: Path, model: str) -> Any:
    """Build a stage runner that drives one LLM stage through the schema-enforced Batch API."""
    from ddharmon.harmonization import write_prompts_jsonl
    from ddharmon.llm import submit_and_wait

    state = {"n": 0}

    def run(records: list[Any]) -> dict[str, Any]:
        state["n"] += 1
        tag = f"stage{state['n']}"
        prompts_path = work_dir / f"prompts_{tag}.jsonl"
        responses_path = work_dir / f"responses_{tag}.jsonl"
        write_prompts_jsonl(records, prompts_path)
        submit_and_wait(prompts_path, responses_path, model=model, max_tokens=1024)
        out: dict[str, Any] = {}
        with open(responses_path) as f:
            for line in f:
                rec = json.loads(line)
                out[rec["id"]] = rec["response"]
        return out

    return run


# ── commands ────────────────────────────────────────────────────────


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, "-V", "--version", prog_name="ddharmon")
@click.pass_context
def main(ctx: click.Context) -> None:
    """ddharmon — Data Dictionary Harmonization Tool.

    Cluster equivalent variables across data dictionaries and assign each to a
    Common Data Element (CDE), routing every recommendation to expert review.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("inputs", nargs=-1)
@click.option("--config", "-c", "config_path", type=click.Path(exists=True, dir_okay=False), help="JSON run config.")
@click.option("--cde", type=click.Path(exists=True, dir_okay=False), help="CDE backbone CSV/TSV (inline mode).")
@click.option("--output-dir", "-o", default="harmonization_output", show_default=True, help="Output directory.")
@click.option("--model", default=_DEFAULT_MODEL, show_default=True, help="LLM model for the three stages.")
@click.option("--min-cluster-size", default=15, show_default=True, type=int)
@click.option("--top-k", default=20, show_default=True, type=int, help="Candidate CDEs retrieved per cluster/group.")
@click.option("--retrieval-floor", default=0.30, show_default=True, type=float)
@click.option("--max-clusters", default=None, type=int, help="Cap clusters harmonized (largest first) to bound cost.")
@click.option("--dry-run", is_flag=True, help="Stop after building prompts ($0; no LLM calls / no API key needed).")
def harmonize(
    inputs: tuple[str, ...],
    config_path: str | None,
    cde: str | None,
    output_dir: str,
    model: str,
    min_cluster_size: int,
    top_k: int,
    retrieval_floor: float,
    max_clusters: int | None,
    dry_run: bool,
) -> None:
    """Run the v2 split-aware harmonization pipeline; write records + an EITL campaign.

    \b
    Inline:  ddharmon harmonize AoU=aou.csv CLSA=clsa.csv --cde all_cdes_flat.tsv -o out/
    Config:  ddharmon harmonize --config harmonize.json -o out/

    Needs ANTHROPIC_API_KEY for the three LLM stages; without it (or with --dry-run) the
    pipeline runs the $0 path (cluster + retrieve) and writes the prompts for later submission.
    """
    from ddharmon.embedding import SentenceTransformerProvider
    from ddharmon.harmonization import harmonize_leanb, write_prompts_jsonl, write_records_json

    cfg = _resolve_config(config_path, inputs, cde)
    opts = cfg.get("options", {})
    model = opts.get("model", model)
    min_cluster_size = opts.get("min_cluster_size", min_cluster_size)
    top_k = opts.get("top_k", top_k)
    retrieval_floor = opts.get("retrieval_floor", retrieval_floor)
    if max_clusters is None:
        max_clusters = opts.get("max_clusters")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    provider = SentenceTransformerProvider()
    cde_spec = cfg["cde"]
    click.echo("Loading + embedding dictionaries:")
    cde_embedded = _load_and_embed(cde_spec, provider)
    cohort_embedded = [_load_and_embed(spec, provider) for spec in cfg["inputs"]]
    embedded = [cde_embedded, *cohort_embedded]

    common: dict[str, Any] = {
        "cde_cohort": cde_spec["name"],
        "min_cluster_size": min_cluster_size,
        "top_k": top_k,
        "retrieval_floor": retrieval_floor,
        "model_tag": model,
        "max_clusters": max_clusters,
    }

    if dry_run or not os.environ.get("ANTHROPIC_API_KEY"):
        result = harmonize_leanb(embedded, **common)
        prompts_path = out / "prompts_generate_ideal.jsonl"
        n = write_prompts_jsonl(result.ideal_prompts, prompts_path)
        reason = "--dry-run" if dry_run else "no ANTHROPIC_API_KEY"
        click.echo(f"\n{reason}: built {n} generate-ideal prompts → {prompts_path}")
        click.echo("Set ANTHROPIC_API_KEY (and drop --dry-run) to run the full pipeline.")
        return

    runner = _make_batch_runner(out, model)
    result = harmonize_leanb(embedded, generate=runner, split=runner, classify=runner, **common)

    records_path = out / "records.json"
    n_records = write_records_json(result, records_path)
    buckets = {k: len(v) for k, v in result.buckets().items()}
    click.echo(f"\n{n_records} routed records {buckets} → {records_path}")
    _export_campaign(result, cfg, out)


def _export_campaign(result: Any, cfg: dict[str, Any], out: Path) -> None:
    """Write the expert-review (EITL) campaign for the routed records."""
    from ddharmon.export.eitl import build_cde_lookup, export_split_eitl_campaign

    cde_dict = _load_dict(cfg["cde"])
    source_dicts = {spec["name"]: _load_dict(spec) for spec in cfg["inputs"]}
    counts = export_split_eitl_campaign(
        result,
        source_dicts,
        build_cde_lookup(cde_dict),
        out_dir=out / "eitl",
        stem="ddharmon",
    )
    click.echo(f"EITL campaign rows: {counts} → {out / 'eitl'}")


@main.command()
@click.argument("inputs", nargs=-1)
@click.option("--config", "-c", "config_path", type=click.Path(exists=True, dir_okay=False), help="JSON run config.")
@click.option("--output", "-o", default="clusters.json", show_default=True, help="Output JSON path.")
@click.option("--min-cluster-size", default=15, show_default=True, type=int)
def cluster(inputs: tuple[str, ...], config_path: str | None, output: str, min_cluster_size: int) -> None:
    """Cluster equivalent variables across dictionaries ($0 — no LLM) and write clusters to JSON.

    \b
    Inline:  ddharmon cluster AoU=aou.csv CLSA=clsa.csv -o clusters.json
    Config:  ddharmon cluster --config harmonize.json -o clusters.json
    """
    from ddharmon.clustering import topic_model_dictionaries
    from ddharmon.embedding import SentenceTransformerProvider

    if config_path:
        cfg = json.loads(Path(config_path).read_text())
        base = Path(config_path).resolve().parent
        specs = cfg["inputs"]
        for spec in specs:
            if not Path(spec["path"]).is_absolute():
                spec["path"] = str(base / spec["path"])
    elif inputs:
        specs = _parse_inline_inputs(inputs)
    else:
        raise click.UsageError("Provide either --config, or one or more NAME=path inputs.")

    provider = SentenceTransformerProvider()
    click.echo("Loading + embedding dictionaries:")
    embedded = [_load_and_embed(spec, provider) for spec in specs]

    tm = topic_model_dictionaries(embedded, min_cluster_size=min_cluster_size)
    payload = [
        {
            "cluster_id": c.cluster_id,
            "label": c.label,
            "cohort_coverage": c.cohort_coverage,
            "n_members": len(c.members),
            "members": [
                {"cohort": m.dictionary_name, "variable_name": m.variable_name, "description": m.description}
                for m in c.members
            ],
        }
        for c in tm.clusters
    ]
    Path(output).write_text(json.dumps(payload, indent=2))
    click.echo(f"\n{len(tm.clusters)} clusters over {len(tm.field_refs)} fields → {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
