"""Tests for the ddharmon CLI (`ddharmon.cli`).

The heavy pieces (embedding model, clustering, LLM stages) are monkeypatched so these run
offline and fast; real dictionary loading + JSON config resolution are exercised for real.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

import ddharmon
from ddharmon.cli import _autodetect_columns, _parse_inline_inputs, _resolve_config, main
from ddharmon.harmonization.leanb import LeanBResult
from ddharmon.models.cluster import FieldCluster, FieldReference


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_cohort(path, header="variable_name,description", rows=(("age", "Age in years"), ("sex", "Sex at birth"))):
    path.write_text("\n".join([header, *(",".join(r) for r in rows)]) + "\n")
    return path


def _write_cde(path):
    path.write_text("variable_name\tfield_id\tdescription\nAge\t1\tAge of participant\nSex\t2\tSex of participant\n")
    return path


# ── top-level ───────────────────────────────────────────────


def test_cli_version_prints_single_sourced_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert ddharmon.__version__ in result.output


def test_cli_help_exits_zero_and_names_program(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "ddharmon" in result.output.lower()


def test_cli_no_args_shows_help(runner):
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "usage" in result.output.lower()


def test_help_lists_subcommands(runner):
    result = runner.invoke(main, ["--help"])
    assert "harmonize" in result.output
    assert "cluster" in result.output


# ── helpers ─────────────────────────────────────────────────


def test_autodetect_columns(tmp_path):
    csv = _write_cohort(tmp_path / "c.csv", header="variable_name,description,units")
    cols = _autodetect_columns(csv)
    assert cols.get("variable_name") == "variable_name"
    assert cols.get("description") == "description"


def test_parse_inline_inputs():
    specs = _parse_inline_inputs(("AoU=a.csv", "b.tsv"))
    assert specs[0]["name"] == "AoU" and specs[0]["path"] == "a.csv"
    assert specs[1]["name"] == "b"  # name derived from filename stem


def test_resolve_config_inline_requires_cde():
    with pytest.raises(click.UsageError):
        _resolve_config(None, (), None)


def test_resolve_config_from_json(tmp_path):
    cde = _write_cde(tmp_path / "cde.tsv")
    aou = _write_cohort(tmp_path / "aou.csv")
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(
        json.dumps(
            {
                "cde": {"name": "NIH_CDE", "path": "cde.tsv", "columns": {"variable_name": "variable_name"}},
                "inputs": [{"name": "AoU", "path": "aou.csv", "columns": {}}],
                "options": {"min_cluster_size": 7},
            }
        )
    )
    cfg = _resolve_config(str(cfg_path), (), None)
    assert cfg["cde"]["path"] == str(cde)  # relative paths resolved against the config dir
    assert cfg["inputs"][0]["path"] == str(aou)
    assert cfg["options"]["min_cluster_size"] == 7


# ── commands (mocked heavy pipeline) ────────────────────────


def _patch_light_pipeline(monkeypatch):
    """Replace the embedding model + embed call with no-ops (no model download)."""
    monkeypatch.setattr("ddharmon.embedding.SentenceTransformerProvider", lambda *a, **k: object())
    monkeypatch.setattr("ddharmon.embedding.embed_dictionary", lambda dd, provider=None: dd)


def test_harmonize_dry_run_writes_prompts(runner, tmp_path, monkeypatch):
    _patch_light_pipeline(monkeypatch)
    monkeypatch.setattr("ddharmon.harmonization.harmonize_leanb", lambda *a, **k: LeanBResult(ideal_prompts=[]))
    cde = _write_cde(tmp_path / "cde.tsv")
    aou = _write_cohort(tmp_path / "aou.csv")
    out = tmp_path / "out"
    result = runner.invoke(main, ["harmonize", f"AoU={aou}", "--cde", str(cde), "-o", str(out), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert (out / "prompts_generate_ideal.jsonl").exists()


def test_harmonize_full_run_writes_records(runner, tmp_path, monkeypatch):
    _patch_light_pipeline(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("ddharmon.harmonization.harmonize_leanb", lambda *a, **k: LeanBResult(records=[]))
    monkeypatch.setattr("ddharmon.cli._export_campaign", lambda *a, **k: None)  # its own tested unit
    cde = _write_cde(tmp_path / "cde.tsv")
    aou = _write_cohort(tmp_path / "aou.csv")
    out = tmp_path / "out"
    result = runner.invoke(main, ["harmonize", f"AoU={aou}", "--cde", str(cde), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "routed records" in result.output
    assert (out / "records.json").exists()


def test_cluster_writes_json(runner, tmp_path, monkeypatch):
    _patch_light_pipeline(monkeypatch)
    fake_tm = SimpleNamespace(
        clusters=[
            FieldCluster(
                cluster_id=0,
                label="age",
                members=[FieldReference("AoU", "age", "Age in years")],
                cohort_coverage={"AoU": 1},
            )
        ],
        field_refs=[FieldReference("AoU", "age", "Age in years")],
    )
    monkeypatch.setattr("ddharmon.clustering.topic_model_dictionaries", lambda *a, **k: fake_tm)
    aou = _write_cohort(tmp_path / "aou.csv")
    out_json = tmp_path / "clusters.json"
    result = runner.invoke(main, ["cluster", f"AoU={aou}", "-o", str(out_json)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out_json.read_text())
    assert len(payload) == 1
    assert payload[0]["label"] == "age"
    assert payload[0]["members"][0]["variable_name"] == "age"
