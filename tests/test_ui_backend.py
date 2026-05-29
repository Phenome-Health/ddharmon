"""Tests for the ddharmon GUI backend (ui/backend).

Covers: /detect role suggestion, the in-memory JobStore, result serialization, the
full HTTP flow with a fake runner, and a deterministic end-to-end runner test that
monkeypatches BERTopic (so no UMAP/HDBSCAN randomness or slowness).
"""

from __future__ import annotations

import json
import time

import numpy as np
from fastapi.testclient import TestClient

from ddharmon.clustering.topic_engine import collect_inputs
from ddharmon.embedding.provider import EmbeddingProvider
from ddharmon.harmonization import HarmonizationResult, HarmonizationVerdict
from ddharmon.models.cluster import FieldCluster, TopicModelResult
from ui.backend import app as app_module
from ui.backend.jobs import JobStore
from ui.backend.runner import run_harmonization, serialize_result

client = TestClient(app_module.app)
DIM = 32


class StubProvider(EmbeddingProvider):
    """Deterministic hash-based embeddings — no model download."""

    @property
    def model_name(self) -> str:
        return "stub-ui"

    @property
    def dimension(self) -> int:
        return DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        import hashlib

        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) or 1.0)
        return out


# ── /detect ────────────────────────────────────────────────────


def test_detect_suggests_roles():
    resp = client.post("/api/harmonize/detect", json={"columns": ["Column Name", "Description", "answer_options"]})
    assert resp.status_code == 200
    roles = resp.json()["columnRoles"]
    assert roles.get("variable_name") == "Column Name"
    assert roles.get("description") == "Description"


# ── JobStore ───────────────────────────────────────────────────


def test_jobstore_lifecycle():
    s = JobStore()
    s.create("j1", "Run 1", {"classify_mode": "none"})
    assert s.get("j1").status == "pending"
    s.update("j1", status="clustering", phase="clustering", completed=3, total=10)
    assert s.get("j1").completed == 3
    assert s.set_decision("j1", "5:0", "approve", "looks good")
    assert s.get("j1").decisions["5:0"]["decision"] == "approve"
    assert [j.job_id for j in s.list()] == ["j1"]
    assert s.delete("j1") is True
    assert s.get("j1") is None


# ── serialization ──────────────────────────────────────────────


def test_serialize_result_camelcase_and_counts():
    verdicts = [
        HarmonizationVerdict(
            sub_cluster_id="1:0",
            parent_topic_id=1,
            sub_label=0,
            mode="harmonize",
            verdict="adopt",
            parent_cde_id="cde_x",
            confidence=0.9,
            anchor_designation="AgeCDE",
        ),
        HarmonizationVerdict(sub_cluster_id="2:0", parent_topic_id=2, sub_label=0, mode="single_cohort", verdict=""),
    ]
    out = serialize_result(HarmonizationResult(verdicts=verdicts, prompt_records=[]))
    assert out["verdicts"][0]["subClusterId"] == "1:0"
    assert out["verdicts"][0]["anchorDesignation"] == "AgeCDE"
    assert out["verdicts"][1]["verdict"] == "pending"  # empty verdict surfaces as 'pending'
    assert out["summary"]["counts"]["adopt"] == 1
    assert out["summary"]["counts"]["single_cohort"] == 1
    assert out["summary"]["nAnchored"] == 1


# ── full HTTP flow with a fake runner ──────────────────────────


def _canned_result() -> dict:
    return {
        "verdicts": [
            {
                "subClusterId": "1:0",
                "label": "age",
                "verdict": "adopt",
                "parentCdeId": "cde_age",
                "anchorDesignation": "AgeCDE",
                "confidence": 0.9,
                "mode": "harmonize",
                "nFields": 3,
                "cohorts": ["CohortA", "CohortB"],
                "decidedBy": "llm",
                "evidence": "match",
            },
            {
                "subClusterId": "2:0",
                "label": "noise",
                "verdict": "pending",
                "parentCdeId": None,
                "anchorDesignation": None,
                "confidence": None,
                "mode": "noise",
                "nFields": 1,
                "cohorts": ["CohortA"],
                "decidedBy": "deterministic",
                "evidence": "",
            },
        ],
        "summary": {"nVerdicts": 2, "nLlmPrompts": 1, "nAnchored": 1, "counts": {"adopt": 1, "noise": 1}},
    }


def test_batch_flow_with_fake_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "_WORK_ROOT", tmp_path)

    def fake_runner(store, job_id, dict_specs, cde_spec, config, *, provider=None, classify=None):
        store.update(job_id, status="complete", phase="complete", result=_canned_result())

    monkeypatch.setattr(app_module, "run_harmonization", fake_runner)

    cfg = {
        "dictionaries": [
            {
                "filename": "cohortA.csv",
                "cohortName": "CohortA",
                "columnRoles": {"variable_name": "var", "description": "desc"},
            }
        ],
        "cdeSet": "none",
        "minClusterSize": 5,
        "classifyMode": "none",
        "displayName": "Test run",
    }
    resp = client.post(
        "/api/harmonize/batch",
        files=[("files", ("cohortA.csv", b"var,desc\nage,Age in years\n", "text/csv"))],
        data={"config": json.dumps(cfg)},
    )
    assert resp.status_code == 200
    job_id = resp.json()["jobId"]

    # fake runner completes in a daemon thread — poll briefly.
    for _ in range(50):
        r = client.get(f"/api/harmonize/result/{job_id}")
        if r.json()["status"] == "complete":
            break
        time.sleep(0.02)
    assert r.json()["status"] == "complete"
    assert len(r.json()["result"]["verdicts"]) == 2

    # jobs list
    assert any(j["jobId"] == job_id for j in client.get("/api/harmonize/jobs").json())

    # human decision
    dec = client.post(
        f"/api/harmonize/jobs/{job_id}/verdict", json={"subClusterId": "1:0", "decision": "approve", "note": "ok"}
    )
    assert dec.status_code == 200

    # export TSV (excludes noise rows) + buckets JSON
    tsv = client.get(f"/api/harmonize/jobs/{job_id}/export", params={"format": "eitl_tsv"})
    assert tsv.status_code == 200
    lines = tsv.text.strip().splitlines()
    assert lines[0].split("\t")[0] == "subClusterId"
    assert any("1:0" in ln for ln in lines)  # harmonize row present
    assert all("2:0" not in ln for ln in lines)  # noise row excluded

    buckets = client.get(f"/api/harmonize/jobs/{job_id}/export", params={"format": "buckets_json"})
    assert "adopt" in buckets.json()

    # delete
    assert client.delete(f"/api/harmonize/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/harmonize/result/{job_id}").status_code == 404


def test_batch_rejects_missing_required_role(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "_WORK_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "run_harmonization", lambda *a, **k: None)
    cfg = {"dictionaries": [{"filename": "x.csv", "cohortName": "X", "columnRoles": {}}], "cdeSet": "none"}
    resp = client.post(
        "/api/harmonize/batch",
        files=[("files", ("x.csv", b"a,b\n1,2\n", "text/csv"))],
        data={"config": json.dumps(cfg)},
    )
    assert resp.status_code == 400


# ── deterministic end-to-end runner (BERTopic monkeypatched) ────


def test_run_harmonization_end_to_end(monkeypatch, tmp_path):
    # Two cohort dicts + a CDE, written as small CSV/TSV files.
    a = tmp_path / "cohortA.csv"
    a.write_text("var,desc,enc\nage,Age in years,1=0-18|2=19+\nsmoke,Do you smoke,1=Yes|2=No\n")
    b = tmp_path / "cohortB.csv"
    b.write_text("var,desc,enc\nage_yrs,Age in years,0-120\nsmoke_b,Current smoker,1=Yes|0=No\n")
    cde = tmp_path / "cde.tsv"
    cde.write_text("designation\tdefinition\tpermissible_values\nAge CDE\tAge of participant\tyears\n")

    dict_specs = [
        {
            "path": str(a),
            "cohort_name": "CohortA",
            "column_roles": {"variable_name": "var", "description": "desc", "value_encoding": "enc"},
        },
        {
            "path": str(b),
            "cohort_name": "CohortB",
            "column_roles": {"variable_name": "var", "description": "desc", "value_encoding": "enc"},
        },
    ]
    cde_spec = {
        "path": str(cde),
        "cohort_name": "NIH_CDE",
        "column_roles": {
            "variable_name": "designation",
            "description": "definition",
            "value_encoding": "permissible_values",
        },
    }

    # Fake BERTopic: one cluster holding every field (built from the embedded dicts).
    def fake_topic_model(embedded, **kwargs):
        docs, embeddings, field_refs, cohorts = collect_inputs(embedded)
        cluster = FieldCluster(cluster_id=0, label="all", members=list(field_refs))
        return TopicModelResult(
            model=None,
            docs=docs,
            embeddings=embeddings,
            field_refs=field_refs,
            clusters=[cluster],
            outlier_cluster=None,
            all_cohort_names=cohorts,
        )

    monkeypatch.setattr("ui.backend.runner.topic_model_dictionaries", fake_topic_model)

    def mock_classify(records):
        return {
            r.id: {"verdict": "refine", "parent_cde_id": "cde_x", "confidence": 0.8, "evidence": "m"} for r in records
        }

    store = JobStore()
    store.create("e2e", "E2E", {"classify_mode": "sync"})
    run_harmonization(
        store,
        "e2e",
        dict_specs,
        cde_spec,
        {"min_cluster_size": 2, "classify_mode": "sync", "cde_cohort": "NIH_CDE"},
        provider=StubProvider(),
        classify=mock_classify,
    )

    job = store.get("e2e")
    assert job.status == "complete", job.error_message
    assert job.result is not None
    assert job.result["summary"]["nVerdicts"] >= 1
