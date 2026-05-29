"""FastAPI app for the ddharmon harmonization GUI.

Endpoints (all under /api/harmonize):
    POST /detect            columns -> suggested column->role map (SchemaRegistry)
    POST /batch             multipart upload of dict files + config -> {jobId}
    GET  /stream/{job_id}   SSE progress (event: progress)
    GET  /result/{job_id}   full job snapshot (REST fallback)
    GET  /jobs              list jobs (summaries)
    DELETE /jobs/{job_id}   delete a job
    POST /jobs/{job_id}/verdict   persist a human approve/refine/reject decision
    GET  /jobs/{job_id}/export    eitl_tsv | buckets_json | decisions_csv

Serves the built frontend (ui/frontend/dist) at / when present.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ui.backend.jobs import TERMINAL_STATES, store
from ui.backend.runner import run_harmonization

# --- CDE catalog (server-side; not uploaded) -------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CDE_DIR = Path(os.environ.get("DDHARMON_CDE_DIR", _REPO_ROOT / "data" / "examples" / "cde"))
CDE_FILES = {"endorsed": _CDE_DIR / "nih_endorsed_flat.tsv", "full": _CDE_DIR / "all_cdes_flat.tsv"}
CDE_COLUMN_ROLES = {
    "variable_name": "designation",
    "field_id": "tinyId",
    "description": "definition",
    "question_text": "question_text",
    "data_type": "datatype",
    "value_encoding": "permissible_values",
    "category": "classification",
    "standard_code": "concept_codes",
}
CDE_COHORT = "NIH_CDE"

_WORK_ROOT = Path(os.environ.get("DDHARMON_UI_WORK", _REPO_ROOT / ".ddharmon_ui"))

app = FastAPI(title="ddharmon Harmonization API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- /detect ---------------------------------------------------------------------------------
class DetectBody(BaseModel):
    columns: list[str]


@app.post("/api/harmonize/detect")
def detect(body: DetectBody) -> dict[str, Any]:
    """Suggest a column->role mapping (load_dictionary kwargs) for the given headers."""
    from ddharmon.ingestion.schema_registry import SchemaRegistry

    mapping = SchemaRegistry().detect_roles(body.columns)
    # Invert role_map (column -> match) into {kwarg: column}, keeping the highest-confidence column per role.
    best: dict[str, tuple[float, str]] = {}
    for column, match in mapping.role_map.items():
        kwarg = match.role.value
        if kwarg not in best or match.confidence > best[kwarg][0]:
            best[kwarg] = (match.confidence, column)
    column_roles = {kwarg: col for kwarg, (_conf, col) in best.items()}
    return {"columnRoles": column_roles, "confidence": mapping.overall_confidence}


# --- /batch ----------------------------------------------------------------------------------
@app.post("/api/harmonize/batch")
async def start_batch(files: Annotated[list[UploadFile], File()], config: Annotated[str, Form()]) -> dict[str, str]:
    """Start a harmonization run. ``config`` is a JSON string:

    ``{dictionaries: [{filename, cohortName, columnRoles}], cdeSet: endorsed|full|none,
       minClusterSize: int, classifyMode: none|sync|batch, displayName?}``
    """
    cfg = json.loads(config)
    job_id = str(uuid.uuid4())
    work_dir = _WORK_ROOT / job_id
    uploads = work_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for up in files:
        dest = uploads / Path(up.filename or "upload.csv").name
        with open(dest, "wb") as fh:
            shutil.copyfileobj(up.file, fh)
        saved[dest.name] = dest

    dict_specs: list[dict[str, Any]] = []
    for d in cfg.get("dictionaries", []):
        fname = Path(d["filename"]).name
        if fname not in saved:
            raise HTTPException(status_code=400, detail=f"Uploaded file missing for {fname!r}")
        roles = {k: v for k, v in d.get("columnRoles", {}).items() if v}
        if "variable_name" not in roles and "description" not in roles and "question_text" not in roles:
            raise HTTPException(
                status_code=400, detail=f"{fname!r} needs at least variable_name/description/question_text"
            )
        dict_specs.append({"path": str(saved[fname]), "cohort_name": d["cohortName"], "column_roles": roles})

    cde_set = cfg.get("cdeSet", "endorsed")
    cde_spec: dict[str, Any] | None = None
    if cde_set in CDE_FILES:
        cde_path = CDE_FILES[cde_set]
        if not cde_path.exists():
            raise HTTPException(
                status_code=400, detail=f"CDE file not found: {cde_path} (run scripts/flatten_cde_repo.py)"
            )
        cde_spec = {"path": str(cde_path), "cohort_name": CDE_COHORT, "column_roles": dict(CDE_COLUMN_ROLES)}

    run_config = {
        "min_cluster_size": int(cfg.get("minClusterSize", 15)),
        "classify_mode": cfg.get("classifyMode", "none"),
        "cde_cohort": CDE_COHORT,
        "work_dir": str(work_dir),
        "cdeSet": cde_set,
    }
    display = cfg.get("displayName") or f"Run {job_id[:8]}"
    store.create(job_id, display, run_config)
    threading.Thread(
        target=run_harmonization,
        args=(store, job_id, dict_specs, cde_spec, run_config),
        daemon=True,
    ).start()
    return {"jobId": job_id}


# --- SSE + result ----------------------------------------------------------------------------
def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/harmonize/stream/{job_id}")
async def stream(job_id: str) -> StreamingResponse:
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def gen() -> Any:
        store.purge_expired()
        while True:
            job = store.get(job_id)
            if job is None:
                yield _sse("error", {"message": "Job not found"})
                return
            yield _sse("progress", job.to_dict())
            if job.status in TERMINAL_STATES:
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/api/harmonize/result/{job_id}")
def result(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


# --- jobs list / delete ----------------------------------------------------------------------
@app.get("/api/harmonize/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return [j.summary_dict() for j in store.list()]


@app.delete("/api/harmonize/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    if not store.delete(job_id):
        raise HTTPException(status_code=404, detail="Job not found")


# --- human decisions -------------------------------------------------------------------------
class VerdictBody(BaseModel):
    subClusterId: str
    decision: str  # approve | refine | reject
    note: str = ""


@app.post("/api/harmonize/jobs/{job_id}/verdict")
def submit_verdict(job_id: str, body: VerdictBody) -> dict[str, bool]:
    if body.decision not in ("approve", "refine", "reject"):
        raise HTTPException(status_code=400, detail="decision must be approve|refine|reject")
    if not store.set_decision(job_id, body.subClusterId, body.decision, body.note):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


# --- export ----------------------------------------------------------------------------------
@app.get("/api/harmonize/jobs/{job_id}/export")
def export(job_id: str, format: str = "eitl_tsv") -> Any:
    job = store.get(job_id)
    if job is None or job.result is None:
        raise HTTPException(status_code=404, detail="Job not found or not complete")
    verdicts: list[dict[str, Any]] = job.result["verdicts"]

    if format == "buckets_json":
        buckets: dict[str, list[dict[str, Any]]] = {}
        for v in verdicts:
            key = v["mode"] if v["mode"] in ("single_cohort", "cde_only", "noise") else v["verdict"]
            buckets.setdefault(key, []).append(v)
        return JSONResponse(
            buckets, headers={"Content-Disposition": f'attachment; filename="buckets_{job_id[:8]}.json"'}
        )

    sep = "," if format == "decisions_csv" else "\t"
    cols = [
        "subClusterId",
        "label",
        "verdict",
        "parentCdeId",
        "anchorDesignation",
        "confidence",
        "mode",
        "nFields",
        "cohorts",
        "humanDecision",
        "humanNote",
        "evidence",
    ]
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=sep)
    w.writerow(cols)
    for v in verdicts:
        if v["mode"] in ("cde_only", "noise"):
            continue
        dec = job.decisions.get(v["subClusterId"], {})
        w.writerow(
            [
                v["subClusterId"],
                v["label"],
                v["verdict"],
                v["parentCdeId"] or "",
                v["anchorDesignation"] or "",
                "" if v["confidence"] is None else round(v["confidence"], 3),
                v["mode"],
                v["nFields"],
                ";".join(v["cohorts"]),
                dec.get("decision", ""),
                dec.get("note", "").replace("\n", " "),
                (v["evidence"] or "").replace("\n", " "),
            ]
        )
    ext = "csv" if format == "decisions_csv" else "tsv"
    media = "text/csv" if ext == "csv" else "text/tab-separated-values"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{format}_{job_id[:8]}.{ext}"'},
    )


# --- static frontend (prod) ------------------------------------------------------------------
_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
