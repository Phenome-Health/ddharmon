"""Background pipeline runner: drives ddharmon.harmonization with phase progress.

Reuses the v1 pipeline stages directly (no new pipeline logic) and reports progress
by mutating the :class:`~ui.backend.jobs.Job` between stages:

    load → embed → cluster (BERTopic) → anchor (sub-cluster + CDE) → classify → assemble

``classify_mode``: ``none`` (deterministic verdicts only; anchored sub-clusters surface
as un-classified) · ``sync`` (inline AnthropicClient per prompt) · ``batch`` (Anthropic
Batch API; minutes-to-hours).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from ddharmon.clustering.topic_engine import topic_model_dictionaries
from ddharmon.embedding.provider import EmbeddingProvider
from ddharmon.embedding.service import embed_dictionary
from ddharmon.harmonization import (
    HarmonizationResult,
    HarmonizationVerdict,
    assemble_verdicts,
    prepare_from_clusters,
)
from ddharmon.harmonization.pipeline import PromptRecord
from ddharmon.ingestion import load_dictionary
from ui.backend.jobs import JobStore

logger = logging.getLogger(__name__)

# Same instruction the Batch API appends, so sync output matches the schema.
_SCHEMA_PREAMBLE = "\n\nRespond with ONLY valid JSON matching this schema (no markdown fences):\n"

ClassifyFn = Callable[[list[PromptRecord]], dict[str, object]]


def _verdict_to_camel(v: HarmonizationVerdict) -> dict[str, Any]:
    d = asdict(v)
    d.pop("raw", None)
    return {
        "subClusterId": d["sub_cluster_id"],
        "parentTopicId": d["parent_topic_id"],
        "subLabel": d["sub_label"],
        "mode": d["mode"],
        "verdict": d["verdict"] or "pending",
        "parentCdeId": d["parent_cde_id"],
        "confidence": d["confidence"],
        "evidence": d["evidence"],
        "label": d["label"],
        "cohorts": d["cohorts"],
        "nFields": d["n_fields"],
        "encodedFraction": d["encoded_fraction"],
        "anchorDesignation": d["anchor_designation"],
        "decidedBy": d["decided_by"],
    }


def serialize_result(result: HarmonizationResult) -> dict[str, Any]:
    """Serialize a HarmonizationResult to the camelCase shape the frontend consumes."""
    verdicts = [_verdict_to_camel(v) for v in result.verdicts]
    counts: dict[str, int] = {}
    for v in verdicts:
        key = v["mode"] if v["mode"] in ("single_cohort", "cde_only", "noise") else v["verdict"]
        counts[key] = counts.get(key, 0) + 1
    n_cohorts_with_cde = sum(1 for v in verdicts if v["anchorDesignation"])
    return {
        "verdicts": verdicts,
        "summary": {
            "nVerdicts": len(verdicts),
            "nLlmPrompts": len(result.prompt_records),
            "nAnchored": n_cohorts_with_cde,
            "counts": counts,
        },
    }


def _sync_classify(prompts: list[PromptRecord], store: JobStore, job_id: str) -> dict[str, object]:
    """Run the classify-only prompts inline via AnthropicClient (needs ANTHROPIC_API_KEY)."""
    from ddharmon.llm.anthropic_client import AnthropicClient

    client = AnthropicClient()
    responses: dict[str, object] = {}
    store.update(job_id, total=len(prompts), completed=0)
    for i, rec in enumerate(prompts):
        system = rec.system_prompt + _SCHEMA_PREAMBLE + rec.schema
        responses[rec.id] = client.complete(rec.user_prompt, system=system, max_tokens=512)
        store.update(job_id, completed=i + 1)
    return responses


def _batch_classify(prompts: list[PromptRecord], work_dir: str) -> dict[str, object]:
    """Run the classify-only prompts via the Anthropic Batch API (blocking poll)."""
    import json
    from pathlib import Path

    from ddharmon.harmonization import write_prompts_jsonl
    from ddharmon.llm.batch import submit_and_wait

    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    prompts_path = wd / "prompts_harmonize_arn.jsonl"
    responses_path = wd / "responses_harmonize_arn.jsonl"
    write_prompts_jsonl(prompts, prompts_path)
    submit_and_wait(prompts_path, responses_path)
    responses: dict[str, object] = {}
    with open(responses_path) as f:
        for line in f:
            rec = json.loads(line)
            responses[rec["id"]] = rec["response"]
    return responses


def run_harmonization(
    store: JobStore,
    job_id: str,
    dict_specs: list[dict[str, Any]],
    cde_spec: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    provider: EmbeddingProvider | None = None,
    classify: ClassifyFn | None = None,
) -> None:
    """Run the full pipeline for a job, reporting phase progress. Safe to run in a thread.

    Args:
        dict_specs: ``[{path, cohort_name, column_roles}]`` for the uploaded cohort dicts.
        cde_spec: ``{path, cohort_name, column_roles}`` for the CDE catalog, or None.
        config: ``{min_cluster_size, classify_mode, cde_cohort, work_dir}``.
        provider: embedding provider (defaults to SentenceTransformerProvider — injected in tests).
        classify: optional ``records -> {id: response}`` override (injected in tests); when None,
            ``config['classify_mode']`` selects none/sync/batch.
    """
    try:
        cde_cohort = config.get("cde_cohort", "NIH_CDE")
        min_cluster_size = int(config.get("min_cluster_size", 15))
        classify_mode = config.get("classify_mode", "none")

        # --- load ---
        store.update(job_id, status="loading", phase="loading")
        specs = list(dict_specs) + ([cde_spec] if cde_spec else [])
        dictionaries = [load_dictionary(s["path"], cohort_name=s["cohort_name"], **s["column_roles"]) for s in specs]

        # --- embed ---
        store.update(job_id, status="embedding", phase="embedding")
        if provider is None:
            from ddharmon.embedding.provider import SentenceTransformerProvider

            provider = SentenceTransformerProvider()
        embedded = [embed_dictionary(dd, provider=provider) for dd in dictionaries]

        # --- cluster ---
        store.update(job_id, status="clustering", phase="clustering")
        topic = topic_model_dictionaries(embedded, min_cluster_size=min_cluster_size)

        # --- sub-cluster + anchor + prompt build ---
        store.update(job_id, status="anchoring", phase="anchoring")
        prompts, deterministic = prepare_from_clusters(
            topic.clusters, embedded, topic.embeddings, topic.field_refs, cde_cohort=cde_cohort
        )

        # --- classify ---
        if classify is not None:
            responses = classify(prompts)
        elif classify_mode == "sync" and prompts:
            store.update(job_id, status="classifying", phase="classifying")
            responses = _sync_classify(prompts, store, job_id)
        elif classify_mode == "batch" and prompts:
            store.update(job_id, status="classifying", phase="classifying")
            responses = _batch_classify(prompts, config.get("work_dir", f".ddharmon_ui/{job_id}"))
        else:  # "none" — anchored sub-clusters surface as un-classified ("pending")
            responses = {}

        result = assemble_verdicts(prompts, responses, deterministic)
        store.update(job_id, status="complete", phase="complete", result=serialize_result(result))
        logger.info("job %s complete: %d verdicts", job_id, len(result.verdicts))
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI rather than crash the thread
        logger.exception("job %s failed", job_id)
        store.update(job_id, status="error", phase="error", error_message=str(exc))
