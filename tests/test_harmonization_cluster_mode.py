"""Cohort-only clustering is the leanb default; the CDE catalog stays in RETRIEVAL only.

``harmonize_leanb`` used to put every embedded dictionary, the CDE catalog included, into UMAP+HDBSCAN and
filter the catalog rows out of the outputs afterwards (inherited from v1's CDE-anchored design). v2 retrieves
CDE candidates from the catalog's OWN vectors + a BM25 index, so the catalog never needed to be clustered.
These tests pin the switch:

1. the default clustering input holds no catalog row, so no catalog-only cluster can exist;
2. ``cluster_with_catalog=True`` reproduces the pre-change behaviour BYTE FOR BYTE (a frozen golden);
3. for a FIXED partition the CDE candidates (and every prompt and record) are identical in both modes;
4. a LEGACY substrate (every one saved before the switch) replays byte-identically to the pre-change code;
5. a new substrate records its mode, and a replay runs in the RECORDED mode whatever the kwarg says;
6. M10 outlier recovery in cohort-only mode never re-clusters a catalog row.

The golden (``fixtures/leanb_cluster_mode_golden.json``) was captured by running :func:`golden_snapshots` on the
code at ``4f8eb18`` — immediately BEFORE the switch was added — and must never be regenerated from later code:
it is the evidence that catalog-in and legacy replays did not move. :func:`golden_snapshots` therefore uses only
APIs that existed then; the new APIs are imported inside the tests that exercise them. It was captured with
``PYTHONHASHSEED=0`` and re-captured identically under other seeds (macOS arm64, numpy 2.4.2). The comparison
is exact, floats included: a different BLAS build could in principle move a last digit of a cosine, so on a new
platform compare the structures before suspecting the code.

No sentence-transformers, no UMAP/HDBSCAN (the fresh-clustering engine is a deterministic stand-in that
clusters whatever rows it is handed), no network, no LLM.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "leanb_cluster_mode_golden.json"
CDE = "NIH_CDE"

# The world: 6-d vectors. dim 0 age, 1 smoke, 2 height, 3 zip, 4 weight, 5 a within-concept axis whose SIGN
# is what M2 chunking bisects on. Cohort and catalog rows of one concept sit on BOTH sides of that axis, so a
# chunk that counts catalog rows (catalog-in) scatters the cohort members across chunks — visibly different
# content-addressed ids from a cohort-only cluster that never needs chunking.
_COHORT_A = [
    ("age", "Age in years", [1, 0, 0, 0, 0, 0.30]),
    ("age_alt", "Age at enrollment visit", [1, 0, 0, 0, 0, -0.30]),
    ("employer_workplace_zip", "ZIP code", [0, 0, 0, 1, 0, -0.30]),
    ("home_residence_zip", "ZIP code", [0, 0, 0, 1, 0, 0.30]),
    ("smoke", "Do you currently smoke", [0, 1, 0, 0, 0, 0.05]),
]
_COHORT_B = [
    ("age_yrs", "Participant age (yrs)", [0.98, 0, 0, 0, 0, -0.32]),
    ("smoke_b", "Current smoker", [0, 0.98, 0, 0, 0, -0.05]),
]
_CATALOG = [
    ("AgeCDE", "Age of the participant in years", [1, 0, 0, 0, 0, 0.32]),
    ("AgeCDE_b", "Age at study entry", [1, 0, 0, 0, 0, -0.28]),
    ("AgeCDE_c", "Age reported at baseline", [1, 0, 0, 0, 0, 0.28]),
    ("HeightCDE", "Standing height in centimeters", [0, 0, 1, 0, 0, 0]),
    ("SmokeCDE", "Current cigarette smoking status", [0, 1, 0, 0, 0, 0]),
    ("WeightCDE", "Body weight in kilograms", [0, 0, 0, 0, 1, 0]),
    ("ZipCDE", "Postal ZIP code of residence", [0, 0, 0, 1, 0, 0.32]),
    ("ZipCDE_b", "Postal ZIP code of employer", [0, 0, 0, 1, 0, -0.32]),
    ("ZipCDE_c", "Postal ZIP code", [0, 0, 0, 1, 0, 0.28]),
]
_CHUNK_CAP = 4  # small enough that a 6-row catalog-in age cluster is chunked and a 3-row cohort-only one is not


def _ed(hf: SimpleNamespace, cohort: str, rows: list) -> object:
    fields = [
        hf.field(
            v,
            d,
            question_text=d,
            field_id=f"id_{v}" if cohort == CDE else None,
            encoding="years" if "Age" in v else None,
        )
        for v, d, _ in rows
    ]
    return hf.embedded_dict(cohort, fields, sem_vecs=hf.l2(np.array([vec for _, _, vec in rows], float)))


def build_world(hf: SimpleNamespace) -> list:
    """The embedded dictionaries, catalog LAST (as the product passes them)."""
    return [_ed(hf, "CohortA", _COHORT_A), _ed(hf, "CohortB", _COHORT_B), _ed(hf, CDE, _CATALOG)]


def fake_topic_model(embedded_dicts, **_kw):
    """A deterministic stand-in for UMAP+HDBSCAN that clusters WHATEVER rows it is handed.

    Topic = the argmax over the five concept dims; the zip concept is left as HDBSCAN noise (topic -1) so M10
    outlier recovery has work. Handed the catalog, it forms catalog rows into clusters with cohort rows AND
    catalog-only clusters (height, weight), as the real engine did; handed cohort rows only, it cannot.
    """
    from ddharmon.clustering.topic_engine import collect_inputs, extract_topic_clusters
    from ddharmon.models.cluster import TopicModelResult

    docs, embeddings, field_refs, cohorts = collect_inputs(embedded_dicts)
    topics = []
    for vec in np.asarray(embeddings):
        t = int(np.argmax(vec[:5]))
        topics.append(-1 if t == 3 else t)
    clusters, outlier = extract_topic_clusters(topics, field_refs, cohorts)
    return TopicModelResult(
        model=None,
        docs=docs,
        embeddings=embeddings,
        field_refs=field_refs,
        clusters=clusters,
        outlier_cluster=outlier,
        all_cohort_names=cohorts,
    )


def _ser(p) -> dict:
    """A prompt as it is SENT (its Batch record) + the context the next stage reads. The long, fixed system
    prompt and schema are pinned by digest to keep the golden readable; every other field is verbatim."""
    rec = p.to_jsonl_record()
    for key in ("system_prompt", "schema"):
        rec[key] = "sha1:" + hashlib.sha1(str(rec[key]).encode("utf-8")).hexdigest()
    return {**rec, "context": p.context}


def recording_stages(log: dict) -> dict:
    """Mock LLM stages: each records the prompts it is sent, then answers as a pure function of each prompt."""

    def _rec(name, prompts):
        log.setdefault(name, []).extend(_ser(p) for p in prompts)

    def generate(prompts):
        _rec("generate", prompts)
        return {p.id: {"ideal_cde": f"ideal for {p.context['cluster_id']}"} for p in prompts}

    def split(prompts):
        _rec("split", prompts)
        out = {}
        for p in prompts:
            ids = [m["member_id"] for m in p.context["members"]]
            groups = (
                [{"member_ids": ids, "concept": "one concept", "verdict": "adopt"}]
                if len(ids) < 3
                else [
                    {"member_ids": ids[:-1], "concept": "main concept", "verdict": "adopt"},
                    {"member_ids": ids[-1:], "concept": "tail concept", "verdict": "novel"},
                ]
            )
            out[p.id] = {"groups": groups}
        return out

    def classify(prompts):
        _rec("classify", prompts)
        return {
            p.id: (
                {"verdict": "novel", "ranking": [], "rationale": "no candidate fits"}
                if p.context["n_members"] == 1
                else {"verdict": "adopt", "cde_id": "1", "ranking": [1, 2], "rationale": "same concept"}
            )
            for p in prompts
        }

    def merge(prompts):
        _rec("merge", prompts)
        return {p.id: {"merge": False} for p in prompts}

    def gencde(prompts):
        _rec("gencde", prompts)
        return {p.id: {"preferred_name": "generated", "definition": "d", "data_type": "categorical"} for p in prompts}

    def specgen(prompts):
        _rec("specgen", prompts)
        return {p.id: {"mappings": []} for p in prompts}

    return {
        "generate": generate,
        "split": split,
        "classify": classify,
        "merge": merge,
        "gencde": gencde,
        "specgen": specgen,
    }


_RESULT_PROMPT_FIELDS = (
    "ideal_prompts",
    "split_prompts",
    "group_assign_prompts",
    "merge_prompts",
    "specgen_prompts",
    "gencde_prompts",
    "concept_gate_prompts",
    "coherence_prompts",
    "kinds_prompts",
    "refine_prompts",
    "group_ideal_prompts",
)


def snapshot(result, log: dict) -> dict:
    """Everything a run emits: every prompt each stage was sent, every prompt on the result, the records, the
    Gate-1 concept groups and the partition the run returns (the fields that existed before the switch)."""
    sub = result.substrate
    return {
        "stages": log,
        "result_prompts": {name: [_ser(p) for p in getattr(result, name)] for name in _RESULT_PROMPT_FIELDS},
        "records": [asdict(r) for r in result.records],
        "concept_groups": [asdict(g) for g in result.concept_groups],
        "substrate": {
            "clusters": [[list(k) for k in cl] for cl in sub.clusters],
            "outlier": [list(k) for k in sub.outlier],
            "n_fields": sub.n_fields,
            "min_cluster_size": sub.min_cluster_size,
            "outliers_recovered": sub.outliers_recovered,
            "substrate_id": sub.substrate_id,
        },
    }


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, default=str)


# The partitions the legacy files hold: what the catalog-in engine produced — catalog rows inside cohort
# clusters, a catalog-only cluster, catalog rows among the HDBSCAN noise.
_LEGACY_CLUSTERS = [
    [
        ["CohortA", "age"],
        ["CohortA", "age_alt"],
        ["CohortB", "age_yrs"],
        [CDE, "AgeCDE"],
        [CDE, "AgeCDE_b"],
        [CDE, "AgeCDE_c"],
    ],
    [["CohortA", "smoke"], ["CohortB", "smoke_b"], [CDE, "SmokeCDE"]],
    [[CDE, "HeightCDE"], [CDE, "WeightCDE"]],
]
_LEGACY_OUTLIER = [
    ["CohortA", "employer_workplace_zip"],
    ["CohortA", "home_residence_zip"],
    [CDE, "ZipCDE"],
    [CDE, "ZipCDE_b"],
    [CDE, "ZipCDE_c"],
]


def legacy_payload(version: int) -> dict:
    """A substrate file as the pre-switch code wrote it: v1 has no ``outliers_recovered`` (loads as recovered),
    v2 records ``outliers_recovered: false`` (a base partition, so M10 runs on the catalog-bearing noise)."""
    payload: dict = {
        "version": version,
        "min_cluster_size": 15,
        "n_fields": len(_COHORT_A) + len(_COHORT_B) + len(_CATALOG),
        "clusters": _LEGACY_CLUSTERS,
        "outlier": _LEGACY_OUTLIER,
    }
    if version >= 2:
        payload["outliers_recovered"] = False
    return payload


def golden_snapshots(hf: SimpleNamespace, tmp_dir: Path) -> dict:
    """The scenarios the golden was captured from (pre-switch APIs only — see the module docstring).

    ``legacy_v1`` / ``legacy_v2_unrecovered`` replay a saved catalog-in file; ``fresh_catalog_in`` clusters
    fresh with the catalog in the clustering input (the old default), engine = :func:`fake_topic_model`.
    """
    import ddharmon.clustering.topic_engine as te
    from ddharmon.harmonization.leanb import harmonize_leanb
    from ddharmon.harmonization.substrate import load_substrate

    embedded = build_world(hf)
    out: dict = {}
    for name, version in (("legacy_v1", 1), ("legacy_v2_unrecovered", 2)):
        path = tmp_dir / f"{name}.json"
        path.write_text(json.dumps(legacy_payload(version)))
        log: dict = {}
        result = harmonize_leanb(
            embedded, substrate=load_substrate(path), chunk_cap=_CHUNK_CAP, **recording_stages(log)
        )
        out[name] = snapshot(result, log)
    real = te.topic_model_dictionaries
    te.topic_model_dictionaries = fake_topic_model
    try:
        log = {}
        result = harmonize_leanb(embedded, chunk_cap=_CHUNK_CAP, **recording_stages(log))
        out["fresh_catalog_in"] = snapshot(result, log)
    finally:
        te.topic_model_dictionaries = real
    return out


# ─────────────────────────────────────────────── tests ────────────────────────────────────────────────


@pytest.fixture
def embedded(hf):
    return build_world(hf)


@pytest.fixture
def spy_engine(monkeypatch):
    """Patch the fresh-clustering engine with :func:`fake_topic_model`, recording the dictionaries it is handed."""
    import ddharmon.clustering.topic_engine as te

    seen: list[list[str]] = []

    def spy(embedded_dicts, **kw):
        seen.append([ed.dictionary.cohort_name for ed in embedded_dicts])
        return fake_topic_model(embedded_dicts, **kw)

    monkeypatch.setattr(te, "topic_model_dictionaries", spy)
    return seen


def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def _catalog_keys(sub) -> list:
    return [k for cl in sub.clusters for k in cl if k[0] == CDE] + [k for k in sub.outlier if k[0] == CDE]


# ── 1. the default clustering input is cohort-only ──


def test_the_default_is_cohort_only_clustering():
    import inspect

    from ddharmon.harmonization.leanb import DEFAULT_CLUSTER_WITH_CATALOG, harmonize_leanb

    assert DEFAULT_CLUSTER_WITH_CATALOG is False
    assert inspect.signature(harmonize_leanb).parameters["cluster_with_catalog"].default is False


def test_default_clustering_input_excludes_the_catalog(embedded, spy_engine):
    from ddharmon.harmonization.leanb import harmonize_leanb

    result = harmonize_leanb(embedded, generate=None)
    assert spy_engine == [["CohortA", "CohortB"]], "the catalog dictionary reached UMAP+HDBSCAN"
    sub = result.substrate
    assert sub.clustered_with_catalog is False
    assert _catalog_keys(sub) == [], "a catalog row is in the partition"
    assert sub.n_fields == len(_COHORT_A) + len(_COHORT_B), "n_fields counts the rows that entered clustering"


def test_no_catalog_only_cluster_can_be_produced(embedded, spy_engine):
    """The engine forms catalog-only clusters (height, weight) when handed the catalog; by default it is not."""
    from ddharmon.harmonization.leanb import harmonize_leanb

    cohort_only = harmonize_leanb(embedded, generate=None).substrate
    assert cohort_only.clusters and all(any(d != CDE for d, _ in cl) for cl in cohort_only.clusters)
    catalog_in = harmonize_leanb(embedded, generate=None, cluster_with_catalog=True).substrate
    assert any(all(d == CDE for d, _ in cl) for cl in catalog_in.clusters), "the stand-in should show the old leak"


def test_m10_recovery_in_cohort_only_mode_never_reclusters_a_catalog_row(embedded):
    """A cohort-only partition is handed to M10 with catalog rows among its noise (hand-built; a real one
    never has them): recovery re-clusters the cohort rows only and leaves the catalog rows where they were."""
    from ddharmon.clustering.topic_engine import collect_inputs
    from ddharmon.harmonization.leanb import recover_outlier_clusters
    from ddharmon.harmonization.substrate import ClusteringSubstrate, clusters_from_substrate

    _docs, emb, refs, _ = collect_inputs(embedded)
    outlier = [tuple(k) for k in _LEGACY_OUTLIER]
    for mode, expect_catalog in ((False, False), (True, True)):
        sub = ClusteringSubstrate(
            clusters=[[("CohortA", "age")]], min_cluster_size=15, outlier=outlier, clustered_with_catalog=mode
        )
        clusters, out = recover_outlier_clusters(clusters_from_substrate(sub, refs), sub, emb, refs, cde_cohort=CDE)
        recovered = [(m.dictionary_name, m.variable_name) for m in clusters[-1].members]
        assert any(d == CDE for d, _ in recovered) is expect_catalog
        assert out.clustered_with_catalog is mode and out.outliers_recovered is True
        if not mode:
            assert sorted(recovered) == [("CohortA", "employer_workplace_zip"), ("CohortA", "home_residence_zip")]
            assert [k for k in out.outlier if k[0] == CDE] == [k for k in outlier if k[0] == CDE]


# ── 2. cluster_with_catalog=True is the old behaviour, byte for byte ──


def test_cluster_with_catalog_true_reproduces_the_pre_change_fresh_run(embedded, spy_engine):
    from ddharmon.harmonization.leanb import harmonize_leanb

    log: dict = {}
    result = harmonize_leanb(embedded, chunk_cap=_CHUNK_CAP, cluster_with_catalog=True, **recording_stages(log))
    assert spy_engine == [["CohortA", "CohortB", CDE]]
    assert canonical(snapshot(result, log)) == canonical(_golden()["fresh_catalog_in"])
    assert result.substrate.clustered_with_catalog is True
    assert result.substrate.n_fields == len(_COHORT_A) + len(_COHORT_B) + len(_CATALOG)


# ── 3. retrieval does not depend on the catalog being clustered ──


def _replay(embedded, sub, **kw):
    from ddharmon.harmonization.leanb import harmonize_leanb

    log: dict = {}
    result = harmonize_leanb(embedded, substrate=sub, **recording_stages(log), **kw)
    return snapshot(result, log)


def test_candidates_are_identical_for_a_fixed_partition_in_both_modes(embedded):
    """One partition, recorded once as cohort-only and once as catalog-in: every prompt (candidate lists
    included), record and concept group is identical. Retrieval reads the catalog's OWN vectors and BM25
    index, never the clustering matrix."""
    from ddharmon.harmonization.substrate import ClusteringSubstrate

    partition = [
        [("CohortA", "age"), ("CohortA", "age_alt"), ("CohortB", "age_yrs")],
        [("CohortA", "smoke"), ("CohortB", "smoke_b")],
    ]
    outlier = [("CohortA", "employer_workplace_zip"), ("CohortA", "home_residence_zip")]
    snaps = []
    for mode in (False, True):
        sub = ClusteringSubstrate(
            clusters=partition,
            min_cluster_size=15,
            outlier=outlier,
            outliers_recovered=True,
            clustered_with_catalog=mode,
        )
        snaps.append(_replay(embedded, sub, chunk_cap=_CHUNK_CAP))
    assert canonical(snaps[0]) == canonical(snaps[1])
    assigned = snaps[0]["result_prompts"]["group_assign_prompts"]
    assert assigned and all(p["context"]["candidates"] for p in assigned)
    assert assigned[0]["context"]["candidates"][0]["designation"].startswith("Age")


def test_catalog_rows_inside_a_cluster_never_reach_retrieval(embedded):
    """The same cohort members with and without catalog co-members (chunking off, so only retrieval could
    tell them apart): the candidate lists, prompts and records are identical."""
    from ddharmon.harmonization.substrate import ClusteringSubstrate

    cohort_only = ClusteringSubstrate(
        clusters=[[tuple(k) for k in cl if k[0] != CDE] for cl in _LEGACY_CLUSTERS[:2]],
        min_cluster_size=15,
        outliers_recovered=True,
        clustered_with_catalog=False,
    )
    catalog_in = ClusteringSubstrate(
        clusters=[[tuple(k) for k in cl] for cl in _LEGACY_CLUSTERS],
        min_cluster_size=15,
        outliers_recovered=True,
        clustered_with_catalog=True,
    )
    a = _replay(embedded, cohort_only, chunk_cap=None)
    b = _replay(embedded, catalog_in, chunk_cap=None)
    for stage in ("generate", "split", "classify"):
        assert canonical(a["stages"][stage]) == canonical(b["stages"][stage])
    assert canonical(a["records"]) == canonical(b["records"])


def test_fresh_runs_retrieve_identical_candidates_in_both_modes(embedded, spy_engine):
    """End to end from FRESH clustering: the stand-in groups the cohort rows identically in both modes
    (catalog rows only ride along), so with chunking off every cluster's candidates and ids match."""
    from ddharmon.harmonization.leanb import harmonize_leanb

    runs = []
    for mode in (False, True):
        log: dict = {}
        harmonize_leanb(embedded, chunk_cap=None, cluster_with_catalog=mode, **recording_stages(log))
        runs.append(log)
    for stage in ("generate", "classify"):
        assert canonical(runs[0][stage]) == canonical(runs[1][stage])


# ── 4. a legacy substrate replays byte-identically to the pre-change code ──


@pytest.mark.parametrize(("name", "version"), [("legacy_v1", 1), ("legacy_v2_unrecovered", 2)])
@pytest.mark.parametrize("kwarg", [None, False, True])
def test_a_legacy_substrate_replays_byte_identically(embedded, tmp_path, name, version, kwarg):
    """Every substrate saved before the switch has no mode field and is CATALOG-IN; the default kwarg (and
    either explicit one) must not change a single prompt, id, record or the returned partition."""
    from ddharmon.harmonization.substrate import load_substrate

    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(legacy_payload(version)))
    sub = load_substrate(path)
    assert sub.clustered_with_catalog is True
    extra = {} if kwarg is None else {"cluster_with_catalog": kwarg}
    got = _replay(embedded, sub, chunk_cap=_CHUNK_CAP, **extra)
    assert canonical(got) == canonical(_golden()[name])


def test_the_legacy_golden_is_sensitive_to_the_mode(embedded):
    """Guard the guard: replaying the SAME legacy partition as cohort-only would change the ids (M2 chunking
    no longer counts the catalog rows), so the byte-identity above really pins the recorded mode."""
    from ddharmon.harmonization.substrate import ClusteringSubstrate

    sub = ClusteringSubstrate(
        clusters=[[tuple(k) for k in cl] for cl in _LEGACY_CLUSTERS],
        min_cluster_size=15,
        outlier=[tuple(k) for k in _LEGACY_OUTLIER],
        outliers_recovered=True,
        clustered_with_catalog=False,
    )
    got = _replay(embedded, sub, chunk_cap=_CHUNK_CAP)
    ids = sorted(p["id"] for p in got["stages"]["generate"])
    legacy_ids = sorted(p["id"] for p in _golden()["legacy_v1"]["stages"]["generate"])
    assert ids != legacy_ids


# ── 5. a new substrate records its mode and replays in it ──


def test_a_fresh_substrate_records_its_mode_and_round_trips(embedded, spy_engine, tmp_path):
    from ddharmon.harmonization.leanb import harmonize_leanb
    from ddharmon.harmonization.substrate import load_substrate, save_substrate

    for mode in (False, True):
        sub = harmonize_leanb(embedded, generate=None, cluster_with_catalog=mode).substrate
        assert sub.clustered_with_catalog is mode
        path = save_substrate(sub, tmp_path / f"sub_{mode}.json")
        assert json.loads(path.read_text())["clustered_with_catalog"] is mode
        back = load_substrate(path)
        assert back.clustered_with_catalog is mode
        assert back.substrate_id == sub.substrate_id and back.n_fields == sub.n_fields


def test_a_replay_runs_in_the_recorded_mode_not_the_kwarg(embedded, spy_engine, tmp_path):
    """Fresh cohort-only -> save -> replay with ``cluster_with_catalog=True``: the identical partition and
    prompts (the kwarg governs only fresh clustering), and the engine is never called on a replay."""
    from ddharmon.harmonization.leanb import harmonize_leanb
    from ddharmon.harmonization.substrate import load_substrate, save_substrate

    log0: dict = {}
    fresh = harmonize_leanb(embedded, chunk_cap=_CHUNK_CAP, **recording_stages(log0))
    path = save_substrate(fresh.substrate, tmp_path / "cohort_only.json")
    for kwarg in (False, True):
        log: dict = {}
        replayed = harmonize_leanb(
            embedded,
            substrate=load_substrate(path),
            chunk_cap=_CHUNK_CAP,
            cluster_with_catalog=kwarg,
            **recording_stages(log),
        )
        assert canonical(snapshot(replayed, log)) == canonical(snapshot(fresh, log0))
        assert replayed.substrate.clustered_with_catalog is False
    assert len(spy_engine) == 1, "a replay re-clustered"


def test_a_cohort_only_replay_never_chunks_a_catalog_row(embedded):
    """A partition recorded cohort-only is reconstructed without catalog rows even if some were written into
    it by hand, so M2 chunking (which is clustering too) never sees one; recorded catalog-in keeps them."""
    from ddharmon.harmonization.substrate import ClusteringSubstrate

    def ids(mode):
        sub = ClusteringSubstrate(
            clusters=[[tuple(k) for k in cl] for cl in _LEGACY_CLUSTERS],
            min_cluster_size=15,
            outliers_recovered=True,
            clustered_with_catalog=mode,
        )
        return sorted(p["id"] for p in _replay(embedded, sub, chunk_cap=_CHUNK_CAP)["stages"]["generate"])

    pure = ClusteringSubstrate(
        clusters=[[tuple(k) for k in cl if k[0] != CDE] for cl in _LEGACY_CLUSTERS[:2]],
        min_cluster_size=15,
        outliers_recovered=True,
        clustered_with_catalog=False,
    )
    pure_ids = sorted(p["id"] for p in _replay(embedded, pure, chunk_cap=_CHUNK_CAP)["stages"]["generate"])
    assert ids(False) == pure_ids
    assert ids(True) != pure_ids


def test_a_substrate_file_without_the_mode_loads_as_catalog_in(tmp_path):
    from ddharmon.harmonization.substrate import ClusteringSubstrate, build_substrate, load_substrate

    p = tmp_path / "v2.json"
    p.write_text(
        json.dumps(
            {
                "version": 2,
                "min_cluster_size": 15,
                "clusters": [[["A", "x"]]],
                "outlier": [],
                "outliers_recovered": True,
            }
        )
    )
    assert load_substrate(p).clustered_with_catalog is True
    assert ClusteringSubstrate(clusters=[], min_cluster_size=15).clustered_with_catalog is True
    assert build_substrate([], min_cluster_size=15).clustered_with_catalog is True


def test_recovery_and_the_recovered_flag_keep_the_mode(embedded, spy_engine):
    """M10 folds its clusters into a NEW substrate; that substrate must still say cohort-only."""
    from ddharmon.harmonization.leanb import harmonize_leanb

    sub = harmonize_leanb(embedded, generate=None).substrate
    assert sub.outliers_recovered is True and sub.clustered_with_catalog is False
    assert any(
        set(cl) == {("CohortA", "employer_workplace_zip"), ("CohortA", "home_residence_zip")} for cl in sub.clusters
    )


def test_a_cohort_only_run_with_no_cohort_rows_is_empty_not_a_crash(hf):
    """Only the catalog was passed: there is nothing to cluster or harmonize. The old code handed the catalog
    alone to UMAP+HDBSCAN; the cohort-only path must return an empty partition, not stack an empty matrix."""
    from ddharmon.harmonization.leanb import harmonize_leanb

    result = harmonize_leanb([_ed(hf, CDE, _CATALOG)], generate=None)
    assert result.ideal_prompts == [] and result.substrate.clusters == []
    assert result.substrate.clustered_with_catalog is False and result.substrate.n_fields == 0
