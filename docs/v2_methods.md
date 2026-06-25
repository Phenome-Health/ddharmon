# ddharmon v2 — Methods & Lineage

**ddharmon v2** replaces v1's sub-cluster-anchored pipeline with a **lean head/tail architecture that
leads with assignment to the given CDE backbone**. Where [v1](v1_methods.md) clustered cohort variables
and CDEs together and anchored each value sub-cluster to its most-central in-cluster CDE, v2 treats
harmonization of a covered concept as **assignment to an existing CDE** and routes only the *uncovered*
tail to generation/clustering. This is the division of labor the research harness settled empirically.

This document describes (1) why the architecture changed, (2) the v2 pipeline and its algorithms,
(3) how v2 is evaluated, and (4) how it relates to v1 and the literature.

---

## 1. Why the architecture changed

The v1 "cluster → sub-cluster → anchor" flow treats clustering as the primary engine. A sequence of
benchmark experiments (against external ground truth, not self-defined metrics) showed that clustering
is the wrong primary engine for the *covered* part of the problem:

- **Two buckets, scored separately.** Harmonization splits into a **head** — concepts that already have
  a CDE in the backbone — and a **tail** with no matching CDE. Blending them hides the truth (a trivial
  "everything is novel" baseline beats a real pipeline on a blended metric because the tail dominates),
  so the two are measured independently.
- **Assignment dominates the head; clustering only helps the diffuse tail.** A cluster-first vs
  assignment-first pooling A/B found assignment-first pools common backbone-covered concepts ~6× better
  (micro), while clustering's advantage is confined to the rare tail and is *diffuse* — not bankable by
  confidence routing. So: **assign the head, cluster/generate the tail.**
- **The head assignment engine is one fused LLM call.** Ranking a wide hybrid-retrieved candidate pool
  *and* committing a verdict in a single call beats a two-call rerank-then-verdict design on both
  accuracy and cost. Unguarded, it over-adopts the tail — which is exactly why it sits behind a
  coverage-aware design rather than being a global swap.
- **Coverage routing is benchmark-capped on exact-ID gold, so the human gate is the arbiter.** Four
  independent routing signals all plateaued on the CDEMapper gold because "exact tinyId absent" is not
  the same as "semantically novel" (~41% of the nominal tail has a defensible near-synonym). The
  adopt/refine/novel cutoff is therefore deliberately **strict**, with final calibration deferred to
  expert (EITL) review of the routed output.

The net design: **assignment-first for the head, GenCDE/clustering residual for the tail, an
independent "ideal CDE" as the coverage anchor, and a human gate for the boundary.**

---

## 2. The v2 pipeline

```
ingest (cohorts + CDE catalog)
  → embed (semantic vectors, SQLite-cached)
    → cluster concepts (retrieval/batching scaffolding, not the decision engine)
      → hybrid retrieve top-k CDE candidates   (BM25 lexical ⊕ dense centroid, RRF)
        → generate-ideal                        (LLM, no candidates → independent coverage anchor)
          → fused assign                        (LLM, rank by the ideal → adopt/refine/novel + pick)
            → route: adopt/refine → CDE ;  novel → GenCDE / clustering residual (tail)
              → EITL review queue
```

| Stage | Method | Notes |
|-------|--------|-------|
| **Ingest / embed** | Same as v1 — role-mapped CSV/TSV loader, `all-mpnet-base-v2` (768-d, L2-normalized, SQLite-cached) | The CDE catalog is loaded as a cohort so its text is embedded into the same space. v2 drops v1's separate *value* sub-clustering vector; value/encoding metadata is routed to the LLM prompt (symbolic), not the geometric vector. |
| **Cluster** | Concept clustering over semantic vectors | Demoted to **scaffolding** — it batches near-duplicate fields for one assignment call and provides a centroid for dense retrieval. It is no longer the decision engine. |
| **Hybrid retrieve** | `BM25` (lexical, over rich CDE text) ⊕ dense centroid cosine, fused by **Reciprocal Rank Fusion**; top-k=20 | The candidate generator. Hybrid beats dense at every k (recall@5 0.447 → 0.632 on the CDEMapper gold); a dense-rich control confirmed the gain is real lexical signal. Reusable in `ddharmon.matching` (`BM25`, `hybrid_topk`). |
| **Generate-ideal** | LLM call describing the *ideal* CDE for the concept **with no candidates shown** | An independent coverage anchor: what *should* exist, formed without being biased by what retrieval happened to surface. Anchors the novel decision. |
| **Fused assign** | One LLM call: rank the retrieved candidates *by the ideal*, then commit `adopt`/`refine`/`novel` **and** the chosen candidate | Beats a two-call rerank-then-verdict design (in-backbone assignment 0.458 → 0.521) at half the cost. The pick resolves to a real CDE designation + NIH tinyId. |
| **Route** | `adopt`/`refine` → CDE assignment; `novel` → GenCDE / clustering residual (the tail) | The head/tail split, applied per concept. |
| **Review** | `export_leanb_eitl_queue(...)` → EITL TSV/CSV; `write_records_json(...)` | Nothing is auto-applied. EITL verdicts are the locked acceptance gate and the source of the cutoff calibration. |

Two expert-review fixes are built into the assign stage:

- **Axis preservation** — a candidate that names a *different* specific qualifier (condition, body site,
  time window) than the source is treated as `novel`, not a refinement, so templated families aren't
  collapsed onto one condition-specific CDE. On the gold this lifts out-of-backbone novel-precision
  0.378 → 0.451 with no loss of in-backbone assignment (0.521) — a *semantic* fix in the engine, not a
  router.
- **Retrieval floor** (`retrieval_floor`, default 0.30) — downgrades an adopt/refine to `novel` when the
  chosen candidate's dense cosine is below the floor (the engine force-fit the least-bad candidate when
  nothing was actually close). A *bottom* floor, not a mid routing threshold; records carry `chosen_cos`
  + `floored` for audit.

The pipeline keeps v1's **split-for-testability** shape: `prepare_leanb` (retrieve + build generate
prompts) → `prepare_assign` (build assign prompts from the ideals) → `assemble_leanb` (parse responses
into routed `LeanBRecord`s). Each stage runs inline *or* via the offline Anthropic Batch API; each prompt
record carries the context to assemble its own decision. The accuracy stack — **hybrid retrieval → LLM
rerank(top-20) → adopt/refine/novel** — is the empirically-selected configuration. Code lives in
`ddharmon.harmonization` (`harmonize_leanb`, `prepare_leanb`, `prepare_assign`, `assemble_leanb`,
`LeanBRecord`, `CdeBackbone`); `ddharmon.matching` holds the reusable hybrid retrieval.

---

## 3. Evaluation

v2 is measured against **three external ground-truth benchmarks** (the `benchmarks/` package; portable,
`$0`, reproducible under `PYTHONHASHSEED=0`) — and a locked in-domain human gate:

| Benchmark | Question | Ground truth | Headline |
|-----------|----------|--------------|----------|
| **A — CDEMapper** | Are we matching the **right CDE**? | Yale CDE-Mapping-Tool (494 field→CDE) | hybrid retrieval recall@5 0.632; fused assignment (in-backbone) 0.521 |
| **B — PhenX** | Do same-concept vars from **different cohorts** co-cluster? | PhenX↔dbGaP crosswalk | embedding separability Δ0.536; clustering's edge is diffuse (motivates assignment-first) |
| **C — ATHLOS** | Are the **value recodes** generated correctly? | ATHLOS harmonisation scripts (284 recode golds) | LLM recode pair-accuracy 0.832 → **0.869 with question_text context** |

Two evaluation principles carry over from the research:

- **Benchmark-usage policy** — CDEMapper is the *development* set (already tuned on), PhenX is a *held-out*
  generalization check (measure, don't tune), and EITL human verdicts are the *locked* in-domain
  acceptance gate. Only mechanistically-justified changes are adopted, never benchmark-chasing.
- **The value layer needs question context.** On Benchmark C, feeding the source field's `question_text`
  (a FieldRole ddharmon already carries) into the recode generator lifts whole-variable accuracy ~7pp by
  resolving polarity/granularity judgment calls — so the transform layer, when built, must include it.

EITL `transform_review` is the value-layer analog of the match-review gate: nothing is auto-applied, and
the strict adopt/refine/novel cutoff is calibrated from human verdicts on the routed output.

---

## 4. How v2 differs from v1

| Axis | v1 (sub-cluster-anchored) | v2 (lean head/tail) |
|------|---------------------------|---------------------|
| Primary engine | Clustering → value sub-cluster → anchor | **Assignment to the backbone**; clustering is retrieval/batching scaffolding |
| Head vs tail | Not distinguished | **Explicit split** — assign the covered head, generate/cluster the tail |
| Candidate generation | Medoid of the sub-cluster → best in-cluster CDE | **Hybrid (BM25 ⊕ dense, RRF) top-20** retrieval |
| Coverage anchor | — | **Generate-ideal** (LLM describes the ideal CDE with no candidates shown) |
| LLM call | One classify-only call (`adopt`/`refine`/`novel`) | One **fused** call (rank by the ideal **+** verdict **+** pick) |
| Vectors | Dual (semantic + value) | Single semantic; value/encoding metadata → prompt, not vector |
| Evaluation | — | Three external benchmarks (A/B/C) + EITL locked gate |

v2 keeps v1's invariants: cohort-agnostic (no cohort identity in clustering or assignment), CDEMapper
cited as prior art rather than lifted, dataclass models, and EITL as the human gate. It supersedes
`harmonize_dictionaries` (v1) as the default pipeline.

---

## 5. Deferred / open

- **Adopt/refine/novel calibration** — the cutoff (and the retrieval floor τ) are intentionally strict
  pending the first EITL human verdicts on the v2 routed output.
- **Transform / value-recode spec authoring** — v2 stops at the assignment decision and hands off to
  EITL; per-variable recode specs (with the validated `question_text` context) are the next build, and
  Benchmark C is the yardstick for them.
- **Tail handling** — GenCDE generation and residual re-clustering for the no-CDE tail are scoped but
  deprioritized relative to the head assignment engine.

See the [Related work](../README.md#related-work) section of the README for the field map, and
[`benchmarks/README.md`](../benchmarks/README.md) for the standing evaluation benchmarks (CDEMapper
retrieval, PhenX cross-cohort co-clustering) used to settle the architecture.
