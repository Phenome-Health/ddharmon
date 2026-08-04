"""Composite / derived-variable builder — can THIS run's harmonized concepts support a published score?

ddharmon harmonizes variables one concept at a time, but researchers work with *composite* variables built
from several concepts — a frailty phenotype, an intrinsic-capacity score, an SES index. This module answers
the two questions standing between a harmonization run and such a score:

  (a) **Feasibility** — given the concepts this run actually harmonized, can the score be computed? Fully,
      partially, or not at all — and in which cohorts?
  (b) **Composition** — which harmonized concepts compose it, and how (per-component coding + combination).

The definition always comes from a real document (:mod:`ddharmon.harmonization.score_sources`) — a paper, a
supplement table, a repo that implements the index — never from a model's recollection of it.

Four stages, only two of which cost an LLM call::

    extract_score_definition()   1 call   document text -> ScoreDefinition (transcription, not invention)
    match_components()           1 call   hybrid-retrieval shortlist per component -> one LLM judge pass
    assess_feasibility()         0 calls  deterministic per-cohort coverage + verdict
    build_composite_spec()       0 calls  the ordered derivation recipe

    derive_composite()                    the entry point that runs all four

Two score shapes drive the design, because published composites split along this line: **criteria-based**
(Fried phenotype — k of n criteria present) and **deficit-accumulation** (FI-Lab — proportion of deficits
outside their reference range). :class:`CompositeKind` covers both plus the plain sum / weighted-index /
z-composite forms.

Grounding is structural, not merely instructed: the judge may only choose among the ids RETRIEVED for that
component, and anything else is dropped — the component is then reported MISSING, never fabricated.
Concepts are referenced by record id rather than label, because real concept labels are whole sentences.

HARD SCOPE (the metadata-only invariant): ddharmon reads data dictionaries, never participant data. This
emits a *recipe* — a spec a human reviews and a notebook applies to their own rows. Feasibility is therefore
about which cohorts CONTAIN the components; effective participant N is not derivable from metadata and is
never claimed. A cutoff or reference range the source does not state is never invented: the component is
marked ``needs_review`` and left to a reviewer.

Library use::

    from ddharmon.harmonization import derive_composite, fetch_source

    run = harmonize_leanb(...)                          # a LeanBResult
    source = fetch_source("10.1007/s11357-017-9993-7")   # the FI-Lab paper
    spec = derive_composite(source, run.records, client.complete).spec
    print(spec.feasibility.verdict, [m.component for m in spec.unmatched])

A composite is ultimately a *score derivation rule* over CDEs — the slot the NIH CDE model calls
``derivationRules`` with ``ruleType: "score"`` (see :class:`~ddharmon.harmonization.models.GenCDE`).
Serializing a :class:`CompositeSpec` into that slot belongs to the export layer, not here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ddharmon.harmonization.models import GenCDE, LeanBRecord, TransformKind, TransformSpec
from ddharmon.harmonization.parse import extract_json, salvage_objects
from ddharmon.harmonization.score_sources import ScoreSource
from ddharmon.matching.lexical import BM25, hybrid_topk

_DEFAULT_TOP_K = 8  # candidate concepts shown to the judge per component
_MAX_COMPONENTS = 80  # FI-Combined is 68 items — the largest published composite we target
_MAX_IDEAL_CHARS = 240  # how much of a concept's generated-ideal text feeds retrieval
_EXTRACT_MAX_TOKENS = 8192
_MATCH_MAX_TOKENS = 8192

# ``complete(prompt, *, system, max_tokens) -> str`` — matches AnthropicClient.complete / LiteLLMClient.
CompleteFn = Callable[..., str]
# ``embed(texts) -> (N, D) L2-normalized array`` — matches EmbeddingProvider.embed. Optional: without it,
# retrieval falls back to BM25 alone (so the builder works without the `embeddings` extra installed).
EmbedFn = Callable[[list[str]], Any]

_METADATA_CAVEAT = (
    "Metadata only: a component counts as present when the cohort's data dictionary describes it, which says "
    "nothing about missingness at the participant level — effective N and statistical power cannot be derived "
    "from metadata."
)


class CompositeKind(StrEnum):
    """How a composite's components combine into one score.

    The two clinically dominant shapes are CRITERIA_COUNT (Fried frailty phenotype: 5 criteria, frail at ≥3)
    and DEFICIT_PROPORTION (FI-Lab: each item coded 1 outside its reference range, score = deficits ÷ items
    considered, range 0–1) — a builder that handles only one of them cannot serve the literature.
    """

    CRITERIA_COUNT = "criteria_count"  # count criteria met, usually with a cut-point (Fried)
    DEFICIT_PROPORTION = "deficit_proportion"  # deficits present ÷ deficits considered (frailty index)
    SUM = "sum"  # plain item sum (PHQ-9, CES-D)
    WEIGHTED_SUM = "weighted_sum"  # per-item weights (Charlson)
    Z_COMPOSITE = "z_composite"  # mean of standardized components (cohort-relative)
    CUSTOM = "custom"  # anything else — the source's rule is carried verbatim for review


class CodingKind(StrEnum):
    """How ONE component's harmonized values become its contribution to the score.

    Deliberately parallel to :class:`~ddharmon.harmonization.models.TransformKind` (the source→CDE recode
    vocabulary), because the same distinctions matter one layer up. UNSTATED is the honest-failure member:
    the source named the component but not how to code it.
    """

    THRESHOLD = "threshold"  # 1 when outside a stated cutoff / reference range, else 0
    CATEGORICAL = "categorical"  # stated response code -> stated points
    IDENTITY = "identity"  # the harmonized value enters the score as-is
    UNIT = "unit"  # needs a unit conversion before it can be compared to the cutoff
    ARITHMETIC = "arithmetic"  # derived from >1 input via a stated formula
    DATA_DEPENDENT = "data_dependent"  # cohort-relative (z-score, quintile) — resolved at apply-time
    UNSTATED = "unstated"  # the document does not say how to code it


# Coding kinds that can never be auto-applied from metadata alone (mirrors the transform layer's rule that
# ARITHMETIC always goes to review, and adds the two that are unresolvable without the source or the data).
_ALWAYS_REVIEW = frozenset({CodingKind.UNSTATED, CodingKind.ARITHMETIC, CodingKind.DATA_DEPENDENT})


@dataclass
class ComponentCoding:
    """How one component is scored — transcribed from the source, never inferred.

    Every threshold/range field is free text held verbatim (``"<130 g/L (men), <120 g/L (women)"``) rather
    than parsed into numbers: sex-specific ranges, unit variants and inequality directions are exactly where
    silent misreading would corrupt a score, so a human sees what the paper said. ``needs_review`` is derived
    in :meth:`__post_init__` and never needs setting by hand.
    """

    kind: CodingKind = CodingKind.UNSTATED
    cutoff: str = ""  # the stated cut-point ("lowest quintile", "≥3 s", "<130 g/L")
    reference_range: str = ""  # the stated normal range a deficit is scored against
    code_map: dict[str, str] = field(default_factory=dict)  # response code -> points, when stated
    formula: str = ""  # for ARITHMETIC, the stated expression
    units: str = ""  # units the cutoff is expressed in
    stated_in_source: bool = False
    needs_review: bool = False

    def __post_init__(self) -> None:
        self.kind = CodingKind(self.kind)
        if self.kind in _ALWAYS_REVIEW or not self.stated_in_source:
            self.needs_review = True


@dataclass
class ScoreComponent:
    """One element of a composite: what it measures, whether the score can omit it, and how it is coded."""

    name: str
    definition: str = ""
    required: bool = True
    weight: float | None = None
    coding: ComponentCoding = field(default_factory=ComponentCoding)


@dataclass
class ScoreDefinition:
    """A published composite score, transcribed from its source document.

    ``combination_rule`` and ``threshold`` hold the source's own wording so a reviewer can check the
    structured fields against it. ``source`` carries provenance (URL/file + sha256 of the text read).
    """

    name: str
    kind: CompositeKind = CompositeKind.CUSTOM
    components: list[ScoreComponent] = field(default_factory=list)
    citation: str = ""
    combination_rule: str = ""
    threshold: str = ""
    notes: str = ""
    # The item count the DOCUMENT claims ("a 32-item index"), independent of how many items we could actually
    # read out of it. When it exceeds `len(components)` the source was incomplete — a publisher page whose
    # item table did not survive text extraction, say — and the gap is surfaced as a caveat rather than
    # quietly filled in.
    stated_n_items: int | None = None
    source: ScoreSource | None = None

    def __post_init__(self) -> None:
        self.kind = CompositeKind(self.kind)

    @property
    def required_components(self) -> list[ScoreComponent]:
        """The components a computable score needs. A transcription where NOTHING was marked required is an
        artifact, not a score with no requirements, so every component counts in that case."""
        required = [c for c in self.components if c.required]
        return required or list(self.components)

    @property
    def under_enumerated(self) -> int:
        """How many items the document claims beyond those actually transcribed (0 when complete)."""
        return max(0, (self.stated_n_items or 0) - len(self.components))

    @property
    def provenance(self) -> str:
        return self.source.provenance if self.source else ""


@dataclass
class ConceptEntry:
    """One harmonized concept from the run, as the closed world a component may be matched to.

    ``column`` is the name the concept's harmonized column carries downstream — the assigned CDE id, else the
    GenCDE's preferred name, else the record id — so a derivation expression lines up with the columns the
    exported harmonization notebook actually produces.
    """

    concept_id: str
    concept: str
    cohorts: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=list)
    verdict: str = ""
    cde_id: str | None = None
    gencde_name: str = ""
    units: str = ""
    data_type: str = ""
    ideal_cde: str = ""

    @property
    def column(self) -> str:
        return self.cde_id or self.gencde_name or self.concept_id

    @property
    def retrieval_text(self) -> str:
        """The text retrieval scores against: the concept label, its CDE name, and a slice of its ideal."""
        parts = [self.concept, self.cde_id or "", self.ideal_cde[:_MAX_IDEAL_CHARS]]
        return " ".join(p for p in parts if p)


@dataclass
class ComponentMatch:
    """The verdict for ONE component: the run concept that measures it, or an honest gap.

    ``shortlist`` is the audit trail — the concept ids retrieval offered the judge — so a missing component
    can be told apart from a component the judge saw good candidates for and still rejected.
    """

    component: str
    concept_id: str | None = None
    concept: str = ""
    column: str = ""
    cohorts: list[str] = field(default_factory=list)
    source_variables: list[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    required: bool = True
    pinned: bool = False  # set by a reviewer override rather than the judge
    shortlist: list[str] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return bool(self.concept_id)


@dataclass
class CohortCoverage:
    """Which of the score's components one cohort can supply, and whether that is enough to compute it."""

    cohort: str
    present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # REQUIRED components this cohort lacks
    computable: bool = False


@dataclass
class FeasibilityReport:
    """The honest answer to "can this run support the score?" — verdict, gaps, per-cohort coverage."""

    verdict: str = "infeasible"  # full | partial | infeasible
    n_required: int = 0
    n_required_matched: int = 0
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    needs_review: list[str] = field(default_factory=list)  # matched, but the coding needs a human decision
    per_cohort: list[CohortCoverage] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def computable_cohorts(self) -> list[str]:
        return [c.cohort for c in self.per_cohort if c.computable]


@dataclass
class DerivationStep:
    """One ordered step of the recipe: code a component, combine them, or apply the score's cut-point."""

    order: int
    kind: str  # code_component | combine | threshold
    description: str
    expression: str = ""
    component: str = ""
    concept_id: str | None = None
    needs_review: bool = False


@dataclass
class CompositeSpec:
    """The deliverable: definition + grounded component matches + feasibility + the derivation recipe."""

    definition: ScoreDefinition
    matches: list[ComponentMatch] = field(default_factory=list)
    feasibility: FeasibilityReport = field(default_factory=FeasibilityReport)
    derivation: list[DerivationStep] = field(default_factory=list)
    units: str = ""
    validation_rules: list[str] = field(default_factory=list)

    @property
    def matched(self) -> list[ComponentMatch]:
        return [m for m in self.matches if m.matched]

    @property
    def unmatched(self) -> list[ComponentMatch]:
        return [m for m in self.matches if not m.matched]


@dataclass
class CompositeResult:
    """Return of :func:`derive_composite`: the spec plus what it cost and what it reasoned over."""

    spec: CompositeSpec
    n_concepts_indexed: int = 0
    calls_made: int = 0


# --- the closed world -------------------------------------------------------------------------


def build_concept_index(records: Sequence[LeanBRecord]) -> list[ConceptEntry]:
    """Index a run's records as the ONLY concepts a composite may be built from.

    Unlike :func:`~ddharmon.harmonization.analysis_ideas.build_concept_digest`, this keeps **single-cohort**
    concepts: a component present in one cohort still makes the score computable *there*, and hiding that
    would misreport feasibility. Records with no concept label and no CDE are skipped (nothing to match).
    """
    index: list[ConceptEntry] = []
    seen: set[str] = set()
    for r in records:
        concept = (r.concept or "").strip()
        concept_id = (r.group_id or r.cluster_id or "").strip()
        if not concept_id or concept_id in seen:
            continue
        if not concept and not r.cde_id:
            continue
        gencde = r.gencde
        index.append(
            ConceptEntry(
                concept_id=concept_id,
                concept=concept or (r.cde_id or ""),
                cohorts=sorted({c for c in (r.cohorts or []) if c}),
                members=list(r.member_variable_names or []),
                verdict=r.verdict or "",
                cde_id=r.cde_id,
                gencde_name=(gencde.preferred_name if gencde else ""),
                units=(gencde.units or "" if gencde else "") or _target_unit(r),
                data_type=(gencde.data_type if gencde else ""),
                ideal_cde=r.ideal_cde or "",
            )
        )
        seen.add(concept_id)
    return index


def _target_unit(record: LeanBRecord) -> str:
    """The unit the record's transforms harmonize onto, when any transform states one."""
    for t in record.transforms or []:
        if getattr(t, "target_unit", None):
            return str(t.target_unit)
    return ""


def _first(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """First present, non-None key among ``names`` — serialized records differ only in camel/snake casing."""
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return default


def records_from_payload(blob: Any) -> list[LeanBRecord]:
    """Rehydrate the records a composite needs from a SERIALIZED run, in either casing.

    Accepts a bare list, ``{"records": [...]}`` (core ``write_records_json``), or
    ``{"result": {"records": [...]}}`` (the UI contract / a demo snapshot), with keys in snake_case or
    camelCase. Deliberately PARTIAL: it recovers only what :func:`build_concept_index` reads — concept, ids,
    cohorts, members, CDE, GenCDE name/units, transform target units — and ignores candidates, cosines and
    coherence flags. It is not a general deserializer.

    Exists so every consumer of a serialized run (the CLI harness, a web backend) shares ONE reader instead
    of each re-deriving the mapping from the contract.
    """
    if isinstance(blob, list):
        payload: list[Any] = list(blob)
    elif isinstance(blob, Mapping):
        nested = blob.get("result")
        candidate = blob.get("records")
        if candidate is None and isinstance(nested, Mapping):
            candidate = nested.get("records")
        if not isinstance(candidate, list):
            raise ValueError("no `records` array found — expected a records.json or a UIResult/demo snapshot")
        payload = list(candidate)
    else:
        raise ValueError(f"cannot read records from {type(blob).__name__}")

    records: list[LeanBRecord] = []
    for raw in payload:
        if not isinstance(raw, Mapping):
            continue
        cde = _first(raw, "cde")
        cde_id = cde.get("id") if isinstance(cde, Mapping) else _first(raw, "cde_id", "cdeId")
        gencde_raw = _first(raw, "gencde")
        gencde = None
        if isinstance(gencde_raw, Mapping):
            gencde = GenCDE(
                gencde_id=str(_first(gencde_raw, "gencde_id", "gencdeId", default="")),
                preferred_name=str(_first(gencde_raw, "preferred_name", "preferredName", default="")),
                definition=str(_first(gencde_raw, "definition", default="")),
                data_type=str(_first(gencde_raw, "data_type", "dataType", default="")),
                units=_first(gencde_raw, "units"),
            )
        transforms = [
            TransformSpec(
                source_variable=str(_first(t, "source_variable", "sourceVariable", default="")),
                target_cde_id=str(_first(t, "target_cde_id", "targetCdeId", default="")),
                kind=_transform_kind(_first(t, "kind")),
                target_unit=_first(t, "target_unit", "targetUnit"),
            )
            for t in (_first(raw, "transforms", default=[]) or [])
            if isinstance(t, Mapping)
        ]
        group_id = str(_first(raw, "group_id", "groupId", "id", default=""))
        records.append(
            LeanBRecord(
                cluster_id=str(_first(raw, "cluster_id", "clusterId", default=group_id.split("#")[0])),
                verdict=str(_first(raw, "verdict", default="")),
                route=str(_first(raw, "route", default="")),
                group_id=group_id,
                concept=str(_first(raw, "concept", default="")),
                cde_id=str(cde_id) if cde_id else None,
                ideal_cde=str(_first(raw, "ideal_cde", "idealCde", default="")),
                cohorts=[str(c) for c in (_first(raw, "cohorts", default=[]) or [])],
                member_variable_names=[
                    str(m) for m in (_first(raw, "member_variable_names", "members", default=[]) or [])
                ],
                n_members=int(_first(raw, "n_members", "nMembers", default=0) or 0),
                gencde=gencde,
                transforms=transforms,
            )
        )
    return records


def _transform_kind(value: Any) -> TransformKind:
    """A serialized transform's kind, tolerating an unknown/absent one (only ``target_unit`` is read here)."""
    try:
        return TransformKind(str(value or "none"))
    except ValueError:
        return TransformKind.NONE


# --- stage 1: transcribe the source's definition ----------------------------------------------


def _extract_prompt(source: ScoreSource, max_components: int) -> tuple[str, str]:
    system = (
        "You transcribe a published composite score (an index, scale, or phenotype) from its source document "
        "into structured JSON. You are a TRANSCRIBER, not an author.\n\n"
        "STRICT RULES:\n"
        "- Record ONLY what the document states. Never supply a component, cutoff, weight, or threshold from "
        "your own knowledge of the score, even if you are confident the document is incomplete.\n"
        "- List ONE component per SCORED ITEM. Never collapse several items into a summary entry such as "
        '"32 laboratory tests" — enumerate the items the document actually names, individually.\n'
        '- Set "statedNItems" to the item count the document CLAIMS the score has (the number in a phrase like '
        '"a 32-item index"), or null if it states none. Report it even when you could enumerate fewer items: '
        "the mismatch tells the reader the document was incomplete, which is information they need.\n"
        '- If the document names a component but not how to code it, set its coding kind to "unstated" and '
        'leave the cutoff empty with "statedInSource": false. An honest gap is the correct answer.\n'
        "- Copy cutoffs and reference ranges VERBATIM as text (keep sex/age strata, units and the direction "
        'of the inequality, e.g. "<130 g/L (men), <120 g/L (women)"). Do not convert or simplify them.\n'
        '- "required" is false only when the document says the score tolerates that item being absent.\n\n'
        "Score kinds:\n"
        '- "criteria_count": count how many criteria are met, usually with a cut-point (Fried phenotype).\n'
        '- "deficit_proportion": each item is 0/1, score = deficits present / items considered (frailty index).\n'
        '- "sum": plain item sum.  "weighted_sum": per-item weights.  "z_composite": mean of standardized '
        'items.  "custom": anything else.\n\n'
        "Coding kinds: threshold | categorical | identity | unit | arithmetic | data_dependent | unstated. "
        'Use "data_dependent" when coding is relative to the sample (lowest quintile, z-score).\n\n'
        "Respond with ONLY valid JSON (no markdown fences) matching this schema:\n"
        '{"name": string, "citation": string, "kind": string, "combinationRule": string, "threshold": string, '
        '"notes": string, "statedNItems": number|null, '
        '"components": [{"name": string, "definition": string, "required": boolean, '
        '"weight": number|null, "coding": {"kind": string, "cutoff": string, "referenceRange": string, '
        '"codeMap": object, "formula": string, "units": string, "statedInSource": boolean}}]}'
    )
    user = (
        f"Source document ({source.kind}: {source.provenance or 'provided text'}):\n"
        f"-----\n{source.text}\n-----\n\n"
        f"Transcribe the composite score this document defines, one entry per scored item, up to "
        f"{max_components} components. If the document defines several related indices, transcribe the one it "
        "presents as primary and name the others in `notes`. If the document names the score and its rule but "
        "does not list its individual items (a table did not survive the text extraction, for instance), "
        "return the items you CAN see and still set `statedNItems` — do not fill the gap from memory."
    )
    return system, user


def _coding_from_payload(payload: Any) -> ComponentCoding:
    data = payload if isinstance(payload, dict) else {}
    raw_kind = str(data.get("kind", "") or "").strip().lower()
    try:
        kind = CodingKind(raw_kind)
    except ValueError:
        kind = CodingKind.UNSTATED
    code_map = data.get("codeMap") or data.get("code_map") or {}
    return ComponentCoding(
        kind=kind,
        cutoff=str(data.get("cutoff", "") or "").strip(),
        reference_range=str(data.get("referenceRange", data.get("reference_range", "")) or "").strip(),
        code_map={str(k): str(v) for k, v in code_map.items()} if isinstance(code_map, dict) else {},
        formula=str(data.get("formula", "") or "").strip(),
        units=str(data.get("units", "") or "").strip(),
        stated_in_source=bool(data.get("statedInSource", data.get("stated_in_source", False))),
    )


def _components_from_payload(items: Sequence[Any], max_components: int) -> list[ScoreComponent]:
    components: list[ScoreComponent] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        weight = item.get("weight")
        components.append(
            ScoreComponent(
                name=name,
                definition=str(item.get("definition", "") or "").strip(),
                required=bool(item.get("required", True)),
                weight=float(weight) if isinstance(weight, (int, float)) else None,
                coding=_coding_from_payload(item.get("coding")),
            )
        )
        if len(components) >= max_components:
            break
    return components


def extract_score_definition(
    source: ScoreSource, complete: CompleteFn, *, max_components: int = _MAX_COMPONENTS
) -> ScoreDefinition:
    """Transcribe a score's definition out of its source document via one LLM call.

    Raises ``ValueError`` when the document yields no usable component list — the honest outcome for a page
    that does not actually define a score (a landing page, a paywall stub), rather than a plausible guess.
    """
    system, user = _extract_prompt(source, max_components)
    raw = complete(user, system=system, max_tokens=_EXTRACT_MAX_TOKENS)
    try:
        payload = extract_json(raw)
    except (ValueError, TypeError):
        payload = {}
    items = payload.get("components")
    if not isinstance(items, list) or not items:
        # A long component list is exactly where the response gets truncated — rescue the complete objects.
        items = salvage_objects(raw, "components")
    components = _components_from_payload(items or [], max_components)
    if not components:
        raise ValueError(
            f"no score components could be read from {source.provenance or 'the provided text'} — "
            "the document may not define a composite score, or the text extraction may be empty"
        )
    raw_kind = str(payload.get("kind", "") or "").strip().lower()
    try:
        kind = CompositeKind(raw_kind)
    except ValueError:
        kind = CompositeKind.CUSTOM
    stated = payload.get("statedNItems", payload.get("stated_n_items"))
    return ScoreDefinition(
        name=str(payload.get("name", "") or "").strip() or "(unnamed composite)",
        kind=kind,
        components=components,
        citation=str(payload.get("citation", "") or "").strip(),
        combination_rule=str(payload.get("combinationRule", payload.get("combination_rule", "")) or "").strip(),
        threshold=str(payload.get("threshold", "") or "").strip(),
        notes=str(payload.get("notes", "") or "").strip(),
        stated_n_items=int(stated) if isinstance(stated, (int, float)) and stated > 0 else None,
        source=source,
    )


# --- stage 2: match components to the run's concepts ------------------------------------------


def shortlist_concepts(
    components: Sequence[ScoreComponent],
    index: Sequence[ConceptEntry],
    *,
    embed: EmbedFn | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> dict[str, list[ConceptEntry]]:
    """Retrieve the candidate concepts for each component — the closed world the judge may choose from.

    Hybrid retrieval (dense cosine + BM25 fused by RRF) is ddharmon's adopted recipe: BM25 alone beats dense
    on field→CDE recall and the fusion beats both (see :mod:`ddharmon.matching.lexical`). With no ``embed``
    callable retrieval is BM25 ALONE — not a fusion against a zero dense array, which would inject an
    index-order ranking into RRF and outrank real lexical hits — so the builder still runs correctly without
    the ``embeddings`` extra. In that mode a concept with no lexical overlap at all is left out rather than
    padding the shortlist with noise the judge would have to reject.
    """
    if not index or not components:
        return {c.name: [] for c in components}

    texts = [e.retrieval_text for e in index]
    bm25 = BM25(texts)
    queries = [f"{c.name}. {c.definition}".strip() for c in components]

    dense: np.ndarray | None = None
    if embed is not None:
        matrix = np.asarray(embed(texts), dtype=np.float32)
        query_vectors = np.asarray(embed(queries), dtype=np.float32)
        dense = query_vectors @ matrix.T  # both L2-normalized -> cosine

    out: dict[str, list[ConceptEntry]] = {}
    for i, component in enumerate(components):
        lexical = bm25.scores(queries[i])
        if dense is None:
            picked = [j for j in np.argsort(-lexical)[:top_k].tolist() if lexical[j] > 0]
        else:
            picked = hybrid_topk(dense[i], lexical, top_k)
        out[component.name] = [index[j] for j in picked]
    return out


def _match_prompt(
    definition: ScoreDefinition, shortlists: Mapping[str, list[ConceptEntry]]
) -> tuple[str, str, dict[str, set[str]]]:
    allowed: dict[str, set[str]] = {}
    blocks: list[str] = []
    for component in definition.components:
        candidates = shortlists.get(component.name) or []
        allowed[component.name] = {c.concept_id for c in candidates}
        lines = [
            f"    [{c.concept_id}] {c.concept}"
            + (f"  · cohorts: {', '.join(c.cohorts)}" if c.cohorts else "")
            + (f"  · units: {c.units}" if c.units else "")
            for c in candidates
        ] or ["    (no candidate concepts retrieved)"]
        head = f"  COMPONENT: {component.name}"
        if component.definition:
            head += f" — {component.definition}"
        blocks.append("\n".join([head, *lines]))

    system = (
        "You decide, for each COMPONENT of a composite score, whether one of the CANDIDATE CONCEPTS from a "
        "harmonization run actually measures it.\n\n"
        "STRICT RULES:\n"
        "- Choose a conceptId ONLY from that component's own candidate list, copied exactly. Never invent an "
        "id, never reuse an id from another component's list.\n"
        '- If no candidate measures the component, return "conceptId": null. A missing component is a useful, '
        "honest result; a wrong match silently corrupts the score. Prefer null when unsure.\n"
        "- Match on WHAT IS MEASURED, not on shared words. A diagnosis of hypertension is not a blood-pressure "
        "measurement; family history of a condition is not the condition; difficulty walking is not gait speed.\n"
        "- A candidate measuring only part of the component (one side, one timepoint) is still a match — say so "
        "in the rationale and lower the confidence.\n"
        "- confidence is 0.0–1.0: your certainty that this concept measures this component.\n\n"
        "Respond with ONLY valid JSON (no markdown fences) matching this schema:\n"
        '{"matches": [{"component": string, "conceptId": string|null, "confidence": number, '
        '"rationale": string}]}'
    )
    user = (
        f"Composite score: {definition.name}"
        + (f" ({definition.citation})" if definition.citation else "")
        + (f"\nCombination rule: {definition.combination_rule}" if definition.combination_rule else "")
        + "\n\nComponents and their candidate concepts from this run:\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn one entry per component, using the component names exactly as given."
    )
    return system, user, allowed


def _parse_matches(raw: str) -> list[dict[str, Any]]:
    try:
        payload = extract_json(raw)
        items = payload.get("matches") if isinstance(payload, dict) else None
    except (ValueError, TypeError):
        items = None
    if not isinstance(items, list) or not items:
        items = salvage_objects(raw, "matches")
    return [i for i in items if isinstance(i, dict)]


def match_components(
    definition: ScoreDefinition,
    index: Sequence[ConceptEntry],
    complete: CompleteFn,
    *,
    embed: EmbedFn | None = None,
    top_k: int = _DEFAULT_TOP_K,
    overrides: Mapping[str, str | None] | None = None,
) -> list[ComponentMatch]:
    """Map each component onto a concept from the run — retrieval bounds the choices, one LLM pass decides.

    ``overrides`` is the reviewer's structured edit: ``{component_name: concept_id}`` pins a match (any
    concept in the index, not just a retrieved one) and ``{component_name: None}`` drops it. Pinned
    components are excluded from the judge pass entirely, so a fully-pinned re-derive costs **no** LLM call.

    Grounding guard: a returned id that was not in that component's shortlist is discarded and the component
    is reported missing.
    """
    by_id = {e.concept_id: e for e in index}
    pins = dict(overrides or {})

    def _match_for(component: ScoreComponent, entry: ConceptEntry | None, **kwargs: Any) -> ComponentMatch:
        return ComponentMatch(
            component=component.name,
            concept_id=entry.concept_id if entry else None,
            concept=entry.concept if entry else "",
            column=entry.column if entry else "",
            cohorts=list(entry.cohorts) if entry else [],
            source_variables=list(entry.members) if entry else [],
            required=component.required,
            **kwargs,
        )

    pending = [c for c in definition.components if c.name not in pins]
    shortlists = shortlist_concepts(pending, index, embed=embed, top_k=top_k) if pending else {}

    decisions: dict[str, dict[str, Any]] = {}
    if pending:
        # One judge pass over every pending component at once: the shortlists are already the closed world,
        # and a single call keeps cost flat in the number of components (FI-Combined has 68).
        pending_def = ScoreDefinition(
            name=definition.name,
            kind=definition.kind,
            components=pending,
            citation=definition.citation,
            combination_rule=definition.combination_rule,
        )
        system, user, allowed = _match_prompt(pending_def, shortlists)
        raw = complete(user, system=system, max_tokens=_MATCH_MAX_TOKENS)
        for item in _parse_matches(raw):
            name = str(item.get("component", "") or "").strip()
            concept_id = item.get("conceptId") or item.get("concept_id")
            concept_id = str(concept_id).strip() if concept_id else None
            if concept_id and concept_id not in allowed.get(name, set()):
                concept_id = None  # hallucinated or cross-component id -> honest gap
            decisions[name] = {
                "concept_id": concept_id,
                "confidence": _confidence(item.get("confidence")),
                "rationale": str(item.get("rationale", "") or "").strip(),
            }

    matches: list[ComponentMatch] = []
    for component in definition.components:
        if component.name in pins:
            pinned_id = pins[component.name]
            entry = by_id.get(str(pinned_id)) if pinned_id else None
            matches.append(
                _match_for(
                    component,
                    entry,
                    pinned=True,
                    confidence=1.0 if entry else 0.0,
                    rationale="pinned by reviewer" if entry else "dropped by reviewer",
                )
            )
            continue
        decision = decisions.get(component.name, {})
        entry = by_id.get(str(decision.get("concept_id"))) if decision.get("concept_id") else None
        matches.append(
            _match_for(
                component,
                entry,
                confidence=float(decision.get("confidence", 0.0)) if entry else 0.0,
                rationale=str(decision.get("rationale", "")),
                shortlist=[c.concept_id for c in shortlists.get(component.name, [])],
            )
        )
    return matches


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# --- stage 3: feasibility (deterministic) -----------------------------------------------------


def assess_feasibility(
    definition: ScoreDefinition,
    matches: Sequence[ComponentMatch],
    *,
    cohorts: Sequence[str] | None = None,
) -> FeasibilityReport:
    """Judge whether the score is computable — fully, partially, or not at all — and in which cohorts.

    Deterministic, no LLM. A cohort is ``computable`` only when every **required** component matched a
    concept that cohort contributes to; optional components never block it. ``cohorts`` defaults to the
    cohorts appearing in the matches, so pass the run's full cohort list to see cohorts that supply nothing.
    """
    by_name = {c.name: c for c in definition.components}
    required_names = {c.name for c in definition.required_components}  # all components when none were flagged
    required = [m for m in matches if m.component in required_names]
    matched_required = [m for m in required if m.matched]

    if required and len(matched_required) == len(required):
        verdict = "full"
    elif matched_required:
        verdict = "partial"
    else:
        verdict = "infeasible"

    all_cohorts = sorted({c for m in matches if m.matched for c in m.cohorts} | {c for c in (cohorts or []) if c})
    per_cohort: list[CohortCoverage] = []
    for cohort in all_cohorts:
        present = [m.component for m in matches if m.matched and cohort in m.cohorts]
        missing = [m.component for m in required if not (m.matched and cohort in m.cohorts)]
        per_cohort.append(
            CohortCoverage(cohort=cohort, present=present, missing=missing, computable=not missing and bool(required))
        )

    needs_review = [
        m.component
        for m in matches
        if m.matched and by_name.get(m.component, ScoreComponent(name=m.component)).coding.needs_review
    ]
    caveats = [_METADATA_CAVEAT]
    if needs_review:
        caveats.append(
            f"{len(needs_review)} of {len(matches)} components have no usable coding rule in the source "
            "(no stated cutoff/reference range, a formula, or a sample-relative rule) — a reviewer must supply "
            "it before the score can be computed. ddharmon does not invent cutoffs."
        )
    if verdict == "partial":
        caveats.append(
            f"{len(required) - len(matched_required)} of {len(required)} required components are missing — a "
            "score computed from the rest is NOT the published score and is not comparable to published values."
        )
    if any(m.matched and m.confidence and m.confidence < 0.6 for m in matches):
        caveats.append("Some component→concept matches are low-confidence — review those before computing.")
    if definition.under_enumerated:
        caveats.append(
            f"The source describes a {definition.stated_n_items}-item score but only "
            f"{len(definition.components)} item(s) could be read out of it — the document is incomplete (a "
            "table may not have survived text extraction). Supply the full item list (the PDF or supplement) "
            "and re-derive; the missing items were NOT filled in from prior knowledge."
        )
    return FeasibilityReport(
        verdict=verdict,
        n_required=len(required),
        n_required_matched=len(matched_required),
        matched=[m.component for m in matches if m.matched],
        missing=[m.component for m in matches if not m.matched],
        needs_review=needs_review,
        per_cohort=per_cohort,
        caveats=caveats,
    )


# --- stage 4: the derivation recipe -----------------------------------------------------------


def _slug(name: str) -> str:
    """A safe identifier for a component's intermediate column."""
    out = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return out or "component"


def _coding_expression(match: ComponentMatch, component: ScoreComponent) -> tuple[str, str]:
    """The per-component coding step: ``(expression, description)``, review-stubbed when it can't be authored."""
    coding = component.coding
    target, column = _slug(component.name), match.column
    bound = coding.cutoff or coding.reference_range
    if coding.kind is CodingKind.THRESHOLD and bound:
        return (
            f"{target} = outside({column!r}, {bound!r})  # 1 if outside the stated range, else 0",
            f"Code {component.name} as a deficit when {column} falls outside {bound}.",
        )
    if coding.kind is CodingKind.CATEGORICAL and coding.code_map:
        return (
            f"{target} = map({column!r}, {coding.code_map!r})",
            f"Score {component.name} from its response codes per the source's mapping.",
        )
    if coding.kind is CodingKind.IDENTITY:
        return f"{target} = {column!r}", f"{component.name} enters the score as its harmonized value."
    if coding.kind is CodingKind.UNIT:
        unit_label = coding.units or "the score's units"
        return (
            f"{target} = convert({column!r}, to={coding.units or '?'!r})  # REVIEW: confirm the conversion",
            f"{component.name} must be converted to {unit_label} before comparison.",
        )
    if coding.kind is CodingKind.ARITHMETIC:
        return (
            f"# REVIEW: {target} = {coding.formula or '<formula not stated>'}  (over {column})",
            f"{component.name} is derived by formula — always reviewed, never auto-applied.",
        )
    if coding.kind is CodingKind.DATA_DEPENDENT:
        return (
            f"# APPLY-TIME: {target} = sample_relative({column!r}, {bound or 'as stated'!r})",
            f"{component.name} is coded relative to the sample ({bound or 'e.g. lowest quintile'}), so it is "
            "computed when the notebook runs on real rows — not derivable from metadata.",
        )
    return (
        f"# REVIEW: {target} = ?  # source states no coding rule for this component (over {column})",
        f"The source does not say how to code {component.name} — a reviewer must supply the rule.",
    )


def _combination(definition: ScoreDefinition, matched: Sequence[ComponentMatch]) -> tuple[str, str, str]:
    """``(expression, description, units)`` for the combine step, per :class:`CompositeKind`."""
    terms = [_slug(m.component) for m in matched]
    joined = " + ".join(terms)
    n = len(terms)
    if not n:
        # Nothing matched: emitting `score = (0) / 0` would be a runnable-looking lie.
        return (
            "# NOT COMPUTABLE: no component of this score matched a concept in this run",
            "No components are available, so there is nothing to combine — see the feasibility gaps above.",
            "",
        )
    if definition.kind is CompositeKind.CRITERIA_COUNT:
        return f"score = {joined}", f"Count the criteria met ({n} of the score's criteria are available).", "count"
    if definition.kind is CompositeKind.DEFICIT_PROPORTION:
        return (
            f"score = ({joined}) / {n}",
            f"Proportion of deficits present over the {n} deficits AVAILABLE in this run — the published "
            "index divides by its own item count, so this denominator differs whenever coverage is partial.",
            "proportion (0-1)",
        )
    if definition.kind is CompositeKind.WEIGHTED_SUM:
        weighted = " + ".join(
            f"{w} * {_slug(m.component)}" for m, w in ((m, _weight_of(definition, m)) for m in matched) if w is not None
        )
        if weighted and len(weighted.split("+")) == n:
            return f"score = {weighted}", "Weighted sum of the components, using the source's weights.", "points"
        return (
            f"# REVIEW: score = weighted sum of ({joined}) — the source's per-item weights are incomplete",
            "Weighted sum, but not every component carries a stated weight — a reviewer must supply them.",
            "points",
        )
    if definition.kind is CompositeKind.Z_COMPOSITE:
        return (
            f"# APPLY-TIME: score = mean(z({'), z('.join(terms) or '…'}))",
            "Mean of standardized components — standardization is computed within the analysis sample at "
            "apply-time, and scores are only comparable across cohorts if standardized on a pooled reference.",
            "z-score",
        )
    if definition.kind is CompositeKind.SUM:
        return f"score = {joined}", f"Sum of the {n} available items.", "points"
    return (
        f"# REVIEW: apply the source's own rule to ({joined})",
        f"The source's combination rule is carried verbatim: {definition.combination_rule or '(not stated)'}",
        "",
    )


def _weight_of(definition: ScoreDefinition, match: ComponentMatch) -> float | None:
    for component in definition.components:
        if component.name == match.component:
            return component.weight
    return None


def _cut_point(threshold: str) -> str:
    """Pull a single numeric cut-point out of the source's threshold wording, e.g. "frail if ≥3 of 5" -> ">= 3".

    Returns "" — a review stub, not a guess — when the wording is a BAND LIST rather than one cut-point
    ("categories 0-0.1, 0.1-0.2, 0.2-0.3, 0.4+", "cut-offs of 5, 10, 15 and 20"). Those are severity strata,
    and collapsing them to the first number produces a threshold the source never stated.
    """
    text = threshold or ""
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    has_operator = bool(re.search(r"(>=|≥|>|<=|≤|<)", text))
    if len(numbers) > 2 or (len(numbers) == 2 and not has_operator and re.search(r"\d\s*(?:[-–]|to)\s*\d", text)):
        return ""  # a list of bands / strata
    m = re.search(r"(>=|≥|>|<=|≤|<)?\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return ""
    operator = {"≥": ">=", "≤": "<=", None: ">="}.get(m.group(1), m.group(1) or ">=")
    return f"{operator} {m.group(2)}"


def build_composite_spec(
    definition: ScoreDefinition, matches: Sequence[ComponentMatch], feasibility: FeasibilityReport
) -> CompositeSpec:
    """Assemble the ordered derivation recipe and its validation rules from the matched components.

    Every step that cannot be authored from metadata alone (an unstated cutoff, a formula, a sample-relative
    rule) is emitted as a clearly-marked review stub rather than a plausible guess — the same discipline the
    transform layer applies to arithmetic and data-dependent recodes.
    """
    by_name = {c.name: c for c in definition.components}
    matched = [m for m in matches if m.matched]

    steps: list[DerivationStep] = []
    for m in matched:
        component = by_name.get(m.component, ScoreComponent(name=m.component))
        expression, description = _coding_expression(m, component)
        steps.append(
            DerivationStep(
                order=len(steps) + 1,
                kind="code_component",
                description=description,
                expression=expression,
                component=m.component,
                concept_id=m.concept_id,
                needs_review=component.coding.needs_review,
            )
        )

    expression, description, units = _combination(definition, matched)
    steps.append(
        DerivationStep(
            order=len(steps) + 1,
            kind="combine",
            description=description,
            expression=expression,
            needs_review=definition.kind in (CompositeKind.CUSTOM, CompositeKind.Z_COMPOSITE),
        )
    )
    if definition.threshold:
        cut = _cut_point(definition.threshold)
        steps.append(
            DerivationStep(
                order=len(steps) + 1,
                kind="threshold",
                description=f"Apply the score's cut-point as stated: {definition.threshold}",
                expression=(f"positive = score {cut}" if cut else f"# REVIEW: {definition.threshold}"),
                needs_review=not cut,
            )
        )

    rules = [
        f"Recompute nothing silently: {len(matched)} of {len(definition.components)} components are wired; "
        f"{len(feasibility.missing)} are missing.",
    ]
    if definition.kind is CompositeKind.DEFICIT_PROPORTION:
        rules.append(
            "score must fall in [0, 1]; the denominator is the number of AVAILABLE deficits, not the published item count."
        )
    if definition.kind is CompositeKind.CRITERIA_COUNT:
        rules.append(f"score must be an integer in [0, {len(matched)}].")
    if feasibility.verdict != "full":
        rules.append(
            "Do not report this as the published score — coverage is incomplete; report it as a modified index and say which items were unavailable."
        )
    if feasibility.needs_review:
        rules.append(f"Components needing a reviewer-supplied coding rule: {', '.join(feasibility.needs_review)}.")
    rules.append("Compute per cohort, then pool; only cohorts listed as computable have every required component.")

    return CompositeSpec(
        definition=definition,
        matches=list(matches),
        feasibility=feasibility,
        derivation=steps,
        units=units,
        validation_rules=rules,
    )


# --- the entry point --------------------------------------------------------------------------


def derive_composite(
    source: ScoreSource | ScoreDefinition,
    records: Sequence[LeanBRecord],
    complete: CompleteFn,
    *,
    embed: EmbedFn | None = None,
    top_k: int = _DEFAULT_TOP_K,
    overrides: Mapping[str, str | None] | None = None,
    max_components: int = _MAX_COMPONENTS,
) -> CompositeResult:
    """Derive a composite-variable spec for ``source`` from a run's harmonized concepts.

    ``source`` is either a :class:`~ddharmon.harmonization.score_sources.ScoreSource` (a fetched/pasted
    document — the definition is transcribed first) or an already-transcribed :class:`ScoreDefinition`,
    which is how a **re-derive** avoids paying for extraction twice.

    ``overrides`` carries the reviewer's structured edits (``{component: concept_id}`` to pin,
    ``{component: None}`` to drop). Re-deriving with every component pinned makes zero LLM calls, so the
    accept/swap/drop loop is free; stages 3–4 are deterministic and always recomputed.

    ``embed`` is an ``EmbeddingProvider.embed``-style callable; without it retrieval is BM25-only.
    """
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return complete(*args, **kwargs)

    definition = (
        source
        if isinstance(source, ScoreDefinition)
        else extract_score_definition(source, counted, max_components=max_components)
    )
    index = build_concept_index(records)
    matches = match_components(definition, index, counted, embed=embed, top_k=top_k, overrides=overrides)
    run_cohorts = sorted({c for e in index for c in e.cohorts})
    feasibility = assess_feasibility(definition, matches, cohorts=run_cohorts)
    spec = build_composite_spec(definition, matches, feasibility)
    return CompositeResult(spec=spec, n_concepts_indexed=len(index), calls_made=calls)


def spec_to_dict(spec: CompositeSpec) -> dict[str, Any]:
    """Serialize a :class:`CompositeSpec` to JSON-ready camelCase — the shape a UI/API layer consumes.

    Kept here (not in the UI) so the contract has exactly one author, per the core↔UI insulation rule.
    """
    definition = spec.definition
    return {
        "definition": {
            "name": definition.name,
            "kind": str(definition.kind),
            "citation": definition.citation,
            "combinationRule": definition.combination_rule,
            "threshold": definition.threshold,
            "notes": definition.notes,
            "statedNItems": definition.stated_n_items,
            "underEnumerated": definition.under_enumerated,
            "provenance": definition.provenance,
            "sourceSha256": definition.source.sha256 if definition.source else "",
            "components": [
                {
                    "name": c.name,
                    "definition": c.definition,
                    "required": c.required,
                    "weight": c.weight,
                    "coding": {
                        "kind": str(c.coding.kind),
                        "cutoff": c.coding.cutoff,
                        "referenceRange": c.coding.reference_range,
                        "codeMap": c.coding.code_map,
                        "formula": c.coding.formula,
                        "units": c.coding.units,
                        "statedInSource": c.coding.stated_in_source,
                        "needsReview": c.coding.needs_review,
                    },
                }
                for c in definition.components
            ],
        },
        "matches": [
            {
                "component": m.component,
                "conceptId": m.concept_id,
                "concept": m.concept,
                "column": m.column,
                "cohorts": m.cohorts,
                "sourceVariables": m.source_variables,
                "confidence": m.confidence,
                "rationale": m.rationale,
                "required": m.required,
                "pinned": m.pinned,
                "shortlist": m.shortlist,
            }
            for m in spec.matches
        ],
        "feasibility": {
            "verdict": spec.feasibility.verdict,
            "nRequired": spec.feasibility.n_required,
            "nRequiredMatched": spec.feasibility.n_required_matched,
            "matched": spec.feasibility.matched,
            "missing": spec.feasibility.missing,
            "needsReview": spec.feasibility.needs_review,
            "computableCohorts": spec.feasibility.computable_cohorts,
            "perCohort": [
                {"cohort": c.cohort, "present": c.present, "missing": c.missing, "computable": c.computable}
                for c in spec.feasibility.per_cohort
            ],
            "caveats": spec.feasibility.caveats,
        },
        "derivation": [
            {
                "order": s.order,
                "kind": s.kind,
                "description": s.description,
                "expression": s.expression,
                "component": s.component,
                "conceptId": s.concept_id,
                "needsReview": s.needs_review,
            }
            for s in spec.derivation
        ],
        "units": spec.units,
        "validationRules": spec.validation_rules,
    }


def spec_to_json(spec: CompositeSpec, *, indent: int = 2) -> str:
    """The spec as a JSON string (thin wrapper over :func:`spec_to_dict`)."""
    return json.dumps(spec_to_dict(spec), indent=indent)
