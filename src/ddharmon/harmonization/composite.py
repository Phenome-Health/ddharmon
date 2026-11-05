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
import logging
import os
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ddharmon.harmonization.models import GenCDE, LeanBRecord, TransformKind, TransformSpec
from ddharmon.harmonization.parse import extract_json, salvage_objects
from ddharmon.harmonization.score_sources import ScoreSource
from ddharmon.matching.lexical import BM25, hybrid_topk, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 8  # candidate concepts shown to the judge per component
_MAX_COMPONENTS = 80  # FI-Combined is 68 items — the largest published composite we target
_MAX_IDEAL_CHARS = 240  # how much of a concept's generated-ideal text feeds retrieval
_MAX_QUESTION_CHARS = 200  # how much of a variable's question/description composes its concept label
_MAX_ANSWER_LABELS = 8  # capped answer-option labels folded into a variable's retrieval text
_MAX_MEMBERS_PER_GROUP = 4  # cap on members of ONE group reaching the judge — enough for cross-cohort
# coverage (the aggregate + the indented member display) while a swarm of near-duplicates stays bounded
_RRF_POOL = 1000  # truncate each ranking before RRF fusion (matches lexical.hybrid_topk's default)
_EXTRACT_MAX_TOKENS = 8192
# The one judge pass rates candidates for EVERY component at once, so its response scales with component
# count x candidates-per-component. Answer-option granularity adds per-option candidates, which
# lengthens the response; a 49-item index with option units overran the old 8192 cap and truncated mid-list
# (late components silently went undecided). Raised to give the whole list headroom.
_MATCH_MAX_TOKENS = 16384

# Answer-option granularity: a MULTI-SELECT checklist variable ("Vascular/heart problems
# diagnosed by doctor" → heart attack | angina | stroke | high blood pressure) measures several DIFFERENT
# components at once — one per option. Such a variable is exploded into per-option coverage units so the
# judge can bind a SPECIFIC option to a component, and coverage names the option that supports it. The
# option unit's id is ``"<cohort:var>#opt=<label>"``; it rolls up to the SAME group as its parent variable.
_OPTION_ID_SEP = "#opt="  # marks an option coverage unit; strip to recover its parent variable id
_MIN_CHECKLIST_OPTIONS = 3  # a variable needs at least this many distinct non-sentinel options to be exploded
# Generic (not cohort-hardcoded) multi-select signal, matched as a substring of the source's data_type: a
# variable whose participant can pick SEVERAL options is a checklist of distinct measurands, whereas a
# single-select (radio) categorical is one ordinal/Likert concept and must NOT be fanned out.
_MULTISELECT_DATATYPE_MARKERS = ("multiple", "checkbox", "multi", "select all", "check all")

# Answer-option labels that carry no topical signal — dropped from a variable's retrieval text so a
# "None"/"Prefer not to answer" option never becomes the reason a component matches. Generic, not tuned
# to any cohort. Negative NUMERIC codes (UKBB's -1/-3/-7 sentinels) are dropped separately, by code.
_SENTINEL_ANSWER_LABELS = frozenset(
    {
        "none",
        "none of the above",
        "prefer not to answer",
        "do not know",
        "don't know",
        "dont know",
        "not applicable",
        "n/a",
        "na",
        "unknown",
        "not answered",
        "missing",
        "no answer",
    }
)

# Union coverage: a component counts as PRESENT in a cohort when at least one judge-accepted member
# in that cohort scores at/above this floor — coverage is a per-cohort union over accepted members across
# ALL the groups the component reached, decoupled from the single surfaced winner group. Set conservatively:
# the judge already OMITS candidates it does not think measure a component, so a rated member is an
# affirmative "this measures it" with a confidence; the floor drops the weakest of those (partial / one-side
# / low-certainty) so recall is never bought with coverage the judge was unsure of. Tunable in one place.
_COVERAGE_CONFIDENCE_FLOOR = 0.5

# Coverage-model ablation hook (idiomatic per the leanb "each mod individually switchable" convention). The
# default UNION model credits a cohort only when a judge-accepted member IN that cohort measures the
# component. Set ``DDHARMON_COMPOSITE_COVERAGE=winner`` to fall back to the earlier winner-only model (the surfaced
# winner GROUP's cohorts) for A/B measurement — NOT for production.
_COVERAGE_MODEL_ENV = "DDHARMON_COMPOSITE_COVERAGE"


def _union_coverage_enabled() -> bool:
    """Whether member-level UNION coverage is on (default). ``DDHARMON_COMPOSITE_COVERAGE=winner`` disables it."""
    return os.environ.get(_COVERAGE_MODEL_ENV, "union").strip().lower() != "winner"


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
    is_variable: bool = False  # True when this entry IS one source variable, not a harmonized concept group
    # Capped, human-readable answer-option labels for a VARIABLE ("Mouth ulcers; Painful gums; …"). For a
    # multi-select variable whose question stem is generic ("Do you have any of the following?"), the answer
    # meanings are the strongest topical signal in the row, so retrieval must see them. Empty for a
    # concept-group entry (a group has no single answer set).
    answer_text: str = ""
    # Answer-option granularity: set when THIS entry is a single answer OPTION of a multi-select
    # checklist variable (``concept_id`` = ``"<parent var>#opt=<label>"``). It is the coverage unit that binds
    # a checklist to ONE component (the option that measures it); it rolls up to the parent variable's group.
    option_label: str = ""

    @property
    def column(self) -> str:
        return self.cde_id or self.gencde_name or self.concept_id

    @property
    def retrieval_text(self) -> str:
        """The text retrieval scores against: the concept label, its CDE name, a slice of its ideal, and —
        for a variable — its answer-option labels (the strongest signal when the question stem is generic)."""
        parts = [self.concept, self.cde_id or "", self.ideal_cde[:_MAX_IDEAL_CHARS], self.answer_text]
        return " ".join(p for p in parts if p)

    @property
    def label(self) -> str:
        """Display name for a surfaced GROUP: its concept, else its generated ideal, else empty. The UI's
        ``groupLabel`` refines this against the full group (adding the reviewer name / "Unnamed group"),
        but a non-UI consumer of the spec still gets a usable name — never a raw internal id."""
        return (self.concept or "").strip() or (self.ideal_cde or "").strip()


class MatchReason(StrEnum):
    """WHY a component ended up matched or unmatched — the difference between a gap and a malfunction.

    Every unmatched component used to look identical from the outside, which made an honest "this run has
    no gait speed" indistinguishable from "the join dropped a valid answer". They have opposite fixes, so
    they get distinct names here and travel all the way out through :func:`spec_to_dict`.
    """

    MATCHED = "matched"
    NO_CANDIDATES = "no_candidates"  # retrieval offered nothing — the judge never had a choice
    JUDGE_DECLINED = "judge_declined"  # real candidates shown, judge returned null: an honest gap
    ID_REJECTED = "id_rejected"  # judge named an id outside this component's shortlist
    NO_DECISION = "no_decision"  # the judge's response contained no entry for this component at all
    PINNED = "pinned"  # reviewer override
    DROPPED = "dropped"  # reviewer override


@dataclass
class ComponentMatch:
    """The verdict for ONE component: the run concept that measures it, or an honest gap.

    ``shortlist`` is the audit trail — the concept ids retrieval offered the judge — so a missing component
    can be told apart from a component the judge saw good candidates for and still rejected. ``reason``
    states which of those happened outright; see :class:`MatchReason`.
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
    reason: MatchReason = MatchReason.NO_DECISION
    is_variable: bool = False  # the chosen entry was a single source variable, not a concept group
    # Variable-only matching (v2 score builder): the surfaced entity is a concept GROUP, reached by rolling
    # up the source VARIABLES the judge rated. `matched_members` are the (variable_id, confidence) pairs that
    # rolled into the surfaced group — shown indented under it; `confidence` above is then the GROUP aggregate
    # (mean of these), not any single variable's. `group_candidates` are every group the component's rated
    # variables rolled up into, (group_id, aggregate, n_matched, n_total) best-first — the deduped Swap list,
    # each carrying "X of Y group members matched".
    matched_members: list[tuple[str, float]] = field(default_factory=list)
    group_candidates: list[tuple[str, float, int, int]] = field(default_factory=list)
    # Union coverage: `cohorts` above is the per-cohort UNION over judge-accepted members (at/above
    # the confidence floor) across ALL reached groups — NOT the surfaced winner group's cohorts. This maps
    # each covered cohort to the (variable_id, confidence) members that support it there, best-first, so a UI
    # can name the supporting variable/option per cohort. Empty in the legacy group-concept path.
    coverage_members: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

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


def build_concept_index(records: Sequence[LeanBRecord], *, include_unlabeled: bool = False) -> list[ConceptEntry]:
    """Index a run's records as the ONLY concepts a composite may be built from.

    Unlike :func:`~ddharmon.harmonization.analysis_ideas.build_concept_digest`, this keeps **single-cohort**
    concepts: a component present in one cohort still makes the score computable *there*, and hiding that
    would misreport feasibility. Records with no concept label and no CDE are skipped (nothing to match) —
    UNLESS ``include_unlabeled`` is set, which keeps every group that has an id (used by
    :func:`build_group_lookup`, where an unnamed/over-merged group is still a valid roll-up target that the
    UI labels from its ``idealCde``).
    """
    index: list[ConceptEntry] = []
    seen: set[str] = set()
    for r in records:
        concept = (r.concept or "").strip()
        concept_id = (r.group_id or r.cluster_id or "").strip()
        if not concept_id or concept_id in seen:
            continue
        if not concept and not r.cde_id and not include_unlabeled:
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


def _answer_labels(fd: Mapping[str, Any]) -> list[str]:
    """Up to :data:`_MAX_ANSWER_LABELS` human-readable answer-option labels for a variable.

    Prefers a structured ``responseOptions`` list (``{code/value, label/text}``); falls back to parsing the
    flat ``valueEncoding`` string, which appears in two shapes across cohorts — ``"1=Mouth ulcers|2=Painful
    gums"`` (UKBB/CLSA, ``code=label``) and ``"DentalCare_Yes, Yes | DentalCare_No, No"`` (AoU, ``code,
    label``). Options are separated by ``|``, ``;`` or a newline. Sentinel options carry no topical signal
    and are dropped: negative NUMERIC codes (UKBB's -1/-3/-7) by code, and generic labels ("None", "Prefer
    not to answer", …) by :data:`_SENTINEL_ANSWER_LABELS`.
    """
    labels: list[str] = []

    def _add(code: str, label: str) -> bool:
        label = label.strip()
        code = code.strip()
        if not label:
            return False
        if code.startswith("-") and code[1:].isdigit():  # UKBB negative sentinel code
            return False
        if label.casefold() in _SENTINEL_ANSWER_LABELS:
            return False
        labels.append(label)
        return len(labels) >= _MAX_ANSWER_LABELS

    options = fd.get("responseOptions")
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        for opt in options:
            if not isinstance(opt, Mapping):
                continue
            code = str(opt.get("value") or opt.get("code") or "")
            label = str(opt.get("label") or opt.get("text") or "")
            if _add(code, label):
                break

    if not labels:
        encoding = str(fd.get("valueEncoding") or "")
        if encoding:
            for token in re.split(r"[|;\n]", encoding):
                if not token.strip():
                    continue
                if "=" in token:
                    code, _, label = token.partition("=")
                elif "," in token:
                    code, _, label = token.partition(",")
                else:
                    code, label = "", token
                if _add(code, label):
                    break

    return labels


def _base_variable_id(concept_id: str) -> str:
    """The parent VARIABLE id of a coverage unit — strips an ``#opt=<label>`` option suffix.

    A plain variable id (no suffix) is returned unchanged, so every group-resolution site can call this
    uniformly and an option coverage unit rolls up to exactly its parent variable's group.
    """
    return concept_id.split(_OPTION_ID_SEP, 1)[0]


def _is_multiselect_checklist(fd: Mapping[str, Any], n_options: int) -> bool:
    """Whether a variable is a multi-select checklist worth exploding into per-option coverage units.

    Generic, discovered-not-hardcoded: the participant can pick SEVERAL options (data_type carries a
    multi-select marker like "multiple"/"checkbox") AND the row offers enough distinct non-sentinel options
    to be a checklist of different measurands rather than one ordinal/Likert concept. A single-select (radio)
    categorical — "Overall health rating": Excellent/Good/Fair/Poor — is one concept and is never exploded.
    """
    if n_options < _MIN_CHECKLIST_OPTIONS:
        return False
    data_type = str(fd.get("dataType") or "").casefold()
    return any(marker in data_type for marker in _MULTISELECT_DATATYPE_MARKERS)


def build_variable_index(
    field_index: Mapping[str, Mapping[str, Any]],
    *,
    checklist_members: set[str] | None = None,
) -> list[ConceptEntry]:
    """Index EACH source variable as a single-member concept — the variable-level matching corpus.

    A component often ties to one specific VARIABLE (e.g. ``UKBB:Miserableness`` — "Do you ever feel 'just
    miserable' for no reason?") that no harmonized concept group is named for, because clustering fused or
    mis-named it. Treating every variable as a degenerate single-member :class:`ConceptEntry` lets the SAME
    retrieval + one-pass judge match a component to a variable directly, ALONGSIDE the group concepts —
    strictly more comprehensive, backward compatible (only added when a ``field_index`` is supplied).

    ``concept_id`` is the ``"cohort:var"`` key (so :attr:`ConceptEntry.column` is that variable — the column
    the exported notebook already produces).

    Retrieval text is composed from more of the row than the old ``questionText or text or name``:
    the specific ``name`` AND the question/description stem TOGETHER (a generic stem like "Do you have any
    of the following?" no longer shadows a specific name like "Mouth/teeth dental problems"), plus a capped
    list of answer-option labels held in :attr:`ConceptEntry.answer_text` — the strongest topical signal on a
    multi-select variable, and previously indexed nowhere. Defensive by design: many entries carry only a
    subset of {name, question, options}, so the label is composed from whatever is present.
    """
    index: list[ConceptEntry] = []
    for key, fd in field_index.items():
        if not isinstance(fd, Mapping):
            continue
        cohort = key.split(":", 1)[0] if ":" in key else ""
        name = str(fd.get("name") or "").strip()
        question = str(fd.get("questionText") or fd.get("text") or fd.get("description") or "").strip()
        if question and len(question) > _MAX_QUESTION_CHARS:
            question = question[:_MAX_QUESTION_CHARS].rstrip()
        # Compose the concept label: name + question when both are present and distinct; otherwise whichever
        # exists. This label feeds BOTH retrieval (retrieval_text) and the judge candidate line.
        if name and question and name.casefold() != question.casefold():
            concept = f"{name} — {question}"
        else:
            concept = name or question
        if not concept:
            continue
        answers = _answer_labels(fd)
        index.append(
            ConceptEntry(
                concept_id=key,
                concept=concept,
                cohorts=[cohort] if cohort else [],
                members=[key],
                verdict="variable",
                data_type=str(fd.get("dataType") or ""),
                is_variable=True,
                answer_text="; ".join(answers),
            )
        )
        # A multi-select checklist ALSO enters the corpus as one coverage unit PER option, so the
        # judge can bind a specific option ("Heart attack") to a specific component (Myocardial infarction).
        # Each option unit retrieves on its own label, carries that label into coverage, and rolls up to the
        # parent variable's group (via _base_variable_id). Precision is the judge's: an option binds only when
        # the judge accepts it above the coverage floor — it is never fanned to every vaguely-related component.
        # When ``checklist_members`` is given, only checklists whose base variable is a run-group member are
        # exploded — a non-member checklist's options can never surface after roll-up, so exploding them would
        # only flood retrieval and lengthen the judge pass for no coverage gain.
        explode = _is_multiselect_checklist(fd, len(answers)) and (
            checklist_members is None or key in checklist_members
        )
        if explode:
            for label in answers:
                index.append(
                    ConceptEntry(
                        concept_id=f"{key}{_OPTION_ID_SEP}{label}",
                        concept=f"{name or question} — {label}" if (name or question) else label,
                        cohorts=[cohort] if cohort else [],
                        members=[key],
                        verdict="variable",
                        data_type=str(fd.get("dataType") or ""),
                        is_variable=True,
                        answer_text=label,
                        option_label=label,
                    )
                )
    return index


def build_group_lookup(
    records: Sequence[LeanBRecord],
) -> tuple[dict[str, str], dict[str, ConceptEntry]]:
    """The reverse of clustering: variable id → its group id, and group id → its :class:`ConceptEntry`.

    Variable-only matching scores VARIABLES, then rolls each matched variable up to the concept GROUP that
    owns it. Every group with members is a valid roll-up target — including unnamed/over-merged ones (the
    fracture group's ``concept`` is empty; the UI labels it from ``idealCde``) — so this uses
    ``include_unlabeled=True`` rather than :func:`build_concept_index`'s default skip. A variable in more
    than one group is bound to the first (records are disjoint by construction; this is a guard, not a case).
    """
    groups = build_concept_index(records, include_unlabeled=True)
    groups_by_id = {g.concept_id: g for g in groups}
    var_to_group: dict[str, str] = {}
    for g in groups:
        for v in g.members:
            var_to_group.setdefault(v, g.concept_id)
    return var_to_group, groups_by_id


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


def _group_diverse_picks(
    order: list[int],
    scores: np.ndarray,
    index: Sequence[ConceptEntry],
    var_to_group: Mapping[str, str],
    top_k: int,
) -> list[int]:
    """Spend the candidate budget on DISTINCT concept GROUPS, not on ``top_k`` near-duplicate variables.

    Walk the fused ranking ``order`` (best first); bucket each candidate variable under the concept group it
    rolls up to, keeping the first ``top_k`` groups to appear and up to :data:`_MAX_MEMBERS_PER_GROUP` of
    each group's members (so a group's cross-cohort members are still all rated, while a swarm of one
    concept's near-duplicates cannot crowd every other concept out of the shortlist). A variable that rolls
    up to NO group is skipped: it can never surface after roll-up, so it must not waste a slot.
    """
    per_group: dict[str, list[int]] = {}
    group_order: list[str] = []
    for j in order:
        if scores[j] <= 0:  # ranking is descending; nothing positive remains
            break
        gid = var_to_group.get(_base_variable_id(index[j].concept_id))
        if gid is None:
            continue
        bucket = per_group.get(gid)
        if bucket is None:
            if len(group_order) >= top_k:
                continue  # budget spent on distinct groups — ignore further NEW groups
            bucket = []
            per_group[gid] = bucket
            group_order.append(gid)
        if len(bucket) < _MAX_MEMBERS_PER_GROUP:
            bucket.append(j)
    return [j for gid in group_order for j in per_group[gid]]


def shortlist_concepts(
    components: Sequence[ScoreComponent],
    index: Sequence[ConceptEntry],
    *,
    embed: EmbedFn | None = None,
    top_k: int = _DEFAULT_TOP_K,
    var_to_group: Mapping[str, str] | None = None,
) -> dict[str, list[ConceptEntry]]:
    """Retrieve the candidate concepts for each component — the closed world the judge may choose from.

    Hybrid retrieval (dense cosine + BM25 fused by RRF) is ddharmon's adopted recipe: BM25 alone beats dense
    on field→CDE recall and the fusion beats both (see :mod:`ddharmon.matching.lexical`). With no ``embed``
    callable retrieval is BM25 ALONE — not a fusion against a zero dense array, which would inject an
    index-order ranking into RRF and outrank real lexical hits — so the builder still runs correctly without
    the ``embeddings`` extra. In that mode a concept with no lexical overlap at all is left out rather than
    padding the shortlist with noise the judge would have to reject.

    ``var_to_group`` switches on GROUP-DIVERSE selection (variable-only matching): ``top_k`` then counts
    distinct concept GROUPS rather than raw variables, so a swarm of near-duplicate variables of one concept
    (29 "…drug(s) you are taking for your diabetes" rows) spends a single slot instead of crowding every
    other concept out of the shortlist. Absent, the classic top-``k``-variables selection runs unchanged.
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
        if var_to_group is not None:
            # Group-diverse: rank the whole candidate space, then keep top_k DISTINCT groups' members.
            if dense is None:
                scores = lexical
            else:
                n = lexical.shape[0]
                dense_order = np.argsort(-dense[i])[:_RRF_POOL].tolist()
                lexical_order = np.argsort(-lexical)[:_RRF_POOL].tolist()
                scores = reciprocal_rank_fusion([dense_order, lexical_order], n)
            order = np.argsort(-scores).tolist()
            picked = _group_diverse_picks(order, scores, index, var_to_group, top_k)
        elif dense is None:
            picked = [j for j in np.argsort(-lexical)[:top_k].tolist() if lexical[j] > 0]
        else:
            picked = hybrid_topk(dense[i], lexical, top_k)
        out[component.name] = [index[j] for j in picked]
    return out


# --- Gate 1 suggestions: the free, retrieval-only half of the match ------------------------------------

#: The cut-off a Gate 1 suggestion must clear: the dense cosine (BioLORD, L2-normalised) between a declared
#: component's query and its group's best-matching member. ABSOLUTE by construction, so one value means the
#: same thing for every component — the RRF fusion the shortlist ranks by is rank-based and is NOT thresholded.
#:
#: CALIBRATED ($0, deterministic sweep). Silver labels: the 43 groups the PAID judge selected (group
#: confidence >= 0.80) on a frailty-index run over three public cohort dictionaries (All of Us + CLSA + UK
#: Biobank; 1317 groups, 11015 variables + checklist options indexed), the 49
#: components queried by NAME ONLY, as a Gate 1 declaration states them. Rule, fixed before the sweep was read:
#: the highest cut-off reaching the best recall of judge-selected groups among cut-offs that suggest at most one
#: group the judge did not select per group it did (extra <= 43). At 0.62: 75 groups suggested, 35 of the 43
#: agree, 40 extra (36 of them groups the judge's own shortlist never showed it), 8 missed; 40 of 49 components
#: get >= 1 suggestion. TUNED ON THE FRAILTY INDEX — FI numbers for this threshold are DEV-optimistic.
GATE1_SUGGEST_MIN_COSINE = 0.62

_NO_DENSE_REASON = (
    "No suggestions: the dense encoder is unavailable, and a suggestion's cut-off is held against the dense "
    "cosine of a group's best member. Lexical (BM25) and fused (RRF) retrieval scores are rank-based and not "
    "comparable across components, so no threshold over them is offered in its place."
)


@dataclass
class GroupSuggestion:
    """One concept group a declared component's free search reached, scored by its best-matching member."""

    group_id: str
    score: float  # dense cosine between the component query and ``best_member`` (rounded to 4 places)
    best_member: str  # the parent VARIABLE id (``"cohort:var"``) — never an option-suffixed coverage-unit id
    best_option: str = ""  # the checklist answer option that scored best, when the best unit is an option


@dataclass
class ComponentSuggestions:
    """A declared component and the groups its search reached, best-first."""

    component: str
    groups: list[GroupSuggestion] = field(default_factory=list)


@dataclass
class GroupSuggestionResult:
    """Return of :func:`suggest_groups`. ``scored`` is False — and every component empty — with no encoder."""

    components: list[ComponentSuggestions]
    scored: bool
    threshold: float = GATE1_SUGGEST_MIN_COSINE
    reason: str = ""
    n_variables_indexed: int = 0


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows, so a dot product is a cosine whatever the embedder returned."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def suggest_groups(
    components: Sequence[ScoreComponent],
    field_index: Mapping[str, Mapping[str, Any]],
    group_members: Mapping[str, Sequence[str]],
    *,
    embed: EmbedFn | None = None,
    top_k: int = _DEFAULT_TOP_K,
    threshold: float = GATE1_SUGGEST_MIN_COSINE,
) -> GroupSuggestionResult:
    """Suggest the concept groups each declared component may measure — retrieval only, no model call, $0.

    The free half of :func:`match_components`, for a screen that has groups but no assigned records (Gate 1):
    the run's VARIABLES are indexed with :func:`build_variable_index` (multi-select checklists exploded into
    option units, as the paid match does), and :func:`shortlist_concepts` picks each component's top ``top_k``
    DISTINCT groups via ``var_to_group``. Each picked group is then scored by the dense cosine between the
    component's query (``"<name>. <definition>"``) and the group's best-matching member — an absolute score,
    comparable across components, which the caller holds against ``threshold``. This returns every picked group
    with its score; it does not filter, so a caller can show or sweep the cut-off.

    ``group_members`` is the membership the reviewer currently sees (moves and New groups applied); the index is
    cut to those members, so a variable in no group neither surfaces nor shapes retrieval. With no ``embed``
    there is no comparable score: every component comes back empty, ``scored`` is False, and ``reason`` says so.
    Deterministic for a deterministic ``embed``.
    """
    var_to_group: dict[str, str] = {}
    for gid, members in group_members.items():
        for m in members or []:
            var_to_group.setdefault(str(m), str(gid))

    empty = [ComponentSuggestions(component=c.name) for c in components]
    if embed is None:
        return GroupSuggestionResult(components=empty, scored=False, threshold=threshold, reason=_NO_DENSE_REASON)

    scoped = {k: v for k, v in field_index.items() if k in var_to_group}
    index = build_variable_index(scoped, checklist_members=set(var_to_group))
    if not index or not components:
        return GroupSuggestionResult(components=empty, scored=True, threshold=threshold, n_variables_indexed=len(index))

    # The shortlist embeds the corpus and then the queries; remember both so the cosines below reuse them.
    seen: dict[tuple[str, ...], np.ndarray] = {}

    def remembered(texts: list[str]) -> np.ndarray:
        key = tuple(texts)
        if key not in seen:
            seen[key] = np.asarray(embed(texts), dtype=np.float32)
        return seen[key]

    shortlists = shortlist_concepts(components, index, embed=remembered, top_k=top_k, var_to_group=var_to_group)
    texts = [e.retrieval_text for e in index]
    queries = [f"{c.name}. {c.definition}".strip() for c in components]
    cosine = _unit_rows(remembered(queries)) @ _unit_rows(remembered(texts)).T

    units_of: dict[str, list[int]] = {}
    for j, entry in enumerate(index):
        gid = var_to_group.get(_base_variable_id(entry.concept_id))
        if gid is not None:
            units_of.setdefault(gid, []).append(j)

    out: list[ComponentSuggestions] = []
    for i, component in enumerate(components):
        picked: list[str] = []
        for entry in shortlists.get(component.name, []):
            gid = var_to_group.get(_base_variable_id(entry.concept_id))
            if gid is not None and gid not in picked:
                picked.append(gid)
        ranked: list[tuple[float, int, GroupSuggestion]] = []
        for position, gid in enumerate(picked):
            rows = units_of[gid]
            best = rows[int(np.argmax(cosine[i, rows]))]
            unit = index[best]
            value = round(float(cosine[i, best]), 4)
            suggestion = GroupSuggestion(
                group_id=gid,
                score=value,
                best_member=_base_variable_id(unit.concept_id),
                best_option=unit.option_label,
            )
            ranked.append((value, position, suggestion))
        ranked.sort(key=lambda s: (-s[0], s[1]))
        out.append(ComponentSuggestions(component=component.name, groups=[s[2] for s in ranked]))
    return GroupSuggestionResult(components=out, scored=True, threshold=threshold, n_variables_indexed=len(index))


def suggestions_to_dict(result: GroupSuggestionResult) -> dict[str, Any]:
    """Serialize :func:`suggest_groups`' result to JSON-ready camelCase — the contract a UI layer consumes.

    ``scoreKind`` names what ``score`` and ``threshold`` measure, so a consumer never mistakes this free
    search's cosine for the paid judge's confidence (both are numbers in [0, 1]; they are not the same claim).
    """

    def _group(g: GroupSuggestion) -> dict[str, Any]:
        payload: dict[str, Any] = {"groupId": g.group_id, "score": g.score, "bestMember": g.best_member}
        if g.best_option:
            payload["bestOption"] = g.best_option
        return payload

    return {
        "scored": result.scored,
        "scoreKind": "dense_cosine",
        "threshold": result.threshold,
        "reason": result.reason,
        "nVariablesIndexed": result.n_variables_indexed,
        "components": [{"component": c.component, "groups": [_group(g) for g in c.groups]} for c in result.components],
    }


def _component_key(position: int) -> str:
    """The stable join token for one component — ``C1``, ``C2``, … — independent of its label."""
    return f"C{position + 1}"


def _normalize_name(value: str) -> str:
    """Fold a component name for the fallback join: NFKC, dash-unified, collapsed space, casefolded."""
    folded = unicodedata.normalize("NFKC", value or "")
    for dash in ("—", "–", "−"):  # em / en / minus -> hyphen
        folded = folded.replace(dash, "-")
    return " ".join(folded.split()).casefold()


def _match_prompt(
    definition: ScoreDefinition, shortlists: Mapping[str, list[ConceptEntry]]
) -> tuple[str, str, dict[str, set[str]]]:
    """Render the judge pass, keyed by a STABLE per-component token rather than the component's label.

    The join used to run on the name the model echoed back, and that name was used twice — once for the
    allowlist and once for the final lookup — so a single character of drift failed a component twice and
    produced a clean ``0/N -> infeasible`` with nothing raised. The old rendering actively invited that
    drift by putting the name and its definition on ONE line separated by an em-dash, then asking for "the
    component names exactly as given": echoing the whole rendered line back is a fair reading of that.

    So the key is now ``C1``/``C2``/… — short, unambiguous, and nothing a model is tempted to reformat —
    and the definition sits on its own line where it cannot be mistaken for part of the name.
    """
    allowed: dict[str, set[str]] = {}
    blocks: list[str] = []
    for position, component in enumerate(definition.components):
        key = _component_key(position)
        candidates = shortlists.get(component.name) or []
        allowed[key] = {c.concept_id for c in candidates}
        lines = [f"  [{key}] COMPONENT: {component.name}"]
        if component.definition:
            lines.append(f"        Definition: {component.definition}")
        lines.append("        Candidates:")
        lines.extend(
            [
                f"          [{c.concept_id}] {c.concept}"
                + (f"  · answers: {c.answer_text[:160]}" if c.answer_text else "")
                + (f"  · cohorts: {', '.join(c.cohorts)}" if c.cohorts else "")
                + (f"  · units: {c.units}" if c.units else "")
                for c in candidates
            ]
            or ["          (no candidate concepts retrieved)"]
        )
        blocks.append("\n".join(lines))

    system = (
        "You decide, for each COMPONENT of a composite score, WHICH of the CANDIDATE CONCEPTS from a "
        "harmonization run measure it — zero, one, or several.\n\n"
        "STRICT RULES:\n"
        "- Identify each component by its componentKey (C1, C2, …), copied exactly. Do not paraphrase it and "
        "do not substitute the component's name.\n"
        "- Choose a conceptId ONLY from that component's own candidate list, copied exactly. Never invent an "
        "id, never reuse an id from another component's list.\n"
        "- List EVERY candidate that measures the component — there are often several, because one real "
        "concept is split across cohort variables. Give each its own entry with its own confidence. OMIT "
        "candidates that do not measure it; if none do, return no entry for that component (its absence is the "
        'honest "not found" — a wrong match silently corrupts the score, so omit when unsure).\n'
        "- Match on WHAT IS MEASURED, not on shared words. A diagnosis of hypertension is not a blood-pressure "
        "measurement; family history of a condition is not the condition; difficulty walking is not gait speed.\n"
        "- A candidate measuring only part of the component (one side, one timepoint) is still a match — say so "
        "in the rationale and lower the confidence.\n"
        "- confidence is 0.0–1.0: your certainty that this candidate measures this component.\n\n"
        "Respond with ONLY valid JSON (no markdown fences) matching this schema:\n"
        '{"matches": [{"componentKey": string, "conceptId": string, "confidence": number, '
        '"rationale": string}]}'
    )
    user = (
        f"Composite score: {definition.name}"
        + (f" ({definition.citation})" if definition.citation else "")
        + (f"\nCombination rule: {definition.combination_rule}" if definition.combination_rule else "")
        + "\n\nComponents and their candidate concepts from this run:\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn one entry per candidate that measures a component — several per component is fine — "
        + "each identified by its componentKey."
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


def _best_decision(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The single strongest decision for a component — the highest-confidence one that named a concept,
    else the first. Preserves the one-pick group-concept path when the judge returns one entry, and picks
    sensibly if it returns several."""
    matched = [d for d in decisions if d.get("concept_id")]
    if matched:
        return dict(max(matched, key=lambda d: float(d.get("confidence", 0.0))))
    return dict(decisions[0]) if decisions else {}


def _rollup_to_groups(
    rated: Sequence[tuple[str, float]],
    var_to_group: Mapping[str, str],
    groups_by_id: Mapping[str, ConceptEntry],
) -> tuple[ConceptEntry | None, float, list[tuple[str, float]], list[tuple[str, float]]]:
    """Roll a component's judge-rated VARIABLES up to their concept GROUPS (variable-only matching).

    Returns ``(surfaced_group, aggregate, matched_members, group_candidates)``:
    - each rated ``(variable_id, confidence)`` binds to its parent group; ungrouped outliers are dropped
      (groups-only surfacing);
    - a group's aggregate is the mean of its rated members' confidences — an over-merged group where only
      one of many members is on-topic scores well below that member;
    - the surfaced group is the highest-aggregate group (tie → the group holding the single best member);
    - ``matched_members`` are the surfaced group's rated members, best-first;
    - ``group_candidates`` are ``(group_id, aggregate)`` for every group the component reached, best-first —
      the deduped Swap list.
    """
    per_group: dict[str, list[tuple[str, float]]] = {}
    for var_id, conf in rated:
        gid = var_to_group.get(_base_variable_id(var_id))
        if gid is None or gid not in groups_by_id:
            continue
        per_group.setdefault(gid, []).append((var_id, conf))
    if not per_group:
        return None, 0.0, [], []

    def _agg(members: list[tuple[str, float]]) -> float:
        return sum(c for _, c in members) / len(members)

    scored = [(gid, _agg(ms), sorted(ms, key=lambda m: -m[1])) for gid, ms in per_group.items()]
    scored.sort(key=lambda s: (-s[1], -(s[2][0][1] if s[2] else 0.0)))
    best_gid, best_agg, best_members = scored[0]
    candidates = [(gid, round(agg, 4)) for gid, agg, _ in scored]
    return groups_by_id[best_gid], round(best_agg, 4), best_members, candidates


def _cohort_of(variable_id: str) -> str:
    """The cohort a ``"cohort:var"`` member id belongs to (empty when unprefixed)."""
    return variable_id.split(":", 1)[0] if ":" in variable_id else ""


def _coverage_from_rated(
    rated: Sequence[tuple[str, float]],
    var_to_group: Mapping[str, str],
    groups_by_id: Mapping[str, ConceptEntry],
    floor: float = _COVERAGE_CONFIDENCE_FLOOR,
) -> tuple[list[str], dict[str, list[tuple[str, float]]], list[tuple[str, float, int, int]]]:
    """Member-level UNION coverage for one component — decoupled from the surfaced winner.

    Returns ``(cohorts, cohort_members, group_candidates)``:
    - ``cohorts`` — every cohort with at least one judge-accepted member AT/ABOVE ``floor`` that rolls up to
      some group, sorted. A single on-topic member in an over-merged (low group-mean) group still credits its
      cohort — coverage is member-level, not group-mean. Ungrouped outliers never contribute.
    - ``cohort_members`` — that cohort → its supporting ``(variable_id, confidence)`` members, best-first.
    - ``group_candidates`` — EVERY group the component reached, ``(group_id, aggregate, n_matched, n_total)``
      best-first: the aggregate is the mean of that group's rated members (same as :func:`_rollup_to_groups`),
      ``n_matched`` is how many of the group's members the judge rated and ``n_total`` its member count — the
      "X of Y group members matched" a review list shows. Aggregate/X-of-Y use ALL rated members (the
      swap list mirrors what the judge saw); only the union `cohorts`/`cohort_members` apply the floor.
    """
    per_group: dict[str, list[tuple[str, float]]] = {}
    for var_id, conf in rated:
        gid = var_to_group.get(_base_variable_id(var_id))
        if gid is None or gid not in groups_by_id:
            continue
        per_group.setdefault(gid, []).append((var_id, conf))

    cohort_members: dict[str, list[tuple[str, float]]] = {}
    for members in per_group.values():
        for var_id, conf in members:
            if conf < floor:
                continue
            cohort = _cohort_of(var_id)
            if not cohort:
                continue
            cohort_members.setdefault(cohort, []).append((var_id, conf))
    for cohort in cohort_members:
        cohort_members[cohort].sort(key=lambda m: -m[1])

    stats: list[tuple[str, float, int, int]] = []
    for gid, members in per_group.items():
        aggregate = round(sum(c for _, c in members) / len(members), 4)
        n_total = len(groups_by_id[gid].members)
        stats.append((gid, aggregate, len(members), n_total))
    stats.sort(key=lambda s: (-s[1], -s[2]))

    return sorted(cohort_members), cohort_members, stats


def match_components(
    definition: ScoreDefinition,
    index: Sequence[ConceptEntry],
    complete: CompleteFn,
    *,
    embed: EmbedFn | None = None,
    top_k: int = _DEFAULT_TOP_K,
    overrides: Mapping[str, str | None] | None = None,
    group_lookup: tuple[Mapping[str, str], Mapping[str, ConceptEntry]] | None = None,
) -> list[ComponentMatch]:
    """Map each component onto a concept from the run — retrieval bounds the choices, one LLM pass decides.

    ``overrides`` is the reviewer's structured edit: ``{component_name: concept_id}`` pins a match (any
    concept in the index, not just a retrieved one) and ``{component_name: None}`` drops it. Pinned
    components are excluded from the judge pass entirely, so a fully-pinned re-derive costs **no** LLM call.

    ``group_lookup`` switches on VARIABLE-ONLY matching (v2 score builder): ``index`` is the run's
    variables (built by :func:`build_variable_index`), the judge rates each relevant variable, and the
    rated variables are rolled up to their concept GROUPS via ``(var→group, group→entry)``. A match then
    surfaces the GROUP (its aggregate confidence + the member variables that reached it), never a bare
    variable. Absent, the legacy group-concept path runs unchanged. Pins in this mode name a GROUP id.

    Grounding guard: a returned id that was not in that component's shortlist is discarded and the component
    is reported missing.
    """
    by_id = {e.concept_id: e for e in index}
    var_to_group, groups_by_id = group_lookup if group_lookup else ({}, {})
    pins = dict(overrides or {})

    def _match_for(component: ScoreComponent, entry: ConceptEntry | None, **kwargs: Any) -> ComponentMatch:
        return ComponentMatch(
            component=component.name,
            concept_id=entry.concept_id if entry else None,
            concept=entry.label if entry else "",
            column=entry.column if entry else "",
            cohorts=list(entry.cohorts) if entry else [],
            source_variables=list(entry.members) if entry else [],
            required=component.required,
            is_variable=bool(entry.is_variable) if entry else False,
            **kwargs,
        )

    pending = [c for c in definition.components if c.name not in pins]
    # In variable-only mode the shortlist is chosen over DISTINCT groups: pass the var→group map so
    # near-duplicate variables of one concept spend a single candidate slot. Absent in the legacy path.
    group_map = var_to_group if group_lookup else None
    shortlists = shortlist_concepts(pending, index, embed=embed, top_k=top_k, var_to_group=group_map) if pending else {}

    decisions: dict[str, list[dict[str, Any]]] = {}
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
        if not any(allowed.values()):
            # Nothing was offered for ANY component, so the judge can only answer null. Worth a call still
            # (a future prompt may reason over the gap), but never worth confusing with a judge's decision.
            logger.warning(
                "composite: retrieval offered no candidates for any of %d component(s) of %r — "
                "every component will report no_candidates",
                len(pending),
                definition.name,
            )
        # Two ways to find the component an entry refers to: its stable key (what we now ask for), or its
        # NAME (what older/looser responses send). The fallback is why upgrading the prompt cannot regress
        # a model that ignores the key — and it deliberately covers the drift actually observed, where the
        # model echoes the whole rendered line, `name — definition`, back as the component.
        by_key = {_component_key(i): c for i, c in enumerate(pending)}
        by_alias: dict[str, str] = {}
        for i, c in enumerate(pending):
            key = _component_key(i)
            for alias in (c.name, f"{c.name} — {c.definition}", f"{c.name}: {c.definition}"):
                by_alias.setdefault(_normalize_name(alias), key)
        raw = complete(user, system=system, max_tokens=_MATCH_MAX_TOKENS)
        rejected_ids = 0
        for item in _parse_matches(raw):
            key = str(item.get("componentKey") or item.get("component_key") or "").strip()
            if key not in by_key:
                echoed = str(item.get("component", "") or "")
                folded = _normalize_name(echoed)
                key = by_alias.get(folded, "")
                if not key:
                    # Last resort: the echo starts with a component's name (a definition, a parenthetical,
                    # or a unit was appended). Longest name first, so "Grip strength, dominant" is not
                    # claimed by "Grip strength" when both are components.
                    for alias, candidate_key in sorted(by_alias.items(), key=lambda kv: -len(kv[0])):
                        if folded.startswith(alias):
                            key = candidate_key
                            break
                if not key:
                    logger.warning(
                        "composite: dropping a judge entry that names no known component "
                        "(componentKey=%r, component=%r)",
                        item.get("componentKey"),
                        echoed[:120],
                    )
                    continue
            concept_id = item.get("conceptId") or item.get("concept_id")
            concept_id = str(concept_id).strip() if concept_id else None
            reason = MatchReason.MATCHED if concept_id else MatchReason.JUDGE_DECLINED
            if concept_id and concept_id not in allowed.get(key, set()):
                concept_id = None  # hallucinated or cross-component id -> honest gap
                reason = MatchReason.ID_REJECTED
                rejected_ids += 1
            decisions.setdefault(key, []).append(
                {
                    "concept_id": concept_id,
                    "confidence": _confidence(item.get("confidence")),
                    "rationale": str(item.get("rationale", "") or "").strip(),
                    "reason": reason,
                }
            )
        if rejected_ids:
            # A nonzero count here alongside zero matches is the signature of a grounding/join problem
            # rather than an honest gap — the judge answered, and we threw its answers away.
            logger.warning(
                "composite: discarded %d id(s) outside their component's shortlist for %r",
                rejected_ids,
                definition.name,
            )

    pending_key = {c.name: _component_key(i) for i, c in enumerate(pending)}
    matches: list[ComponentMatch] = []
    for component in definition.components:
        if component.name in pins:
            pinned_id = pins[component.name]
            # A pin names a GROUP id in variable-only mode, a concept/variable id in the legacy path.
            entry = (
                (groups_by_id.get(str(pinned_id)) if group_lookup else by_id.get(str(pinned_id))) if pinned_id else None
            )
            matches.append(
                _match_for(
                    component,
                    entry,
                    pinned=True,
                    confidence=1.0 if entry else 0.0,
                    rationale="pinned by reviewer" if entry else "dropped by reviewer",
                    reason=MatchReason.PINNED if entry else MatchReason.DROPPED,
                )
            )
            continue
        shortlist = [c.concept_id for c in shortlists.get(component.name, [])]
        decision_list = decisions.get(pending_key.get(component.name, ""), [])

        if group_lookup is not None:
            # Variable-only: roll the judge's rated variables up to their groups; surface the top group.
            rated = [
                (str(d["concept_id"]), float(d.get("confidence", 0.0))) for d in decision_list if d.get("concept_id")
            ]
            # Surface the winner GROUP (concept/column/confidence), but measure COVERAGE as a member-level
            # per-cohort UNION across ALL reached groups: the winner answers "what is the
            # canonical concept?", the union answers "which cohorts have it?" — two questions the old single
            # winner conflated. `candidates` from the rollup (2-tuples) is discarded in favour of the union's
            # X-of-Y candidate stats.
            entry, aggregate, members, _ = _rollup_to_groups(rated, var_to_group, groups_by_id)
            union_cohorts, cohort_members, candidate_stats = _coverage_from_rated(rated, var_to_group, groups_by_id)
            if entry is not None:
                reason = MatchReason.MATCHED
            elif not shortlist:
                reason = MatchReason.NO_CANDIDATES
            elif not decision_list:
                reason = MatchReason.NO_DECISION
            else:
                # The judge rated candidates, but every one was an ungrouped outlier (or none matched) —
                # an honest gap, not a retrieval failure.
                reason = MatchReason.JUDGE_DECLINED
            best = _best_decision(decision_list)
            match = _match_for(
                component,
                entry,
                confidence=aggregate,
                rationale=str(best.get("rationale", "")),
                shortlist=shortlist,
                reason=reason,
                matched_members=members,
                group_candidates=candidate_stats,
            )
            if entry is not None and _union_coverage_enabled():
                match.cohorts = union_cohorts  # decouple coverage from the winner group's cohorts
                match.coverage_members = cohort_members
            matches.append(match)
            continue

        # Legacy group-concept path: one best pick per component.
        decision = _best_decision(decision_list)
        entry = by_id.get(str(decision.get("concept_id"))) if decision.get("concept_id") else None
        if entry:
            reason = MatchReason.MATCHED
        elif not shortlist:
            # Checked BEFORE the decision: retrieval offering nothing is the upstream cause, and reporting
            # a judge that "declined" a list it was never shown is the exact conflation this field exists
            # to end. True even when the judge dutifully returned a null entry for it.
            reason = MatchReason.NO_CANDIDATES
        elif not decision_list:
            # The judge returned nothing for this component. Distinct from a decline: with the keyed join
            # this should be rare, and a run where EVERY component lands here is a parse/join failure.
            reason = MatchReason.NO_DECISION
        else:
            reason = MatchReason(decision.get("reason") or MatchReason.JUDGE_DECLINED)
        matches.append(
            _match_for(
                component,
                entry,
                confidence=float(decision.get("confidence", 0.0)) if entry else 0.0,
                rationale=str(decision.get("rationale", "")),
                shortlist=shortlist,
                reason=reason,
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
    field_index: Mapping[str, Mapping[str, Any]] | None = None,
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
    # Variable-only matching (v2, opt-in via `field_index`): the closed world is the run's VARIABLES, not
    # its group concepts — variable text is ground truth, whereas a group's concept label is an LLM summary
    # that is empty or wrong for an over-merged cluster. The judge rates variables; matched variables roll
    # up to their concept GROUPS, which are what a match surfaces (a group's score is the aggregate of its
    # on-topic members). Without a `field_index` the legacy group-concept path runs unchanged.
    if field_index:
        group_lookup = build_group_lookup(records)
        # Explode a multi-select checklist into per-option coverage units only when its base variable is a
        # run-group member: options of an unclustered checklist can never surface after roll-up.
        index = build_variable_index(field_index, checklist_members=set(group_lookup[0]))
        matches = match_components(
            definition, index, counted, embed=embed, top_k=top_k, overrides=overrides, group_lookup=group_lookup
        )
        run_cohorts = sorted({c for e in group_lookup[1].values() for c in e.cohorts})
    else:
        index = build_concept_index(records)
        matches = match_components(definition, index, counted, embed=embed, top_k=top_k, overrides=overrides)
        run_cohorts = sorted({c for e in index for c in e.cohorts})
    feasibility = assess_feasibility(definition, matches, cohorts=run_cohorts)
    spec = build_composite_spec(definition, matches, feasibility)
    return CompositeResult(spec=spec, n_concepts_indexed=len(index), calls_made=calls)


def _member_payload(variable_id: str, confidence: float) -> dict[str, Any]:
    """One member/coverage entry for the UI contract. A checklist OPTION unit (id ``…#opt=<label>``) carries
    its parent ``variableId`` and the ``optionLabel`` that binds it, so the panel can name the option."""
    if _OPTION_ID_SEP in variable_id:
        base, _, label = variable_id.partition(_OPTION_ID_SEP)
        return {"variableId": base, "confidence": confidence, "optionLabel": label}
    return {"variableId": variable_id, "confidence": confidence}


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
                # Why this component is (un)matched: matched | no_candidates | judge_declined |
                # id_rejected | no_decision | pinned | dropped. A UI must not render every unmatched
                # component the same way — "we found nothing to offer" and "the judge saw good options and
                # said no" are different answers to the reviewer, with different next actions.
                "reason": str(m.reason),
                # True when the match/candidate is a single source variable (variable-level matching)
                # rather than a harmonized concept group.
                "isVariable": m.is_variable,
                # Variable-only matching: the surfaced concept above is a GROUP; these are the member
                # variables (id + confidence) that rolled up into it — shown indented under the group —
                # and the deduped Swap list of every group the component's rated variables reached
                # (groupId + aggregate confidence + "X of Y group members matched", best-first). Empty in
                # the legacy group-concept path.
                "matchedMembers": [_member_payload(vid, conf) for vid, conf in m.matched_members],
                "groupCandidates": [
                    {"groupId": gid, "confidence": conf, "nMatched": n_matched, "nTotal": n_total}
                    for gid, conf, n_matched, n_total in m.group_candidates
                ],
                # Union coverage: `cohorts` is the per-cohort union over accepted members across all
                # reached groups (not the winner group's cohorts); this maps each covered cohort to the
                # supporting member variables so a UI can name the supporting variable/option per cohort. A
                # member that is a checklist OPTION carries its ``optionLabel``.
                "coverageMembers": {
                    cohort: [_member_payload(vid, conf) for vid, conf in members]
                    for cohort, members in m.coverage_members.items()
                },
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
