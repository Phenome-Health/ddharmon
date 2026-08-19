"""Selective export of transform specs — the *consumption* side of spec generation.

Spec generation emits one :class:`~ddharmon.harmonization.models.TransformSpec` per adopt/refine edge; a
real cohort yields hundreds. A researcher usually needs only the handful of concepts relevant to THEIR
analysis. These primitives let a user **mark concepts for export** and filter the record set down to the
marked ones before :func:`~ddharmon.harmonization.transform.export_transform_review` renders the campaign.

Design goals (so the notebook flag and a future GUI checkbox column share ONE mechanism):

- **Headless + dependency-light** — depends only on the models, never pandas / ipywidgets / the LLM stack,
  so the GUI backend can import it without the transform pipeline.
- **Concept = one** :class:`~ddharmon.harmonization.models.LeanBRecord` **(a distinct concept-group)** — the
  atomic export unit, keyed by its stable ``group_id``. Coarser selection (a whole semantic cluster, all
  ``refine`` edges, a cohort) is expressed as a set of ids drawn from the same table.
- **Opt-in by default** (lean output): nothing is marked unless the caller flips ``default_export`` or marks
  concepts explicitly.
- **Export-eligibility is separate from the EITL accept/reject verdict** — a concept can be a *correct*
  mapping yet irrelevant to a given study. An ``eligible`` predicate optionally gates *which* concepts may be
  marked (e.g. only EITL-accepted ones), without touching the verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import cast

from ddharmon.harmonization.models import LeanBRecord, TransformKind


@dataclass
class ExportConcept:
    """A selectable, serializable view of one harmonized concept (a :class:`LeanBRecord`).

    One row of the "mark for export" table. ``export`` is the flag the user toggles; ``eligible`` is False
    when an ``eligible`` predicate rejected the concept (shown, but not markable). A GUI renders these as a
    checkbox column; the notebook renders them as an editable DataFrame — both round-trip back through
    :func:`select_records`.
    """

    group_id: str
    cluster_id: str
    concept: str
    verdict: str
    route: str
    cde_id: str | None
    cohorts: list[str]
    cross_cohort: bool
    n_members: int
    n_specs: int  # non-identity transform specs on this concept (the export payload size)
    export: bool = False
    eligible: bool = True

    @property
    def key(self) -> str:
        """The stable id this concept selects on (``group_id``, falling back to ``cluster_id``)."""
        return self.group_id or self.cluster_id


def _non_identity_specs(rec: LeanBRecord) -> int:
    """Count of the record's transform specs that actually carry a recode (identity is a no-op)."""
    return sum(1 for t in rec.transforms if t.kind != TransformKind.IDENTITY)


def _key(rec: LeanBRecord) -> str:
    """The id a record selects on — its ``group_id`` (unique per concept-group), else ``cluster_id``."""
    return rec.group_id or rec.cluster_id


def list_export_concepts(
    records: Iterable[LeanBRecord],
    *,
    default_export: bool = False,
    eligible: Callable[[LeanBRecord], bool] | None = None,
    only_with_specs: bool = True,
) -> list[ExportConcept]:
    """Build the mark-for-export table — one :class:`ExportConcept` per record.

    ``default_export`` sets the initial mark: ``False`` = opt-in / lean (mark what you want), ``True`` =
    opt-out / export-all (unmark what you don't). ``eligible`` optionally restricts which concepts may be
    exported — e.g. ``lambda r: r.group_id in accepted_ids`` for accepted-only; ineligible concepts get
    ``eligible=False`` and ``export=False`` regardless of ``default_export``. ``only_with_specs`` (default)
    drops concepts carrying no non-identity spec — the transform-spec surface has nothing to export for them;
    pass ``False`` for a general concept picker.
    """
    out: list[ExportConcept] = []
    for rec in records:
        n = _non_identity_specs(rec)
        if only_with_specs and n == 0:
            continue
        ok = eligible(rec) if eligible else True
        out.append(
            ExportConcept(
                group_id=rec.group_id or rec.cluster_id,
                cluster_id=rec.cluster_id,
                concept=rec.concept,
                verdict=rec.verdict,
                route=rec.route,
                cde_id=rec.cde_id,
                cohorts=list(rec.cohorts),
                cross_cohort=rec.cross_cohort,
                n_members=rec.n_members,
                n_specs=n,
                export=bool(default_export and ok),
                eligible=ok,
            )
        )
    return out


def concepts_matching(
    records: Iterable[LeanBRecord],
    *,
    verdict: str | None = None,
    cohort: str | None = None,
    cluster_id: str | None = None,
    cross_cohort: bool | None = None,
    with_specs: bool = True,
) -> set[str]:
    """Ids of records matching **all** the given filters — for one-action coarse marking.

    Grab "all ``refine`` edges", "everything touching UKBB", "this whole cluster", or "only cross-cohort
    concepts". Any argument left ``None`` is not filtered on. ``with_specs`` (default) keeps only concepts
    that carry a non-identity spec. Union the returned sets and hand them to :func:`select_records`.
    """
    out: set[str] = set()
    for rec in records:
        if verdict is not None and rec.verdict != verdict:
            continue
        if cluster_id is not None and rec.cluster_id != cluster_id:
            continue
        if cross_cohort is not None and rec.cross_cohort != cross_cohort:
            continue
        if cohort is not None and cohort not in rec.cohorts:
            continue
        if with_specs and _non_identity_specs(rec) == 0:
            continue
        out.add(_key(rec))
    return out


def select_records(records: Iterable[LeanBRecord], selection: object) -> list[LeanBRecord]:
    """Filter ``records`` to the concepts a selection picks. ``selection`` may be, in priority order:

    - ``None`` → all records (no filtering; the backward-compatible export-all default).
    - a callable ``predicate(record) -> bool``.
    - an iterable of :class:`ExportConcept` (the edited table) → keeps those with ``.export`` truthy.
    - a mapping ``{id: keep}`` → keeps ids that map truthy.
    - an iterable of string ids → keeps records whose ``group_id`` **or** ``cluster_id`` is in the set, so a
      *cluster* id transparently selects all of its concept-groups (coarse selection for free).
    """
    recs = list(records)
    if selection is None:
        return recs
    if callable(selection):
        return [r for r in recs if selection(r)]
    if isinstance(selection, dict):
        ids = {k for k, v in selection.items() if v}
        return [r for r in recs if r.group_id in ids or r.cluster_id in ids]
    items = list(cast("Iterable[object]", selection))  # materialize the iterable once
    if items and all(isinstance(x, ExportConcept) for x in items):
        ids = {c.key for c in items if isinstance(c, ExportConcept) and c.export}
        return [r for r in recs if _key(r) in ids]
    ids = {str(x) for x in items}
    return [r for r in recs if r.group_id in ids or r.cluster_id in ids]
