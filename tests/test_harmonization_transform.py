"""Tests for the transform-spec data model (C1). Generator tests are added with the spec-gen stage."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from ddharmon.harmonization import (
    TransformKind,
    TransformSpec,
    apply_coherence_gate,
    assemble_arith_specgen,
    assemble_concept_gate,
    assemble_gencde_specgen,
    assemble_specgen,
    eval_formula,
    generate_unit_specs,
    generate_wide_to_long_specs,
    prepare_arith_specgen,
    prepare_concept_gate,
    prepare_gencde_specgen,
    prepare_specgen,
    verify_formula,
)
from ddharmon.harmonization.models import ROUTE_RESIDUAL, GenCDE, LeanBRecord
from ddharmon.harmonization.transform import (
    FormulaError,
    formula_names,
    is_identity_formula,
    is_safe_formula,
    monotonic_ordinal_fill,
)
from ddharmon.models.data_dictionary import Field, ResponseOption

APPROX = pytest.approx


class TestTransformSpecModel:
    def test_defaults(self):
        s = TransformSpec(source_variable="CohortA:smoke", target_cde_id="SmokeCDE", kind=TransformKind.CATEGORICAL)
        assert s.kind == "categorical"  # StrEnum compares to its value
        assert s.confidence == 0.0 and s.coverage == 0.0
        assert s.code_map == {} and s.unmapped_source_codes == []
        assert not (s.needs_units or s.needs_data or s.needs_review)
        assert s.generated_by == "llm"

    def test_categorical_round_trip(self):
        s = TransformSpec(
            source_variable="CohortA:smoke",
            target_cde_id="SmokeCDE",
            kind=TransformKind.CATEGORICAL,
            code_map={"1": "1", "2": "0"},
            coverage=1.0,
            confidence=0.9,
        )
        d = asdict(s)
        assert d["code_map"] == {"1": "1", "2": "0"}
        assert d["kind"] == "categorical"
        assert d["coverage"] == 1.0

    def test_leanbrecord_carries_per_edge_transforms(self):
        rec = LeanBRecord(cluster_id="c0", verdict="refine", route="assigned")
        assert rec.transforms == []
        rec.transforms.append(
            TransformSpec(source_variable="CohortA:smoke", target_cde_id="SmokeCDE", kind=TransformKind.CATEGORICAL)
        )
        assert len(rec.transforms) == 1 and rec.transforms[0].kind == TransformKind.CATEGORICAL

    def test_kind_members(self):
        assert {k.value for k in TransformKind} == {
            "identity",
            "categorical",
            "unit",
            "arithmetic",
            "data_dependent",
            "wide_to_long",
            "none",
        }


# ── C1 categorical spec-gen generator (prepare -> mock LLM -> assemble) ──


def _world(hf, *, source_encoding="1=Yes|2=No", cde_encoding="1=Yes|0=No", verdict="refine"):
    """A 1-member adopt/refine record assigned to a coded CDE, with prepared spec-gen prompts."""
    src = hf.field("smoke", "Do you currently smoke", encoding=source_encoding, data_type="categorical")
    cde = hf.field("SmokeCDE", "Current smoking status", field_id="cde_smoke", encoding=cde_encoding)
    ed_a = hf.embedded_dict("CohortA", [src], sem_vecs=np.array([[1.0]]))
    ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
    cde_fields = dict(ed_cde.dictionary.fields)
    rec = LeanBRecord(
        cluster_id="c0",
        verdict=verdict,
        route="assigned",
        cde_id="SmokeCDE",
        member_variable_names=["CohortA:smoke"],
    )
    sg = prepare_specgen([rec], [ed_a, ed_cde], cde_fields)
    return sg, rec


class TestPrepareSpecgen:
    def test_builds_prompt_with_both_value_sets(self, hf):
        sg, _ = _world(hf)
        assert len(sg) == 1  # one prompt per unique (cde_id, source value set)
        p = sg[0]
        assert p.id.startswith("leanb:specgen:")  # content-addressed by the (cde_id, source-encoding) signature
        # the source AND target value sets are both visible to the recode model
        assert "1=Yes|2=No" in p.user_prompt and "1=Yes|0=No" in p.user_prompt
        assert p.context["source_value_set"] == "1=Yes|2=No"
        assert p.context["edges"] == [("c0", "CohortA:smoke")]  # the edges this recode fans out to

    def test_skips_novel(self, hf):
        sg, _ = _world(hf, verdict="novel")
        assert sg == []

    def test_skips_when_cde_has_no_codes(self, hf):
        sg, _ = _world(hf, cde_encoding=None)
        assert sg == []

    def test_skips_when_no_coded_members(self, hf):
        sg, _ = _world(hf, source_encoding=None)
        assert sg == []


class TestAssembleSpecgen:
    def _run(self, hf, recode, **kw):
        """``recode`` = the single per-edge response object (the new bare-object contract)."""
        sg, rec = _world(hf, **kw)
        assemble_specgen(sg, {sg[0].id: recode}, [rec])
        return rec

    def test_full_coverage_categorical(self, hf):
        rec = self._run(hf, {"code_map": {"1": "1", "2": "0"}, "confidence": 0.9})
        assert len(rec.transforms) == 1
        t = rec.transforms[0]
        assert t.kind == TransformKind.CATEGORICAL
        assert t.code_map == {"1": "1", "2": "0"}
        assert t.coverage == 1.0 and t.needs_review is False
        assert t.unmapped_source_codes == []
        assert t.source_variable == "CohortA:smoke"  # bound from context, not the response

    def test_partial_coverage_flags_review(self, hf):
        rec = self._run(hf, {"code_map": {"1": "1"}, "confidence": 0.9})
        t = rec.transforms[0]
        assert t.coverage == 0.5 and t.needs_review is True
        assert t.unmapped_source_codes == ["2"]

    def test_identity_map_is_identity_kind(self, hf):
        rec = self._run(hf, {"code_map": {"1": "1", "2": "2"}, "confidence": 0.9}, cde_encoding="1=Yes|2=No")
        assert rec.transforms[0].kind == TransformKind.IDENTITY

    def test_drops_hallucinated_target(self, hf):
        rec = self._run(hf, {"code_map": {"1": "5", "2": "0"}, "confidence": 0.9})
        t = rec.transforms[0]
        assert t.code_map == {"2": "0"}  # "1"->"5" dropped: 5 is not in the CDE value set {1,0}
        assert "1" in t.unmapped_source_codes and t.needs_review is True

    def test_missing_recode_is_none_kind(self, hf):
        rec = self._run(hf, {})  # model returned no recode for the edge
        t = rec.transforms[0]
        assert t.kind == TransformKind.NONE and t.needs_review is True and t.code_map == {}

    def test_no_response_for_edge_is_none_kind(self, hf):
        sg, rec = _world(hf)
        assemble_specgen(sg, {}, [rec])  # the edge's prompt id is absent from responses
        assert rec.transforms[0].kind == TransformKind.NONE and rec.transforms[0].needs_review is True

    def test_tolerates_legacy_recodes_wrapper(self, hf):
        # soft Batch schema: the model may still wrap the single recode as {"recodes": [ {...} ]}
        rec = self._run(hf, {"recodes": [{"code_map": {"1": "1", "2": "0"}, "confidence": 0.9}]})
        assert rec.transforms[0].kind == TransformKind.CATEGORICAL and rec.transforms[0].coverage == 1.0

    def test_tolerates_bare_list_response(self, hf):
        rec = self._run(hf, [{"code_map": {"1": "1", "2": "0"}, "confidence": 0.9}])
        assert rec.transforms[0].code_map == {"1": "1", "2": "0"}

    # ── M9: monotonic ordinal fill (equal-length ordinal scales) ──
    _LIKERT = "1=Never|2=Rarely|3=Sometimes|4=Often|5=Always"

    def test_ordinal_fill_completes_offset_scale(self, hf):
        # source 1..5, CDE 0..4; LLM maps only the two endpoints -> interior filled by monotonic position
        rec = self._run(
            hf,
            {"code_map": {"1": "0", "5": "4"}, "confidence": 0.9},
            source_encoding=self._LIKERT,
            cde_encoding="0=Never|1=Rarely|2=Sometimes|3=Often|4=Always",
        )
        t = rec.transforms[0]
        assert t.kind == TransformKind.CATEGORICAL
        assert t.code_map == {"1": "0", "2": "1", "3": "2", "4": "3", "5": "4"}
        assert t.coverage == 1.0 and t.needs_review is False
        assert "ordinal fill" in t.rationale

    def test_ordinal_fill_equal_scale_is_identity(self, hf):
        # identical 1..5 scales, partial map -> filled to a full identity map -> IDENTITY kind
        rec = self._run(
            hf,
            {"code_map": {"1": "1", "5": "5"}, "confidence": 0.9},
            source_encoding=self._LIKERT,
            cde_encoding=self._LIKERT,
        )
        assert rec.transforms[0].kind == TransformKind.IDENTITY and rec.transforms[0].coverage == 1.0

    def test_ordinal_fill_skips_reverse_coded(self, hf):
        # reverse-coded anchors are off the monotonic diagonal -> NOT filled, left for review
        rec = self._run(
            hf,
            {"code_map": {"1": "5", "5": "1"}, "confidence": 0.9},
            source_encoding=self._LIKERT,
            cde_encoding=self._LIKERT,
        )
        t = rec.transforms[0]
        assert t.code_map == {"1": "5", "5": "1"} and t.coverage == 0.4 and t.needs_review is True

    def test_ordinal_fill_needs_two_anchors(self, hf):
        # a single mapped code is not enough evidence to fill the scale
        rec = self._run(
            hf,
            {"code_map": {"3": "3"}, "confidence": 0.9},
            source_encoding=self._LIKERT,
            cde_encoding=self._LIKERT,
        )
        assert rec.transforms[0].code_map == {"3": "3"} and rec.transforms[0].coverage == 0.2

    def test_ordinal_fill_skips_unequal_length(self, hf):
        # unequal scale lengths -> not a positional alignment -> not filled
        rec = self._run(
            hf,
            {"code_map": {"1": "1", "3": "3"}, "confidence": 0.9},
            source_encoding=self._LIKERT,
            cde_encoding="1=Low|2=Mid|3=High",
        )
        assert rec.transforms[0].code_map == {"1": "1", "3": "3"} and rec.transforms[0].needs_review is True


class TestMonotonicOrdinalFill:
    """The pure M9 helper: complete a rank-aligned partial recode between equal-length integer ordinal scales."""

    def test_fills_offset_scale(self):
        out = monotonic_ordinal_fill(["1", "2", "3", "4", "5"], ["0", "1", "2", "3", "4"], {"1": "0", "5": "4"})
        assert out == {"1": "0", "2": "1", "3": "2", "4": "3", "5": "4"}

    def test_reverse_coded_returns_none(self):
        assert monotonic_ordinal_fill(["1", "2", "3"], ["1", "2", "3"], {"1": "3", "3": "1"}) is None

    def test_unequal_length_returns_none(self):
        assert monotonic_ordinal_fill(["1", "2", "3", "4"], ["1", "2", "3"], {"1": "1", "3": "3"}) is None

    def test_non_integer_codes_return_none(self):
        assert monotonic_ordinal_fill(["a", "b", "c"], ["x", "y", "z"], {"a": "x", "c": "z"}) is None

    def test_single_anchor_returns_none(self):
        assert monotonic_ordinal_fill(["1", "2", "3"], ["1", "2", "3"], {"2": "2"}) is None

    def test_already_complete_returns_none(self):
        assert monotonic_ordinal_fill(["1", "2", "3"], ["1", "2", "3"], {"1": "1", "2": "2", "3": "3"}) is None

    def test_binary_scale_below_min_length_returns_none(self):
        assert monotonic_ordinal_fill(["1", "2"], ["1", "0"], {"1": "1"}) is None


class TestSpecgenDedupAndFanout:
    """Edges sharing (cde_id, source value set) collapse to one prompt whose recode fans out to all of
    them; distinct encodings stay distinct prompts; a missing response NONEs only its own edges (the
    under-return regression)."""

    def _group(self, hf, encodings):
        """One adopt/refine record whose members carry the given source value-encodings."""
        members = [hf.field(f"v{i}", f"Question {i}", encoding=enc) for i, enc in enumerate(encodings)]
        cde = hf.field("CDE", "Concept", field_id="tiny", encoding="1=Yes|0=No")
        ed = hf.embedded_dict("CohortA", members, sem_vecs=np.ones((len(members), 1)))
        ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
        rec = LeanBRecord(
            cluster_id="c0",
            verdict="refine",
            route="assigned",
            cde_id="CDE",
            member_variable_names=[f"CohortA:v{i}" for i in range(len(members))],
        )
        sg = prepare_specgen([rec], [ed, ed_cde], dict(ed_cde.dictionary.fields))
        return sg, rec

    def test_identical_encodings_collapse_to_one_prompt(self, hf):
        sg, rec = self._group(hf, ["1=Yes|2=No"] * 5)  # 5 members, same encoding -> ONE prompt
        assert len(sg) == 1
        assert sorted(sg[0].context["edges"]) == [("c0", f"CohortA:v{i}") for i in range(5)]
        # the single recode fans out to all 5 edges
        assemble_specgen(sg, {sg[0].id: {"code_map": {"1": "1", "2": "0"}, "confidence": 0.9}}, [rec])
        assert len(rec.transforms) == 5
        assert all(t.kind == TransformKind.CATEGORICAL and t.code_map == {"1": "1", "2": "0"} for t in rec.transforms)
        assert {t.source_variable for t in rec.transforms} == {f"CohortA:v{i}" for i in range(5)}
        # per-edge copies, not a shared dict
        rec.transforms[0].code_map["1"] = "MUT"
        assert rec.transforms[1].code_map["1"] == "1"

    def test_distinct_encodings_stay_distinct_prompts(self, hf):
        sg, _ = self._group(hf, ["1=Yes|2=No", "1=Y|0=N", "1=Yes|2=No|3=Maybe"])
        assert len(sg) == 3  # three different source value sets -> three prompts

    def test_missing_response_nones_only_its_own_edges(self, hf):
        # two DISTINCT encodings -> two prompts; answer one, leave the other unanswered (the under-return case)
        sg, rec = self._group(hf, ["1=Yes|2=No", "1=low|2=mid|3=high"])
        assert len(sg) == 2
        answered = next(p for p in sg if "1=Yes|2=No" in p.user_prompt)
        assemble_specgen(sg, {answered.id: {"code_map": {"1": "1", "2": "0"}, "confidence": 0.9}}, [rec])
        by_sv = {t.source_variable: t for t in rec.transforms}
        assert len(rec.transforms) == 2
        assert by_sv["CohortA:v0"].kind == TransformKind.CATEGORICAL  # the answered yes/no edge
        # the unanswered edge is an honest NONE — not a side effect of the other being answered
        assert by_sv["CohortA:v1"].kind == TransformKind.NONE and by_sv["CohortA:v1"].needs_review is True


# ── C2 N1 deterministic unit-spec generator ──


class TestGenerateUnitSpecs:
    def _numeric_world(self, hf, *, src_units, cde_units, src_encoding=None, verdict="refine"):
        """A 1-member adopt/refine record over a NUMERIC source -> numeric CDE, after N1 generation."""
        src = hf.field("weight", "Body weight", units=src_units, encoding=src_encoding, data_type="continuous")
        cde = hf.field("WeightCDE", "Body weight", field_id="cde_wt", units=cde_units, data_type="continuous")
        ed_a = hf.embedded_dict("CohortA", [src], sem_vecs=np.array([[1.0]]))
        ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
        rec = LeanBRecord(
            cluster_id="c0",
            verdict=verdict,
            route="assigned",
            cde_id="WeightCDE",
            member_variable_names=["CohortA:weight"],
        )
        generate_unit_specs([rec], [ed_a, ed_cde], dict(ed_cde.dictionary.fields))
        return rec, [ed_a, ed_cde]

    def test_known_conversion_kg_to_lb(self, hf):
        rec, _ = self._numeric_world(hf, src_units="kg", cde_units="lb")
        assert len(rec.transforms) == 1
        t = rec.transforms[0]
        assert t.kind == TransformKind.UNIT
        assert t.factor == APPROX(1 / 0.45359237) and t.offset == APPROX(0.0)
        assert t.source_unit == "kg" and t.target_unit == "lb"
        assert t.generated_by == "rule" and t.needs_units is False
        assert t.confidence == 0.9  # known conversion -> high band

    def test_temperature_conversion_has_offset(self, hf):
        rec, _ = self._numeric_world(hf, src_units="degF", cde_units="degC")
        t = rec.transforms[0]
        assert t.kind == TransformKind.UNIT
        assert t.factor == APPROX(5 / 9) and t.offset == APPROX(-160 / 9)

    def test_identity_same_unit(self, hf):
        rec, _ = self._numeric_world(hf, src_units="kg", cde_units="kg")
        assert rec.transforms[0].kind == TransformKind.IDENTITY

    def test_identity_same_unrecognized_unit(self, hf):
        rec, _ = self._numeric_world(hf, src_units="score", cde_units="score")
        assert rec.transforms[0].kind == TransformKind.IDENTITY

    def test_missing_units_flags_needs_units(self, hf):
        rec, _ = self._numeric_world(hf, src_units=None, cde_units=None)
        t = rec.transforms[0]
        assert t.kind == TransformKind.UNIT and t.needs_units is True
        assert t.needs_review is True and t.confidence == 0.4

    def test_cross_family_flags_needs_units(self, hf):
        rec, _ = self._numeric_world(hf, src_units="kg", cde_units="cm")
        t = rec.transforms[0]
        assert t.kind == TransformKind.UNIT and t.needs_units is True

    def test_skips_categorical_edge(self, hf):
        # a coded source is a categorical edge (C1 handles it) -> N1 emits nothing
        rec, _ = self._numeric_world(hf, src_units=None, cde_units=None, src_encoding="1=Yes|2=No")
        assert rec.transforms == []

    def test_skips_novel(self, hf):
        rec, _ = self._numeric_world(hf, src_units="kg", cde_units="lb", verdict="novel")
        assert rec.transforms == []

    def test_idempotent(self, hf):
        rec, embedded = self._numeric_world(hf, src_units="kg", cde_units="lb")
        cde_fields = dict(embedded[1].dictionary.fields)
        generate_unit_specs([rec], embedded, cde_fields)  # second pass
        assert len(rec.transforms) == 1  # not duplicated


# ── C2 N2 arithmetic verify harness ──


class TestEvalFormula:
    def test_basic_arithmetic(self):
        assert eval_formula("source / 12", {"source": 24}) == APPROX(2.0)
        assert eval_formula("source * 2.20462", {"source": 10}) == APPROX(22.0462)
        assert eval_formula("a + b * c", {"a": 1, "b": 2, "c": 3}) == APPROX(7.0)

    def test_power_and_functions(self):
        assert eval_formula("weight / (height ** 2)", {"weight": 80, "height": 2}) == APPROX(20.0)
        assert eval_formula("sqrt(source)", {"source": 9}) == APPROX(3.0)

    def test_unknown_variable_raises(self):
        with pytest.raises(FormulaError):
            eval_formula("source + missing", {"source": 1})

    def test_disallowed_syntax_raises(self):
        for bad in ("__import__('os')", "source.attr", "[x for x in range(3)]", "open('f')"):
            with pytest.raises(FormulaError):
                eval_formula(bad, {"source": 1, "x": 1})


class TestFormulaHelpers:
    def test_formula_names_excludes_funcs(self):
        assert formula_names("sqrt(source) + weight") == {"source", "weight"}
        assert formula_names("source / 12") == {"source"}

    def test_is_safe_formula(self):
        assert is_safe_formula("source / 12", ["source"])
        assert is_safe_formula("1 / (source - 1)", ["source"])  # divide-by-zero on dummy != unsafe
        assert not is_safe_formula("__import__('os').system('x')", ["source"])
        assert not is_safe_formula("", ["source"])

    def test_is_identity_formula(self):
        # M8: no-ops in `source` alone
        assert is_identity_formula("source")
        assert is_identity_formula("source * 1")
        assert is_identity_formula("source + 0")
        assert is_identity_formula("(source)")
        # NOT identities
        assert not is_identity_formula("source / 12")
        assert not is_identity_formula("source + 1")
        assert not is_identity_formula("source * height")  # references another input
        assert not is_identity_formula("")
        assert not is_identity_formula("null")


class TestVerifyFormula:
    def test_perfect_formula(self):
        cases = [
            {"weight": 80, "height": 2, "expected": 20.0},
            {"weight": 50, "height": 2, "expected": 12.5},
        ]
        m = verify_formula("weight / (height ** 2)", cases)
        assert m == {"n": 2, "correct": 2, "accuracy": 1.0, "errors": 0}

    def test_wrong_formula_scores_low(self):
        cases = [{"source": 24, "expected": 2.0}, {"source": 12, "expected": 1.0}]
        m = verify_formula("source / 6", cases)  # should be /12
        assert m["correct"] == 0 and m["accuracy"] == 0.0

    def test_arithmetic_error_counts_as_wrong(self):
        cases = [{"source": 0, "expected": 5.0}]
        m = verify_formula("1 / source", cases)  # div by zero
        assert m["errors"] == 1 and m["correct"] == 0


# ── C2 N2 arithmetic spec generation (residual -> LLM formula -> upgrade) ──


class TestArithSpecgen:
    def _residual_world(self, hf, *, src_units=None, cde_units=None):
        """A numeric edge N1 left as a needs_units UNIT residual, ready for the N2 LLM pass."""
        src = hf.field("agemo", "Age in months", units=src_units, data_type="continuous")
        cde = hf.field("AgeCDE", "Age", field_id="cde_age", units=cde_units, data_type="continuous")
        ed_a = hf.embedded_dict("CohortA", [src], sem_vecs=np.array([[1.0]]))
        ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
        rec = LeanBRecord(
            cluster_id="c0",
            verdict="refine",
            route="assigned",
            cde_id="AgeCDE",
            member_variable_names=["CohortA:agemo"],
        )
        embedded = [ed_a, ed_cde]
        cde_fields = dict(ed_cde.dictionary.fields)
        generate_unit_specs([rec], embedded, cde_fields)
        prompts = prepare_arith_specgen([rec], embedded, cde_fields)
        return rec, prompts

    def test_prompt_built_for_residual(self, hf):
        rec, prompts = self._residual_world(hf)
        assert rec.transforms[0].kind == TransformKind.UNIT and rec.transforms[0].needs_units
        assert len(prompts) == 1
        p = prompts[0]
        assert p.id == "leanb:arith:c0:0"
        assert p.context["source_variable"] == "CohortA:agemo"
        assert "source" in p.user_prompt.lower()

    def test_no_prompt_for_known_conversion(self, hf):
        # kg->lb is a deterministic N1 conversion -> no residual -> no arithmetic prompt
        rec, prompts = self._residual_world(hf, src_units="kg", cde_units="lb")
        assert rec.transforms[0].kind == TransformKind.UNIT and not rec.transforms[0].needs_units
        assert prompts == []

    def test_valid_formula_upgrades_residual(self, hf):
        rec, prompts = self._residual_world(hf)
        resp = {prompts[0].id: {"formula": "source / 12", "confidence": 0.9, "notes": "months to years"}}
        assemble_arith_specgen(prompts, resp, [rec])
        assert len(rec.transforms) == 1  # residual replaced, not appended
        t = rec.transforms[0]
        assert t.kind == TransformKind.ARITHMETIC
        assert t.formula == "source / 12" and t.inputs == ["source"]
        assert t.generated_by == "llm" and t.needs_review is True
        assert t.confidence == 0.6  # unverified arithmetic -> medium (review)
        assert eval_formula(t.formula, {"source": 24}) == APPROX(2.0)

    def test_null_formula_keeps_residual(self, hf):
        rec, prompts = self._residual_world(hf)
        assemble_arith_specgen(prompts, {prompts[0].id: {"formula": None}}, [rec])
        assert rec.transforms[0].kind == TransformKind.UNIT and rec.transforms[0].needs_units

    def test_unsafe_formula_keeps_residual(self, hf):
        rec, prompts = self._residual_world(hf)
        assemble_arith_specgen(prompts, {prompts[0].id: {"formula": "__import__('os')"}}, [rec])
        assert rec.transforms[0].kind == TransformKind.UNIT and rec.transforms[0].needs_units

    def test_multi_input_formula_flags_needs_data(self, hf):
        rec, prompts = self._residual_world(hf)
        resp = {prompts[0].id: {"formula": "source * 703 / (height ** 2)", "confidence": 0.8}}
        assemble_arith_specgen(prompts, resp, [rec])
        t = rec.transforms[0]
        assert t.kind == TransformKind.ARITHMETIC and t.needs_data is True
        assert t.confidence == 0.4  # needs_data -> low (review)

    def test_identity_formula_becomes_identity_spec(self, hf):
        # M8: a no-op "source" formula is not an arithmetic conversion -> deterministic IDENTITY, no review
        rec, prompts = self._residual_world(hf)
        assemble_arith_specgen(prompts, {prompts[0].id: {"formula": "source", "confidence": 0.9}}, [rec])
        assert len(rec.transforms) == 1  # residual replaced, not appended
        t = rec.transforms[0]
        assert t.kind == TransformKind.IDENTITY and t.generated_by == "rule"
        assert t.needs_review is False and t.confidence == 0.9  # IDENTITY band (cfg.high)

    def test_scaled_identity_formula_becomes_identity_spec(self, hf):
        # M8: "source * 1", "source + 0" etc. are also no-ops -> IDENTITY
        rec, prompts = self._residual_world(hf)
        assemble_arith_specgen(prompts, {prompts[0].id: {"formula": "source * 1", "confidence": 0.7}}, [rec])
        assert rec.transforms[0].kind == TransformKind.IDENTITY and rec.transforms[0].needs_review is False


# ── C1 EITL transform_review export ──


_PAIR_CONTRACT = (
    "source_text",
    "source_id",
    "source_dataset",
    "target_text",
    "target_id",
    "target_dataset",
    "pair_type",
)


class TestExportTransformReview:
    def _world(self, hf):
        src = hf.field("smoke", "Smoking", encoding="1=Yes|2=No", question_text="Do you currently smoke?")
        cde = hf.field(
            "SmokeCDE",
            "Current smoking status",
            field_id="TINY123",
            question_text="Do you currently smoke cigarettes?",
            encoding="1=Yes|0=No",
        )
        ed_a = hf.embedded_dict("CohortA", [src], sem_vecs=np.array([[1.0]]))
        ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
        rec = LeanBRecord(
            cluster_id="c0",
            group_id="c0#g0",
            verdict="refine",
            route="assigned",
            cde_id="SmokeCDE",
            cde_external_id="TINY123",
        )
        rec.transforms = [
            TransformSpec(
                source_variable="CohortA:smoke",
                target_cde_id="SmokeCDE",
                kind=TransformKind.CATEGORICAL,
                code_map={"1": "1", "2": "0"},
                coverage=1.0,
                confidence=0.95,
            ),
            TransformSpec(  # identity -> skipped (no-op)
                source_variable="CohortA:smoke",
                target_cde_id="SmokeCDE",
                kind=TransformKind.IDENTITY,
                code_map={"1": "1"},
            ),
        ]
        return [ed_a, ed_cde], dict(ed_cde.dictionary.fields), rec

    def test_emits_importable_pair_contract(self, hf, tmp_path):
        import csv as _csv

        from ddharmon.harmonization import export_transform_review

        embedded, cde_fields, rec = self._world(hf)
        path = tmp_path / "transform_review.csv"
        n = export_transform_review([rec], embedded, cde_fields, path)
        assert n == 1  # the IDENTITY recode is skipped
        with open(path, encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        r = rows[0]
        assert set(_PAIR_CONTRACT) <= set(r)  # EITL importer's required fields are all present
        assert r["source_id"] == "CohortA:smoke" and r["source_dataset"] == "CohortA"
        assert r["target_id"] == "TINY123" and r["target_dataset"] == "NIH_CDE"
        assert r["pair_type"] == "value_map/refine"
        # the proposed recode renders inline in target_text with labels resolved from both value sets
        assert "1=Yes→1=Yes" in r["target_text"] and "2=No→0=No" in r["target_text"]
        # source response values + question are visible in source_text
        assert "1=Yes" in r["source_text"] and "smoke" in r["source_text"].lower()

    def test_empty_records_is_zero(self, hf, tmp_path):
        from ddharmon.harmonization import export_transform_review

        embedded, cde_fields, _ = self._world(hf)
        assert export_transform_review([], embedded, cde_fields, tmp_path / "x.csv") == 0

    def _numeric_world(self, hf, spec):
        src = hf.field("weight", "Weight", units="kg", data_type="continuous", question_text="Body weight")
        cde = hf.field("WeightCDE", "Body weight", field_id="TINYW", units="lb", data_type="continuous")
        ed_a = hf.embedded_dict("CohortA", [src], sem_vecs=np.array([[1.0]]))
        ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
        rec = LeanBRecord(
            cluster_id="c0",
            verdict="refine",
            route="assigned",
            cde_id="WeightCDE",
            cde_external_id="TINYW",
        )
        rec.transforms = [spec]
        return [ed_a, ed_cde], dict(ed_cde.dictionary.fields), rec

    def _row(self, hf, tmp_path, spec):
        import csv as _csv

        from ddharmon.harmonization import export_transform_review

        embedded, cde_fields, rec = self._numeric_world(hf, spec)
        path = tmp_path / "tr.csv"
        n = export_transform_review([rec], embedded, cde_fields, path)
        with open(path, encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        return n, rows[0] if rows else None

    def test_exports_unit_conversion_row(self, hf, tmp_path):
        spec = TransformSpec(
            source_variable="CohortA:weight",
            target_cde_id="WeightCDE",
            kind=TransformKind.UNIT,
            factor=2.20462,
            offset=0.0,
            source_unit="kg",
            target_unit="lb",
            confidence=0.9,
        )
        n, r = self._row(hf, tmp_path, spec)
        assert n == 1
        assert set(_PAIR_CONTRACT) <= set(r)
        assert r["pair_type"] == "unit_convert/refine"
        assert r["transform_kind"] == "unit"
        assert "target = source * 2.20462" in r["target_text"]
        assert "kg -> lb" in r["target_text"]
        assert "kg" in r["source_text"]  # units shown in source_text for numeric edges

    def test_exports_arithmetic_row(self, hf, tmp_path):
        spec = TransformSpec(
            source_variable="CohortA:weight",
            target_cde_id="WeightCDE",
            kind=TransformKind.ARITHMETIC,
            formula="source / 12",
            inputs=["source"],
            generated_by="llm",
            needs_review=True,
            confidence=0.6,
        )
        n, r = self._row(hf, tmp_path, spec)
        assert n == 1 and r["pair_type"] == "arithmetic/refine"
        assert "target = source / 12" in r["target_text"]

    def test_exports_needs_units_residual(self, hf, tmp_path):
        spec = TransformSpec(
            source_variable="CohortA:weight",
            target_cde_id="WeightCDE",
            kind=TransformKind.UNIT,
            needs_units=True,
            needs_review=True,
            confidence=0.4,
        )
        n, r = self._row(hf, tmp_path, spec)
        assert n == 1 and r["pair_type"] == "unit_convert/refine"
        assert "none authored" in r["target_text"]
        assert "NEEDS REVIEW" in r["llm_reasoning"]


# ── Structural: wide->long repeating-measure routing ──


def _repeating_world(hf, n=6, encoding="1=Yes|2=No"):
    """An adopt/refine record whose members are numbered occurrence columns (a repeating measure)."""
    members = [
        hf.field(
            f"med{i}", f"Prescribed - Medication {i}", encoding=encoding, question_text=f"Prescribed - Medication {i}"
        )
        for i in range(1, n + 1)
    ]
    cde = hf.field("MedCDE", "Medication", field_id="cde_med", encoding="1=Yes|0=No", question_text="Medication taken")
    ed_a = hf.embedded_dict("CohortA", members, sem_vecs=np.ones((n, 1)))
    ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
    rec = LeanBRecord(
        cluster_id="c0",
        verdict="refine",
        route="assigned",
        cde_id="MedCDE",
        cde_external_id="cde_med",
        member_variable_names=[f"CohortA:med{i}" for i in range(1, n + 1)],
    )
    return [ed_a, ed_cde], dict(ed_cde.dictionary.fields), rec


class TestWideToLong:
    def test_detects_and_attaches_one_spec(self, hf):
        embedded, cde_fields, rec = _repeating_world(hf, n=6)
        generate_wide_to_long_specs([rec], embedded, cde_fields)
        w = [t for t in rec.transforms if t.kind == TransformKind.WIDE_TO_LONG]
        assert len(w) == 1
        t = w[0]
        assert t.generated_by == "rule" and t.needs_review is True and t.method == "wide_to_long"
        assert t.params["n_occurrences"] == 6 and t.params["int_range"] == [1, 6]
        assert sorted(t.inputs) == [f"CohortA:med{i}" for i in range(1, 7)]

    def test_claims_record_so_n1_and_c1_skip(self, hf):
        embedded, cde_fields, rec = _repeating_world(hf, n=6)
        generate_wide_to_long_specs([rec], embedded, cde_fields)
        assert prepare_specgen([rec], embedded, cde_fields) == []  # C1: no per-edge prompts
        before = len(rec.transforms)
        generate_unit_specs([rec], embedded, cde_fields)
        assert len(rec.transforms) == before  # N1: record skipped, nothing added

    def test_non_repeating_record_untouched(self, hf):
        # distinct concepts -> not a positional enumeration -> no wide->long spec; C1 still runs
        pairs = [
            ("sbp", "Systolic BP"),
            ("dbp", "Diastolic BP"),
            ("hr", "Heart rate"),
            ("temp", "Body temp"),
            ("rr", "Resp rate"),
        ]
        members = [hf.field(v, q, encoding="1=Yes|2=No", question_text=q) for v, q in pairs]
        cde = hf.field("CDE", "Concept", field_id="tiny", encoding="1=Yes|0=No")
        ed = hf.embedded_dict("CohortA", members, sem_vecs=np.ones((len(members), 1)))
        ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
        cde_fields = dict(ed_cde.dictionary.fields)
        rec = LeanBRecord(
            cluster_id="c0",
            verdict="refine",
            route="assigned",
            cde_id="CDE",
            member_variable_names=[f"CohortA:{v}" for v, _ in pairs],
        )
        generate_wide_to_long_specs([rec], [ed, ed_cde], cde_fields)
        assert not any(t.kind == TransformKind.WIDE_TO_LONG for t in rec.transforms)
        assert len(prepare_specgen([rec], [ed, ed_cde], cde_fields)) >= 1  # C1 still emits

    def test_eitl_exports_one_wide_to_long_row(self, hf, tmp_path):
        import csv as _csv

        from ddharmon.harmonization import export_transform_review

        embedded, cde_fields, rec = _repeating_world(hf, n=6)
        generate_wide_to_long_specs([rec], embedded, cde_fields)
        path = tmp_path / "tr.csv"
        assert export_transform_review([rec], embedded, cde_fields, path) == 1  # ONE row for the whole group
        with open(path, encoding="utf-8") as f:
            r = list(_csv.DictReader(f))[0]
        assert r["pair_type"] == "wide_to_long/refine" and r["transform_kind"] == "wide_to_long"
        assert "repeating measure" in r["llm_reasoning"].lower()
        assert "reshape" in r["target_text"].lower()

    # ── M6: detect on the variable-NAME signature + exclude aggregates ──

    def _name_numbered_rec(self, hf, names, label):
        """A record whose members carry ``names`` (occurrence in the name) under a UNIFORM ``label``."""
        members = [hf.field(n, label, encoding="1=Yes|2=No", question_text=label) for n in names]
        cde = hf.field("HdlCDE", "HDL", field_id="cde_hdl", encoding="1=Yes|0=No", question_text=label)
        ed = hf.embedded_dict("CohortA", members, sem_vecs=np.ones((len(names), 1)))
        ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
        rec = LeanBRecord(
            cluster_id="c0",
            verdict="refine",
            route="assigned",
            cde_id="HdlCDE",
            member_variable_names=[f"CohortA:{n}" for n in names],
        )
        return [ed, ed_cde], dict(ed_cde.dictionary.fields), rec

    def test_detects_on_variable_name_when_label_uniform(self, hf):
        # M6: the occurrence digit lives ONLY in the name (hdl1..hdl6); question_text is uniform, so the
        # label-signature path can't see it. Detect on the variable-name signature instead.
        embedded, cde_fields, rec = self._name_numbered_rec(hf, [f"hdl{i}" for i in range(1, 7)], "HDL cholesterol")
        generate_wide_to_long_specs([rec], embedded, cde_fields)
        w = [t for t in rec.transforms if t.kind == TransformKind.WIDE_TO_LONG]
        assert len(w) == 1
        assert w[0].params["n_occurrences"] == 6 and w[0].params["int_range"] == [1, 6]
        assert sorted(w[0].inputs) == [f"CohortA:hdl{i}" for i in range(1, 7)]

    def test_excludes_aggregate_columns_from_family(self, hf):
        # M6: an _AVG/_MEAN column is a summary over the measure, not an occurrence -> excluded from the family
        names = [f"hdl{i}" for i in range(1, 7)] + ["hdl_avg", "hdl_mean"]
        embedded, cde_fields, rec = self._name_numbered_rec(hf, names, "HDL cholesterol")
        generate_wide_to_long_specs([rec], embedded, cde_fields)
        w = [t for t in rec.transforms if t.kind == TransformKind.WIDE_TO_LONG]
        assert len(w) == 1
        assert "CohortA:hdl_avg" not in w[0].inputs and "CohortA:hdl_mean" not in w[0].inputs
        assert sorted(w[0].inputs) == [f"CohortA:hdl{i}" for i in range(1, 7)]
        assert w[0].params["n_occurrences"] == 6

    def test_name_path_skipped_when_labels_distinguish_members(self, hf):
        # sequential NAMES (item1..item6) but DISTINCT question_texts = a qualifier matrix, NOT a repeating
        # measure -> the name-signature path must NOT collapse it (guards against false positives).
        qs = ["Chest pain", "Shortness of breath", "Palpitations", "Dizziness", "Fatigue", "Ankle swelling"]
        members = [hf.field(f"item{i}", qs[i - 1], encoding="1=Yes|2=No", question_text=qs[i - 1]) for i in range(1, 7)]
        cde = hf.field("CDE", "Concept", field_id="tiny", encoding="1=Yes|0=No")
        ed = hf.embedded_dict("CohortA", members, sem_vecs=np.ones((6, 1)))
        ed_cde = hf.embedded_dict("NIH_CDE", [cde], sem_vecs=np.array([[1.0]]))
        rec = LeanBRecord(
            cluster_id="c0",
            verdict="refine",
            route="assigned",
            cde_id="CDE",
            member_variable_names=[f"CohortA:item{i}" for i in range(1, 7)],
        )
        generate_wide_to_long_specs([rec], [ed, ed_cde], dict(ed_cde.dictionary.fields))
        assert not any(t.kind == TransformKind.WIDE_TO_LONG for t in rec.transforms)


# ── M3: NONE-fraction coherence gate ──


class TestCoherenceGate:
    def _rec(self, kinds, *, verdict="adopt"):
        return LeanBRecord(
            cluster_id="c",
            verdict=verdict,
            route="assigned",
            cde_id="X",
            transforms=[
                TransformSpec(source_variable=f"C:v{i}", target_cde_id="X", kind=k) for i, k in enumerate(kinds)
            ],
        )

    def test_mostly_none_adopt_is_flagged_and_demoted(self):
        tk = TransformKind
        rec = self._rec([tk.NONE, tk.NONE, tk.NONE, tk.CATEGORICAL])  # 3/4 = 0.75 >= tau
        apply_coherence_gate([rec])
        assert rec.coherence_gap is True
        assert rec.verdict == "refine" and rec.route == "assigned"  # demoted, route unchanged
        assert "coherence gate" in rec.rationale

    def test_coherent_record_untouched(self):
        rec = self._rec([TransformKind.CATEGORICAL] * 4)
        apply_coherence_gate([rec])
        assert rec.coherence_gap is False and rec.verdict == "adopt"

    def test_below_min_edges_not_gated(self):
        rec = self._rec([TransformKind.NONE, TransformKind.NONE])  # 100% NONE but only 2 < min_coded_edges
        apply_coherence_gate([rec])
        assert rec.coherence_gap is False and rec.verdict == "adopt"

    def test_refine_flagged_but_verdict_kept(self):
        rec = self._rec([TransformKind.NONE] * 3, verdict="refine")
        apply_coherence_gate([rec])
        assert rec.coherence_gap is True and rec.verdict == "refine"  # never hardened past refine

    def test_demote_false_flags_without_demoting(self):
        rec = self._rec([TransformKind.NONE] * 3)
        apply_coherence_gate([rec], demote=False)
        assert rec.coherence_gap is True and rec.verdict == "adopt"

    def test_numeric_and_structural_edges_are_not_counted(self):
        tk = TransformKind
        rec = self._rec([tk.UNIT, tk.ARITHMETIC, tk.WIDE_TO_LONG, tk.NONE])  # only 1 coded edge (NONE) < min
        apply_coherence_gate([rec])
        assert rec.coherence_gap is False

    def test_novel_record_skipped(self):
        rec = self._rec([TransformKind.NONE] * 5, verdict="novel")
        apply_coherence_gate([rec])
        assert rec.coherence_gap is False and rec.verdict == "novel"


# ── M7: concept-match gate ──


class TestConceptGate:
    def _rec(self, *, cde_id="X", verdict="adopt", members=("C:v1",), concept="Smoking status"):
        return LeanBRecord(
            cluster_id="c",
            group_id="c#g0",
            verdict=verdict,
            route="assigned" if verdict in ("adopt", "refine") else "gencde_residual",
            cde_id=cde_id,
            concept=concept,
            ideal_cde="ideal: current smoking status",
            member_variable_names=list(members),
            transforms=[
                TransformSpec(source_variable=members[0], target_cde_id=cde_id or "X", kind=TransformKind.CATEGORICAL)
            ],
        )

    def _cde_fields(self):
        return {"X": Field(variable_name="X", question_text="Do you currently smoke?", description="smoking")}

    def test_prepare_one_prompt_per_assigned_record_skips_novel_and_no_cde(self):
        recs = [
            self._rec(verdict="adopt"),
            self._rec(verdict="refine"),
            self._rec(verdict="novel"),
            self._rec(cde_id=None),
        ]
        prompts = prepare_concept_gate(recs, [], self._cde_fields())
        assert len(prompts) == 2  # adopt + refine with a cde only
        assert "Do you currently smoke" in prompts[0].user_prompt  # CDE concept shown
        assert "Smoking status" in prompts[0].user_prompt  # source concept shown

    def test_mismatch_flags_and_sets_needs_review_without_flipping_verdict(self):
        rec = self._rec(verdict="adopt")
        prompts = prepare_concept_gate([rec], [], self._cde_fields())
        resp = {prompts[0].id: {"match": False, "reason": "different concept"}}
        assemble_concept_gate(prompts, resp, [rec])
        assert rec.concept_mismatch is True
        assert rec.verdict == "adopt"  # gate never flips the verdict
        assert all(t.needs_review for t in rec.transforms)
        assert "concept-gate" in rec.rationale

    def test_match_leaves_record_untouched(self):
        rec = self._rec()
        prompts = prepare_concept_gate([rec], [], self._cde_fields())
        assemble_concept_gate(prompts, {prompts[0].id: {"match": True}}, [rec])
        assert rec.concept_mismatch is False and not any(t.needs_review for t in rec.transforms)

    def test_string_false_is_treated_as_mismatch(self):
        rec = self._rec()
        prompts = prepare_concept_gate([rec], [], self._cde_fields())
        assemble_concept_gate(prompts, {prompts[0].id: {"match": "false"}}, [rec])
        assert rec.concept_mismatch is True

    def test_unparseable_response_does_not_flag(self):
        rec = self._rec()
        prompts = prepare_concept_gate([rec], [], self._cde_fields())
        assemble_concept_gate(prompts, {prompts[0].id: "garbage, not json"}, [rec])
        assert rec.concept_mismatch is False


# ── M12: GenCDE tail spec-gen — member -> synthesized-GenCDE recodes for novel records ──


def _gencde_world(hf, *, source_encoding="1=Yes|2=No", gencde_pv=(("1", "Yes"), ("0", "No"))):
    """A 1-member ``novel`` record that synthesized a categorical GenCDE, with prepared tail spec-gen prompts."""
    src = hf.field("smoke", "Do you currently smoke", encoding=source_encoding, data_type="categorical")
    ed_a = hf.embedded_dict("CohortA", [src], sem_vecs=np.array([[1.0]]))
    rec = LeanBRecord(
        cluster_id="c0",
        verdict="novel",
        route=ROUTE_RESIDUAL,
        group_id="c0#g0",
        member_variable_names=["CohortA:smoke"],
    )
    pv = [ResponseOption(code=c, label=lb) for c, lb in gencde_pv] if gencde_pv else []
    rec.gencde = GenCDE(gencde_id="GENCDE:c0#g0", preferred_name="ever_smoked", permissible_values=pv)
    sg = prepare_gencde_specgen([rec], [ed_a])
    return sg, rec


class TestPrepareGenCDESpecgen:
    def test_builds_prompt_targeting_the_gencde(self, hf):
        sg, _ = _gencde_world(hf)
        assert len(sg) == 1
        p = sg[0]
        assert p.id.startswith("leanb:gencde_specgen:")  # own content-addressed namespace (not leanb:specgen:)
        # both the source encoding AND the synthesized GenCDE domain are visible to the recode model
        assert "1=Yes|2=No" in p.user_prompt and "1=Yes|0=No" in p.user_prompt
        assert "GENCDE:c0#g0" in p.user_prompt
        assert p.context["gencde_id"] == "GENCDE:c0#g0"
        assert p.context["cde_value_set"] == "1=Yes|0=No"  # the key _compute_recode reads
        assert p.context["edges"] == [("c0#g0", "CohortA:smoke")]  # fanned back by (group_id, source var)

    def test_skips_record_without_a_gencde(self, hf):
        src = hf.field("smoke", "Do you currently smoke", encoding="1=Yes|2=No", data_type="categorical")
        ed_a = hf.embedded_dict("CohortA", [src], sem_vecs=np.array([[1.0]]))
        rec = LeanBRecord(
            cluster_id="c0", verdict="novel", route=ROUTE_RESIDUAL, member_variable_names=["CohortA:smoke"]
        )
        assert prepare_gencde_specgen([rec], [ed_a]) == []  # no gencde attached -> nothing to recode into

    def test_skips_numeric_gencde_no_permissible_values(self, hf):
        sg, _ = _gencde_world(hf, gencde_pv=None)  # numeric GenCDE (units/bounds) -> not a categorical recode
        assert sg == []

    def test_skips_when_source_has_no_codes(self, hf):
        sg, _ = _gencde_world(hf, source_encoding=None)
        assert sg == []


class TestAssembleGenCDESpecgen:
    def test_attaches_recode_with_gencde_target(self, hf):
        sg, rec = _gencde_world(hf)
        assemble_gencde_specgen(sg, {sg[0].id: {"code_map": {"1": "1", "2": "0"}, "confidence": 0.9}}, [rec])
        assert len(rec.transforms) == 1
        t = rec.transforms[0]
        assert t.target_cde_id == "GENCDE:c0#g0"  # the tail's synthesized target, not a catalog cde_id
        assert t.source_variable == "CohortA:smoke"
        assert t.kind == TransformKind.CATEGORICAL
        assert t.code_map == {"1": "1", "2": "0"}
        assert t.coverage == 1.0 and t.needs_review is False

    def test_hallucinated_target_codes_dropped(self, hf):
        sg, rec = _gencde_world(hf)  # GenCDE domain codes are {1, 0}
        assemble_gencde_specgen(sg, {sg[0].id: {"code_map": {"1": "1", "2": "9"}, "confidence": 0.9}}, [rec])
        t = rec.transforms[0]
        assert "2" not in t.code_map  # "9" is not in the GenCDE value set -> dropped, "2" left unmapped
        assert "2" in t.unmapped_source_codes
        assert t.coverage == 0.5 and t.needs_review is True  # partial coverage flags review
