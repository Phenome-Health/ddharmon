"""Per-model request rules: one request shape per model, shared by the Batch API and the synchronous path.

Why this file exists (2026-10-14): a synchronous run failed every split call with ``TypeError: Messages.create() got
an unexpected keyword argument 'temperature'`` — the Anthropic SDK 1.x signature no longer has sampling parameters —
and the models users now want (Claude Sonnet 5.5, Claude Haiku 5.5) reject the request ddharmon sent in other ways:
Sonnet 5.5 refuses a forced ``tool_choice`` and a non-default ``temperature``, Haiku 5.5 refuses any ``temperature``
but ``1``, and both think by default and answer with a ``thinking`` block before the text. Every test that used to
cover these paths mocked the client with ``MagicMock``, which accepts any keyword — so the TypeError could never
fail one. The signature tests here bind the request against the INSTALLED SDK's real ``Messages.create``; run this
file under both SDK generations (0.x and 1.x).
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest

from ddharmon.llm.models import DEFAULT_MODEL, MODELS, model_info
from ddharmon.llm.params import message_params, structured_output_schema

SONNET_46 = "claude-sonnet-4-6"
SONNET_55 = "claude-sonnet-5-5"
HAIKU_55 = "claude-haiku-5-5"

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "member_ids": {"type": "array", "items": {"type": "string"}},
                    "cde_id": {"type": ["string", "null"], "description": "null for novel"},
                    "verdict": {"type": "string", "enum": ["adopt", "refine", "novel"]},
                },
                "required": ["member_ids", "verdict"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
}


def _params(model: str, **kw: Any) -> dict:
    return message_params(model=model, system="SYS", user_prompt="USER", max_tokens=2048, temperature=0.0, **kw)


# ── the registry carries the user-selectable models ──────────────────────────────────────────────────


class TestRegistry:
    def test_new_models_are_selectable_but_not_yet_benchmarked(self) -> None:
        for mid in (SONNET_55, HAIKU_55):
            info = model_info(mid)
            assert info is not None, f"{mid} missing from the registry"
            assert info.selectable, f"{mid} must be selectable"
            assert not info.validated, f"{mid} is not benchmark-validated yet"

    def test_the_default_stays_validated_and_selectable(self) -> None:
        info = model_info(DEFAULT_MODEL)
        assert info is not None and info.validated and info.selectable

    def test_every_selectable_model_has_a_price(self) -> None:
        for m in MODELS:
            if m.selectable:
                assert m.input_per_mtok is not None and m.output_per_mtok is not None, m.id

    def test_token_ratio_marks_the_newer_tokenizer(self) -> None:
        # The 5.x models count the same text as ~30% more tokens than the validated default (platform.claude.com,
        # Haiku 5.5 migration guide); a cost estimate calibrated on the default scales by it.
        assert model_info(SONNET_46).token_ratio == 1.0
        assert model_info(SONNET_55).token_ratio == pytest.approx(1.3)
        assert model_info(HAIKU_55).token_ratio == pytest.approx(1.3)

    def test_published_prices(self) -> None:
        # platform.claude.com pricing, checked 2026-10-14 (Haiku 5.5: the <=100K-token prompt tier).
        assert (model_info(SONNET_46).input_per_mtok, model_info(SONNET_46).output_per_mtok) == (3.0, 15.0)
        assert (model_info(SONNET_55).input_per_mtok, model_info(SONNET_55).output_per_mtok) == (2.0, 10.0)
        assert (model_info(HAIKU_55).input_per_mtok, model_info(HAIKU_55).output_per_mtok) == (0.10, 0.50)

    def test_models_that_think_carry_headroom_for_it(self) -> None:
        # 2026-10-15: a Claude Haiku 5.5 run's four largest split prompts came back with no answer. Replayed, each
        # stopped on `max_tokens` at the 2,048 budget with only a thinking block — thinking is billed as output and
        # counts against `max_tokens`, so a budget sized for the answer alone does not survive it. The validated
        # default does not think, so its budget must not move.
        assert model_info(SONNET_46).thinking_headroom == 0
        for mid in (SONNET_55, HAIKU_55):
            assert model_info(mid).thinking_headroom >= 4096, mid


# ── one request shape per model ─────────────────────────────────────────────────────────────────────


class TestMessageParams:
    def test_sonnet_46_is_unchanged_greedy_and_forced_tool(self) -> None:
        # The validated model's request must not move: same keys and values the pipeline was benchmarked with.
        p = _params(SONNET_46, tool_schema=TOOL_SCHEMA, tool_name="emit_groups")
        assert p == {
            "model": SONNET_46,
            "max_tokens": 2048,
            "temperature": 0.0,
            "system": "SYS",
            "messages": [{"role": "user", "content": "USER"}],
            "tools": [
                {
                    "name": "emit_groups",
                    "description": "Return the result as structured input conforming to the schema.",
                    "input_schema": TOOL_SCHEMA,
                }
            ],
            "tool_choice": {"type": "tool", "name": "emit_groups"},
        }

    def test_sonnet_55_uses_structured_outputs_no_temperature_thinking_off(self) -> None:
        p = _params(SONNET_55, tool_schema=TOOL_SCHEMA, tool_name="emit_groups")
        assert "temperature" not in p
        assert "tool_choice" not in p and "tools" not in p, "Sonnet 5.5 rejects a forced tool call"
        assert p["output_config"]["format"] == {"type": "json_schema", "schema": structured_output_schema(TOOL_SCHEMA)}
        assert p["thinking"] == {"type": "between_tools"}

    def test_sonnet_55_text_prompt(self) -> None:
        p = _params(SONNET_55)
        assert "temperature" not in p and "output_config" not in p
        assert p["thinking"] == {"type": "between_tools"}

    def test_haiku_55_keeps_the_forced_tool_drops_temperature_low_effort(self) -> None:
        p = _params(HAIKU_55, tool_schema=TOOL_SCHEMA, tool_name="emit_groups")
        assert "temperature" not in p
        assert p["tool_choice"] == {"type": "tool", "name": "emit_groups"}
        assert p["output_config"] == {"effort": "low"}
        assert "thinking" not in p

    def test_a_thinking_model_gets_its_headroom_on_top_of_the_answer_budget(self) -> None:
        # The caller's budget is what the ANSWER needs; the thinking model's headroom rides on top, on every
        # transport (submit_batch and complete_request both build their request here). Sonnet 4.6's exact request
        # is pinned above.
        for mid in (SONNET_55, HAIKU_55):
            assert _params(mid)["max_tokens"] == 2048 + model_info(mid).thinking_headroom, mid

    def test_an_unknown_model_keeps_the_legacy_shape(self) -> None:
        # Back-compat for API callers passing ids the registry does not list.
        p = _params("claude-some-future-model", tool_schema=TOOL_SCHEMA)
        assert p["temperature"] == 0.0 and p["tool_choice"] == {"type": "tool", "name": "emit"}


class TestStructuredOutputSchema:
    def test_every_object_closes_and_type_lists_become_any_of(self) -> None:
        s = structured_output_schema(TOOL_SCHEMA)
        assert s["additionalProperties"] is False, "the top-level object did not declare additionalProperties"
        item = s["properties"]["groups"]["items"]
        assert item["additionalProperties"] is False
        cde = item["properties"]["cde_id"]
        assert "type" not in cde and cde["anyOf"] == [{"type": "string"}, {"type": "null"}]
        assert cde["description"] == "null for novel"
        assert s["properties"]["groups"]["minItems"] == 1  # 0 and 1 are the supported values
        assert TOOL_SCHEMA["properties"]["groups"]["items"]["properties"]["cde_id"]["type"] == [
            "string",
            "null",
        ], "the caller's schema was mutated"

    def test_unsupported_constraints_move_into_the_description(self) -> None:
        s = structured_output_schema({"type": "array", "minItems": 3, "items": {"type": "string", "maxLength": 9}})
        assert "minItems" not in s and "minItems: 3" in s["description"]
        assert "maxLength" not in s["items"] and "maxLength: 9" in s["items"]["description"]


# ── the request binds against the installed SDK's real signature ────────────────────────────────────


def _create_signature() -> inspect.Signature:
    from anthropic.resources.messages import Messages

    return inspect.signature(Messages.create)


class _StrictMessages:
    """A ``messages`` resource whose ``create`` HAS the installed SDK's signature and binds every call against it."""

    def __init__(self, *responses: Any) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses)

    def create(self, **kwargs: Any) -> Any:
        _create_signature().bind(None, **kwargs)  # TypeError here = the SDK would have raised it
        self.calls.append(kwargs)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


# The fake presents the REAL signature, so code that inspects it (to route what the SDK lacks into extra_body)
# sees exactly what it would see on the installed SDK.
_StrictMessages.create.__signature__ = _create_signature()  # type: ignore[attr-defined]


def _client_with(*responses: Any):
    from ddharmon.llm.anthropic_client import AnthropicClient

    client = AnthropicClient.__new__(AnthropicClient)
    client._client = SimpleNamespace(messages=_StrictMessages(*responses))
    client._model_name = DEFAULT_MODEL
    client._max_tokens = 2048
    client._use_structured = None
    client.usage_log = []
    return client


def _msg(*blocks: Any, stop_reason: str = "end_turn", stop_details: Any = None) -> Any:
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop_reason, stop_details=stop_details, usage=usage, model="m"
    )


def _text(t: str) -> Any:
    return SimpleNamespace(type="text", text=t)


def _thinking() -> Any:
    return SimpleNamespace(type="thinking", thinking="", signature="sig")


def _tool(inp: dict) -> Any:
    return SimpleNamespace(type="tool_use", input=inp, name="emit", id="t1")


class TestCompleteRequestOnTheRealSignature:
    @pytest.mark.parametrize("model", [SONNET_46, SONNET_55, HAIKU_55])
    def test_no_keyword_the_installed_sdk_rejects(self, model: str) -> None:
        answer = _tool({"ok": True}) if model != SONNET_55 else _text(json.dumps({"ok": True}))
        client = _client_with(_msg(_thinking(), answer), _msg(_thinking(), _text("plain")))
        client.complete_request(
            "U",
            system="S",
            max_tokens=100,
            temperature=0.0,
            model=model,
            tool_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        )
        client.complete_request("U", system="S", max_tokens=100, temperature=0.0, model=model)

    def test_sonnet_46_still_sends_temperature_zero(self) -> None:
        client = _client_with(_msg(_text("{}")))
        client.complete_request("U", system="S", max_tokens=100, temperature=0.0, model=SONNET_46)
        sent = client._client.messages.calls[0]
        temperature = sent.get("temperature", (sent.get("extra_body") or {}).get("temperature"))
        assert temperature == 0.0, "the validated model's greedy decoding was dropped"

    def test_sonnet_55_structured_output_returns_the_parsed_dict(self) -> None:
        client = _client_with(_msg(_thinking(), _text('{"groups": []}')))
        out = client.complete_request(
            "U", system="S", max_tokens=100, temperature=0.0, model=SONNET_55, tool_schema=TOOL_SCHEMA
        )
        assert out == {"groups": []}

    def test_a_thinking_block_first_does_not_hide_the_text(self) -> None:
        client = _client_with(_msg(_thinking(), _text("answer")))
        assert client.complete_request("U", system="S", max_tokens=100, temperature=0.0, model=HAIKU_55) == "answer"

    def test_complete_sends_the_headroom_too(self) -> None:
        # The plain-text path (analysis ideas, labels) builds its own request; it must not lose the headroom.
        client = _client_with(_msg(_thinking(), _text("ok")))
        client._model_name = HAIKU_55
        assert client.complete("U", max_tokens=512) == "ok"
        assert client._client.messages.calls[0]["max_tokens"] == 512 + model_info(HAIKU_55).thinking_headroom

    @pytest.mark.parametrize("model", [HAIKU_55, SONNET_46])
    def test_an_answer_cut_off_by_the_output_limit_says_so(self, model: str) -> None:
        # What the Haiku 5.5 run saw as "Anthropic response contained no text content": the budget ran out before
        # the answer began. Name the cause, so the next one is not a replay away from diagnosis.
        client = _client_with(_msg(_thinking(), stop_reason="max_tokens"))
        with pytest.raises(ValueError, match="output limit"):
            client.complete_request("U", system="S", max_tokens=100, temperature=0.0, model=model)
        tool = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        with pytest.raises(ValueError, match="output limit"):
            client.complete_request("U", system="S", max_tokens=100, temperature=0.0, model=model, tool_schema=tool)

    def test_a_refusal_raises_naming_its_category(self) -> None:
        from ddharmon.llm.params import RefusalError

        client = _client_with(
            _msg(stop_reason="refusal", stop_details=SimpleNamespace(category="bio", explanation="x"))
        )
        with pytest.raises(RefusalError, match="bio"):
            client.complete_request("U", system="S", max_tokens=100, temperature=0.0, model=SONNET_55)


# ── the Batch API path: same request, answers read by block type ────────────────────────────────────


class TestBatchPath:
    def test_submit_batch_sends_the_shared_request(self, tmp_path, monkeypatch) -> None:
        import ddharmon.llm.batch as batch_mod

        prompts = tmp_path / "p.jsonl"
        prompts.write_text(
            json.dumps(
                {
                    "id": "a",
                    "system_prompt": "S",
                    "user_prompt": "U",
                    "model_tag": SONNET_55,
                    "tool_schema": TOOL_SCHEMA,
                    "tool_name": "emit_groups",
                }
            )
            + "\n"
        )
        sent: list = []

        class _Batches:
            def create(self, requests):
                sent.extend(requests)
                return SimpleNamespace(id="b1", created_at=None, expires_at=None)

        fake = SimpleNamespace(Anthropic=lambda **k: SimpleNamespace(messages=SimpleNamespace(batches=_Batches())))
        monkeypatch.setitem(__import__("sys").modules, "anthropic", fake)
        batch_mod.submit_batch(prompts)
        params = sent[0]["params"]
        expected = message_params(
            model=SONNET_55,
            system="S",
            user_prompt="U",
            max_tokens=2048,
            temperature=0.0,
            tool_schema=TOOL_SCHEMA,
            tool_name="emit_groups",
        )
        assert params == expected

    def test_retrieve_reads_text_after_a_thinking_block_and_skips_a_refusal(self, tmp_path, monkeypatch) -> None:
        import ddharmon.llm.batch as batch_mod

        def result(cid, message):
            return SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="succeeded", message=message))

        results = [
            result("ok", _msg(_thinking(), _text('{"x": 1}'))),
            result(
                "refused", _msg(stop_reason="refusal", stop_details=SimpleNamespace(category="bio", explanation="x"))
            ),
        ]
        batches = SimpleNamespace(
            retrieve=lambda bid: SimpleNamespace(
                processing_status="ended",
                request_counts=SimpleNamespace(succeeded=2, processing=0, errored=0, expired=0, canceled=0),
            ),
            results=lambda bid: iter(results),
        )
        fake = SimpleNamespace(Anthropic=lambda **k: SimpleNamespace(messages=SimpleNamespace(batches=batches)))
        monkeypatch.setitem(__import__("sys").modules, "anthropic", fake)
        out = tmp_path / "r.jsonl"
        written = batch_mod.retrieve_batch("b1", out)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["id"] for r in rows] == ["ok"], "a refusal must stay unanswered so the stage retries and names it"
        assert rows[0]["response"] == {"x": 1}
        assert written == 1


# ── cost: the registry's price wins over a stale price map ──────────────────────────────────────────


class TestRegistryPrices:
    def test_new_models_price_from_the_registry(self) -> None:
        from ddharmon.llm.cost import price_usage

        assert price_usage(HAIKU_55, 1_000_000, 1_000_000) == pytest.approx(0.60)
        assert price_usage(SONNET_55, 1_000_000, 1_000_000) == pytest.approx(12.0)
        assert price_usage(SONNET_55, 1_000_000, 1_000_000, batch=True) == pytest.approx(6.0)
