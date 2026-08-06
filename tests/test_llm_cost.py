"""Unit tests for realized LLM cost accounting (ddharmon.llm.cost) + per-client usage capture.

Cost is priced against LiteLLM's local model→price map (``cost_per_token`` — no network), so these run
offline. The map is pinned by the installed litellm version; the Sonnet rate ($3/$15 per M tokens) is stable
and asserted here as the accuracy anchor.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# Sonnet standard rate from LiteLLM's map: $3 / 1M input, $15 / 1M output.
_SONNET = "claude-sonnet-4-6"
_ONE_M_IN_ONE_M_OUT = 3.0 + 15.0  # pricing 1M input + 1M output tokens


class TestPriceUsage:
    def test_prices_tokens_against_litellm_map(self) -> None:
        from ddharmon.llm.cost import price_usage

        cost = price_usage(_SONNET, 1_000_000, 1_000_000)
        assert abs(cost - _ONE_M_IN_ONE_M_OUT) < 1e-6

    def test_batch_gets_half_price(self) -> None:
        from ddharmon.llm.cost import price_usage

        full = price_usage(_SONNET, 1_000_000, 1_000_000)
        batch = price_usage(_SONNET, 1_000_000, 1_000_000, batch=True)
        assert abs(batch - full * 0.5) < 1e-6

    def test_unknown_model_reports_zero_not_a_guess(self) -> None:
        from ddharmon.llm.cost import price_usage

        assert price_usage("no-such-model-xyz", 1000, 1000) == 0.0

    def test_empty_model_reports_zero(self) -> None:
        from ddharmon.llm.cost import price_usage

        assert price_usage(None, 1000, 1000) == 0.0

    def test_prefers_provider_reported_response_cost(self) -> None:
        from ddharmon.llm.cost import price_usage

        # response_cost short-circuits the map (used for litellm-routed calls); batch halves it.
        assert price_usage("any", 0, 0, response_cost=0.02) == 0.02
        assert price_usage("any", 0, 0, response_cost=0.02, batch=True) == 0.01


class TestPriceUsageFallback:
    """The built-in rate table prices accurately when the optional litellm package isn't installed."""

    def test_built_in_table_prices_when_litellm_absent(self, monkeypatch) -> None:
        # Make `import litellm` inside price_usage raise ImportError, simulating a litellm-free install.
        monkeypatch.setitem(sys.modules, "litellm", None)
        from ddharmon.llm.cost import price_usage

        # Sonnet fallback: $3/$15 per M -> 1M in + 1M out = $18; batch halves to $9.
        assert abs(price_usage(_SONNET, 1_000_000, 1_000_000) - 18.0) < 1e-6
        assert abs(price_usage(_SONNET, 1_000_000, 1_000_000, batch=True) - 9.0) < 1e-6
        # A model neither litellm nor the table knows still reports $0 (never guesses).
        assert price_usage("no-such-model-xyz", 1000, 1000) == 0.0

    def test_fallback_longest_match_disambiguates_gpt4o_mini(self) -> None:
        from ddharmon.llm.cost import _fallback_price

        # "gpt-4o-mini" contains "gpt-4o" — longest-key match must pick the mini rate, not gpt-4o's.
        mini = _fallback_price("gpt-4o-mini", 1_000_000, 0)  # $0.15/M input
        full = _fallback_price("gpt-4o", 1_000_000, 0)  # $2.50/M input
        assert mini is not None and full is not None
        assert abs(mini - 0.15) < 1e-6 and abs(full - 2.5) < 1e-6

    def test_dated_sonnet_id_resolves_by_substring(self) -> None:
        from ddharmon.llm.cost import _fallback_price

        assert abs((_fallback_price("claude-sonnet-4-20250514", 1_000_000, 0) or 0) - 3.0) < 1e-6


class TestCostLedger:
    def test_accumulates_per_stage_and_total(self) -> None:
        from ddharmon.llm.cost import CostLedger, TokenUsage

        ledger = CostLedger()
        ledger.add("splitting", [TokenUsage(_SONNET, 1_000_000, 1_000_000)], batch=True)  # 18 * 0.5 = 9
        ledger.add("specs", [TokenUsage(_SONNET, 1_000_000, 0)], batch=False)  # 3
        assert abs(ledger.total_usd - 12.0) < 1e-6
        assert ledger.total_tokens == {"input": 2_000_000, "output": 1_000_000}

        d = ledger.to_dict()
        assert d["perStage"]["splitting"]["calls"] == 1
        assert abs(d["perStage"]["splitting"]["usd"] - 9.0) < 1e-6
        assert d["tokens"]["input"] == 2_000_000

    def test_multiple_calls_same_stage_sum(self) -> None:
        from ddharmon.llm.cost import CostLedger, TokenUsage

        ledger = CostLedger()
        ledger.add("assigning", [TokenUsage(_SONNET, 1000, 100), TokenUsage(_SONNET, 2000, 200)], batch=False)
        assert ledger.to_dict()["perStage"]["assigning"]["calls"] == 2
        assert ledger.total_tokens == {"input": 3000, "output": 300}


class TestAnthropicUsageCapture:
    def _client(self):
        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
            from ddharmon.llm.anthropic_client import AnthropicClient

            return AnthropicClient(model_name=_SONNET)

    def test_record_usage_appends_and_drain_clears(self) -> None:
        client = self._client()
        resp = MagicMock()
        resp.usage.input_tokens = 120
        resp.usage.output_tokens = 30
        client._record_usage(resp)

        assert len(client.usage_log) == 1
        u = client.usage_log[0]
        assert u.input_tokens == 120 and u.output_tokens == 30 and u.model == _SONNET

        drained = client.drain_usage()
        assert len(drained) == 1
        assert client.usage_log == []
        assert client.drain_usage() == []  # second drain is empty

    def test_record_usage_missing_block_is_noop(self) -> None:
        client = self._client()
        resp = MagicMock()
        resp.usage = None
        client._record_usage(resp)
        assert client.usage_log == []


class TestBatchUsageCapture:
    def test_attaches_usage_and_model(self) -> None:
        from ddharmon.llm.batch import _attach_batch_usage

        msg = MagicMock()
        msg.usage.input_tokens = 200
        msg.usage.output_tokens = 40
        msg.model = _SONNET
        record: dict = {"id": "x", "response": {}}
        _attach_batch_usage(record, msg)

        assert record["usage"] == {"input_tokens": 200, "output_tokens": 40}
        assert record["model"] == _SONNET

    def test_no_usage_leaves_record_unchanged(self) -> None:
        from ddharmon.llm.batch import _attach_batch_usage

        class _Msg:  # a message object with no usage attribute
            pass

        record: dict = {"id": "x", "response": {}}
        _attach_batch_usage(record, _Msg())
        assert "usage" not in record
        assert record == {"id": "x", "response": {}}


class TestDrainUsageDefault:
    def test_client_without_usage_log_returns_empty(self) -> None:
        # A minimal BaseLLMClient subclass that never sets usage_log still drains cleanly.
        from ddharmon.llm.base import BaseLLMClient
        from ddharmon.llm.prompts import RerankerResponse

        class _Bare(BaseLLMClient):
            @property
            def provider_name(self) -> str:
                return "bare"

            @property
            def model_name(self) -> str:
                return "bare"

            def rerank_candidates(self, source_context, candidate_contexts, candidate_names) -> RerankerResponse:
                return RerankerResponse(judgments=[])

        assert _Bare().drain_usage() == []
