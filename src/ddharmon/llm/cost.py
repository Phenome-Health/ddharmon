"""Realized LLM cost accounting — the single source of truth for "what did this run actually cost".

Every cost number ddharmon shows should come from REAL token usage, not a guess. The clients
(:class:`~ddharmon.llm.anthropic_client.AnthropicClient`, :class:`~ddharmon.llm.litellm_client.LiteLLMClient`)
append a :class:`TokenUsage` per call to ``self.usage_log`` when the provider returns a usage block, and
``batch.retrieve_batch`` preserves the batch usage into its responses JSONL. This module prices those tokens
against **LiteLLM's maintained model→price map** (``litellm.cost_per_token``) so there is no hardcoded
per-token table to drift when a provider changes prices.

The Anthropic Message Batches API bills at 50% of standard rates; LiteLLM's map is standard-rate, so the
batch discount is applied here (``batch=True``). For BYOK users the resulting number is literally their own
provider bill.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Anthropic Message Batches API bills at 50% of standard token rates (batch.py header). LiteLLM's price map
# is standard-rate, so we halve batch-sourced usage here rather than maintaining a second discounted table.
BATCH_DISCOUNT = 0.5

# Published standard rates (USD per 1M tokens: input, output) for the models ddharmon runs — a small FALLBACK
# used only when the optional ``litellm`` package isn't installed (so realized cost still prices accurately
# without a heavy dependency). LiteLLM's maintained map is PREFERRED whenever available: it auto-tracks price
# changes across every provider. Keys are matched as case-insensitive substrings (longest match wins), so a
# dated model id like ``claude-sonnet-4-20250514`` still resolves. Update if a provider changes these rates.
_FALLBACK_RATES_PER_M: dict[str, tuple[float, float]] = {
    "claude-haiku": (1.0, 5.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-opus": (15.0, 75.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
}


def _fallback_price(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Price via the built-in rate table (longest substring match), or ``None`` if the model isn't known."""
    m = model.lower()
    best_key: str | None = None
    for key in _FALLBACK_RATES_PER_M:
        if key in m and (best_key is None or len(key) > len(best_key)):
            best_key = key
    if best_key is None:
        return None
    rate_in, rate_out = _FALLBACK_RATES_PER_M[best_key]
    return input_tokens / 1_000_000 * rate_in + output_tokens / 1_000_000 * rate_out


@dataclass
class TokenUsage:
    """Real token usage from a single LLM call.

    ``response_cost`` is a provider-/LiteLLM-reported realized cost when available (litellm attaches it via
    ``_hidden_params["response_cost"]``); when ``None`` the cost is computed from the token counts.
    """

    model: str
    input_tokens: int
    output_tokens: int
    response_cost: float | None = None


def price_usage(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    *,
    batch: bool = False,
    response_cost: float | None = None,
) -> float:
    """Realized USD for one call.

    Prefers a provider-reported ``response_cost`` when present; otherwise prices the captured token counts —
    via ``litellm.cost_per_token`` against LiteLLM's maintained model→price map when litellm is installed, else
    against a small built-in rate table (:data:`_FALLBACK_RATES_PER_M`) so cost still works without the
    optional dependency. Batch usage gets the 50% discount. Returns ``0.0`` for a model neither source can
    price rather than fabricating a number (never guess).
    """
    discount = BATCH_DISCOUNT if batch else 1.0
    if response_cost is not None:
        return float(response_cost) * discount
    if not model:
        return 0.0
    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=int(input_tokens),
            completion_tokens=int(output_tokens),
        )
        return ((prompt_cost or 0.0) + (completion_cost or 0.0)) * discount
    except Exception:  # noqa: BLE001 — litellm absent OR its map missed this model: try the built-in table
        logger.debug("price_usage: litellm could not price model %r; trying the built-in table", model)
    fallback = _fallback_price(model, int(input_tokens), int(output_tokens))
    if fallback is None:
        logger.debug("price_usage: no rate for model %r; reporting $0 (not guessing)", model)
        return 0.0
    return fallback * discount


@dataclass
class StageCost:
    """Accumulated realized cost + token totals for one pipeline stage."""

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    calls: int = 0


class CostLedger:
    """Accumulates realized cost per pipeline stage (and a run total).

    A stage runner drains its client's ``usage_log`` (sync) or reads the batch ``usage`` records (batch) and
    calls :meth:`add`; the total feeds a live "spent so far" counter and the final per-stage breakdown.
    """

    def __init__(self) -> None:
        self._stages: dict[str, StageCost] = {}

    def add(self, stage: str, usages: list[TokenUsage], *, batch: bool = False) -> float:
        """Price a stage's usage records and fold them into the ledger. Returns the USD added for this call."""
        sc = self._stages.setdefault(stage, StageCost())
        added = 0.0
        for u in usages:
            usd = price_usage(u.model, u.input_tokens, u.output_tokens, batch=batch, response_cost=u.response_cost)
            sc.input_tokens += int(u.input_tokens)
            sc.output_tokens += int(u.output_tokens)
            sc.usd += usd
            sc.calls += 1
            added += usd
        return added

    @property
    def total_usd(self) -> float:
        return sum(s.usd for s in self._stages.values())

    @property
    def total_tokens(self) -> dict[str, int]:
        return {
            "input": sum(s.input_tokens for s in self._stages.values()),
            "output": sum(s.output_tokens for s in self._stages.values()),
        }

    def to_dict(self) -> dict:
        """Serialize for the UI result contract (camelCase to match the rest of the contract)."""
        return {
            "actualUsd": round(self.total_usd, 6),
            "tokens": {"input": self.total_tokens["input"], "output": self.total_tokens["output"]},
            "perStage": {
                stage: {
                    "usd": round(sc.usd, 6),
                    "inputTokens": sc.input_tokens,
                    "outputTokens": sc.output_tokens,
                    "calls": sc.calls,
                }
                for stage, sc in self._stages.items()
            },
        }
