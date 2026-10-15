"""LLM model registry: the single source of truth for which model ddharmon calls by default.

Every default in the library (``AnthropicClient``, ``get_client``, the Batch API fallback, the harmonization
stages' ``model_tag``, the CLI's ``--model``) reads :data:`DEFAULT_MODEL` from here, so moving to a new model is
a one-line change. That change is benchmark-gated, never automatic: the pipeline's thresholds and prompts were
validated on one model, and a different model can shift the assignment verdicts. Bump ``DEFAULT_MODEL`` only
after the new model has been re-run against the benchmarks.

Why this module exists: a stale hardcoded default 404s on every call once the provider retires it. Sonnet 4
(``claude-sonnet-4-20250514``) retired 2026-06-15 and broke bare ``AnthropicClient()`` calls; Sonnet 4.5
(``claude-sonnet-4-5``) retires 2026-11-30. Keep retired or retiring ids out of :data:`MODELS`.

Stdlib-only on purpose: the thin Batch API submitter imports this and must stay importable without the heavy
stack.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL: str = "claude-sonnet-4-6"


@dataclass(frozen=True)
class ModelInfo:
    """One model ddharmon knows how to call: what it is, whether a user may pick it, and how to ask it.

    The request rules exist because the current model lines reject what older ones accepted (checked against
    platform.claude.com, 2026-10-14): Claude Sonnet 5.5 refuses a forced ``tool_choice`` and a non-default
    ``temperature``; Claude Haiku 5.5 refuses any ``temperature`` but ``1``; both think by default. One request
    builder (:func:`ddharmon.llm.params.message_params`) reads these fields for the Batch API and the synchronous
    path alike, so a model added here is called correctly everywhere.
    """

    id: str  # provider model id, e.g. "claude-sonnet-4-6"
    label: str  # human label, e.g. "Claude Sonnet 4.6"
    provider: str  # "anthropic" | "openai" | ...
    validated: bool  # True only for models the pipeline was benchmark-validated on
    # Offered in a model picker. A validated model always is; a newer model can be offered BEFORE its benchmark
    # so users get the cheaper or stronger option as it is released — the picker says it is not yet benchmarked.
    selectable: bool = False
    accepts_temperature: bool = True  # False: send no sampling parameter (the model 400s on a non-default one)
    forced_tool_choice: bool = True  # False: a forced tool call 400s — structured outputs carry the schema instead
    thinking_type: str | None = None  # the `thinking.type` to send ("between_tools" = Sonnet 5.5's thinking-off)
    effort: str | None = None  # `output_config.effort` to send; None leaves the model's own default
    # Standard (non-batch) USD per million tokens. Priced from here first, so a model newer than the installed
    # LiteLLM price map is not priced at an older sibling's rate. Haiku 5.5: its <=100K-token prompt tier.
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    # About how many tokens the same text counts as, relative to DEFAULT_MODEL's tokenizer. The 5.x models use the
    # tokenizer introduced with Claude Opus 4.7, which counts ~30% more (Haiku 5.5 migration guide); a cost
    # estimate calibrated on the default scales by it alongside the price.
    token_ratio: float = 1.0


# DEFAULT_MODEL first and the only validated entry. The 5.5 models are selectable ahead of their benchmark; the
# rest are listed (greyed out in a picker) but not offered. Thinking is kept minimal on the 5.5 models so a run
# costs and behaves like the validated default (Sonnet 5.5: `between_tools`; Haiku 5.5 has no thinking-off
# setting, so `low` effort); raise them only on benchmark evidence.
MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id=DEFAULT_MODEL,
        label="Claude Sonnet 4.6",
        provider="anthropic",
        validated=True,
        selectable=True,
        input_per_mtok=3.0,
        output_per_mtok=15.0,
    ),
    ModelInfo(
        id="claude-sonnet-5-5",
        label="Claude Sonnet 5.5",
        provider="anthropic",
        validated=False,
        selectable=True,
        accepts_temperature=False,
        forced_tool_choice=False,
        thinking_type="between_tools",
        input_per_mtok=2.0,
        output_per_mtok=10.0,
        token_ratio=1.3,
    ),
    ModelInfo(
        id="claude-haiku-5-5",
        label="Claude Haiku 5.5",
        provider="anthropic",
        validated=False,
        selectable=True,
        accepts_temperature=False,
        effort="low",
        input_per_mtok=0.10,
        output_per_mtok=0.50,
        token_ratio=1.3,
    ),
    ModelInfo(
        id="claude-opus-5-5",
        label="Claude Opus 5.5",
        provider="anthropic",
        validated=False,
        accepts_temperature=False,
        forced_tool_choice=False,
        effort="low",
        input_per_mtok=4.0,
        output_per_mtok=20.0,
        token_ratio=1.3,
    ),
    ModelInfo(
        id="gpt-4o", label="GPT-4o", provider="openai", validated=False, input_per_mtok=2.5, output_per_mtok=10.0
    ),
)

_BY_ID: dict[str, ModelInfo] = {m.id: m for m in MODELS}


def model_info(model_id: str) -> ModelInfo | None:
    """The registry entry for ``model_id``, or ``None`` if the model isn't in the registry."""
    return _BY_ID.get(model_id)
