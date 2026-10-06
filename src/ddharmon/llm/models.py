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
    """One selectable model."""

    id: str  # provider model id, e.g. "claude-sonnet-4-6"
    label: str  # human label, e.g. "Claude Sonnet 4.6"
    provider: str  # "anthropic" | "openai" | ...
    validated: bool  # True only for models the pipeline was benchmark-validated on


# DEFAULT_MODEL first and the only validated entry; the rest are offered for experimentation, not validated.
MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(id=DEFAULT_MODEL, label="Claude Sonnet 4.6", provider="anthropic", validated=True),
    ModelInfo(id="claude-opus-4-8", label="Claude Opus 4.8", provider="anthropic", validated=False),
    ModelInfo(id="gpt-4o", label="GPT-4o", provider="openai", validated=False),
)

_BY_ID: dict[str, ModelInfo] = {m.id: m for m in MODELS}


def model_info(model_id: str) -> ModelInfo | None:
    """The registry entry for ``model_id``, or ``None`` if the model isn't in the registry."""
    return _BY_ID.get(model_id)
