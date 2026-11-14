"""Tests for the LLM model registry (``ddharmon.llm.models``) — the single source of truth for model ids."""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ddharmon.llm.models import DEFAULT_MODEL, MODELS, ModelInfo, model_info

# Ids the provider has retired or announced a retirement date for. A default or registry entry naming one of
# these 404s on every call once the date passes (this happened with Sonnet 4 on 2026-06-15).
_RETIRED_IDS = frozenset(
    {
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-0",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "gemini/gemini-1.5-pro",
        "gemini-1.5-pro",
    }
)

# A versioned Claude model id: ``claude-`` + optional family words + a version digit
# (``claude-sonnet-4-6``, ``claude-3-5-haiku``, ``anthropic/claude-opus-4-8``). Bare family prefixes such as
# ``claude-sonnet`` (used as price-table keys) don't match.
_MODEL_ID_RE = re.compile(r"\bclaude-(?:[a-z]+-)*\d")

_SRC = Path(__file__).resolve().parents[1] / "src" / "ddharmon"
_REGISTRY = _SRC / "llm" / "models.py"


class TestRegistry:
    def test_default_is_first_and_validated(self) -> None:
        assert MODELS[0].id == DEFAULT_MODEL
        assert MODELS[0].validated

    def test_only_the_default_is_validated(self) -> None:
        assert [m.id for m in MODELS if m.validated] == [DEFAULT_MODEL]

    def test_no_entry_is_retired(self) -> None:
        assert DEFAULT_MODEL not in _RETIRED_IDS
        assert not [m.id for m in MODELS if m.id in _RETIRED_IDS]

    def test_ids_are_unique(self) -> None:
        ids = [m.id for m in MODELS]
        assert len(ids) == len(set(ids))

    def test_entries_are_complete(self) -> None:
        for m in MODELS:
            assert isinstance(m, ModelInfo)
            assert m.id and m.label and m.provider

    def test_model_info_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            MODELS[0].id = "something-else"  # type: ignore[misc]

    def test_model_info_lookup(self) -> None:
        info = model_info(DEFAULT_MODEL)
        assert info is not None
        assert info is MODELS[0]
        assert model_info("no-such-model") is None


class TestDefaultsPointAtRegistry:
    def test_anthropic_client_default(self) -> None:
        mock_anthropic_mod = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
            from ddharmon.llm.anthropic_client import AnthropicClient

            assert AnthropicClient().model_name == DEFAULT_MODEL

    def test_get_client_anthropic_default(self) -> None:
        mock_anthropic_mod = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
            from ddharmon.llm import get_client

            assert get_client("anthropic").model_name == DEFAULT_MODEL

    def test_cli_model_option_default(self) -> None:
        from ddharmon import cli

        assert cli._DEFAULT_MODEL == DEFAULT_MODEL

    def test_batch_fallback(self) -> None:
        from ddharmon.llm import batch

        assert batch._FALLBACK_MODEL == DEFAULT_MODEL

    def test_harmonization_model_tags(self) -> None:
        from ddharmon.harmonization import leanb, merge, pipeline

        assert leanb.DEFAULT_MODEL_TAG == DEFAULT_MODEL
        assert pipeline.DEFAULT_MODEL_TAG == DEFAULT_MODEL
        assert inspect.signature(merge.prepare_merge).parameters["model_tag"].default == DEFAULT_MODEL


def test_no_hardcoded_claude_model_ids_outside_registry() -> None:
    """Only ``ddharmon/llm/models.py`` may name a Claude model id, so the next bump is a one-line change."""
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        if path == _REGISTRY:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _MODEL_ID_RE.search(line):
                offenders.append(f"{path.relative_to(_SRC.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "hardcoded Claude model id(s) — import DEFAULT_MODEL instead:\n" + "\n".join(offenders)
