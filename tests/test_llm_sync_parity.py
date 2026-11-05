"""``AnthropicClient.complete_request`` — one prompt, sent the way the Batch API sends it.

A synchronous run and a batch run must verify the SAME pipeline. Before this, a sync call ran at the API's
default temperature (1.0, batch is 0), was capped at 1024 output tokens (batch honours a per-prompt budget,
else 2048 — the enforced split has returned empty at 1024), ignored a forced tool call (batch sends
``tool_choice``), and always used the client's model where batch uses the prompt's own ``model_tag``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _client(create_return: object):
    mod = MagicMock()
    inner = MagicMock()
    inner.messages.create.return_value = create_return
    mod.Anthropic.return_value = inner
    with patch.dict(sys.modules, {"anthropic": mod}):
        from ddharmon.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model_name="claude-run-model"), inner


def _message(*blocks: object, model: str = "claude-prompt-model") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), model=model, usage=SimpleNamespace(input_tokens=11, output_tokens=7))


def test_a_text_prompt_is_sent_at_temperature_zero_with_the_prompts_model_and_budget() -> None:
    client, inner = _client(_message(SimpleNamespace(type="text", text='{"a": 1}')))

    out = client.complete_request("u", system="s", max_tokens=2048, temperature=0.0, model="claude-prompt-model")

    kw = inner.messages.create.call_args.kwargs
    assert kw["temperature"] == 0.0
    assert kw["max_tokens"] == 2048
    assert kw["model"] == "claude-prompt-model"
    assert kw["system"] == "s"
    assert "tools" not in kw and "tool_choice" not in kw
    assert out == '{"a": 1}'


def test_a_forced_tool_call_is_honoured_and_returns_the_tool_input() -> None:
    schema = {"type": "object", "properties": {"kinds": {"type": "string"}}}
    client, inner = _client(_message(SimpleNamespace(type="tool_use", input={"kinds": "distinct"})))

    out = client.complete_request(
        "u", system="s", max_tokens=2048, temperature=0.0, model="m", tool_schema=schema, tool_name="emit_kinds"
    )

    kw = inner.messages.create.call_args.kwargs
    assert kw["tools"] == [
        {
            "name": "emit_kinds",
            "description": "Return the result as structured input conforming to the schema.",
            "input_schema": schema,
        }
    ]
    assert kw["tool_choice"] == {"type": "tool", "name": "emit_kinds"}
    assert out == {"kinds": "distinct"}


def test_usage_is_priced_against_the_model_that_actually_ran() -> None:
    client, _ = _client(_message(SimpleNamespace(type="text", text="{}"), model="claude-prompt-model"))

    client.complete_request("u", system="s", max_tokens=16, temperature=0.0, model="claude-prompt-model")

    (usage,) = client.drain_usage()
    assert usage.model == "claude-prompt-model"
    assert (usage.input_tokens, usage.output_tokens) == (11, 7)


def test_no_model_falls_back_to_the_clients_own() -> None:
    client, inner = _client(_message(SimpleNamespace(type="text", text="{}"), model="claude-run-model"))

    client.complete_request("u", system="s", max_tokens=16, temperature=0.0)

    assert inner.messages.create.call_args.kwargs["model"] == "claude-run-model"


def test_plain_complete_is_unchanged() -> None:
    """``complete`` keeps its signature and sends no temperature — existing callers get what they got."""
    client, inner = _client(_message(SimpleNamespace(type="text", text="hi")))

    with patch.dict(sys.modules, {"anthropic.types": MagicMock(TextBlock=SimpleNamespace)}):
        assert client.complete("p", system="s", max_tokens=5) == "hi"
    kw = inner.messages.create.call_args.kwargs
    assert "temperature" not in kw and kw["model"] == "claude-run-model"
