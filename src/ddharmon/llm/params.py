"""One Messages API request per model, shared by the Batch API and the synchronous path.

The pipeline sends every prompt the same way whichever transport carries it (``batch.submit_batch`` and
``AnthropicClient.complete_request`` both build their request here), so what a model accepts is decided in ONE
place — the model's entry in :mod:`ddharmon.llm.models`:

- ``temperature`` only for a model that accepts it. The validated default keeps greedy decoding (``0.0``);
  Claude Sonnet 5.5 and Claude Haiku 5.5 reject a non-default value with a 400.
- A ``tool_schema`` becomes a FORCED tool call where the model accepts one, and structured outputs
  (``output_config.format``) where it does not (Claude Sonnet 5.5): both constrain the answer to the same schema,
  so the caller gets the same dict either way.
- ``thinking`` / ``output_config.effort`` as the registry sets them, to keep the newer models' default thinking
  minimal (it is billed as output and, on a small ``max_tokens``, can use up the whole budget).

The newer models also answer with a ``thinking`` block BEFORE the text, and can decline (``stop_reason:
"refusal"``). :func:`message_text` and :func:`refusal_of` read a response by block type rather than by position.

Stdlib-only, like :mod:`ddharmon.llm.models`: the thin Batch API submitter imports it.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

from ddharmon.llm.models import model_info

TOOL_DESCRIPTION = "Return the result as structured input conforming to the schema."

# JSON Schema keywords structured outputs accept beside the ones handled structurally below. Anything else is moved
# into the description (the model may still follow it; the API would reject it). Mirrors the Anthropic SDK's own
# `transform_schema`, which is private and does not accept a `type` list.
_KEPT_STRING_FORMATS = {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}


class RefusalError(ValueError):
    """The model declined the request (``stop_reason == "refusal"``). The prompt stays unanswered."""

    def __init__(self, category: str | None, explanation: str | None = None) -> None:
        self.category = category
        self.explanation = explanation
        super().__init__(f"the model declined this request (category: {category or 'unspecified'})")


def structured_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` rewritten into the JSON Schema subset structured outputs accept. The input is not mutated.

    Every object gets ``additionalProperties: false`` (required); a ``type`` list such as ``["string", "null"]``
    becomes ``anyOf``; ``minItems`` survives only as 0 or 1; other unsupported constraints move into the description.
    """
    return _transform(copy.deepcopy(schema))


def _transform(node: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ref = node.pop("$ref", None)
    if ref is not None:
        out["$ref"] = ref
        return out
    type_ = node.pop("type", None)
    any_of = node.pop("anyOf", None) or node.pop("oneOf", None)
    all_of = node.pop("allOf", None)
    if isinstance(type_, list):
        any_of, type_ = [{"type": t} for t in type_], None
    if any_of is not None:
        out["anyOf"] = [_transform(dict(v)) for v in any_of]
    elif all_of is not None:
        out["allOf"] = [_transform(dict(v)) for v in all_of]
    elif type_ is not None:
        out["type"] = type_
    for key in ("enum", "const", "description", "title"):
        if key in node:
            out[key] = node.pop(key)
    if type_ == "object":
        out["properties"] = {k: _transform(dict(v)) for k, v in node.pop("properties", {}).items()}
        node.pop("additionalProperties", None)
        out["additionalProperties"] = False
        if "required" in node:
            out["required"] = node.pop("required")
    elif type_ == "array":
        if "items" in node:
            out["items"] = _transform(dict(node.pop("items")))
        if node.get("minItems") in (0, 1):
            out["minItems"] = node.pop("minItems")
    elif type_ == "string" and node.get("format") in _KEPT_STRING_FORMATS:
        out["format"] = node.pop("format")
    for key in ("$defs", "definitions"):
        if key in node:
            out[key] = {k: _transform(dict(v)) for k, v in node.pop(key).items()}
    if node:  # what is left is unsupported: say it in the description so the model may still follow it
        extra = ", ".join(f"{k}: {v}" for k, v in node.items())
        out["description"] = (out["description"] + "\n\n" if "description" in out else "") + "{" + extra + "}"
    return out


def message_params(
    *,
    model: str,
    system: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    tool_schema: dict[str, Any] | None = None,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """The request for one prompt to ``model``, shaped by that model's registry entry.

    A model the registry does not list gets the long-standing shape (``temperature`` + a forced tool), so callers
    passing their own ids see no change.
    """
    info = model_info(model)
    accepts_temperature = info.accepts_temperature if info else True
    forced_tool = info.forced_tool_choice if info else True
    params: dict[str, Any] = {"model": model, "max_tokens": max_tokens}
    if accepts_temperature:
        params["temperature"] = temperature
    params["system"] = system
    params["messages"] = [{"role": "user", "content": user_prompt}]
    output_config: dict[str, Any] = {}
    if tool_schema:
        if forced_tool:
            name = tool_name or "emit"
            params["tools"] = [{"name": name, "description": TOOL_DESCRIPTION, "input_schema": tool_schema}]
            params["tool_choice"] = {"type": "tool", "name": name}
        else:
            output_config["format"] = {"type": "json_schema", "schema": structured_output_schema(tool_schema)}
    params.update(thinking_params(model))
    if info and info.effort:
        output_config["effort"] = info.effort
    if output_config:
        params["output_config"] = output_config
    return params


def thinking_params(model: str) -> dict[str, Any]:
    """The ``thinking`` field the registry sets for ``model`` (empty when it sets none)."""
    info = model_info(model)
    return {"thinking": {"type": info.thinking_type}} if info and info.thinking_type else {}


def uses_structured_output(params: dict[str, Any]) -> bool:
    """Whether ``params`` carries its schema as structured outputs (the answer is JSON text, not a tool call)."""
    return "format" in (params.get("output_config") or {})


def refusal_of(message: Any) -> RefusalError | None:
    """A :class:`RefusalError` when ``message`` was declined, else ``None``."""
    if getattr(message, "stop_reason", None) != "refusal":
        return None
    details = getattr(message, "stop_details", None)
    return RefusalError(getattr(details, "category", None), getattr(details, "explanation", None))


def message_text(message: Any) -> str | None:
    """The response's text, joined across ``text`` blocks — ``None`` when it has none (thinking-only, a refusal)."""
    parts = [b.text for b in getattr(message, "content", None) or [] if getattr(b, "type", None) == "text"]
    return "".join(parts) if parts else None


def tool_input(message: Any) -> Any | None:
    """The first ``tool_use`` block's input, or ``None``."""
    block = next((b for b in getattr(message, "content", None) or [] if getattr(b, "type", None) == "tool_use"), None)
    if block is None:
        return None
    return dict(block.input) if isinstance(block.input, dict) else block.input


def for_installed_sdk(kwargs: dict[str, Any], create: Any) -> dict[str, Any]:
    """``kwargs`` with every key the installed SDK's ``create`` does not accept moved into ``extra_body``.

    Anthropic SDK 1.x removed ``temperature`` (and ``top_p`` / ``top_k``) from ``Messages.create`` — passing one is a
    TypeError — while the API still honours it for the models that accept it; an older SDK may predate a newer field.
    ``extra_body`` is merged into the request JSON as-is on both, so one request works on either SDK generation.
    """
    try:
        sig = inspect.signature(create)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    out = {k: v for k, v in kwargs.items() if k in sig.parameters}
    extra = {k: v for k, v in kwargs.items() if k not in sig.parameters}
    if extra:
        out["extra_body"] = {**(kwargs.get("extra_body") or {}), **extra}
    return out
