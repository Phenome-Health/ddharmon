"""Anthropic (Claude) LLM client for reranking."""

from __future__ import annotations

import json
import logging

from ddharmon.llm.base import BaseLLMClient
from ddharmon.llm.prompts import RERANKER_SYSTEM_PROMPT, RerankerResponse, build_reranker_prompt

logger = logging.getLogger(__name__)

_JSON_INSTRUCTION = """

Respond with ONLY valid JSON matching this schema (no markdown fences):
{
  "judgments": [
    {
      "candidate_variable": "variable_name",
      "relation": "exact|broader|narrower|composite|derivable|no_match",
      "confidence": 0.0,
      "rationale": "brief explanation"
    }
  ]
}"""


def _parse_json_response(text: str) -> RerankerResponse:
    """Parse a JSON response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return RerankerResponse.model_validate(json.loads(text))


def _message_text(message) -> str:
    """Concatenate text from an Anthropic message's TextBlock content blocks.

    ``Message.content`` is a union of block types (text, thinking, tool-use, …);
    only ``TextBlock`` carries ``.text``. These reranking/labeling calls always
    return text, so a response with no TextBlock is an error worth surfacing.
    """
    from anthropic.types import TextBlock

    parts = [block.text for block in message.content if isinstance(block, TextBlock)]
    if not parts:
        raise ValueError("Anthropic response contained no text content")
    return "".join(parts)


class AnthropicClient(BaseLLMClient):
    """LLM client using the Anthropic Claude API.

    Attempts structured output via ``messages.parse()`` first; falls back to
    ``messages.create()`` with JSON instructions for models that don't support
    output_format.
    """

    def __init__(
        # Default per project decision 03-01 ("default to Claude Sonnet 4.5 for Anthropic"). The prior pin
        # (claude-sonnet-4-20250514, Sonnet 4) retired 2026-06-15 and now 404s on a bare AnthropicClient().
        self,
        model_name: str = "claude-sonnet-4-5",
        max_tokens: int = 2048,
        *,
        api_key: str | None = None,
    ) -> None:
        import anthropic

        # api_key=None lets the SDK fall back to ANTHROPIC_API_KEY (unchanged default). A caller-supplied
        # key (e.g. a per-request BYOK key from a web backend) is scoped to this client only — never written
        # to os.environ, which would race/leak across concurrent jobs.
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._use_structured: bool | None = None  # auto-detect on first call
        # Realized token usage per call, drained per-stage by cost accounting (see cost.CostLedger). Best-effort:
        # capture never raises into a run — a missing/odd usage block just means that call isn't priced.
        self.usage_log: list = []

    def _record_usage(self, response: object, *, model: str | None = None) -> None:
        """Append this response's token usage to ``usage_log`` (best-effort; never raises into a run).

        ``model`` = the model that ran when it is not this client's own (``complete_request``'s per-prompt model).
        """
        from ddharmon.llm.cost import TokenUsage

        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return
            self.usage_log.append(
                TokenUsage(
                    model=model or self._model_name,
                    input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                )
            )
        except Exception:  # noqa: BLE001 — usage capture must never break a run
            logger.debug("usage capture failed", exc_info=True)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank_candidates(
        self,
        source_context: dict[str, str],
        candidate_contexts: list[dict[str, str]],
        candidate_names: list[str],
    ) -> RerankerResponse:
        """Rerank candidates using Claude.

        Tries structured output on the first call; if the model doesn't support
        it, permanently falls back to JSON parsing for all subsequent calls.
        """
        user_prompt = build_reranker_prompt(source_context, candidate_contexts)

        if self._use_structured is not False:
            try:
                response = self._client.messages.parse(
                    model=self._model_name,
                    max_tokens=self._max_tokens,
                    system=RERANKER_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                    output_format=RerankerResponse,
                )
                self._use_structured = True
                self._record_usage(response)
                if response.parsed_output is None:
                    raise ValueError("Anthropic structured output returned no parsed result")
                return response.parsed_output
            except Exception as e:
                err_msg = str(e).lower()
                if "output format" not in err_msg and "output_format" not in err_msg:
                    raise
                logger.info(
                    "Model %s does not support structured output, using JSON fallback",
                    self._model_name,
                )
                self._use_structured = False

        # JSON fallback
        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=self._max_tokens,
            system=RERANKER_SYSTEM_PROMPT + _JSON_INSTRUCTION,
            messages=[{"role": "user", "content": user_prompt}],
        )
        self._record_usage(response)
        return _parse_json_response(_message_text(response))

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> str:
        """Send a plain text prompt and return a plain text response."""
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": self._model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        self._record_usage(response)
        return _message_text(response)

    def complete_request(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int,
        temperature: float,
        model: str | None = None,
        tool_schema: dict | None = None,
        tool_name: str | None = None,
    ) -> str | dict:
        """Send ONE prompt exactly as ``batch.submit_batch`` would, synchronously.

        Same request shape as a batch request's ``params``: the caller's ``temperature`` and per-prompt
        ``max_tokens``, the prompt's own ``model`` (else this client's), and — when ``tool_schema`` is set — a
        FORCED tool call whose ``input`` is returned as the structured dict, mirroring ``retrieve_batch``. With no
        tool the response text is returned for the caller to parse. Usage is recorded against the model that
        actually ran, so a prompt pinned to a different model than the run's is priced as that model.
        """
        kwargs: dict = {
            "model": model or self._model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if tool_schema:
            name = tool_name or "emit"
            kwargs["tools"] = [
                {
                    "name": name,
                    "description": "Return the result as structured input conforming to the schema.",
                    "input_schema": tool_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": name}
        response = self._client.messages.create(**kwargs)
        ran = getattr(response, "model", None)
        self._record_usage(response, model=ran if isinstance(ran, str) and ran else kwargs["model"])
        if tool_schema:
            block = next((b for b in response.content if getattr(b, "type", None) == "tool_use"), None)
            if block is None:
                raise ValueError("Anthropic response to a forced tool call contained no tool_use block")
            return dict(block.input) if isinstance(block.input, dict) else block.input
        parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        if not parts:
            raise ValueError("Anthropic response contained no text content")
        return "".join(parts)
