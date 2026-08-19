"""OpenAI LLM client for reranking."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

from ddharmon.llm.base import BaseLLMClient
from ddharmon.llm.prompts import RERANKER_SYSTEM_PROMPT, RerankerResponse, build_reranker_prompt

if TYPE_CHECKING:  # import-time only for type checkers; openai stays an optional runtime dep
    from openai.types.chat import ChatCompletionMessageParam

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


def _require_content(content: str | None) -> str:
    """Return the message text, or raise if OpenAI returned no content."""
    if content is None:
        raise ValueError("OpenAI returned no message content")
    return content


class OpenAIClient(BaseLLMClient):
    """LLM client using the OpenAI API.

    Attempts structured output via ``beta.chat.completions.parse()`` first;
    falls back to ``chat.completions.create()`` with JSON instructions for
    models that don't support response_format with Pydantic.
    """

    def __init__(self, model_name: str = "gpt-4o", max_tokens: int = 2048) -> None:
        import openai

        self._client = openai.OpenAI()
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._use_structured: bool | None = None  # auto-detect on first call
        # Realized token usage per call, drained per-stage by cost accounting (see cost.CostLedger).
        self.usage_log: list = []

    def _record_usage(self, response: object) -> None:
        """Append this response's token usage to ``usage_log`` (best-effort; never raises into a run)."""
        from ddharmon.llm.cost import TokenUsage

        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return
            self.usage_log.append(
                TokenUsage(
                    model=self._model_name,
                    input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                )
            )
        except Exception:  # noqa: BLE001 — usage capture must never break a run
            logger.debug("usage capture failed", exc_info=True)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank_candidates(
        self,
        source_context: dict[str, str],
        candidate_contexts: list[dict[str, str]],
        candidate_names: list[str],
    ) -> RerankerResponse:
        """Rerank candidates using OpenAI.

        Tries structured output on the first call; if the model doesn't support
        it, permanently falls back to JSON parsing for all subsequent calls.
        """
        user_prompt = build_reranker_prompt(source_context, candidate_contexts)
        messages = [
            {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if self._use_structured is not False:
            try:
                response = self._client.beta.chat.completions.parse(
                    model=self._model_name,
                    max_tokens=self._max_tokens,
                    messages=cast("list[ChatCompletionMessageParam]", messages),
                    response_format=RerankerResponse,
                )
                self._use_structured = True
                self._record_usage(response)
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("OpenAI structured output returned no parsed result")
                return parsed
            except Exception as e:
                err_msg = str(e).lower()
                if "response_format" not in err_msg and "response format" not in err_msg:
                    raise
                logger.info(
                    "Model %s does not support structured output, using JSON fallback",
                    self._model_name,
                )
                self._use_structured = False

        # JSON fallback
        messages[0]["content"] = RERANKER_SYSTEM_PROMPT + _JSON_INSTRUCTION
        response = self._client.chat.completions.create(
            model=self._model_name,
            max_tokens=self._max_tokens,
            messages=cast("list[ChatCompletionMessageParam]", messages),
        )
        self._record_usage(response)
        return _parse_json_response(_require_content(response.choices[0].message.content))

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> str:
        """Send a plain text prompt and return a plain text response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self._model_name,
            max_tokens=max_tokens,
            messages=cast("list[ChatCompletionMessageParam]", messages),
        )
        self._record_usage(response)
        return _require_content(response.choices[0].message.content)
