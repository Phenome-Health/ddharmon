"""Tests for the Anthropic Batch API submit path — determinism-relevant params.

The LLM stages are extraction/classification, so the batch requests must default to greedy decoding
(``temperature=0.0``) to minimise run-to-run generation variance. These tests pin that default (and the
override) by capturing the request bodies handed to the client, without hitting the network.
"""

from __future__ import annotations

import json
from pathlib import Path


def _write_prompts(tmp_path: Path) -> Path:
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        json.dumps(
            {
                "id": "cluster:1",
                "system_prompt": "sys",
                "user_prompt": "hi",
                "schema": "{}",
                "model_tag": "claude-sonnet-4-6",
            }
        )
        + "\n"
    )
    return p


class _FakeBatch:
    id = "batch_test"
    created_at = None
    expires_at = None


class _FakeBatches:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def create(self, *, requests):
        self._captured["requests"] = requests
        return _FakeBatch()


class _FakeClient:
    def __init__(self, captured: dict) -> None:
        self.messages = type("M", (), {"batches": _FakeBatches(captured)})()


def test_submit_batch_defaults_to_temperature_zero(tmp_path, monkeypatch):
    import anthropic

    from ddharmon.llm.batch import submit_batch

    captured: dict = {}
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: _FakeClient(captured))

    batch_id = submit_batch(_write_prompts(tmp_path), manifest_path=tmp_path / "manifest.json")

    assert batch_id == "batch_test"
    reqs = captured["requests"]
    assert len(reqs) == 1
    # greedy decoding is the deterministic-leaning default for every request
    assert reqs[0]["params"]["temperature"] == 0.0


def test_submit_batch_temperature_override(tmp_path, monkeypatch):
    import anthropic

    from ddharmon.llm.batch import submit_batch

    captured: dict = {}
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: _FakeClient(captured))

    submit_batch(_write_prompts(tmp_path), temperature=0.7, manifest_path=tmp_path / "m.json")

    assert captured["requests"][0]["params"]["temperature"] == 0.7


def test_submit_batch_forwards_api_key(tmp_path, monkeypatch):
    """A caller-supplied api_key (BYOK) is passed to anthropic.Anthropic — not read from env."""
    import anthropic

    from ddharmon.llm.batch import submit_batch

    captured: dict = {}

    def _fake_anthropic(*args, **kwargs):
        captured["ctor_kwargs"] = kwargs
        return _FakeClient(captured)

    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic)

    submit_batch(_write_prompts(tmp_path), api_key="sk-ant-byok", manifest_path=tmp_path / "m.json")

    assert captured["ctor_kwargs"] == {"api_key": "sk-ant-byok"}


def test_submit_batch_api_key_defaults_none(tmp_path, monkeypatch):
    """Default construction forwards api_key=None so the SDK reads ANTHROPIC_API_KEY (unchanged behavior)."""
    import anthropic

    from ddharmon.llm.batch import submit_batch

    captured: dict = {}

    def _fake_anthropic(*args, **kwargs):
        captured["ctor_kwargs"] = kwargs
        return _FakeClient(captured)

    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic)

    submit_batch(_write_prompts(tmp_path), manifest_path=tmp_path / "m.json")

    assert captured["ctor_kwargs"] == {"api_key": None}


def test_resume_and_wait_forwards_api_key(tmp_path, monkeypatch):
    """resume_and_wait forwards the BYOK api_key to submit_and_wait (no prior responses -> full run)."""
    from ddharmon.llm import batch as batch_mod

    captured: dict = {}

    def _fake_submit_and_wait(prompts_path, output_path, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(batch_mod, "submit_and_wait", _fake_submit_and_wait)

    # output_path does not exist -> resume_and_wait delegates the full pass to submit_and_wait
    batch_mod.resume_and_wait(_write_prompts(tmp_path), tmp_path / "responses.jsonl", api_key="sk-ant-byok")

    assert captured.get("api_key") == "sk-ant-byok"


# --- base_url opt-in proxy passthrough ---
# base_url threads to anthropic.Anthropic so the leanb Batch path can route through the
# self-hosted proxy's /anthropic passthrough. Default None = direct Anthropic, byte-for-byte
# (the no-proxy path must construct the client exactly as before).

_PROXY_URL = "http://localhost:4000/anthropic"


def test_submit_batch_forwards_base_url(tmp_path, monkeypatch):
    """A caller-supplied base_url is passed to anthropic.Anthropic (proxy passthrough route)."""
    import anthropic

    from ddharmon.llm.batch import submit_batch

    captured: dict = {}

    def _fake_anthropic(*args, **kwargs):
        captured["ctor_kwargs"] = kwargs
        return _FakeClient(captured)

    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic)

    submit_batch(_write_prompts(tmp_path), base_url=_PROXY_URL, manifest_path=tmp_path / "m.json")

    assert captured["ctor_kwargs"] == {"api_key": None, "base_url": _PROXY_URL}
    # request shape unchanged by the transport hook
    assert captured["requests"][0]["params"]["temperature"] == 0.0
    assert captured["requests"][0]["custom_id"] == "cluster_1"


def test_submit_batch_base_url_defaults_none(tmp_path, monkeypatch):
    """Default construction adds NO base_url kwarg — direct-Anthropic behavior is byte-for-byte."""
    import anthropic

    from ddharmon.llm.batch import submit_batch

    captured: dict = {}

    def _fake_anthropic(*args, **kwargs):
        captured["ctor_kwargs"] = kwargs
        return _FakeClient(captured)

    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic)

    submit_batch(_write_prompts(tmp_path), manifest_path=tmp_path / "m.json")

    assert "base_url" not in captured["ctor_kwargs"]
    assert captured["ctor_kwargs"] == {"api_key": None}


class _PollBatch:
    def __init__(self, status: str) -> None:
        self.processing_status = status
        self.request_counts = type(
            "C", (), {"succeeded": 0, "processing": 1, "errored": 0, "expired": 0, "canceled": 0}
        )()


class _PollBatches:
    def __init__(self, status: str) -> None:
        self._status = status

    def retrieve(self, _batch_id):
        return _PollBatch(self._status)


class _PollClient:
    def __init__(self, status: str) -> None:
        self.messages = type("M", (), {"batches": _PollBatches(status)})()


def test_retrieve_batch_forwards_base_url(tmp_path, monkeypatch):
    """retrieve_batch threads base_url to its client (status != ended -> early 0 return, no streaming)."""
    import anthropic

    from ddharmon.llm.batch import retrieve_batch

    captured: dict = {}

    def _fake_anthropic(*args, **kwargs):
        captured["ctor_kwargs"] = kwargs
        return _PollClient("in_progress")

    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic)

    n = retrieve_batch("batch_x", tmp_path / "out.jsonl", base_url=_PROXY_URL)

    assert n == 0
    assert captured["ctor_kwargs"]["base_url"] == _PROXY_URL


def test_submit_and_wait_forwards_base_url(tmp_path, monkeypatch):
    """submit_and_wait threads base_url into submit_batch, the poll-loop client, and retrieve_batch."""
    import anthropic

    from ddharmon.llm import batch as batch_mod

    captured: dict = {}

    def _fake_submit_batch(prompts_path, **kwargs):
        captured["submit"] = kwargs
        return "batch_sw"

    def _fake_retrieve_batch(batch_id, output_path, **kwargs):
        captured["retrieve"] = kwargs
        return 3

    def _fake_anthropic(*args, **kwargs):
        captured["poll_ctor"] = kwargs
        return _PollClient("ended")

    monkeypatch.setattr(batch_mod, "submit_batch", _fake_submit_batch)
    monkeypatch.setattr(batch_mod, "retrieve_batch", _fake_retrieve_batch)
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic)

    n = batch_mod.submit_and_wait(_write_prompts(tmp_path), tmp_path / "out.jsonl", base_url=_PROXY_URL)

    assert n == 3
    assert captured["submit"]["base_url"] == _PROXY_URL
    assert captured["retrieve"]["base_url"] == _PROXY_URL
    assert captured["poll_ctor"]["base_url"] == _PROXY_URL


def test_resume_and_wait_forwards_base_url(tmp_path, monkeypatch):
    """resume_and_wait forwards base_url to submit_and_wait (no prior responses -> full run)."""
    from ddharmon.llm import batch as batch_mod

    captured: dict = {}

    def _fake_submit_and_wait(prompts_path, output_path, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(batch_mod, "submit_and_wait", _fake_submit_and_wait)

    batch_mod.resume_and_wait(_write_prompts(tmp_path), tmp_path / "responses.jsonl", base_url=_PROXY_URL)

    assert captured.get("base_url") == _PROXY_URL


# --- forced tool calls: a prompt that sets ``tool_schema`` is submitted and read back as a tool call ---


_TOOL_SCHEMA = {"type": "object", "properties": {"groups": {"type": "array"}}, "required": ["groups"]}


def _write_tool_prompts(tmp_path: Path) -> Path:
    p = tmp_path / "tool_prompts.jsonl"
    rows = [
        {
            "id": "split:1",
            "system_prompt": "sys",
            "user_prompt": "hi",
            "schema": '{"groups": []}',
            "model_tag": "claude-sonnet-4-6",
            "tool_schema": _TOOL_SCHEMA,
            "tool_name": "emit_groups",
            "max_tokens": 4096,
        },
        {
            "id": "plain:1",
            "system_prompt": "sys",
            "user_prompt": "hi",
            "schema": "{}",
            "model_tag": "claude-sonnet-4-6",
        },
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_submit_batch_issues_a_forced_tool_call_for_a_tool_schema_record(tmp_path, monkeypatch):
    """A record with ``tool_schema`` is sent as a forced tool call with the bare system prompt; others are not."""
    import anthropic

    from ddharmon.llm.batch import submit_batch

    captured: dict = {}
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: _FakeClient(captured))

    submit_batch(_write_tool_prompts(tmp_path), manifest_path=tmp_path / "manifest.json")

    tool_req, plain_req = (r["params"] for r in captured["requests"])
    assert tool_req["system"] == "sys"  # no soft text schema appended: the tool enforces the shape
    assert tool_req["tools"] == [
        {
            "name": "emit_groups",
            "description": "Return the result as structured input conforming to the schema.",
            "input_schema": _TOOL_SCHEMA,
        }
    ]
    assert tool_req["tool_choice"] == {"type": "tool", "name": "emit_groups"}
    assert tool_req["max_tokens"] == 4096
    # A record without tool_schema keeps the text-preamble path, byte-for-byte as before.
    assert "tools" not in plain_req and "tool_choice" not in plain_req
    assert plain_req["system"].startswith("sys") and plain_req["system"].endswith("{}")


def test_retrieve_batch_writes_a_tool_use_blocks_input_as_the_response(tmp_path, monkeypatch):
    """A forced tool call's answer is the ToolUseBlock's structured ``input`` — no JSON parsing involved."""
    import anthropic
    from anthropic.types import TextBlock, ToolUseBlock

    from ddharmon.llm.batch import retrieve_batch

    def _message(content):
        return type("Msg", (), {"content": content, "usage": None, "model": "claude-sonnet-4-6"})()

    def _result(custom_id, content):
        inner = type("R", (), {"type": "succeeded", "message": _message(content)})()
        return type("Res", (), {"custom_id": custom_id, "result": inner})()

    results = [
        _result("split_1", [ToolUseBlock(id="t1", type="tool_use", name="emit_groups", input={"groups": [1, 2]})]),
        _result("plain_1", [TextBlock(type="text", text='{"verdict": "adopt"}')]),
    ]

    class _Batches:
        def retrieve(self, _batch_id):
            counts = type("C", (), {"succeeded": 2, "processing": 0, "errored": 0, "expired": 0, "canceled": 0})()
            return type("B", (), {"processing_status": "ended", "request_counts": counts})()

        def results(self, _batch_id):
            return iter(results)

    client = type("Client", (), {"messages": type("M", (), {"batches": _Batches()})()})()
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: client)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"id_map": {"split_1": "split:1", "plain_1": "plain:1"}}))

    out = tmp_path / "out.jsonl"
    assert retrieve_batch("batch_x", out, manifest_path=manifest) == 2

    rows = {r["id"]: r["response"] for r in map(json.loads, out.read_text().splitlines())}
    assert rows == {"split:1": {"groups": [1, 2]}, "plain:1": {"verdict": "adopt"}}
