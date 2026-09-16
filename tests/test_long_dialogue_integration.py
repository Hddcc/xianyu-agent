"""Long-dialogue storage, compression, fallback and snapshot behavior."""
import pytest

from benchmarks.scenario_cases import GOLD_SPEAKER_SUMMARY, QUOTE_CASES, long_dialogue
from xianyu_agent import llm
from xianyu_agent.agent import maybe_compact
from xianyu_agent.cancel import CancellationToken
from xianyu_agent.kf_tools import kf_tools
from xianyu_agent.session import append_snapshot, load_latest
from xianyu_agent.store import Store
from xianyu_agent.tools import ToolContext
from xianyu_agent.types import (
    Context, Message, ToolResultBlock, ToolUseBlock, context_to_dict,
)


@pytest.mark.parametrize("rounds", [30, 40, 60])
async def test_compression_preserves_recent_messages_archive_and_snapshot(
        tmp_path, monkeypatch, rounds):
    ctx = long_dialogue(QUOTE_CASES[0], rounds)
    original = context_to_dict(ctx)
    recent = list(ctx.messages[-20:])
    store = Store(str(tmp_path / "history.db"), max_history=1000)
    for message in ctx.messages:
        await store.add_message("c1", message.role, "i1", message.role, message.content)

    async def stream(sub, **kwargs):
        assert "第1轮议价，买家出价300元" in sub.messages[0].content
        assert "第4轮议价，卖家报价348元" in sub.messages[0].content
        yield {"type": "text_delta", "delta": GOLD_SPEAKER_SUMMARY}
        yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    assert await maybe_compact(ctx, CancellationToken())
    assert len(ctx.messages) == 21
    assert ctx.messages[0].content == f"[会话画像]\n{GOLD_SPEAKER_SUMMARY}"
    assert ctx.messages[1:] == recent
    assert ctx.system_prompt == original["system_prompt"]
    assert ctx.meta == original["meta"]
    history = await store.get_context_by_chat("c1")
    assert len(history) == rounds * 2
    assert history[0]["content"] == "第1轮议价，买家出价300元。"
    tool = next(tool for tool in kf_tools() if tool.name == "read_earlier_history")
    earlier = await tool.execute({"limit": 100}, ToolContext(chat_id="c1", store=store))
    if rounds <= 40:
        assert "第1轮议价，买家出价300元" in earlier
    path = tmp_path / "session.jsonl"
    append_snapshot(path, ctx)
    restored = load_latest(path)
    assert context_to_dict(restored) == context_to_dict(ctx)


async def test_compression_does_not_split_call_and_result_at_recent_boundary(monkeypatch):
    ctx = Context("system", [Message("user", f"old-{i}") for i in range(30)])
    ctx.messages.extend([
        Message("assistant", [ToolUseBlock(id="call-1", name="get_item_info", input={})]),
        Message("user", [ToolResultBlock(tool_use_id="call-1", content="known item")]),
        *[Message("user", f"recent-{i}") for i in range(19)],
    ])

    async def stream(sub, **kwargs):
        yield {"type": "text_delta", "delta": "Earlier conversation summary"}
        yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    assert await maybe_compact(ctx, CancellationToken())
    provider = llm.context_to_provider_messages(ctx)
    calls = [message for message in provider if message.get("tool_calls")]
    results = [message for message in provider if message["role"] == "tool"]
    assert len(calls) == len(results) == 1
    assert calls[0]["tool_calls"][0]["id"] == results[0]["tool_call_id"] == "call-1"
    assert provider.index(results[0]) == provider.index(calls[0]) + 1


async def test_short_dialogue_is_not_compressed(monkeypatch):
    ctx = Context("system", [Message("user", "short") for _ in range(49)])
    before = context_to_dict(ctx)

    async def stream(*args, **kwargs):
        raise AssertionError("A short conversation must not request a summary")
        yield

    monkeypatch.setattr(llm, "stream", stream)
    assert await maybe_compact(ctx, CancellationToken()) is False
    assert context_to_dict(ctx) == before


@pytest.mark.parametrize("stop", ["error", "max_tokens", "aborted", "end_turn"])
async def test_unsuccessful_or_empty_summary_keeps_original_history(monkeypatch, stop):
    ctx = long_dialogue(QUOTE_CASES[0])
    before = context_to_dict(ctx)

    async def stream(*args, **kwargs):
        if stop != "end_turn":
            yield {"type": "text_delta", "delta": "incomplete summary"}
        yield {"type": "done", "stopReason": stop}

    monkeypatch.setattr(llm, "stream", stream)
    changed = await maybe_compact(ctx, CancellationToken())
    assert changed is False
    assert context_to_dict(ctx) == before


async def test_takeover_during_summary_keeps_original_history(monkeypatch):
    ctx = long_dialogue(QUOTE_CASES[0])
    before = context_to_dict(ctx)
    signal = CancellationToken()

    async def stream(*args, **kwargs):
        yield {"type": "text_delta", "delta": "partial summary"}
        signal.cancel()
        yield {"type": "done", "stopReason": "aborted"}

    monkeypatch.setattr(llm, "stream", stream)
    assert await maybe_compact(ctx, signal) is False
    assert context_to_dict(ctx) == before
