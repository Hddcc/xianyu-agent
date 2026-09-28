"""Session concurrency and takeover through the public registry interface."""
import asyncio
from collections import defaultdict

import pytest

from xianyu_agent import llm
from xianyu_agent.session import load_latest, session_path
from xianyu_agent.telemetry import MemoryEventSink
from xianyu_agent.types import ToolResultBlock, ToolUseBlock

from .integration_support import incoming, make_registry


async def test_run_events_share_one_id_and_capture_final_decisions(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    registry, sent = make_registry(tmp_path, monkeypatch, event_sink=sink)

    async def stream(ctx, **kwargs):
        yield {"type": "text_delta", "delta": "正常回复"}
        yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    run_id = await registry.handle(incoming("c1", "在吗"))
    events = sink.for_run(run_id)

    assert [event["type"] for event in events] == [
        "run_started", "route_decided", "agent_started",
        "model_call_started", "model_call_finished", "agent_finished",
        "reply_finalized", "message_delivery", "run_finished",
    ]
    assert events[1]["source"] == "model"
    assert events[6]["fallback_used"] is False
    assert events[7]["status"] == "sent"
    assert events[-1]["outcome"] == "sent"
    assert sent == [("c1", "buyer-c1", "正常回复")]


async def test_event_sink_failure_does_not_break_reply(tmp_path, monkeypatch):
    def broken_sink(event):
        raise OSError("disk unavailable")

    registry, sent = make_registry(tmp_path, monkeypatch, event_sink=broken_sink)

    async def stream(ctx, **kwargs):
        yield {"type": "text_delta", "delta": "仍然回复"}
        yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "在吗"))
    assert sent == [("c1", "buyer-c1", "仍然回复")]


async def test_sender_failure_is_recorded_and_propagated(tmp_path, monkeypatch):
    sink = MemoryEventSink()
    registry, _ = make_registry(tmp_path, monkeypatch, event_sink=sink)

    async def broken_sender(*args):
        raise ConnectionError("offline")

    async def stream(ctx, **kwargs):
        yield {"type": "text_delta", "delta": "待发送"}
        yield {"type": "done", "stopReason": "end_turn"}

    registry.set_sender(broken_sender)
    monkeypatch.setattr(llm, "stream", stream)
    with pytest.raises(ConnectionError):
        await registry.handle(incoming("c1", "在吗"))

    deliveries = [event for event in sink.events if event["type"] == "message_delivery"]
    assert deliveries[0]["status"] == "error"
    assert deliveries[0]["error"] == "ConnectionError"
    assert sink.events[-1]["type"] == "run_finished"
    assert sink.events[-1]["outcome"] == "error"


async def test_incomplete_idle_profile_preserves_previous_state(tmp_path, monkeypatch):
    from xianyu_agent import registry as registry_module

    sink = MemoryEventSink()
    registry, sent = make_registry(
        tmp_path, monkeypatch, event_sink=sink, idle_compact_hours=0.001)
    await registry.store.add_message("c1", "buyer-c1", "", "user", "旧消息")
    monkeypatch.setattr(registry_module.time, "time", lambda: 10**12)

    async def stream(ctx, **kwargs):
        if ctx.system_prompt == registry_module.SUMMARY_PROMPT:
            yield {"type": "text_delta", "delta": "不完整画像"}
            yield {"type": "done", "stopReason": "max_tokens"}
            return
        yield {"type": "text_delta", "delta": "正常回复"}
        yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    run_id = await registry.handle(incoming("c1", "新消息"))

    profile, _ = await registry.store.get_profile("c1")
    assert profile is None
    profile_events = [event for event in sink.for_run(run_id)
                      if event["type"] == "idle_profile_finished"]
    assert profile_events[0]["status"] == "preserved"
    assert profile_events[0]["stop_reason"] == "max_tokens"
    assert sent == [("c1", "buyer-c1", "正常回复")]


@pytest.mark.parametrize("sessions", [1, 5, 10, 20])
async def test_sessions_overlap_and_each_session_replies_in_order(
        tmp_path, monkeypatch, sessions):
    registry, sent = make_registry(tmp_path, monkeypatch)
    active = set()
    entered = set()
    all_entered = asyncio.Event()
    accepted = defaultdict(list)

    async def stream(ctx, **kwargs):
        chat_id = ctx.meta["chat_id"]
        assert chat_id not in active, "One session must not run two messages at once"
        active.add(chat_id)
        text = ctx.messages[-1].content
        accepted[chat_id].append(text)
        entered.add(chat_id)
        if len(entered) == sessions:
            all_entered.set()
        try:
            await asyncio.wait_for(all_entered.wait(), timeout=10)
            yield {"type": "text_delta", "delta": f"reply:{text}"}
            yield {"type": "done", "stopReason": "end_turn"}
        finally:
            active.remove(chat_id)

    monkeypatch.setattr(llm, "stream", stream)
    tasks = [asyncio.create_task(registry.handle(incoming(f"c{s}", f"message-{m}")))
             for m in range(5) for s in range(sessions)]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    expected = [f"message-{m}" for m in range(5)]
    assert len(sent) == sessions * 5
    for s in range(sessions):
        chat_id = f"c{s}"
        assert accepted[chat_id] == expected
        replies = [(buyer, text) for cid, buyer, text in sent if cid == chat_id]
        assert replies == [(f"buyer-{chat_id}", f"reply:{text}") for text in expected]
        history = await registry.store.get_context_by_chat(chat_id)
        assert [m["content"] for m in history if m["role"] == "assistant"] == [
            f"reply:{text}" for text in expected]


async def test_takeover_while_generating_suppresses_reply_and_allows_resumption(
        tmp_path, monkeypatch):
    registry, sent = make_registry(tmp_path, monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def stream(ctx, signal=None, **kwargs):
        if ctx.messages[-1].content == "first":
            yield {"type": "text_delta", "delta": "obsolete partial reply"}
            started.set()
            await asyncio.wait_for(release.wait(), timeout=10)
            yield {"type": "done", "stopReason":
                   "aborted" if signal.cancelled else "end_turn"}
        else:
            yield {"type": "text_delta", "delta": "resumed reply"}
            yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    task = asyncio.create_task(registry.handle(incoming("c1", "first")))
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        registry.abort("c1")
        release.set()
        await asyncio.wait_for(task, timeout=10)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert sent == []
    assert [m["role"] for m in await registry.store.get_context_by_chat("c1")] == ["user"]
    await registry.record_manual_reply(incoming("c1", "seller handled it"), "assistant")
    await registry.handle(incoming("c1", "second"))
    assert sent == [("c1", "buyer-c1", "resumed reply")]
    assert any(m["content"] == "seller handled it"
               for m in await registry.store.get_context_by_chat("c1"))


async def test_takeover_during_tool_execution_stops_remaining_tools_and_pairs_results(
        tmp_path, monkeypatch):
    from xianyu_agent import notify

    registry, sent = make_registry(tmp_path, monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()
    deliveries = []
    loop = asyncio.get_running_loop()

    def deliver(title, content):
        deliveries.append((title, content))
        loop.call_soon_threadsafe(entered.set)
        # The external notification is synchronous, just as in production.
        import threading
        done = threading.Event()
        loop.call_soon_threadsafe(lambda: asyncio.create_task(wait_release(done)))
        assert done.wait(10), "Takeover test did not release the notification"
        return "Notification delivered"

    async def wait_release(done):
        await release.wait()
        done.set()

    async def stream(ctx, **kwargs):
        yield {"type": "text_delta", "delta": "obsolete tool reply"}
        for tool_id, kind in [("first", "image"), ("second", "price")]:
            yield {"type": "tool_call", "id": tool_id, "name": "notify_seller",
                   "args": {"kind": kind, "detail": "test detail"}}
        yield {"type": "done", "stopReason": "tool_use"}

    monkeypatch.setattr(llm, "stream", stream)
    monkeypatch.setattr(notify, "send_notification", deliver)
    task = asyncio.create_task(registry.handle(incoming("c1", "show photo")))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        registry.abort("c1")
        release.set()
        await asyncio.wait_for(task, timeout=10)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert sent == []
    assert len(deliveries) == 1
    ctx = load_latest(session_path("c1"))
    calls, results = [], []
    for message in ctx.messages:
        if isinstance(message.content, list):
            calls.extend(b.id for b in message.content if isinstance(b, ToolUseBlock))
            results.extend(b for b in message.content if isinstance(b, ToolResultBlock))
    assert calls == ["first", "second"]
    assert [r.tool_use_id for r in results] == calls
    assert results[1].is_error and results[1].content == "error: aborted"


async def test_aborting_one_session_does_not_cancel_another(tmp_path, monkeypatch):
    registry, sent = make_registry(tmp_path, monkeypatch)
    entered = set()
    ready = asyncio.Event()
    release = asyncio.Event()

    async def stream(ctx, signal=None, **kwargs):
        entered.add(ctx.meta["chat_id"])
        if len(entered) == 2:
            ready.set()
        await asyncio.wait_for(release.wait(), timeout=10)
        yield {"type": "text_delta", "delta": "valid reply"}
        yield {"type": "done", "stopReason":
               "aborted" if signal.cancelled else "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    tasks = [asyncio.create_task(registry.handle(incoming(cid, "hello")))
             for cid in ("c1", "c2")]
    try:
        await asyncio.wait_for(ready.wait(), timeout=10)
        registry.abort("c1")
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert sent == [("c2", "buyer-c2", "valid reply")]


async def test_manual_recording_and_idle_abort_do_not_send_replies(tmp_path, monkeypatch):
    registry, sent = make_registry(tmp_path, monkeypatch)
    registry.abort("missing")
    await registry.record_manual_reply(incoming("c1", "buyer question"), "user")
    await registry.record_manual_reply(incoming("c1", "seller answer"), "assistant")
    registry.abort("c1")
    assert sent == []
    assert await registry.store.get_context_by_chat("c1") == [
        {"role": "user", "content": "buyer question"},
        {"role": "assistant", "content": "seller answer"},
    ]
