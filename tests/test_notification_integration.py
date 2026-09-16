"""Notification outcomes are tested independently from model reply quality."""
import pytest

from xianyu_agent import llm, notify
from xianyu_agent.cancel import CancellationToken
from xianyu_agent.deadline import Deadline
from xianyu_agent.kf_tools import kf_tools
from xianyu_agent.tools import ToolContext
from xianyu_agent.types import ToolResultBlock

from .integration_support import incoming, make_registry


@pytest.mark.parametrize("kind", ["image", "price"])
@pytest.mark.parametrize("outcome", ["success", "failure", "exception", "unconfigured"])
async def test_notification_result_reaches_model_and_preserves_recipient(
        tmp_path, monkeypatch, kind, outcome):
    registry, sent = make_registry(tmp_path, monkeypatch)
    deliveries = []
    feedback = []
    expected = {
        "success": "Notification delivered",
        "failure": "error: notification service unavailable",
        "exception": "error: RuntimeError: notification transport failed",
        "unconfigured": "error: 未配置通知渠道（NOTIFY_METHOD），无法通知卖家",
    }[outcome]

    def deliver(title, content):
        deliveries.append((title, content))
        if outcome == "exception":
            raise RuntimeError("notification transport failed")
        return expected

    if outcome == "unconfigured":
        monkeypatch.delenv("NOTIFY_METHOD", raising=False)
    else:
        monkeypatch.setattr(notify, "send_notification", deliver)

    async def stream(ctx, tools=None, **kwargs):
        results = [block for message in ctx.messages
                   if isinstance(message.content, list) for block in message.content
                   if isinstance(block, ToolResultBlock)]
        if results:
            feedback.extend(results)
            yield {"type": "text_delta", "delta": "Please wait for seller assistance"}
            yield {"type": "done", "stopReason": "end_turn"}
        else:
            assert {tool["name"] for tool in tools} == {
                "get_item_info", "get_bargain_status", "read_earlier_history",
                "web_search", "notify_seller",
            }
            yield {"type": "tool_call", "id": "notify-1", "name": "notify_seller",
                   "args": {"kind": kind, "detail": "agreed amount 90; photo angle front"}}
            yield {"type": "done", "stopReason": "tool_use"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "please assist", item_id="item-1"))
    assert [(result.tool_use_id, result.content.split("\n")[0]) for result in feedback] == [
        ("notify-1", expected)]
    assert sent == [("c1", "buyer-c1", "Please wait for seller assistance")]
    if outcome != "unconfigured":
        assert len(deliveries) == 1
        title, content = deliveries[0]
        assert ("买家要看实物图" if kind == "image" else "买家谈妥需要改价") in title
        assert "buyer-c1" in content and "item-1" in content
        assert "c1" in content and "agreed amount 90; photo angle front" in content


@pytest.mark.parametrize("kind", ["image", "price"])
async def test_expired_deadline_does_not_deliver_notification(kind):
    delivered = []

    async def deliver(kind, detail):
        delivered.append((kind, detail))
        return "Notification delivered"

    tctx = ToolContext(cancel=CancellationToken(), notify=deliver,
                       deadline=Deadline(seconds=-1))
    tool = next(tool for tool in kf_tools() if tool.name == "notify_seller")
    result = await tool.execute({"kind": kind, "detail": "test"}, tctx)
    assert result.startswith("error: 时限已到")
    assert delivered == []


@pytest.mark.parametrize("forbidden", ["send_image", "update_price"])
async def test_unavailable_image_and_price_tools_are_not_executable(
        tmp_path, monkeypatch, forbidden):
    registry, sent = make_registry(tmp_path, monkeypatch)
    delivered = []
    feedback = []
    monkeypatch.setattr(notify, "send_notification",
                        lambda title, content: delivered.append((title, content)) or "delivered")

    async def stream(ctx, tools=None, **kwargs):
        results = [block for message in ctx.messages
                   if isinstance(message.content, list) for block in message.content
                   if isinstance(block, ToolResultBlock)]
        if results:
            feedback.extend(results)
            yield {"type": "text_delta", "delta": "Seller assistance is required"}
            yield {"type": "done", "stopReason": "end_turn"}
        else:
            assert forbidden not in {tool["name"] for tool in tools}
            yield {"type": "tool_call", "id": "unsupported", "name": forbidden,
                   "args": {"price": 90, "url": "https://example.invalid/photo.png"}}
            yield {"type": "done", "stopReason": "tool_use"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "please assist"))
    assert len(feedback) == 1
    assert feedback[0].content == f'error: 没有名为 "{forbidden}" 的工具'
    assert delivered == []
    assert sent == [("c1", "buyer-c1", "Seller assistance is required")]
