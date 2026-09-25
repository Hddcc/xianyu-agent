"""Price refresh and bargaining rules through the real registry and tools."""
from decimal import Decimal
from types import SimpleNamespace

import pytest

from xianyu_agent import llm
from xianyu_agent.deadline import Deadline
from xianyu_agent.kf_tools import build_item_description, kf_tools
from xianyu_agent.pricing import BargainPolicy
from xianyu_agent.registry import load_config
from xianyu_agent.tools import ToolContext
from xianyu_agent.types import ToolResultBlock

from .integration_support import incoming, make_registry


def tool(name):
    return next(spec for spec in kf_tools() if spec.name == name)


async def test_item_price_is_refreshed_for_each_message(tmp_path, monkeypatch):
    registry, _ = make_registry(tmp_path, monkeypatch)
    await registry.store.save_item_info("item-1", {"soldPrice": 100})
    platform = {"soldPrice": 1000}
    calls, prompts = [], []

    def fetch(item_id):
        calls.append(item_id)
        return {"data": {"itemDO": dict(platform)}}

    async def stream(ctx, **kwargs):
        prompts.append(ctx.system_prompt)
        yield {"type": "text_delta", "delta": "已了解"}
        yield {"type": "done", "stopReason": "end_turn"}

    registry.api = SimpleNamespace(get_item_info=fetch)
    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "在吗", "item-1"))
    platform["soldPrice"] = 2000
    await registry.handle(incoming("c1", "现在呢", "item-1"))
    assert calls == ["item-1", "item-1"]
    assert "1000" in prompts[0] and "2000" in prompts[1]
    assert (await registry.store.get_item_info("item-1"))["soldPrice"] == 2000


@pytest.mark.parametrize("failure", [
    "exception", "missing_data", "invalid_price", "null_response", "null_data", "invalid_item",
])
async def test_failed_refresh_does_not_use_cached_price(tmp_path, monkeypatch, failure):
    registry, sent = make_registry(tmp_path, monkeypatch)
    await registry.store.save_item_info("item-1", {"soldPrice": 100})

    def fetch(item_id):
        if failure == "exception":
            raise RuntimeError("offline")
        if failure == "invalid_price":
            return {"data": {"itemDO": {"soldPrice": "not-a-price"}}}
        if failure == "null_response":
            return None
        if failure == "null_data":
            return {"data": None}
        if failure == "invalid_item":
            return {"data": {"itemDO": "invalid"}}
        return {}

    registry.api = SimpleNamespace(get_item_info=fetch)

    async def stream(ctx, **kwargs):
        assert "暂停报价" in ctx.system_prompt
        if any(m.role == "tool" for m in ctx.messages):
            yield {"type": "text_delta", "delta": "95元可以"}
            yield {"type": "done", "stopReason": "end_turn"}
        else:
            yield {"type": "tool_call", "id": "quote", "name": "quote_price",
                   "args": {"price": 95}}
            yield {"type": "done", "stopReason": "tool_use"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "能便宜吗", "item-1"))
    assert sent == [("c1", "buyer-c1", "价格需要卖家确认，请稍等")]
    tctx = ToolContext(store=registry.store, item_id="item-1")
    assert (await tool("get_item_info").execute({}, tctx)).startswith("error:")


@pytest.mark.parametrize("intent", ["price", "tech", "default"])
async def test_old_offer_is_blocked_after_repricing(tmp_path, monkeypatch, intent):
    registry, sent = make_registry(tmp_path, monkeypatch)
    registry.router.classify_llm = lambda *args: intent
    registry.api = SimpleNamespace(
        get_item_info=lambda _: {"data": {"itemDO": {"soldPrice": 1000}}})
    await registry.store.save_item_info("item-1", {"soldPrice": 100})
    await registry.store.add_message("c1", "seller", "item-1", "assistant", "95元可以")

    async def stream(ctx, **kwargs):
        assert "1000" in ctx.system_prompt and "950" in ctx.system_prompt
        assert "历史" in ctx.system_prompt
        yield {"type": "text_delta", "delta": "95元包邮可以吗"}
        yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "80出吗", "item-1"))
    assert "95元包邮" not in sent[0][2]
    assert (await registry.store.get_context_by_chat("c1"))[-1]["content"] == sent[0][2]


async def test_structured_quote_controls_reply_and_cumulative_discount(tmp_path, monkeypatch):
    registry, sent = make_registry(tmp_path, monkeypatch)
    registry.router.classify_llm = lambda *args: "price"
    registry.api = SimpleNamespace(
        get_item_info=lambda _: {"data": {"itemDO": {"soldPrice": 1000}}})
    proposed = [950, 925]
    feedback = []

    async def stream(ctx, **kwargs):
        results = [b for m in ctx.messages if isinstance(m.content, list)
                   for b in m.content if isinstance(b, ToolResultBlock)]
        if not results:
            yield {"type": "tool_call", "id": "quote-1", "name": "quote_price",
                   "args": {"price": proposed.pop(0)}}
            yield {"type": "done", "stopReason": "tool_use"}
        else:
            feedback.append(results[-1].content)
            yield {"type": "text_delta", "delta": "80元成交，已经改好了"}
            yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "能优惠吗", "item-1"))
    await registry.handle(incoming("c1", "还能少点吗", "item-1"))
    assert sent == [("c1", "buyer-c1", "可以 950 元，您看怎么样"),
                    ("c1", "buyer-c1", "可以 925 元，您看怎么样")]
    assert all(not result.startswith("error:") for result in feedback)


@pytest.mark.parametrize("reply", ["95元可以", "九十五元包邮", "95包邮", "可以打八折"])
async def test_unverified_monetary_replies_are_blocked(tmp_path, monkeypatch, reply):
    registry, sent = make_registry(tmp_path, monkeypatch)
    registry.api = SimpleNamespace(
        get_item_info=lambda _: {"data": {"itemDO": {"soldPrice": 1000}}})

    async def stream(ctx, **kwargs):
        yield {"type": "text_delta", "delta": reply}
        yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "看看实物", "item-1"))
    assert sent[0][2] != reply


async def test_technical_numbers_are_not_treated_as_quotes(tmp_path, monkeypatch):
    registry, sent = make_registry(tmp_path, monkeypatch)

    async def stream(ctx, **kwargs):
        yield {"type": "text_delta", "delta": "支持蓝牙5.0，续航8小时，电量80%"}
        yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "续航多久"))
    assert sent[0][2] == "支持蓝牙5.0，续航8小时，电量80%"


@pytest.mark.parametrize("settings,accepted,rejected", [
    ({}, 900, 899.99),
    ({"BARGAIN_ENABLED": "False"}, 1000, 999),
    ({"BARGAIN_MAX_DISCOUNT_PERCENT": "5"}, 950, 949),
    ({"BARGAIN_MAX_DISCOUNT_PERCENT": "20"}, 800, 799.99),
    ({"BARGAIN_MAX_DISCOUNT_PERCENT": "0"}, 1000, 999.99),
    ({"BARGAIN_MAX_DISCOUNT_PERCENT": "100"}, 0.01, 0),
])
async def test_price_policy_bounds(monkeypatch, settings, accepted, rejected):
    for name in ("BARGAIN_ENABLED", "BARGAIN_MIN_PRICE", "BARGAIN_MAX_DISCOUNT_PERCENT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    policy = load_config().bargain_policy
    tctx = ToolContext(item=build_item_description({"soldPrice": 1000}), bargain_policy=policy)
    result = await tool("quote_price").execute({"price": accepted}, tctx)
    assert not result.startswith("error:")
    assert str(accepted) in tctx.price_reply
    result = await tool("quote_price").execute({"price": rejected}, tctx)
    assert result.startswith("error:")
    assert str(rejected) not in tctx.price_reply


async def test_percent_policy_uses_each_items_latest_price(tmp_path, monkeypatch):
    monkeypatch.setenv("BARGAIN_ENABLED", "True")
    monkeypatch.setenv("BARGAIN_MAX_DISCOUNT_PERCENT", "10")
    monkeypatch.setenv("BARGAIN_MIN_PRICE", "950")
    registry, sent = make_registry(tmp_path, monkeypatch,
                                   bargain_policy=load_config().bargain_policy)
    registry.router.classify_llm = lambda *args: "price"
    prices = {"cheap": 100, "expensive": 1000}
    registry.api = SimpleNamespace(
        get_item_info=lambda item_id: {"data": {"itemDO": {"soldPrice": prices[item_id]}}})
    offers = iter([95, 950, 185, 180])
    feedback = []

    async def stream(ctx, **kwargs):
        results = [b for m in ctx.messages if isinstance(m.content, list)
                   for b in m.content if isinstance(b, ToolResultBlock)]
        if not results:
            yield {"type": "tool_call", "id": "quote", "name": "quote_price",
                   "args": {"price": next(offers)}}
            yield {"type": "done", "stopReason": "tool_use"}
        else:
            feedback.append(results[-1].content)
            yield {"type": "text_delta", "delta": "已了解"}
            yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "便宜点", "cheap"))
    await registry.handle(incoming("c2", "便宜点", "expensive"))
    prices["cheap"] = 200
    await registry.handle(incoming("c1", "现在呢", "cheap"))
    await registry.handle(incoming("c1", "再少点", "cheap"))
    assert [entry[2] for entry in sent] == [
        "可以 95 元，您看怎么样", "可以 950 元，您看怎么样",
        "可以 185 元，您看怎么样", "可以 180 元，您看怎么样",
    ]
    assert feedback[0].startswith("已验证报价")
    assert feedback[1].startswith("已验证报价")
    assert feedback[2].startswith("已验证报价")
    assert feedback[3].startswith("已验证报价")


async def test_price_rounding_does_not_exceed_discount_limit():
    tctx = ToolContext(item=build_item_description({"soldPrice": "33.33"}),
                       bargain_policy=BargainPolicy())
    assert tctx.bargain_policy.minimum(tctx.item) == Decimal("30.00")
    assert (await tool("quote_price").execute({"price": 29.99}, tctx)).startswith("error:")
    assert not (await tool("quote_price").execute({"price": 30}, tctx)).startswith("error:")


def test_progressive_discount_uses_current_bargain_round():
    policy = BargainPolicy()
    item = build_item_description({"soldPrice": 1000})
    assert policy.minimum(item, 0) == Decimal("950.00")
    assert policy.minimum(item, 1) == Decimal("925.00")
    assert policy.minimum(item, 2) == Decimal("900.00")
    assert policy.minimum(item, 5) == Decimal("900.00")


async def test_numeric_price_message_requires_quote_tool(tmp_path, monkeypatch):
    registry, sent = make_registry(tmp_path, monkeypatch)
    registry.router.classify_llm = lambda *args: "price"
    registry.api = SimpleNamespace(
        get_item_info=lambda _: {"data": {"itemDO": {"soldPrice": 1000}}})
    choices = []

    async def stream(ctx, **kwargs):
        choices.append(kwargs.get("tool_choice"))
        yield {"type": "text_delta", "delta": "500元可以"}
        yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "500出吗", "item-1"))

    assert choices[0] == {"type": "function", "function": {"name": "quote_price"}}
    assert sent == [("c1", "buyer-c1", "目前最低 950.00 元，您看可以吗")]


async def test_over_limit_quote_returns_progressive_counter_offer(tmp_path, monkeypatch):
    registry, sent = make_registry(tmp_path, monkeypatch)
    registry.router.classify_llm = lambda *args: "price"
    registry.api = SimpleNamespace(
        get_item_info=lambda _: {"data": {"itemDO": {"soldPrice": 1000}}})
    feedback = []

    async def stream(ctx, **kwargs):
        results = [b for m in ctx.messages if isinstance(m.content, list)
                   for b in m.content if isinstance(b, ToolResultBlock)]
        if not results:
            yield {"type": "tool_call", "id": "quote", "name": "quote_price",
                   "args": {"price": 800}}
            yield {"type": "done", "stopReason": "tool_use"}
        else:
            feedback.append(results[-1].content)
            yield {"type": "text_delta", "delta": "这个价格可以吗"}
            yield {"type": "done", "stopReason": "end_turn"}

    monkeypatch.setattr(llm, "stream", stream)
    await registry.handle(incoming("c1", "800出吗", "item-1"))

    assert feedback[0].startswith("error:")
    assert sent == [("c1", "buyer-c1", "目前最低 950.00 元，您看可以吗")]


@pytest.mark.parametrize("price", [None, 0, -1, "NaN", "Infinity", "invalid"])
async def test_unknown_or_invalid_price_cannot_be_quoted(price):
    tctx = ToolContext(item=build_item_description({"soldPrice": price}))
    assert (await tool("quote_price").execute({"price": 95}, tctx)).startswith("error:")


async def test_different_sku_prices_require_seller_confirmation():
    tctx = ToolContext(item=build_item_description({"soldPrice": 1000, "skuList": [
        {"price": 100000}, {"price": 200000},
    ]}))
    assert (await tool("quote_price").execute({"price": 950}, tctx)).startswith("error:")


@pytest.mark.parametrize("price", [None, 95, 1001, "NaN", True])
async def test_unsafe_price_notification_is_not_delivered(price):
    delivered = []

    async def notify(kind, detail):
        delivered.append(detail)
        return "已通知卖家"

    tctx = ToolContext(item=build_item_description({"soldPrice": 1000}), notify=notify)
    result = await tool("notify_seller").execute(
        {"kind": "price", "price": price, "detail": "改成95元"}, tctx)
    assert result.startswith("error:")
    assert delivered == []


async def test_price_notification_uses_validated_amount():
    delivered = []

    async def notify(kind, detail):
        delivered.append(detail)
        return "已通知卖家"

    tctx = ToolContext(item=build_item_description({"soldPrice": 1000}), notify=notify)
    result = await tool("notify_seller").execute(
        {"kind": "price", "price": 950, "detail": "改成95元"}, tctx)
    assert not result.startswith("error:")
    assert "950" in delivered[0] and "95元" not in delivered[0]
    assert "卖家" in tctx.price_reply


@pytest.mark.parametrize("price", [900.001, 1001, True, "Infinity", None])
async def test_failed_quote_replaces_previous_reply(price):
    tctx = ToolContext(item=build_item_description({"soldPrice": 1000}))
    assert not (await tool("quote_price").execute({"price": 950}, tctx)).startswith("error:")
    assert (await tool("quote_price").execute({"price": price}, tctx)).startswith("error:")
    assert tctx.price_reply == "价格需要卖家确认，请稍等"


async def test_expired_quote_replaces_previous_reply():
    tctx = ToolContext(item=build_item_description({"soldPrice": 1000}),
                       deadline=Deadline(seconds=-1), price_reply="可以 950 元")
    assert (await tool("quote_price").execute({"price": 900}, tctx)).startswith("error:")
    assert tctx.price_reply == "价格需要卖家确认，请稍等"


async def test_price_notification_without_capability_replaces_previous_reply():
    tctx = ToolContext(item=build_item_description({"soldPrice": 1000}),
                       price_reply="可以 950 元")
    result = await tool("notify_seller").execute({"kind": "price", "price": 950}, tctx)
    assert result.startswith("error:")
    assert tctx.price_reply == "改价需要卖家确认，请稍等"


@pytest.mark.parametrize("name,value", [
    ("BARGAIN_ENABLED", "typo"),
    ("BARGAIN_MAX_DISCOUNT_PERCENT", "101"), ("BARGAIN_MAX_DISCOUNT_PERCENT", "-1"),
    ("BARGAIN_MAX_DISCOUNT_PERCENT", "NaN"),
    ("BARGAIN_MAX_DISCOUNT_PERCENT", "invalid"),
])
def test_invalid_policy_configuration_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_config()
