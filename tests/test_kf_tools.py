"""领域工具测试：正常、异常、时限耗尽，不启动服务、不调真实接口。"""
from xianyu_agent.deadline import Deadline
from xianyu_agent.kf_tools import (build_item_description, format_price, kf_tools)
from xianyu_agent.tools import ToolContext


class FakeStore:
    def __init__(self, bargain_count=0, item=None, history=None):
        self.bargain_count = bargain_count
        self.item = item
        self.history = history or []

    async def get_bargain_count(self, chat_id):
        return self.bargain_count

    async def get_item_info(self, item_id):
        return self.item

    async def get_context_by_chat(self, chat_id, limit=None):
        return self.history[-(limit or 100):]


def make_tctx(**kwargs):
    defaults = dict(chat_id="c1", item_id="i1", store=FakeStore())
    defaults.update(kwargs)
    return ToolContext(**defaults)


def tool_by_name(name):
    return {t.name: t for t in kf_tools()}[name]


async def test_get_item_info_normal():
    item = {"title": "蓝牙音箱", "desc": "99新", "soldPrice": 199,
            "quantity": 3, "skuList": []}
    tctx = make_tctx(item={"title": "蓝牙音箱", "desc": "99新",
                           "price_range": "¥199", "total_stock": 3,
                           "sku_details": []})
    result = await tool_by_name("get_item_info").execute({}, tctx)
    assert "蓝牙音箱" in result and "¥199" in result


async def test_get_item_info_unavailable():
    tctx = make_tctx(item=None, store=FakeStore(item=None))
    result = await tool_by_name("get_item_info").execute({}, tctx)
    assert result.startswith("error:")


async def test_get_bargain_status():
    tctx = make_tctx(store=FakeStore(bargain_count=2),
                     item={"price_range": "¥399"},
                     floor_note="350 以下不出")
    result = await tool_by_name("get_bargain_status").execute({}, tctx)
    assert "2 轮" in result and "350" in result


async def test_read_earlier_history():
    history = [{"role": "user", "content": "便宜点"},
               {"role": "assistant", "content": "348 包邮"}]
    tctx = make_tctx(store=FakeStore(history=history))
    result = await tool_by_name("read_earlier_history").execute({"limit": 10}, tctx)
    assert "348 包邮" in result


async def test_deadline_exhausted_returns_error():
    tctx = make_tctx(deadline=Deadline(seconds=-1.0))
    for name in ("get_item_info", "get_bargain_status", "read_earlier_history"):
        result = await tool_by_name(name).execute({}, tctx)
        assert result.startswith("error: 时限已到"), name


async def test_web_search_disabled():
    tctx = make_tctx(search=None)
    result = await tool_by_name("web_search").execute({"query": "型号对比"}, tctx)
    assert result.startswith("error:")


async def test_web_search_normal():
    async def fake_search(query):
        return f"搜索结果：{query}"

    tctx = make_tctx(search=fake_search)
    result = await tool_by_name("web_search").execute({"query": "A10 和 A20 区别"}, tctx)
    assert "A10 和 A20 区别" in result


async def test_notify_seller_disabled():
    tctx = make_tctx(notify=None)
    result = await tool_by_name("notify_seller").execute({"kind": "image"}, tctx)
    assert result.startswith("error:")


async def test_notify_seller_normal():
    async def fake_notify(kind, detail):
        return f"已通知: {kind} {detail}"

    tctx = make_tctx(notify=fake_notify, item=build_item_description({"soldPrice": 100}))
    result = await tool_by_name("notify_seller").execute(
        {"kind": "price", "price": 90, "detail": "90元"}, tctx)
    assert "已通知" in result and "买家确认 90 元" in result


async def test_notify_seller_deadline():
    async def fake_notify(kind, detail):
        return "ok"

    tctx = make_tctx(notify=fake_notify, deadline=Deadline(seconds=-1.0))
    result = await tool_by_name("notify_seller").execute({"kind": "image"}, tctx)
    assert result.startswith("error: 时限已到")


# ---------------------------------------------------------------- 商品摘要

def test_format_price_cents_to_yuan():
    assert format_price(34800) == 348.0
    assert format_price(None) == 0.0


def test_build_item_description_with_skus():
    item = {
        "title": "音箱", "desc": "99新", "soldPrice": 199, "quantity": 5,
        "skuList": [
            {"propertyList": [{"valueText": "黑色"}], "price": 19900, "quantity": 3},
            {"propertyList": [{"valueText": "白色"}], "price": 21900, "quantity": 2},
        ],
    }
    summary = build_item_description(item)
    assert summary["title"] == "音箱"
    assert summary["price_range"] == "¥199.0 - ¥219.0"
    assert summary["total_stock"] == 5
    assert summary["sku_details"][0]["spec"] == "黑色"
    assert summary["sku_details"][0]["price"] == 199.0


def test_build_item_description_fallback_to_main_price():
    summary = build_item_description({"title": "x", "desc": "", "soldPrice": 88,
                                      "quantity": 1, "skuList": []})
    assert summary["price_range"] == "¥88.0"
