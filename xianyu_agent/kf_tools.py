"""闲鱼客服领域的工具。循环完全不知道这些工具是干什么的。

安全约束落在参数定义上：get_item_info 和 get_bargain_status 的参数是空的——
查哪个商品、哪个会话，由 ToolContext 里的编号决定，模型连表达
"我想查别的会话"的入口都没有。回复发给谁，永远由通道决定。
"""
from __future__ import annotations

import json

from .tools import ToolSpec


def format_price(price) -> float:
    """SKU 价格：分转元（原项目逻辑）。"""
    try:
        return round(float(price) / 100, 2)
    except (ValueError, TypeError):
        return 0.0


def build_item_description(item_info: dict) -> dict:
    """构建商品摘要（原项目逻辑）。通道和工具共用这一份。"""
    clean_skus = []
    for sku in item_info.get("skuList", []) or []:
        specs = [p["valueText"] for p in sku.get("propertyList", []) if p.get("valueText")]
        spec_text = " ".join(specs) if specs else "默认规格"
        clean_skus.append({
            "spec": spec_text,
            "price": format_price(sku.get("price", 0)),
            "stock": sku.get("quantity", 0),
        })

    valid_prices = [s["price"] for s in clean_skus if s["price"] > 0]
    if valid_prices:
        min_price, max_price = min(valid_prices), max(valid_prices)
        price_display = (f"¥{min_price}" if min_price == max_price
                         else f"¥{min_price} - ¥{max_price}")
    else:
        try:
            main_price = round(float(item_info.get("soldPrice", 0)), 2)
        except (ValueError, TypeError):
            main_price = 0.0
        price_display = f"¥{main_price}"

    return {
        "title": item_info.get("title", ""),
        "desc": item_info.get("desc", ""),
        "price_range": price_display,
        "total_stock": item_info.get("quantity", 0),
        "sku_details": clean_skus,
    }


# ---------------------------------------------------------------- 时限约定

def _deadline_check(tctx) -> str | None:
    """工具层拦截：超时就不干活，把剩余时间还给模型让它直接作答。"""
    if tctx.deadline is not None and tctx.deadline.expired:
        return "error: 时限已到，请立即用已有信息回复买家，不要再调用工具。"
    return None


def _remaining_note(tctx) -> str:
    """工具返回里附带剩余时间，让模型能感知并收敛。"""
    if tctx.deadline is not None:
        return f"\n（剩余 {tctx.deadline.remaining():.0f} 秒）"
    return ""


# ---------------------------------------------------------------- 工具实现

async def _get_item_info(args: dict, tctx) -> str:
    err = _deadline_check(tctx)
    if err:
        return err
    item = tctx.item
    if item is None and tctx.store is not None:
        raw = await tctx.store.get_item_info(tctx.item_id)
        if raw:
            item = build_item_description(raw)
    if item is None:
        return "error: 商品信息暂不可用，请凭已有信息礼貌回复买家。"
    return (f"商品：{item['title']}\n"
            f"价格：{item['price_range']}\n"
            f"库存：{item['total_stock']}\n"
            f"规格：{json.dumps(item['sku_details'], ensure_ascii=False)}\n"
            f"描述：{str(item['desc'])[:500]}"
            + _remaining_note(tctx))


async def _get_bargain_status(args: dict, tctx) -> str:
    err = _deadline_check(tctx)
    if err:
        return err
    count = 0
    if tctx.store is not None:
        count = await tctx.store.get_bargain_count(tctx.chat_id)
    price = tctx.item["price_range"] if tctx.item else "未知"
    floor = tctx.floor_note or "按系统提示词中的策略执行"
    return (f"当前会话已进行 {count} 轮议价。\n"
            f"标价 {price}。卖家底线：{floor}。"
            + _remaining_note(tctx))


async def _read_earlier_history(args: dict, tctx) -> str:
    err = _deadline_check(tctx)
    if err:
        return err
    n = min(int(args.get("limit", 30)), 100)
    msgs = []
    if tctx.store is not None:
        msgs = await tctx.store.get_context_by_chat(tctx.chat_id, limit=n)
    if not msgs:
        return "没有更早的记录了。"
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + _remaining_note(tctx)


async def _web_search(args: dict, tctx) -> str:
    err = _deadline_check(tctx)
    if err:
        return err
    if tctx.search is None:
        return "error: 搜索能力未启用，请凭已有信息回答。"
    result = await tctx.search(args["query"])
    if not result:
        return "error: 搜索没有返回结果，请换个说法或凭已有信息回答。"
    return result + _remaining_note(tctx)


_KIND_ZH = {"image": "买家要看实物图", "price": "买家谈妥需要改价", "other": "需要人工介入"}


async def _notify_seller(args: dict, tctx) -> str:
    """AI 做不了的事（发图/改价等），通知卖家人工介入，而不是瞎承诺。"""
    err = _deadline_check(tctx)
    if err:
        return err
    if tctx.notify is None:
        return "error: 通知能力未启用，请诚实回复买家'稍后处理'，不要承诺已经处理好了"
    kind = args.get("kind", "other")
    detail = args.get("detail", "")
    result = await tctx.notify(kind, detail)
    return result + _remaining_note(tctx)


def kf_tools() -> list[ToolSpec]:
    return [
        ToolSpec("get_item_info",
                 "查看当前会话买家正在咨询的商品信息（价格、规格、库存、描述）。",
                 {"type": "object", "properties": {}}, _get_item_info),
        ToolSpec("get_bargain_status",
                 "查看当前会话的议价状态：已进行几轮、卖家底线。",
                 {"type": "object", "properties": {}}, _get_bargain_status),
        ToolSpec("read_earlier_history",
                 "读取更早的对话原文。当会话画像里的信息不够时使用。",
                 {"type": "object",
                  "properties": {"limit": {"type": "integer",
                                           "description": "要读的条数，默认 30"}}},
                 _read_earlier_history),
        ToolSpec("web_search",
                 "联网搜索。回答参数对比、行情、型号差异等技术问题时使用。",
                 {"type": "object",
                  "properties": {"query": {"type": "string", "description": "搜索词"}},
                  "required": ["query"]}, _web_search),
        ToolSpec("notify_seller",
                 "当买家提出你无法完成、需要卖家本人处理的要求时，通知卖家介入。"
                 "你无法发送图片、无法修改价格，遇到'看实物图''改价'等请求必须用本工具，"
                 "绝不能承诺'已经处理好了'。",
                 {"type": "object",
                  "properties": {
                      "kind": {"type": "string",
                               "enum": ["image", "price", "other"],
                               "description": "image=要看实物图, price=谈妥需改价, other=其他人工事项"},
                      "detail": {"type": "string",
                                 "description": "补充说明，如买家接受的价格、想看的角度等"},
                  },
                  "required": ["kind"]}, _notify_seller),
    ]
