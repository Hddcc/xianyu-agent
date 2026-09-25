"""闲鱼客服领域的工具。循环完全不知道这些工具是干什么的。

安全约束落在参数定义上：get_item_info 和 get_bargain_status 的参数是空的——
查哪个商品、哪个会话，由 ToolContext 里的编号决定，模型连表达
"我想查别的会话"的入口都没有。回复发给谁，永远由通道决定。
"""
from __future__ import annotations

import json
import math

from .tools import ToolSpec
from .pricing import BargainPolicy


def format_price(price) -> float:
    """SKU 价格：分转元（原项目逻辑）。"""
    try:
        value = float(price) / 100
        return round(value, 2) if math.isfinite(value) and value > 0 else 0.0
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
        # 多规格价格不同时，自动报价需要先明确规格。
        current_price = min_price if min_price == max_price else None
    else:
        try:
            main_price = round(float(item_info.get("soldPrice", 0)), 2)
        except (ValueError, TypeError):
            main_price = 0.0
        current_price = (main_price if math.isfinite(main_price) and main_price > 0
                         and not isinstance(item_info.get("soldPrice"), bool) else None)
        price_display = f"¥{current_price}" if current_price is not None else "未知"

    return {
        "title": item_info.get("title", ""),
        "desc": item_info.get("desc", ""),
        "price_range": price_display,
        "price": current_price,
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
    policy = tctx.bargain_policy or BargainPolicy()
    return (f"当前会话已进行 {count} 轮议价。\n"
            f"标价 {price}。卖家底线：{floor}。"
            + "\n" + policy.prompt(tctx.item, tctx.bargain_count)
            + _remaining_note(tctx))


async def _quote_price(args: dict, tctx) -> str:
    policy = tctx.bargain_policy or BargainPolicy()
    tctx.price_reply = policy.fallback_reply()
    err = _deadline_check(tctx)
    if err:
        return err
    try:
        price = policy.validate(args.get("price"), tctx.item, tctx.bargain_count)
    except ValueError as e:
        try:
            proposed_raw = args.get("price")
            if isinstance(proposed_raw, bool):
                raise ValueError
            proposed = float(proposed_raw)
            minimum = policy.minimum(tctx.item, tctx.bargain_count)
            if proposed < float(minimum):
                tctx.price_reply = policy.counter_reply(tctx.item, tctx.bargain_count)
        except (TypeError, ValueError):
            pass
        return f"error: {e}"
    tctx.price_reply = (f"可以 {price:f} 元，您看怎么样" if policy.enabled
                       else f"这款一口价 {price:f} 元，暂不接受议价")
    return f"已验证报价 {price:f} 元。" + _remaining_note(tctx)


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
    kind = args.get("kind", "other")
    if kind == "price":
        tctx.price_reply = "改价需要卖家确认，请稍等"
    err = _deadline_check(tctx)
    if err:
        return err
    if tctx.notify is None:
        return "error: 通知能力未启用，请诚实回复买家'稍后处理'，不要承诺已经处理好了"
    detail = args.get("detail", "")
    if kind == "price":
        policy = tctx.bargain_policy or BargainPolicy()
        try:
            price = policy.validate(args.get("price"), tctx.item, tctx.bargain_count)
        except ValueError as e:
            tctx.price_reply = policy.counter_reply(tctx.item, tctx.bargain_count)
            return f"error: {e}"
        detail = f"买家确认 {price:f} 元，请卖家确认并手动改价"
    result = await tctx.notify(kind, detail)
    if kind == "price":
        tctx.price_reply = ("价格已记下，请等待卖家操作" if not result.startswith("error:")
                           else "改价需要卖家确认，请稍等")
    return result + _remaining_note(tctx)


def kf_tools() -> list[ToolSpec]:
    return [
        ToolSpec("get_item_info",
                 "查看当前会话买家正在咨询的商品信息（价格、规格、库存、描述）。",
                 {"type": "object", "properties": {}}, _get_item_info),
        ToolSpec("get_bargain_status",
                 "查看当前会话的议价状态：已进行几轮、卖家底线。",
                 {"type": "object", "properties": {}}, _get_bargain_status),
        ToolSpec("quote_price",
                 "向买家报价前必须调用。程序按最新标价、议价开关和累计优惠百分比上限校验，"
                 "并生成实际发送的报价回复。报价不包含运费或赠品承诺。",
                 {"type": "object", "properties": {"price": {"type": "number"}},
                  "required": ["price"]}, _quote_price),
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
                      "price": {"type": "number",
                                "description": "kind=price 时必填，买家确认的成交金额（元），程序会校验"},
                  },
                  "required": ["kind"]}, _notify_seller),
    ]
