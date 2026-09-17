"""Shared bargaining policy for prompts, structured quotes and price notifications."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import re


CENT = Decimal("0.01")
_NUMBER = r"[\d零〇一二两三四五六七八九十百千万点.,]+"
_MONEY = re.compile(
    rf"[¥￥]|{_NUMBER}\s*(?:元|块|包邮|折)|"
    rf"(?:报价|成交价|价格|最低)\s*(?:是|为)?\s*{_NUMBER}|"
    rf"(?:优惠|降价|便宜)\s*(?:百分之)?{_NUMBER}\s*%?")


def _decimal(value, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"{name} 必须是有效数字") from None
    if not number.is_finite():
        raise ValueError(f"{name} 必须是有限数字")
    return number


@dataclass
class BargainPolicy:
    enabled: bool = True
    max_discount_percent: Decimal = Decimal("10")

    def __post_init__(self):
        self.max_discount_percent = _decimal(
            self.max_discount_percent, "BARGAIN_MAX_DISCOUNT_PERCENT")
        if not 0 <= self.max_discount_percent <= 100:
            raise ValueError("BARGAIN_MAX_DISCOUNT_PERCENT 必须在 0 到 100 之间")

    @staticmethod
    def list_price(item) -> Decimal:
        price = _decimal(item.get("price") if item else None, "当前商品价格")
        if price <= 0:
            raise ValueError("当前商品价格无效，请让卖家确认价格和规格")
        return price

    def minimum(self, item) -> Decimal:
        price = self.list_price(item)
        ratio = 1 - self.max_discount_percent / 100 if self.enabled else Decimal("1")
        return (price * ratio).quantize(CENT, rounding=ROUND_CEILING)

    def validate(self, amount, item) -> Decimal:
        price = self.list_price(item)
        minimum = self.minimum(item)
        amount = _decimal(amount, "报价")
        if amount <= 0 or amount != amount.quantize(CENT):
            raise ValueError("报价必须大于 0，且最多保留两位小数")
        if not minimum <= amount <= price:
            raise ValueError(f"报价必须在 {minimum:f} 到 {price:f} 元之间")
        return amount

    def prompt(self, item) -> str:
        rules = (
            "【价格规则：优先于专家话术和历史报价】\n"
            "商品信息是本次读取的最新信息，历史报价不得覆盖当前标价和规则。\n"
            "报价必须调用 quote_price；改价通知必须提供数值 price。\n"
            "最终报价由程序生成；不要自行承诺金额、折扣、赠品或免运费。\n"
            "不要向买家透露内部最低价。\n")
        try:
            price, minimum = self.list_price(item), self.minimum(item)
        except ValueError:
            return rules + "当前价格或具体规格尚未确认，暂停报价和改价通知，交由卖家确认。"
        mode = "允许议价" if self.enabled else "一口价，不接受议价"
        return (rules + f"{mode}；最新标价 {price:f} 元。\n"
                f"允许报价范围 {minimum:f} 到 {price:f} 元，累计优惠始终基于最新标价计算。")

    def fallback_reply(self) -> str:
        if not self.enabled:
            return "这款一口价，暂不接受议价"
        return "价格需要卖家确认，请稍等"

    def guard_reply(self, text: str, intent: str, price_reply: str | None) -> str:
        if price_reply is not None:
            return price_reply
        if intent == "price" or _MONEY.search(text):
            return self.fallback_reply()
        return text
