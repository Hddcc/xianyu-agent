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


def has_numeric_offer(text: str) -> bool:
    """识别带数字的报价，供价格路由决定是否强制调用报价工具。"""
    return bool(re.search(r"(?<![\d.])\d+(?:\.\d{1,2})?(?![\d.])", text or ""))


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

    def discount_percent_for_round(self, bargain_count: int | None = None) -> Decimal:
        """首轮只开放一半上限，后续逐轮放宽，最终不超过总上限。"""
        if bargain_count is None:
            return self.max_discount_percent if self.enabled else Decimal("0")
        try:
            count = max(int(bargain_count), 0)
        except (TypeError, ValueError):
            count = 0
        progress = min(Decimal("1"), Decimal("0.5") + Decimal("0.25") * count)
        return (self.max_discount_percent * progress).quantize(CENT,
                                                                rounding=ROUND_CEILING)

    def minimum(self, item, bargain_count: int | None = None) -> Decimal:
        price = self.list_price(item)
        discount = self.discount_percent_for_round(bargain_count) if self.enabled else Decimal("0")
        ratio = 1 - discount / 100
        return (price * ratio).quantize(CENT, rounding=ROUND_CEILING)

    def validate(self, amount, item, bargain_count: int | None = None) -> Decimal:
        price = self.list_price(item)
        minimum = self.minimum(item, bargain_count)
        amount = _decimal(amount, "报价")
        if amount <= 0 or amount != amount.quantize(CENT):
            raise ValueError("报价必须大于 0，且最多保留两位小数")
        if not minimum <= amount <= price:
            raise ValueError(f"报价必须在 {minimum:f} 到 {price:f} 元之间")
        return amount

    def prompt(self, item, bargain_count: int | None = None) -> str:
        rules = (
            "【价格规则：优先于专家话术和历史报价】\n"
            "商品信息是本次读取的最新信息，历史报价不得覆盖当前标价和规则。\n"
            "报价必须调用 quote_price；改价通知必须提供数值 price。\n"
            "最终报价由程序生成；不要自行承诺金额、折扣、赠品或免运费。\n"
            "不要向买家透露内部最低价。\n")
        try:
            price = self.list_price(item)
            minimum = self.minimum(item, bargain_count)
        except ValueError:
            return rules + "当前价格或具体规格尚未确认，暂停报价和改价通知，交由卖家确认。"
        mode = "允许议价" if self.enabled else "一口价，不接受议价"
        current_discount = self.discount_percent_for_round(bargain_count) if self.enabled else Decimal("0")
        round_number = max(int(bargain_count), 0) + 1 if bargain_count is not None else "当前"
        return (rules + f"{mode}；最新标价 {price:f} 元。\n"
                f"{'' if bargain_count is None else f'当前第 {round_number} 轮，'}当前最多优惠 {current_discount:f}%；"
                f"允许报价范围 {minimum:f} 到 {price:f} 元；总优惠上限 {self.max_discount_percent:f}%。")

    def fallback_reply(self) -> str:
        if not self.enabled:
            return "这款一口价，暂不接受议价"
        return "价格需要卖家确认，请稍等"

    def counter_reply(self, item, bargain_count: int | None = None) -> str:
        try:
            minimum = self.minimum(item, bargain_count)
        except ValueError:
            return self.fallback_reply()
        return f"目前最低 {minimum:f} 元，您看可以吗"

    def guard_reply(self, text: str, intent: str, price_reply: str | None,
                    item=None, bargain_count: int | None = None,
                    offer_detected: bool = False) -> str:
        if price_reply is not None:
            return price_reply
        if intent == "price" or _MONEY.search(text):
            if offer_detected:
                return self.counter_reply(item, bargain_count)
            return self.fallback_reply()
        return text
