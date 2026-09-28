"""意图路由。规则优先，模型兜底——被真实流量验证过的省钱经验。

三级策略（技术优先，原项目的经验：参数问题答错了比贵一块钱严重）：
1. 技术类关键词/正则；
2. 价格类关键词/正则；
3. 分类模型兜底（可能返回 no_reply）。

路由器不知道循环的存在，循环也不知道自己是被谁路由进来的。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

from . import llm
from .llm import Models
from .types import Context, Message

VALID_INTENTS = ("price", "tech", "default", "no_reply")


@dataclass(frozen=True)
class ClassifierResult:
    intent: str
    raw_output: str = ""
    model: str = ""
    fallback_reason: str | None = None


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    source: str
    duration_ms: float
    matched_rule: str | None = None
    raw_output: str | None = None
    model: str | None = None
    fallback_reason: str | None = None


class IntentRouter:
    RULES = {
        "tech": {   # 技术类优先判定
            "keywords": ["参数", "规格", "型号", "连接", "对比"],
            "patterns": [r"和.+比"],
        },
        "price": {
            "keywords": ["便宜", "价", "砍价", "少点"],
            "patterns": [r"\d+元", r"能少\d+"],
        },
    }

    def __init__(self, classify_llm: Callable[
            [str, str, str], str | ClassifierResult]):
        """
        classify_llm: (user_msg, item_desc, history) -> 意图字符串，
        可能返回 price / tech / default / no_reply。
        """
        self.classify_llm = classify_llm

    def decide(self, user_msg: str, item_desc: str, history: str) -> IntentDecision:
        """返回分类结果和可解释的判断来源。"""
        started = time.monotonic()
        text = re.sub(r"[^\w\u4e00-\u9fa5]", "", user_msg)

        for intent in ("tech", "price"):       # 技术类优先
            rule = self.RULES[intent]
            if any(kw in text for kw in rule["keywords"]):
                keyword = next(kw for kw in rule["keywords"] if kw in text)
                return IntentDecision(
                    intent, "rule", (time.monotonic() - started) * 1000,
                    matched_rule=f"keyword:{keyword}",
                )
            for pattern in rule["patterns"]:
                if re.search(pattern, text):
                    return IntentDecision(
                        intent, "rule", (time.monotonic() - started) * 1000,
                        matched_rule=f"pattern:{pattern}",
                    )

        classified = self.classify_llm(user_msg, item_desc, history)
        if isinstance(classified, ClassifierResult):
            result = classified
        else:
            intent = classified if classified in VALID_INTENTS else "default"
            reason = None if classified in VALID_INTENTS else "invalid_output"
            result = ClassifierResult(intent=intent, raw_output=str(classified or ""),
                                      fallback_reason=reason)
        if result.intent not in VALID_INTENTS:
            result = ClassifierResult(
                intent="default", raw_output=result.raw_output, model=result.model,
                fallback_reason=result.fallback_reason or "invalid_output",
            )
        return IntentDecision(
            result.intent, "model", (time.monotonic() - started) * 1000,
            raw_output=result.raw_output, model=result.model,
            fallback_reason=result.fallback_reason,
        )

    def detect(self, user_msg: str, item_desc: str, history: str) -> str:
        """兼容原有调用方，只返回 tech / price / default / no_reply。"""
        return self.decide(user_msg, item_desc, history).intent


def make_classify_llm(classify_prompt: str, models: Models) -> Callable[
        [str, str, str], ClassifierResult]:
    """构造兜底分类函数。同步 HTTP 调用，由调用方丢进线程池。"""

    def classify(user_msg: str, item_desc: str, history: str) -> ClassifierResult:
        ctx = Context(
            system_prompt=classify_prompt,
            messages=[Message(role="user",
                              content=(f"【商品信息】{item_desc}\n"
                                       f"【对话历史】\n{history}\n"
                                       f"【买家最新消息】{user_msg}"))],
        )
        raw = llm.complete(ctx, model=models.classify, temperature=0.1, max_tokens=20)
        if not raw:
            return ClassifierResult("default", raw_output="", model=models.classify.name,
                                    fallback_reason="empty_or_error")
        # 模型偶尔会在类别名前后带说明文字，提取第一个出现的合法类别
        low = raw.lower()
        for kw in ("no_reply", "price", "tech", "default"):
            if kw in low:
                return ClassifierResult(kw, raw_output=raw, model=models.classify.name)
        return ClassifierResult("default", raw_output=raw, model=models.classify.name,
                                fallback_reason="invalid_output")

    return classify
