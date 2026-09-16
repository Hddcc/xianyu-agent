"""意图路由。规则优先，模型兜底——被真实流量验证过的省钱经验。

三级策略（技术优先，原项目的经验：参数问题答错了比贵一块钱严重）：
1. 技术类关键词/正则；
2. 价格类关键词/正则；
3. 分类模型兜底（可能返回 no_reply）。

路由器不知道循环的存在，循环也不知道自己是被谁路由进来的。
"""
from __future__ import annotations

import re
from typing import Callable

from . import llm
from .llm import Models
from .types import Context, Message

VALID_INTENTS = ("price", "tech", "default", "no_reply")


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

    def __init__(self, classify_llm: Callable[[str, str, str], str]):
        """
        classify_llm: (user_msg, item_desc, history) -> 意图字符串，
        可能返回 price / tech / default / no_reply。
        """
        self.classify_llm = classify_llm

    def detect(self, user_msg: str, item_desc: str, history: str) -> str:
        """返回 tech / price / default / no_reply。"""
        text = re.sub(r"[^\w\u4e00-\u9fa5]", "", user_msg)

        for intent in ("tech", "price"):       # 技术类优先
            rule = self.RULES[intent]
            if any(kw in text for kw in rule["keywords"]):
                return intent
            for pattern in rule["patterns"]:
                if re.search(pattern, text):
                    return intent

        return self.classify_llm(user_msg, item_desc, history)   # 兜底


def make_classify_llm(classify_prompt: str, models: Models) -> Callable[[str, str, str], str]:
    """构造兜底分类函数。同步 HTTP 调用，由调用方丢进线程池。"""

    def classify(user_msg: str, item_desc: str, history: str) -> str:
        ctx = Context(
            system_prompt=classify_prompt,
            messages=[Message(role="user",
                              content=(f"【商品信息】{item_desc}\n"
                                       f"【对话历史】\n{history}\n"
                                       f"【买家最新消息】{user_msg}"))],
        )
        raw = llm.complete(ctx, model=models.classify, temperature=0.1, max_tokens=20)
        if not raw:
            return "default"                    # 分类失败按默认处理，宁可答泛不可不答
        # 模型偶尔会在类别名前后带说明文字，提取第一个出现的合法类别
        low = raw.lower()
        for kw in ("no_reply", "price", "tech", "default"):
            if kw in low:
                return kw
        return "default"

    return classify
