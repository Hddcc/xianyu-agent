"""评测脚本共用工具。

三个评测：
- intent_accuracy.py  意图路由准确率（质量）
- concurrency.py      多会话并发端到端延迟（性能）
- hallucination.py    幻觉率 / 能力越界率（可靠性）
"""
from __future__ import annotations

import os
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
load_dotenv(".env.example")


def build_eval_registry():
    """构造真实 registry（真实模型），发送口换成记录函数（不真发闲鱼消息）。

    并发评测下必须按 chat_id 归集回复，否则会取到别的会话的内容。
    """
    from xianyu_agent.cli import build_registry

    registry, api = build_registry()
    sent: dict[str, str] = {}

    async def _capture(chat_id, to_user_id, text):
        sent[chat_id] = text

    registry.config.myid = "seller"
    registry.set_sender(_capture)
    return registry, sent


def build_router_with_counter():
    """构造真实的意图路由，并统计走 LLM 兜底的次数。"""
    from xianyu_agent.experts import _load_prompt
    from xianyu_agent.llm import default_models
    from xianyu_agent.router import IntentRouter, make_classify_llm

    prompt = _load_prompt("classify_prompt", os.getenv("PROMPTS_DIR", "prompts"))
    base = make_classify_llm(prompt, default_models())
    counter = {"llm": 0}

    def counting_classify(user_msg, item_desc, history):
        counter["llm"] += 1
        return base(user_msg, item_desc, history)

    return IntentRouter(counting_classify), counter


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize(latencies: list[float]) -> dict:
    if not latencies:
        return {"n": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": len(latencies),
        "avg": statistics.mean(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "max": max(latencies),
    }
