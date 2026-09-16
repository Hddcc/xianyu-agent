"""多会话并发端到端延迟评测。

跑法：python benchmarks/concurrency.py
对每个并发档位，起 N 个不同会话同时发一条消息，测「消息进入 -> 回复产出」耗时。
不同会话并发、同一会话串行（被测对象的核心设计）。
"""
from __future__ import annotations

import asyncio
import time

from common import build_eval_registry, summarize
from xianyu_agent.types import IncomingChat

MESSAGE = "这个多少钱"
LEVELS = (1, 5, 10, 20)


async def run_level(registry, sent, n: int) -> tuple[list[float], list[str], float]:
    latencies: list[float] = []
    errors: list[str] = []

    async def one(i: int):
        chat = IncomingChat(chat_id=f"bench-{n}-{i}", item_id="",
                            sender_id=f"buyer{i}", sender_name=f"买家{i}",
                            text=MESSAGE)
        t0 = time.monotonic()
        try:
            await registry.handle(chat)
            latencies.append(time.monotonic() - t0)
        except Exception as e:                     # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    wall0 = time.monotonic()
    await asyncio.gather(*[one(i) for i in range(n)])
    wall = time.monotonic() - wall0
    return latencies, errors, wall


async def main() -> None:
    registry, sent = build_eval_registry()

    print("=" * 72)
    print("多会话并发端到端延迟评测（每条消息走完整链路：路由 -> 专家 -> 循环 -> 回复）")
    print("=" * 72)
    print(f"{'并发数':<8}{'成功':<8}{'平均(s)':<10}{'P50(s)':<10}{'P95(s)':<10}"
          f"{'P99(s)':<10}{'最大(s)':<10}{'墙钟(s)':<10}{'失败':<6}")

    baseline = None
    for n in LEVELS:
        latencies, errors, wall = await run_level(registry, sent, n)
        s = summarize(latencies)
        if n == 1:
            baseline = s["avg"]
        print(f"{n:<8}{s['n']:<8}{s['avg']:<10.2f}{s['p50']:<10.2f}{s['p95']:<10.2f}"
              f"{s['p99']:<10.2f}{s['max']:<10.2f}{wall:<10.2f}{len(errors):<6}")

    if baseline:
        print("-" * 72)
        print(f"基线（单会话）平均 {baseline:.2f}s；"
              f"并发下平均延迟无明显劣化即为达标的并发能力。")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
