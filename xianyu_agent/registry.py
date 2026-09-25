"""会话注册表与单次运行流程。并发的单位是会话，不是消息。

- 不同会话之间天然并发（一个买家触发模型调用，别的买家不用排队）；
- 同一个会话严格串行（两条消息交叉回复，顺序一乱，买家立刻穿帮）；
- 卖家接管时能打断该会话正在生成的回复（abort -> signal.cancel）。

这里也是"一条买家消息只发送一次回复"这条硬规则的守门人：
发送动作只有一处、且只执行一次，模型永远无法决定发给谁。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field

from . import llm
from .agent import SUMMARY_PROMPT, run_agent
from .cancel import CancellationToken
from .deadline import Deadline
from .kf_tools import build_item_description
from .pricing import BargainPolicy, has_numeric_offer
from .session import append_snapshot, session_path
from .tools import ToolContext
from .types import Context, IncomingChat, Message

logger = logging.getLogger(__name__)

BLOCKED_PHRASES = ["微信", "QQ", "支付宝", "银行卡", "线下"]


def safety_filter(text: str) -> str:
    """发送前的最后一道闸。挂在发送端，因为出口只有一个。"""
    if any(p in text for p in BLOCKED_PHRASES):
        return "[安全提醒]请通过平台沟通"
    return text


@dataclass
class AppConfig:
    myid: str = "me"
    simulate_typing: bool = False
    max_context_messages: int = 20
    idle_compact_hours: float = 12.0
    agent_max_turns: int = 4
    agent_deadline_seconds: float = 30.0
    fallback_reply: str = "稍等哈，我确认下"
    bargain_policy: BargainPolicy = field(default_factory=BargainPolicy)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def load_config(myid: str = "me") -> AppConfig:
    enabled = (os.getenv("BARGAIN_ENABLED", "True") or "True").strip().lower()
    if enabled not in ("true", "false"):
        raise ValueError("BARGAIN_ENABLED 必须是 True 或 False")
    return AppConfig(
        myid=myid,
        simulate_typing=os.getenv("SIMULATE_HUMAN_TYPING", "False").lower() == "true",
        max_context_messages=_env_int("MAX_CONTEXT_MESSAGES", 20),
        idle_compact_hours=_env_float("IDLE_COMPACT_HOURS", 12.0),
        agent_max_turns=_env_int("AGENT_MAX_TURNS", 4),
        agent_deadline_seconds=_env_float("AGENT_DEADLINE_SECONDS", 30.0),
        bargain_policy=BargainPolicy(
            enabled=enabled == "true",
            max_discount_percent=os.getenv("BARGAIN_MAX_DISCOUNT_PERCENT") or "10",
        ),
    )


class SessionRegistry:
    def __init__(self, store, experts, router, models, config,
                 sender=None, api=None):
        """
        sender: async (chat_id, to_user_id, text) -> None，由通道注册；
        dev 模式下注入一个打印函数。发送入口只有这一个。
        """
        self.store = store
        self.experts = experts
        self.router = router
        self.models = models
        self.config = config
        self.api = api
        self._sender = sender
        self._locks: dict[str, asyncio.Lock] = {}
        self._signals: dict[str, CancellationToken] = {}
        self._search_enabled = os.getenv("ENABLE_SEARCH", "True").lower() == "true"

    def set_sender(self, sender):
        self._sender = sender

    # ---------------------------------------------------------------- 入口

    async def handle(self, chat: IncomingChat):
        """通道调用的唯一入口。不同会话并发，同会话串行。"""
        lock = self._locks.setdefault(chat.chat_id, asyncio.Lock())
        async with lock:
            await self._run_once(chat)

    def abort(self, chat_id: str):
        """卖家接管 -> 打断这个会话正在生成的回复。"""
        signal = self._signals.get(chat_id)
        if signal:
            signal.cancel()
            logger.info("已打断会话 %s 正在生成的回复", chat_id)

    async def record_manual_reply(self, chat: IncomingChat, role: str):
        """人工模式下（或卖家亲自回复）只记录，不回复。AI 接回来时知道发生过什么。"""
        await self.store.add_message(chat.chat_id, chat.sender_id, chat.item_id,
                                     role, chat.text)
        logger.info("人工记录 (会话: %s, 商品: %s, %s): %s",
                    chat.chat_id, chat.item_id, role, chat.text)

    # ---------------------------------------------------------------- 内部

    async def _ensure_item(self, item_id: str) -> dict | None:
        """每条消息重新读取商品；失败时暂停报价，保留数据库供排查。"""
        if not item_id:
            return None
        if self.api is None:
            return None
        try:
            result = await asyncio.to_thread(self.api.get_item_info, item_id)
        except Exception:
            logger.exception("获取商品信息失败: %s", item_id)
            return None
        data = result.get("data") if isinstance(result, dict) else None
        item_info = data.get("itemDO") if isinstance(data, dict) else None
        if isinstance(item_info, dict) and item_info:
            await self.store.save_item_info(item_id, item_info)
            logger.info("从平台获取商品信息并缓存: %s", item_id)
            return item_info
        logger.warning("获取商品信息失败: %s", result)
        return None

    async def _refresh_profile_if_idle(self, chat_id: str) -> str | None:
        """会话空闲超时 -> 把旧对话压成会话画像。活跃会话直接用现有画像。"""
        hours = self.config.idle_compact_hours
        if hours <= 0:
            return None

        last = await self.store.last_activity(chat_id)
        text, updated = await self.store.get_profile(chat_id)
        if last <= 0 or time.time() - last < hours * 3600:
            return text            # 会话还活跃：用现有画像（可能为 None）

        if updated >= last:
            return text            # 画像比最后一条消息新：上次空闲后已经刷新过

        msgs = await self.store.get_context_by_chat(chat_id, limit=None)
        if not msgs:
            return text

        logger.info("会话 %s 空闲超时，刷新会话画像", chat_id)
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        if text:
            rendered = f"[上一段会话画像]\n{text}\n\n[其后的新对话]\n{rendered}"

        sub = Context(system_prompt=SUMMARY_PROMPT,
                      messages=[Message(role="user", content=rendered)])
        summary = ""
        async for ev in llm.stream(sub, model=self.models.reply):
            if ev["type"] == "text_delta":
                summary += ev["delta"]
        if not summary.strip():
            return text            # 摘要失败（比如请求报错）：宁可不用也不用空摘要

        await self.store.set_profile(chat_id, summary)
        return summary

    async def _search(self, query: str) -> str:
        """工具的搜索入口：同步 HTTP 丢线程池，别卡事件循环。"""
        return await asyncio.to_thread(llm.search_web, query, self.models.reply)

    def _make_notifier(self, chat):
        """构造 notify_seller 工具的通知入口，带上会话上下文。"""
        from .notify import send_notification
        kind_zh = {"image": "买家要看实物图", "price": "买家谈妥需要改价",
                   "other": "需要人工介入"}

        async def notify(kind: str, detail: str) -> str:
            title = f"[闲鱼客服] {kind_zh.get(kind, kind)}"
            content = (f"买家：{chat.sender_name}（ID {chat.sender_id}）\n"
                       f"商品ID：{chat.item_id}\n"
                       f"会话ID：{chat.chat_id}\n"
                       f"事项：{kind_zh.get(kind, kind)}\n"
                       f"说明：{detail or '无'}\n")
            return await asyncio.to_thread(send_notification, title, content)

        return notify

    async def _run_once(self, chat: IncomingChat):
        cfg = self.config
        chat_id, item_id = chat.chat_id, chat.item_id

        # 1. 最新商品信息（通道不管这件事）
        item_raw = await self._ensure_item(item_id)
        item = build_item_description(item_raw) if item_raw else None
        item_desc = json.dumps(item, ensure_ascii=False) if item else "商品信息暂不可用"

        # 2. 空闲画像
        profile_text = await self._refresh_profile_if_idle(chat_id)

        # 3. 路由（规则优先，模型兜底；分类是同步 HTTP，丢线程池）
        history = await self.store.get_context_by_chat(chat_id, limit=cfg.max_context_messages)
        history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        intent = await asyncio.to_thread(self.router.detect, chat.text, item_desc, history_str)
        logger.info("[会话 %s] 意图识别完成: %s", chat_id, intent)

        if intent == "no_reply":
            logger.info("[会话 %s] 无需回复，跳过", chat_id)
            return

        # 4. 落库买家消息（no_reply 不落库，与原项目一致）
        await self.store.add_message(chat_id, chat.sender_id, item_id, "user", chat.text)

        # 5. 专家与上下文
        profile = self.experts.get(intent, self.experts["default"])
        bargain_count = await self.store.get_bargain_count(chat_id)
        system_prompt = f"【商品信息】{item_desc}\n{profile.system_prompt}"
        system_prompt += "\n" + cfg.bargain_policy.prompt(item, bargain_count)
        floor_note = os.getenv("FLOOR_NOTE", "")
        if floor_note:
            system_prompt += f"\n【卖家补充要求】{floor_note}；上述数值价格规则优先。"
        if intent == "price":
            system_prompt += f"\n▲当前议价轮次：{bargain_count}"

        messages: list[Message] = []
        if profile_text:
            messages.append(Message(role="user", content=f"[会话画像]\n{profile_text}"))
        messages += [Message(role=m["role"], content=m["content"]) for m in history]
        messages.append(Message(role="user", content=chat.text))

        ctx = Context(system_prompt=system_prompt, messages=messages,
                      meta={"chat_id": chat_id, "item_id": item_id,
                            "expert": profile.name, "bargain_count": bargain_count})

        # 6. 跑循环
        signal = CancellationToken()
        self._signals[chat_id] = signal
        deadline = Deadline(max_turns=cfg.agent_max_turns,
                            seconds=cfg.agent_deadline_seconds)
        tctx = ToolContext(
            chat_id=chat_id, item_id=item_id, cancel=signal,
            deadline=deadline, store=self.store, item=item,
            floor_note=floor_note,
            bargain_policy=cfg.bargain_policy,
            search=self._search if self._search_enabled else None,
            notify=self._make_notifier(chat),
            bargain_count=bargain_count,
        )
        ctx.meta["require_quote"] = intent == "price" and has_numeric_offer(chat.text)
        if ctx.meta["require_quote"]:
            logger.info("[会话 %s] 检测到数字报价，首轮要求调用 quote_price", chat_id)

        final_text, stop_reason = "", "end_turn"
        try:
            async for ev in run_agent(ctx, profile.tools, tctx, signal,
                                      max_turns=deadline.max_turns,
                                      temperature=profile.temperature_for(bargain_count),
                                      max_tokens=profile.max_tokens,
                                      model=self.models.reply, deadline=deadline):
                if ev["type"] == "tool_call":
                    logger.info("[会话 %s] 调用工具 %s %s", chat_id, ev["name"], ev["args"])
                elif ev["type"] == "tool_result":
                    logger.info("[会话 %s] 工具结果 %s: %s",
                                chat_id, ev["name"], str(ev["result"])[:200])
                elif ev["type"] == "turn_end":
                    final_text = ev.get("text", "")
                    stop_reason = ev["stopReason"]
        finally:
            self._signals.pop(chat_id, None)

        # 7. 收尾：中断的不发，其余强制有一条回复
        if stop_reason == "aborted":
            logger.info("[会话 %s] 回复被打断（人工接管），不发送", chat_id)
            append_snapshot(session_path(chat_id), ctx)
            return

        reply = (final_text or "").strip() or cfg.fallback_reply
        logger.info("[会话 %s] 回复收尾: stop_reason=%s, price_reply=%r, final_text=%r",
                    chat_id, stop_reason, tctx.price_reply, final_text)
        reply = cfg.bargain_policy.guard_reply(
            reply, intent, tctx.price_reply, item=item,
            bargain_count=bargain_count,
            offer_detected=has_numeric_offer(chat.text),
        )
        safe = safety_filter(reply)

        await self.store.add_message(chat_id, cfg.myid, item_id, "assistant", safe)
        if intent == "price":
            await self.store.increment_bargain_count(chat_id)
            logger.info("[会话 %s] 议价次数已递增", chat_id)

        append_snapshot(session_path(chat_id), ctx)

        # 8. 拟人延迟 + 发送（发送入口只有这一处，且只执行一次）
        if cfg.simulate_typing:
            delay = min(random.uniform(0, 1) + len(safe) * random.uniform(0.1, 0.3), 10.0)
            logger.info("[会话 %s] 模拟人工输入，延迟 %.1f 秒", chat_id, delay)
            await asyncio.sleep(delay)

        if self._sender is not None:
            await self._sender(chat_id, chat.sender_id, safe)
        logger.info("[会话 %s] 已回复: %s", chat_id, safe)
        self._emit({"type": "message_sent", "chat_id": chat_id,
                    "intent": intent, "bargain_count": bargain_count, "text": safe})

    def _emit(self, ev: dict) -> None:
        """领域事件：message_sent / bargain_counted。通道与管理台只消费，不参与决策。"""
        if ev.get("type") == "message_sent" and ev.get("intent") == "price":
            logger.info("[事件] bargain_counted 会话=%s 次数=%s",
                        ev["chat_id"], ev.get("bargain_count", 0) + 1)
