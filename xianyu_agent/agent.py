"""Agent 循环。整个项目的核心。

问模型 -> 写回 -> 判断 -> 执行 -> 写回，直到模型说"好了"。
四种意外的处理都在这里：
1. 长度截断：一个工具都不执行，回灌提示让模型重发；
2. 中断：等模型时丢掉未执行的调用；执行中补齐"error: aborted"配对；
3. 请求失败：直接收尾，绝不原地重试；
4. 畸形响应：声明要调工具但列表为空，按正常结束处理。

外加本项目的第三种停法：时限到了，用已生成的文字强制收尾。
"""
from __future__ import annotations

import inspect
import logging

from . import llm
from .cancel import Cancelled
from .types import Context, Message

logger = logging.getLogger(__name__)

COMPACT_THRESHOLD = 50     # 消息超过这么多条就压（安全网，常规触发靠会话画像）
KEEP_RECENT = 20           # 最近这二十条原样保留

SUMMARY_PROMPT = """请把下面的客服对话压缩成一段简洁的会话画像，保留：
1. 买家的核心诉求
2. 已报价的完整轨迹（每一轮谁报了什么价，逐条列出，一条都不能少）
3. 当前议价轮次
4. 对买家类型的判断（捡漏型 / 爽快型 / 比价型）
不要保留完整原文，需要时可以用工具重新读取。"""


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def build_assistant_message(text, tool_calls):
    return llm.build_assistant_message(text, tool_calls)


def build_tool_result_message(results):
    return llm.build_tool_result_message(results)


async def run_agent(ctx, tools, tool_context, signal, *,
                    max_turns: int = 24,
                    temperature: float = 0.4,
                    max_tokens: int = 500,
                    top_p: float = 0.8,
                    model=None,
                    deadline=None):
    """异步生成器，逐个产出 AgentEvent。

    事件只有四种：assistant_text / tool_call / tool_result / turn_end。
    这是"业务语义"，比模型那边碎片的事件高一层。
    turn_end 事件附带 text：多轮时取最后一轮的文字，作为最终回复。
    """
    tool_map = {t.name: t for t in tools}
    tool_defs = [{"name": t.name, "description": t.description, "parameters": t.parameters}
                 for t in tools]
    turn = ctx.meta.get("turn", 0)
    last_text = ""

    while turn < max_turns:
        # 0. 时限检查（循环层拦截）+ 压缩
        if deadline is not None and deadline.expired:
            yield {"type": "turn_end", "stopReason": "deadline", "text": last_text}
            return
        await maybe_compact(ctx, signal, model=model)

        turn += 1
        ctx.meta["turn"] = turn

        # 1. 问模型
        text, tool_calls, stop = "", [], "end_turn"
        async for ev in llm.stream(ctx, tools=tool_defs, signal=signal, model=model,
                                   temperature=temperature,
                                   max_tokens=max_tokens, top_p=top_p):
            if ev["type"] == "text_delta":
                text += ev["delta"]
                yield {"type": "assistant_text", "delta": ev["delta"]}
            elif ev["type"] == "tool_call":
                tool_calls.append(ev)
                yield {"type": "tool_call", "id": ev["id"],
                       "name": ev["name"], "args": ev["args"]}
            elif ev["type"] == "done":
                stop = ev["stopReason"]

        if text:
            last_text = text        # 多轮时，最终回复取最后一轮的文字

        # 2. 把回复写回上下文
        ctx.messages.append(build_assistant_message(text, tool_calls))

        # 3. 判断要不要停（顺序固定：error -> aborted -> max_tokens -> not tool_calls）
        if stop == "error":
            yield {"type": "turn_end", "stopReason": "error", "text": last_text}
            return
        if stop == "aborted":
            # 只保存文字，丢掉还没执行的工具调用，避免留下没有结果的调用
            if text:
                ctx.messages[-1] = build_assistant_message(text, [])
            else:
                ctx.messages.pop()   # 一个字都没收到，这条空消息不能留，有厂商会拒收
            yield {"type": "turn_end", "stopReason": "aborted", "text": ""}
            return
        if stop == "max_tokens" and tool_calls:
            # 意外一：长度截断。残缺的参数可能拼出错误的报价，一个都不执行
            ctx.messages.append(build_tool_result_message([
                {"tool_use_id": tc["id"], "is_error": True,
                 "content": "error: 输出被长度上限截断，参数可能不完整，请重新完整发起一次调用。"}
                for tc in tool_calls]))
            continue          # 进下一轮让模型重发

        if not tool_calls:
            # 意外四：声明要调工具但列表为空的畸形响应，天然按正常结束处理
            yield {"type": "turn_end", "stopReason": "end_turn", "text": last_text}
            return

        # 4. 执行工具
        results = []
        aborted = False
        for tc in tool_calls:
            if signal.cancelled or (deadline is not None and deadline.expired):
                aborted = True
                break
            spec = tool_map.get(tc["name"])
            if spec is None:
                content = f"error: 没有名为 \"{tc['name']}\" 的工具"
            else:
                try:
                    signal.raise_if_cancelled()
                    content = await _maybe_await(spec.execute(tc["args"], tool_context))
                except Cancelled:
                    aborted = True
                    break
                except Exception as e:
                    logger.exception("工具 %s 执行失败", tc["name"])
                    content = f"error: {type(e).__name__}: {e}"
            results.append({"tool_use_id": tc["id"], "content": content})
            yield {"type": "tool_result", "id": tc["id"],
                   "name": tc["name"], "result": content}

        # 配对补齐：每个工具调用都必须有对应的结果，否则下次发出去接口直接报错
        for tc in tool_calls[len(results):]:
            results.append({"tool_use_id": tc["id"], "content": "error: aborted",
                            "is_error": True})

        ctx.messages.append(build_tool_result_message(results))

        if aborted:
            reason = "aborted" if signal.cancelled else "deadline"
            yield {"type": "turn_end", "stopReason": reason, "text": last_text}
            return

        # 5. 把结果写回上下文（已在上面的 append 完成），进入下一轮

    yield {"type": "turn_end", "stopReason": "max_turns", "text": last_text}


# ---------------------------------------------------------------- 压缩

async def maybe_compact(ctx: Context, signal, model=None) -> bool:
    """消息太多就压。中断时跳过——半截摘要比不压缩更危险。"""
    if len(ctx.messages) < COMPACT_THRESHOLD:
        return False
    if signal is not None and signal.cancelled:
        return False

    cut = _safe_cut(ctx.messages, len(ctx.messages) - KEEP_RECENT)
    old, recent = ctx.messages[:cut], ctx.messages[cut:]
    if not old:
        return False

    # 摘要请求用一份独立的上下文，不污染主上下文
    sub = Context(system_prompt=SUMMARY_PROMPT,
                  messages=[Message(role="user", content=_render(old))])
    summary = ""
    async for ev in llm.stream(sub, signal=signal, model=model):
        if ev["type"] == "text_delta":
            summary += ev["delta"]

    ctx.messages = [Message(role="user", content=f"[会话画像]\n{summary}")] + recent
    return True


def _is_tool_result(m: Message) -> bool:
    return (not isinstance(m.content, str)
            and any(b.type == "tool_result" for b in m.content))


def _safe_cut(msgs: list[Message], cut: int) -> int:
    """把切割点往前挪到一个干净边界。

    不能让 recent 以一条孤儿工具结果开头——发起调用的那条 assistant 消息
    要是被压进了摘要里，接口会因为"有结果没有调用"直接报错。
    """
    while cut > 0 and _is_tool_result(msgs[cut]):
        cut -= 1          # 把发起调用的那条一起留在 recent 里
    return cut


def _render(msgs: list[Message]) -> str:
    parts = []
    for m in msgs:
        if isinstance(m.content, str):
            parts.append(f"{m.role}: {m.content}")
        else:
            for b in m.content:
                parts.append(f"{m.role}/{b.type}: "
                             f"{getattr(b, 'text', None) or getattr(b, 'content', '')}")
    return "\n".join(parts)
