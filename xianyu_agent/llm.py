"""跟模型通信。输入 Context，输出一串流式事件。

全项目只有这一个模块知道厂商格式的差异：
- 内部格式 -> 厂商格式，只有 context_to_provider_messages 一个函数做；
- 流式响应 -> 统一事件（text_delta / tool_call / done）；
- 分片到达的工具调用参数，先暂存、流读完再拼完整。

模型分工（客服领域的成本经验）：分类要快和便宜，回复要好和稳。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import httpx

from .types import Context, Message, TextBlock, ToolUseBlock, ToolResultBlock

logger = logging.getLogger(__name__)

_STOP_MAP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


@dataclass
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    format: str = "openai"          # openai / anthropic / gemini / ...（预留）


@dataclass
class Models:
    classify: ModelConfig           # 意图分类：高频、低难度，便宜快就够
    reply: ModelConfig              # 专家回复：直接面对买家，质量优先


def load_models_from_env() -> Models:
    api_key = os.getenv("API_KEY", "")
    base_url = os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    reply_name = os.getenv("MODEL_NAME", "qwen-max")
    classify_name = os.getenv("CLASSIFY_MODEL_NAME") or reply_name
    return Models(
        classify=ModelConfig(classify_name, base_url, api_key),
        reply=ModelConfig(reply_name, base_url, api_key),
    )


_models: Models | None = None


def default_models() -> Models:
    global _models
    if _models is None:
        _models = load_models_from_env()
    return _models


def context_to_provider_messages(ctx: Context) -> list[dict]:
    """内部格式 -> 厂商要求的格式。全项目只有这一处做这种转换。

    工具结果在内部嵌在用户消息里；厂商要求拆成独立的 tool 消息，
    在这里转。工具结果嵌在用户消息里返回时也一样在这里转回去。
    """
    out = []
    if ctx.system_prompt:
        out.append({"role": "system", "content": ctx.system_prompt})

    for m in ctx.messages:
        if isinstance(m.content, str):
            out.append({"role": m.role, "content": m.content})
            continue

        text_parts, tool_calls, tool_results = [], [], []
        for b in m.content:
            if isinstance(b, TextBlock) and b.text:
                text_parts.append(b.text)
            elif isinstance(b, ToolUseBlock):
                tool_calls.append({
                    "id": b.id, "type": "function",
                    "function": {"name": b.name,
                                 "arguments": json.dumps(b.input, ensure_ascii=False)},
                })
            elif isinstance(b, ToolResultBlock):
                tool_results.append(b)

        if m.role == "assistant":
            msg = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        else:
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})
            for tr in tool_results:
                out.append({"role": "tool", "tool_call_id": tr.tool_use_id,
                            "content": tr.content})
    return out


def build_assistant_message(text: str, tool_calls: list[dict]) -> Message:
    """把这一轮的文字和工具调用打包成一条助手消息。"""
    content = []
    if text:
        content.append(TextBlock(text=text))
    for tc in tool_calls:
        content.append(ToolUseBlock(id=tc["id"], name=tc["name"], input=tc["args"]))
    return Message(role="assistant", content=content)


def build_tool_result_message(results: list[dict]) -> Message:
    """把一批工具结果打包成一条用户消息。"""
    return Message(role="user", content=[
        ToolResultBlock(tool_use_id=r["tool_use_id"],
                        content=r["content"],
                        is_error=r.get("is_error", False))
        for r in results])


async def stream(ctx: Context, tools: list[dict] | None = None, signal=None,
                 model: ModelConfig | None = None, *,
                 temperature: float = 0.7, max_tokens: int = 500, top_p: float = 0.8,
                 tool_choice=None):
    """异步生成器，逐个产出流式事件。

    事件只有四种：text_delta / tool_call / done。
    这是"传输语义"，比循环对外产出的事件要碎。
    """
    cfg = model or default_models().reply
    body = {"model": cfg.name,
            "messages": context_to_provider_messages(ctx),
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p}
    if tools:
        body["tools"] = [{"type": "function",
                          "function": {"name": t["name"],
                                       "description": t["description"],
                                       "parameters": t["parameters"]}}
                         for t in tools]
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    finish = None
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST", f"{cfg.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                json=body,
            ) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "ignore")[:500]
                    yield {"type": "done", "stopReason": "error",
                           "message": f"HTTP {resp.status_code}: {detail}"}
                    return

                buffers: dict[int, dict] = {}
                async for line in resp.aiter_lines():
                    if signal is not None and signal.cancelled:
                        yield {"type": "done", "stopReason": "aborted"}
                        return
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    chunk = json.loads(data)
                    choice = chunk["choices"][0]
                    delta = choice.get("delta") or {}

                    if delta.get("content"):
                        yield {"type": "text_delta", "delta": delta["content"]}

                    # 工具调用的参数是分片来的，先按序号暂存
                    for tc in delta.get("tool_calls") or []:
                        i = tc["index"]
                        buf = buffers.setdefault(i, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            buf["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            buf["name"] = fn["name"]
                        if fn.get("arguments"):
                            buf["args"] += fn["arguments"]

                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]

                # 流读完了，把暂存的参数拼完整再发出去
                for buf in buffers.values():
                    try:
                        args = json.loads(buf["args"]) if buf["args"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    yield {"type": "tool_call", "id": buf["id"],
                           "name": buf["name"], "args": args}

                yield {"type": "done", "stopReason": _STOP_MAP.get(finish, "end_turn")}

    except Exception as e:                      # 网络问题、解析问题，统一变成 error 事件
        yield {"type": "done", "stopReason": "error", "message": str(e)}


def complete(ctx: Context, model: ModelConfig | None = None, *,
             temperature: float = 0.1, max_tokens: int = 50,
             extra_body: dict | None = None) -> str:
    """非流式一次性调用。给分类器这种"只要一个词"的场景用。"""
    cfg = model or default_models().classify
    body = {"model": cfg.name,
            "messages": context_to_provider_messages(ctx),
            "temperature": temperature,
            "max_tokens": max_tokens}
    if extra_body:
        body.update(extra_body)
    try:
        resp = httpx.post(f"{cfg.base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {cfg.api_key}"},
                          json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""
    except Exception as e:
        logger.error("complete 调用失败: %s", e)
        return ""


def search_web(query: str, model: ModelConfig | None = None) -> str:
    """联网搜索。走 DashScope 的 enable_search（模型需支持）。

    原项目把搜索绑死在 tech 专家的生成参数上；我们把它做成独立能力，
    由 web_search 工具调用，模型自己决定要不要搜——换厂商时能力不丢。
    """
    cfg = model or default_models().reply
    ctx = Context(system_prompt="你是一个联网搜索助手。请联网搜索并简要总结答案，"
                                "如果搜不到就直说搜不到。控制在 200 字以内。",
                  messages=[Message(role="user", content=query)])
    return complete(ctx, model=cfg, temperature=0.3, max_tokens=400,
                    extra_body={"enable_search": True})
