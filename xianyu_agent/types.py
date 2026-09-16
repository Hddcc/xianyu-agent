"""所有数据形状都放在这里。

上下文是一个纯 JSON 对象：一段系统提示词 + 一个消息数组。
消息里可以有三种内容块：文字、工具调用、工具结果。
内部只认这一种格式，厂商差异全部关在 llm.py 里。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False


@dataclass
class Message:
    role: str          # "user" 或 "assistant"
    content: Any       # 字符串，或者内容块组成的列表


@dataclass
class Context:
    """一次对话的全部状态。纯数据，可以直接变成 JSON。

    meta 是内核搬运行李时贴的标签（chat_id、item_id、expert、bargain_count），
    模型看不到，但落盘、恢复、审计都要用。
    """
    system_prompt: str = ""
    messages: list[Message] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class IncomingChat:
    """通道解出的一条聊天消息。通道只负责解出它，处理交给内核。"""
    chat_id: str
    item_id: str
    sender_id: str
    sender_name: str
    text: str


_BLOCKS = {"text": TextBlock, "tool_use": ToolUseBlock, "tool_result": ToolResultBlock}


def context_to_dict(ctx: Context) -> dict:
    def enc(m: Message):
        if isinstance(m.content, str):
            return {"role": m.role, "content": m.content}
        return {"role": m.role, "content": [b.__dict__ for b in m.content]}

    return {"system_prompt": ctx.system_prompt,
            "messages": [enc(m) for m in ctx.messages],
            "meta": ctx.meta}


def context_from_dict(d: dict) -> Context:
    msgs = []
    for m in d["messages"]:
        if isinstance(m["content"], str):
            msgs.append(Message(role=m["role"], content=m["content"]))
        else:
            blocks = [_BLOCKS[b["type"]](**{k: v for k, v in b.items() if k != "type"})
                      for b in m["content"]]
            msgs.append(Message(role=m["role"], content=blocks))
    return Context(system_prompt=d["system_prompt"],
                   messages=msgs,
                   meta=d.get("meta", {}))


def context_to_json(ctx: Context) -> str:
    return json.dumps(context_to_dict(ctx), ensure_ascii=False)
