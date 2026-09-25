"""工具。每个工具是一个纯函数：接收参数，返回字符串。

三条纪律，后面所有工具都要遵守：
1. 工具不抛异常——所有错误都变成 "error: ..." 开头的字符串；
2. 工具不做重试——重试是模型的决策；
3. 跑得久的工具要自己认中断信号。

ToolContext 是工具唯一能碰到的外部世界。注意里面没有连接、没有 Cookie、
没有"发给谁"的任何入口——回复发给谁，永远由通道决定。
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .cancel import CancellationToken
from .deadline import Deadline

if TYPE_CHECKING:
    from .pricing import BargainPolicy


@dataclass
class ToolContext:
    """工具唯一能碰到的外部世界。"""
    chat_id: str = ""
    item_id: str = ""
    cancel: CancellationToken = field(default_factory=CancellationToken)
    emit: Callable[[dict], None] = lambda ev: None
    deadline: Deadline | None = None
    store: Any = None              # 能力层：SQLite 存储（商品缓存、议价计数、历史）
    search: Any = None             # 能力层：联网搜索入口
    notify: Any = None             # 能力层：通知卖家（发图/改价等需人工介入时）
    item: dict | None = None       # 组装上下文时已带上的商品摘要
    floor_note: str = ""           # 卖家自述的议价底线
    bargain_policy: BargainPolicy | None = None
    bargain_count: int | None = None
    price_reply: str | None = None  # 由报价/改价工具生成，发送端优先使用


@dataclass
class ToolSpec:
    name: str
    description: str               # 给模型看，决定要不要调
    parameters: dict               # JSON Schema，告诉模型参数长什么样
    execute: Callable[[dict, ToolContext], str]


def truncate(text: str, max_lines: int = 200, keep: str = "tail") -> str:
    """输出太长就在工具内部截断。完整内容落盘，把路径告诉模型。"""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    kept = lines[-max_lines:] if keep == "tail" else lines[:max_lines]
    # 文件名必须唯一，否则并发的两次截断会互相覆盖，模型拿到的路径就是错的
    fd, tmp = tempfile.mkstemp(prefix="truncated_", suffix=".txt")
    with open(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return "\n".join(kept) + f"\n...(共 {len(lines)} 行，完整内容见 {tmp})"


def _read_file(args: dict, tctx: ToolContext) -> str:
    try:
        return truncate(Path(args["path"]).read_text(encoding="utf-8"))
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def _write_file(args: dict, tctx: ToolContext) -> str:
    try:
        p = Path(args["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"], encoding="utf-8")
        return f"已写入 {p}，共 {len(args['content'])} 个字符"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def _edit(args: dict, tctx: ToolContext) -> str:
    """用精确字符串替换，不用行号——连续编辑之后行号会漂移。"""
    p = Path(args["path"])
    try:
        src = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"

    old, new = args["old_string"], args["new_string"]
    n = src.count(old)
    if n == 0:
        return "error: old_string 在文件里找不到"
    if n > 1:
        return f"error: old_string 在文件里出现了 {n} 次，必须唯一匹配"

    p.write_text(src.replace(old, new), encoding="utf-8")
    return f"已替换 {p}"


def _run_bash(args: dict, tctx: ToolContext) -> str:
    """命令失败不抛异常，把退出码和输出一起返回——失败也是信息。"""
    proc = subprocess.Popen(args["command"], shell=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:                                   # 中断信号要一路传到子进程
        try:
            out, _ = proc.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if tctx.cancel.cancelled:
                proc.kill()
                proc.communicate()
                return "error: aborted（命令已被中断）"
    return f"exit={proc.returncode}\n{truncate(out or '')}"


def builtin_tools() -> list[ToolSpec]:
    """四个通用工具：先把循环跑通用，第三十步之后换成领域工具。"""
    return [
        ToolSpec("read_file", "读取文件内容。",
                 {"type": "object",
                  "properties": {"path": {"type": "string", "description": "文件路径"}},
                  "required": ["path"]},
                 _read_file),
        ToolSpec("write_file", "把内容写入文件。",
                 {"type": "object",
                  "properties": {"path": {"type": "string"},
                                 "content": {"type": "string"}},
                  "required": ["path", "content"]},
                 _write_file),
        ToolSpec("edit", "用精确字符串替换文件中的一段。old_string 必须在文件中唯一。",
                 {"type": "object",
                  "properties": {"path": {"type": "string"},
                                 "old_string": {"type": "string"},
                                 "new_string": {"type": "string"}},
                  "required": ["path", "old_string", "new_string"]},
                 _edit),
        ToolSpec("run_bash", "执行一条 shell 命令。",
                 {"type": "object",
                  "properties": {"command": {"type": "string"}},
                  "required": ["command"]},
                 _run_bash),
    ]
