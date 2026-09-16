"""会话落盘。每轮结束后把整份上下文写成一行 JSONL。

这是"行车记录仪"：回答"模型当时看到了什么"。与之相对，SQLite（store.py）
是"对账簿"，回答"到底发生过什么"。两者分工，缺一不可。
"""
from __future__ import annotations

import json
from pathlib import Path

from .types import Context, context_to_dict, context_from_dict


def session_path(chat_id: str) -> Path:
    p = Path("data/sessions") / chat_id
    p.mkdir(parents=True, exist_ok=True)
    return p / "session.jsonl"


def append_snapshot(path: Path, ctx: Context) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "snapshot", "ctx": context_to_dict(ctx)},
                           ensure_ascii=False) + "\n")


def load_latest(path: Path) -> Context | None:
    """从最后一行往前找第一个能解析的快照。坏行直接跳过。"""
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return context_from_dict(json.loads(line)["ctx"])
        except (json.JSONDecodeError, KeyError):
            continue                       # 这就是 JSONL 的容错性
    return None
