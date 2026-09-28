"""结构化运行事件。

业务模块只依赖一个可调用的记录接口。生产环境默认静默，评测可以换成
内存或 JSONL 适配器；记录失败不会打断客服主流程。
"""
from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path
from threading import Lock
import time
from typing import Callable

logger = logging.getLogger(__name__)

EventSink = Callable[[dict], None]


def noop_sink(event: dict) -> None:
    """默认适配器：保持生产行为不变。"""


def emit(sink: EventSink | None, event_type: str, **fields) -> dict:
    """补齐公共字段并安全地交给记录适配器。"""
    event = {"type": event_type, "timestamp": time.time(), **fields}
    if sink is None:
        return event
    try:
        sink(event)
    except Exception:
        logger.exception("记录运行事件失败: %s", event_type)
    return event


class MemoryEventSink:
    """测试和离线评测使用的内存适配器。"""

    def __init__(self):
        self.events: list[dict] = []
        self._lock = Lock()

    def __call__(self, event: dict) -> None:
        with self._lock:
            self.events.append(deepcopy(event))

    def for_run(self, run_id: str) -> list[dict]:
        with self._lock:
            return [deepcopy(event) for event in self.events
                    if event.get("run_id") == run_id]


class JsonlEventSink:
    """逐行持久化事件，便于离线聚合和复查单次运行。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def __call__(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
