"""中断信号。Pi 里 AbortController 的等价物。一路传给请求、工具、子进程。

本项目中它的典型触发场景：卖家发了接管关键词，而此时一条回复正在生成——
正在生成的回复必须立刻作废，否则人工接管了还会蹦出一条 AI 回复。
"""
import threading


class Cancelled(Exception):
    pass


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self):
        if self._event.is_set():
            raise Cancelled()
