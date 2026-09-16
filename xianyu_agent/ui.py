"""终端界面。只认识事件，不知道模型和工具的存在。"""
from __future__ import annotations

import sys


class Ui:
    def print_text(self, delta: str):
        sys.stdout.write(delta)
        sys.stdout.flush()

    def print_tool_call(self, name: str, args: dict):
        sys.stdout.write(f"\n[工具] {name} {args}\n")
        sys.stdout.flush()

    def print_tool_result(self, name: str, result: str):
        sys.stdout.write(f"[结果] {name}: {result[:400]}\n")
        sys.stdout.flush()

    def print_turn_end(self, reason: str):
        sys.stdout.write(f"\n—— 本轮结束（{reason}）——\n\n")
        sys.stdout.flush()

    def print_event(self, ev: dict):
        """领域事件（message_sent / manual_mode / bargain_counted）的流水。"""
        sys.stdout.write(f"[事件] {ev}\n")
        sys.stdout.flush()
