"""专家配置。每个专家 = 一份提示词 + 一套工具 + 一条温度策略。

原项目用 Agent 子类承载差异（PriceAgent 重写 generate、TechAgent 加搜索参数）；
我们把差异收敛成数据：提示词是数据，工具集是数据，温度曲线是数据。
议价专家的"越谈越松"从散落在子类里的魔法数字，变成一个有名字、
可单独测试的函数。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from .kf_tools import kf_tools
from .tools import ToolSpec


@dataclass
class ExpertProfile:
    name: str
    system_prompt: str
    tools: list[ToolSpec] = field(default_factory=kf_tools)
    temperature: float = 0.4
    temperature_curve: Callable[[int], float] | None = None
    max_tokens: int = 500

    def temperature_for(self, bargain_count: int) -> float:
        if self.temperature_curve:
            return self.temperature_curve(bargain_count)
        return self.temperature


def _price_curve(bargain_count: int) -> float:
    """越谈越松。原项目验证过的策略：0.3 起步，每轮加 0.15，封顶 0.9。"""
    return min(0.3 + bargain_count * 0.15, 0.9)


def _load_prompt(name: str, prompt_dir: str) -> str:
    """用户文件优先于 example 文件——热加载就靠重新调 load_experts()。"""
    for filename in (f"{name}.txt", f"{name}_example.txt"):
        path = os.path.join(prompt_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(f"找不到提示词 {name}（在 {prompt_dir} 下）")


def load_experts(prompt_dir: str | None = None) -> dict[str, ExpertProfile]:
    prompt_dir = prompt_dir or os.getenv("PROMPTS_DIR", "prompts")
    return {
        "price": ExpertProfile("price", _load_prompt("price_prompt", prompt_dir),
                               temperature_curve=_price_curve),
        "tech": ExpertProfile("tech", _load_prompt("tech_prompt", prompt_dir)),
        "default": ExpertProfile("default", _load_prompt("default_prompt", prompt_dir),
                                 temperature=0.7),
    }
