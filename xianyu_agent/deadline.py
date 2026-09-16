"""时限。买家不会等：一条买家消息从进来到回复发出，必须有硬顶。

管三件事：最大轮数（很小）、总时限（到点用已生成的文字强制收尾）、
以及一条隐含的硬规则——一条买家消息只发送一次回复（由 registry 守）。
"""
import time
from dataclasses import dataclass, field


@dataclass
class Deadline:
    max_turns: int = 4
    seconds: float = 30.0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.started_at > self.seconds

    def remaining(self) -> float:
        return max(0.0, self.seconds - (time.monotonic() - self.started_at))
