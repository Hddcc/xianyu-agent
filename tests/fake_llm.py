"""按脚本产出流式事件的假模型，替换真实调用——测试不依赖任何网络。"""


class FakeLLM:
    def __init__(self, script=None):
        # script: 每一轮要产出的事件列表；耗尽后默认"说一句话就结束"
        self.script = script or []
        self.i = 0
        self.calls = []

    async def stream(self, ctx, tools=None, signal=None, model=None, *,
                     temperature=0.7, max_tokens=500, top_p=0.8, **kwargs):
        self.calls.append({"messages": list(ctx.messages), "tools": tools})
        if self.i < len(self.script):
            events = self.script[self.i]
        else:
            events = [
                {"type": "text_delta", "delta": "好的。"},
                {"type": "done", "stopReason": "end_turn"},
            ]
        self.i += 1
        for ev in events:
            yield ev
