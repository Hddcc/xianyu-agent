"""xianyu_agent：基于 Pi 架构的闲鱼智能客服 Agent。

分层结构（依赖单向：传输层 -> 内核 -> 工具 -> 能力层）：
- 传输层：channel.py（闲鱼通道）、cli.py（入口/终端调试）、ui.py
- 内核：agent.py（循环）、types.py（上下文）、llm.py（模型适配）、router.py（意图路由）
- 工具层：tools.py（通用工具）、kf_tools.py（领域工具）
- 能力层：store.py（SQLite）、xianyu_api.py / xianyu_utils.py（平台对接）
"""

__version__ = "0.1.0"
