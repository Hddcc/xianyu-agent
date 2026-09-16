"""循环的单元测试。用假的模型响应替换真实调用，不依赖任何网络。

覆盖内核必须处理的全部场景：
正常结束 / 工具回灌 / 长度截断 / 两种中断 / 请求失败 / 畸形响应 /
轮数上限 / 时限硬停 / 压缩不切断配对。
"""
import pytest

from xianyu_agent import agent as agent_mod
from xianyu_agent.cancel import CancellationToken
from xianyu_agent.deadline import Deadline
from xianyu_agent.tools import ToolContext, ToolSpec
from xianyu_agent.types import Context, Message, ToolResultBlock, ToolUseBlock

from .fake_llm import FakeLLM


def make_tool(name, fn):
    return ToolSpec(name, f"test tool {name}", {"type": "object", "properties": {}}, fn)


def make_ctx(user_text="你好"):
    return Context(system_prompt="test",
                   messages=[Message(role="user", content=user_text)])


async def collect(agen):
    return [ev async for ev in agen]


@pytest.fixture
def patch_llm(monkeypatch):
    def _patch(fake):
        monkeypatch.setattr(agent_mod.llm, "stream", fake.stream)
        return fake
    return _patch


def last_end(events):
    return [e for e in events if e["type"] == "turn_end"][-1]


# ---------------------------------------------------------------- 正常路径

async def test_no_tool_calls_ends_normally(patch_llm):
    patch_llm(FakeLLM([[{"type": "text_delta", "delta": "在的"},
                        {"type": "done", "stopReason": "end_turn"}]]))
    ctx = make_ctx()
    signal = CancellationToken()
    tctx = ToolContext(cancel=signal)

    events = await collect(agent_mod.run_agent(ctx, [], tctx, signal))

    assert last_end(events)["stopReason"] == "end_turn"
    assert last_end(events)["text"] == "在的"
    assert len(ctx.messages) == 2          # 用户一条 + 助手一条


async def test_tool_calls_execute_and_feed_back(patch_llm):
    script = [
        [{"type": "text_delta", "delta": "我看下"},
         {"type": "tool_call", "id": "t1", "name": "echo", "args": {"x": 1}},
         {"type": "done", "stopReason": "tool_use"}],
        [{"type": "text_delta", "delta": "结果是1"},
         {"type": "done", "stopReason": "end_turn"}],
    ]
    fake = patch_llm(FakeLLM(script))
    executed = []
    tool = make_tool("echo", lambda args, tctx: executed.append(args) or f"echo:{args['x']}")

    ctx = make_ctx()
    events = await collect(agent_mod.run_agent(ctx, [tool], ToolContext(cancel=CancellationToken()),
                                               CancellationToken()))

    assert executed == [{"x": 1}]
    assert last_end(events)["text"] == "结果是1"
    # 第二次问模型时，最后一条消息是嵌着工具结果的 user 消息
    last_seen = fake.calls[1]["messages"][-1]
    assert last_seen.role == "user"
    assert any(isinstance(b, ToolResultBlock) and b.content == "echo:1"
               for b in last_seen.content)
    # 上下文：user / assistant(text+tool_use) / user(tool_result) / assistant(text)
    assert len(ctx.messages) == 4


# ---------------------------------------------------------------- 四种意外

async def test_max_tokens_truncation_skips_tools_and_reprompts(patch_llm):
    script = [
        [{"type": "text_delta", "delta": "我调一下"},
         {"type": "tool_call", "id": "t1", "name": "echo", "args": {"x": 1}},
         {"type": "done", "stopReason": "max_tokens"}],      # 被截断
        [{"type": "text_delta", "delta": "重发好了"},
         {"type": "done", "stopReason": "end_turn"}],
    ]
    patch_llm(FakeLLM(script))
    executed = []
    tool = make_tool("echo", lambda args, tctx: executed.append(args) or "ok")

    ctx = make_ctx()
    events = await collect(agent_mod.run_agent(ctx, [tool], ToolContext(cancel=CancellationToken()),
                                               CancellationToken()))

    assert executed == []                  # 截断的工具调用一个都不执行
    assert last_end(events)["stopReason"] == "end_turn"
    # 回灌的提示以 is_error 的工具结果出现
    second_seen = [m for m in ctx.messages if not isinstance(m.content, str)]
    assert any(b.is_error and "截断" in b.content
               for m in second_seen for b in m.content
               if isinstance(b, ToolResultBlock))


async def test_abort_while_streaming_drops_pending_calls(patch_llm):
    patch_llm(FakeLLM([[{"type": "text_delta", "delta": "部分文字"},
                        {"type": "tool_call", "id": "t1", "name": "echo", "args": {}},
                        {"type": "done", "stopReason": "aborted"}]]))
    executed = []
    tool = make_tool("echo", lambda args, tctx: executed.append(1) or "ok")

    ctx = make_ctx()
    signal = CancellationToken()
    events = await collect(agent_mod.run_agent(ctx, [tool], ToolContext(cancel=signal), signal))

    assert executed == []                  # 未执行的工具调用直接丢掉
    assert last_end(events)["stopReason"] == "aborted"
    # 保存的文字留下，但没有孤儿工具调用
    last_msg = ctx.messages[-1]
    assert last_msg.role == "assistant"
    assert not any(isinstance(b, ToolUseBlock) for b in last_msg.content)


async def test_abort_during_execution_pads_pairs(patch_llm):
    patch_llm(FakeLLM([[{"type": "tool_call", "id": "t1", "name": "first", "args": {}},
                        {"type": "tool_call", "id": "t2", "name": "second", "args": {}},
                        {"type": "done", "stopReason": "tool_use"}]]))

    def _first(args, tctx):
        tctx.cancel.cancel()               # 第一个工具执行中触发中断
        return "ok"

    def _second(args, tctx):
        raise AssertionError("第二个工具不应被执行")

    tools = [make_tool("first", _first), make_tool("second", _second)]
    ctx = make_ctx()
    signal = CancellationToken()
    events = await collect(agent_mod.run_agent(ctx, tools, ToolContext(cancel=signal), signal))

    assert last_end(events)["stopReason"] == "aborted"
    # 配对补齐：两个调用都有结果，第二个是 error: aborted
    last_msg = ctx.messages[-1]
    results = [b for b in last_msg.content if isinstance(b, ToolResultBlock)]
    assert len(results) == 2
    assert results[0].content == "ok"
    assert results[1].is_error and results[1].content == "error: aborted"


async def test_request_error_never_retries(patch_llm):
    fake = patch_llm(FakeLLM([[{"type": "done", "stopReason": "error",
                                "message": "HTTP 500"}]]))
    ctx = make_ctx()
    events = await collect(agent_mod.run_agent(ctx, [], ToolContext(cancel=CancellationToken()),
                                               CancellationToken()))

    assert last_end(events)["stopReason"] == "error"
    assert len(fake.calls) == 1            # 只调了一次，没有原地重试


async def test_empty_tool_list_treated_as_end(patch_llm):
    # 声明要调工具（stopReason=tool_use）但列表为空的畸形响应
    patch_llm(FakeLLM([[{"type": "text_delta", "delta": "答完了"},
                        {"type": "done", "stopReason": "tool_use"}]]))
    ctx = make_ctx()
    events = await collect(agent_mod.run_agent(ctx, [], ToolContext(cancel=CancellationToken()),
                                               CancellationToken()))
    assert last_end(events)["stopReason"] == "end_turn"


# ---------------------------------------------------------------- 上限

async def test_max_turns_hard_stop(patch_llm):
    turn = [{"type": "tool_call", "id": "t", "name": "echo", "args": {}},
            {"type": "done", "stopReason": "tool_use"}]
    patch_llm(FakeLLM([turn, turn, turn, turn]))     # 永远想调工具
    tool = make_tool("echo", lambda args, tctx: "ok")

    ctx = make_ctx()
    events = await collect(agent_mod.run_agent(ctx, [tool],
                                               ToolContext(cancel=CancellationToken()),
                                               CancellationToken(), max_turns=3))
    assert last_end(events)["stopReason"] == "max_turns"


async def test_deadline_hard_stop(patch_llm):
    fake = patch_llm(FakeLLM())
    ctx = make_ctx()
    deadline = Deadline(seconds=-1.0)      # 一出生就过期
    events = await collect(agent_mod.run_agent(ctx, [], ToolContext(cancel=CancellationToken()),
                                               CancellationToken(), deadline=deadline))
    assert last_end(events)["stopReason"] == "deadline"
    assert fake.calls == []                # 模型一次都没被调用


# ---------------------------------------------------------------- 压缩

def test_safe_cut_avoids_orphan_tool_result():
    msgs = [Message("user", f"m{i}") for i in range(30)]
    msgs.append(Message("assistant", [ToolUseBlock(id="a", name="t", input={})]))   # 30
    msgs.append(Message("user", [ToolResultBlock(tool_use_id="a", content="ok")]))  # 31
    msgs += [Message("user", f"t{i}") for i in range(19)]                           # 共 51

    cut = agent_mod._safe_cut(msgs, 31)
    assert cut == 30                       # 挪到 assistant，把配对整体留在 recent


async def test_maybe_compact_replaces_old_messages(patch_llm):
    patch_llm(FakeLLM([[{"type": "text_delta", "delta": "买家想砍价到300，已报348。"},
                        {"type": "done", "stopReason": "end_turn"}]]))
    msgs = [Message("user", f"m{i}") for i in range(40)]
    msgs.append(Message("assistant", [ToolUseBlock(id="a", name="t", input={})]))
    msgs.append(Message("user", [ToolResultBlock(tool_use_id="a", content="ok")]))
    msgs += [Message("user", f"t{i}") for i in range(10)]        # 共 52 > 阈值 50
    ctx = Context(system_prompt="s", messages=msgs)

    changed = await agent_mod.maybe_compact(ctx, None)

    assert changed
    assert len(ctx.messages) < 52
    assert isinstance(ctx.messages[0].content, str)
    assert ctx.messages[0].content.startswith("[会话画像]")
    # recent 保留的部分不能以孤儿工具结果开头
    assert not agent_mod._is_tool_result(ctx.messages[1])


async def test_maybe_compact_skips_when_cancelled(patch_llm):
    fake = patch_llm(FakeLLM())
    ctx = Context(system_prompt="s",
                  messages=[Message("user", f"m{i}") for i in range(60)])
    signal = CancellationToken()
    signal.cancel()

    changed = await agent_mod.maybe_compact(ctx, signal)

    assert changed is False                # 中断时跳过，半截摘要比不压缩更危险
    assert len(ctx.messages) == 60
    assert fake.calls == []
