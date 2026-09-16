"""数据形状测试：序列化往返必须一模一样。"""


from xianyu_agent.types import (Context, IncomingChat, Message, TextBlock,
                                ToolResultBlock, ToolUseBlock,
                                context_from_dict, context_to_dict, context_to_json)


def test_roundtrip_plain_messages():
    ctx = Context(system_prompt="你是卖家",
                  messages=[Message(role="user", content="在吗"),
                            Message(role="assistant", content="在的")],
                  meta={"chat_id": "c1", "item_id": "i1"})
    restored = context_from_dict(context_to_dict(ctx))
    assert restored.system_prompt == ctx.system_prompt
    assert restored.meta == ctx.meta
    assert [m.content for m in restored.messages] == ["在吗", "在的"]


def test_roundtrip_with_blocks():
    ctx = Context(system_prompt="s", messages=[
        Message(role="assistant", content=[
            TextBlock(text="我看下商品"),
            ToolUseBlock(id="t1", name="get_item_info", input={}),
        ]),
        Message(role="user", content=[
            ToolResultBlock(tool_use_id="t1", content="商品：音响", is_error=False),
        ]),
    ])
    restored = context_from_dict(context_to_dict(ctx))
    blocks_a = restored.messages[0].content
    blocks_u = restored.messages[1].content
    assert isinstance(blocks_a[0], TextBlock) and blocks_a[0].text == "我看下商品"
    assert isinstance(blocks_a[1], ToolUseBlock)
    assert blocks_a[1].name == "get_item_info" and blocks_a[1].input == {}
    assert isinstance(blocks_u[0], ToolResultBlock)
    assert blocks_u[0].tool_use_id == "t1" and blocks_u[0].content == "商品：音响"


def test_json_serializable():
    ctx = Context(system_prompt="s", messages=[Message(role="user", content="你好")])
    assert '"你好"' in context_to_json(ctx)     # ensure_ascii=False


def test_incoming_chat():
    chat = IncomingChat(chat_id="c", item_id="i", sender_id="u",
                        sender_name="买家", text="在吗")
    assert chat.text == "在吗"
