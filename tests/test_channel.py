from xianyu_agent.channel import XianyuChannel


def make_channel():
    channel = object.__new__(XianyuChannel)
    channel.message_dedup_time = 300000
    channel._seen_message_keys = {}
    return channel


def test_duplicate_chat_message_is_seen_once():
    channel = make_channel()
    assert channel._is_duplicate_message("same-message") is False
    assert channel._is_duplicate_message("same-message") is True
    assert channel._is_duplicate_message("different-message") is False
