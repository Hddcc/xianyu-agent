import json
import time

from xianyu_agent import channel as channel_module
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


class RecordingDispatcher:
    def __init__(self):
        self.events = []
        self.handled = []

    def record_event(self, event_type, **fields):
        self.events.append({"type": event_type, **fields})

    async def handle(self, chat, **kwargs):
        self.handled.append((chat, kwargs))


def raw_chat_message(created_at: int, *, include_url: bool = True):
    detail = {
        "reminderTitle": "买家", "senderUserId": "buyer",
        "reminderContent": "还在吗",
    }
    if include_url:
        detail["reminderUrl"] = "https://example.test/?itemId=item-1&x=1"
    decrypted = {"1": {"2": "chat-1@goofish", "5": str(created_at), "10": detail}}
    envelope = {"body": {"syncPushPackage": {"data": [{"data": "invalid"}]}}}
    return envelope, decrypted


def observable_channel(dispatcher):
    channel = make_channel()
    channel.dispatcher = dispatcher
    channel.myid = "seller"
    channel.message_expire_time = 300000
    channel._is_sync_package = lambda message: True
    channel._is_typing_status = lambda message: False
    channel._is_chat_message = lambda message: True
    channel._is_system_message = lambda message: False
    channel._is_manual_mode = lambda chat_id: False
    channel._check_toggle_keywords = lambda message: False
    return channel


async def test_accepted_message_and_run_share_identifiers(monkeypatch):
    dispatcher = RecordingDispatcher()
    channel = observable_channel(dispatcher)
    envelope, decrypted = raw_chat_message(int(time.time() * 1000))
    monkeypatch.setattr(channel_module, "decrypt", lambda data: json.dumps(decrypted))

    await channel._handle_raw(envelope, None)

    accepted = dispatcher.events[0]
    _, handle_fields = dispatcher.handled[0]
    assert accepted["type"] == "message_accepted"
    assert accepted["run_id"] == handle_fields["run_id"]
    assert accepted["message_key"] == handle_fields["message_key"]


async def test_expired_message_without_url_is_observed_without_dispatch(monkeypatch):
    dispatcher = RecordingDispatcher()
    channel = observable_channel(dispatcher)
    envelope, decrypted = raw_chat_message(0, include_url=False)
    monkeypatch.setattr(channel_module, "decrypt", lambda data: json.dumps(decrypted))

    await channel._handle_raw(envelope, None)

    assert dispatcher.handled == []
    assert dispatcher.events[0]["type"] == "message_filtered"
    assert dispatcher.events[0]["reason"] == "expired"
    assert dispatcher.events[0]["run_id"]
