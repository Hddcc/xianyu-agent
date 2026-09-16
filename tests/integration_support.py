"""Real registry and store, with only external service boundaries replaced."""
from xianyu_agent.experts import ExpertProfile
from xianyu_agent.llm import ModelConfig, Models
from xianyu_agent.registry import AppConfig, SessionRegistry
from xianyu_agent.router import IntentRouter
from xianyu_agent.store import Store
from xianyu_agent.types import IncomingChat


def make_registry(tmp_path, monkeypatch, **config_overrides):
    monkeypatch.chdir(tmp_path)
    model = ModelConfig("offline-test", "https://unused.invalid/v1", "unused")
    experts = {name: ExpertProfile(name, "Test customer service agent.")
               for name in ("default", "price", "tech")}
    sent = []

    async def capture(chat_id, to_user_id, text):
        sent.append((chat_id, to_user_id, text))

    config = AppConfig(myid="seller", idle_compact_hours=0,
                       **config_overrides)
    registry = SessionRegistry(
        Store(str(tmp_path / "history.db"), max_history=1000), experts,
        IntentRouter(lambda user, item, history: "default"),
        Models(model, model), config, sender=capture,
    )
    return registry, sent


def incoming(chat_id, text, item_id=""):
    return IncomingChat(chat_id=chat_id, item_id=item_id,
                        sender_id=f"buyer-{chat_id}", sender_name="Test buyer",
                        text=text)
