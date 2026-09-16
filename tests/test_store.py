"""SQLite 存储测试：消息、议价计数、商品缓存、会话画像。"""
import time

from xianyu_agent.store import Store


def make_store(tmp_path):
    return Store(db_path=str(tmp_path / "test.db"))


async def test_add_and_get_messages(tmp_path):
    store = make_store(tmp_path)
    await store.add_message("c1", "buyer", "i1", "user", "在吗")
    await store.add_message("c1", "seller", "i1", "assistant", "在的")

    msgs = await store.get_context_by_chat("c1")
    assert [(m["role"], m["content"]) for m in msgs] == [("user", "在吗"), ("assistant", "在的")]

    # limit 取最近 N 条且保持时间正序
    await store.add_message("c1", "buyer", "i1", "user", "多少钱")
    msgs = await store.get_context_by_chat("c1", limit=2)
    assert [m["content"] for m in msgs] == ["在的", "多少钱"]


async def test_bargain_count(tmp_path):
    store = make_store(tmp_path)
    assert await store.get_bargain_count("c1") == 0
    await store.increment_bargain_count("c1")
    await store.increment_bargain_count("c1")
    assert await store.get_bargain_count("c1") == 2
    assert await store.get_bargain_count("c2") == 0


async def test_item_cache(tmp_path):
    store = make_store(tmp_path)
    assert await store.get_item_info("i1") is None
    await store.save_item_info("i1", {"title": "音箱", "soldPrice": 199})
    assert (await store.get_item_info("i1"))["title"] == "音箱"


async def test_session_profile(tmp_path):
    store = make_store(tmp_path)
    text, updated = await store.get_profile("c1")
    assert text is None and updated == 0.0

    await store.set_profile("c1", "买家想砍价，已报348")
    text, updated = await store.get_profile("c1")
    assert text == "买家想砍价，已报348"
    assert updated > 0


async def test_last_activity(tmp_path):
    store = make_store(tmp_path)
    assert await store.last_activity("c1") == 0.0
    await store.add_message("c1", "buyer", "i1", "user", "在吗")
    assert time.time() - await store.last_activity("c1") < 10
