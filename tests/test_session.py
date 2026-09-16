"""会话落盘测试：快照往返一致，坏行跳过。"""
from xianyu_agent.session import append_snapshot, load_latest, session_path
from xianyu_agent.types import Context, Message


def test_snapshot_roundtrip(tmp_path):
    ctx = Context(system_prompt="s",
                  messages=[Message(role="user", content="在吗")],
                  meta={"chat_id": "c1"})
    path = tmp_path / "session.jsonl"

    append_snapshot(path, ctx)
    append_snapshot(path, Context(system_prompt="s2",
                                  messages=[Message(role="user", content="多少钱")]))

    latest = load_latest(path)
    assert latest.system_prompt == "s2"
    assert latest.messages[-1].content == "多少钱"


def test_load_latest_skips_broken_lines(tmp_path):
    path = tmp_path / "session.jsonl"
    append_snapshot(path, Context(system_prompt="good",
                                  messages=[Message(role="user", content="好")]))
    with open(path, "a", encoding="utf-8") as f:
        f.write("{broken json\n")                 # 模拟写了一半崩掉
    append_snapshot(path, Context(system_prompt="newer",
                                  messages=[Message(role="user", content="新")]))

    # 最后一行是好的
    assert load_latest(path).system_prompt == "newer"

    # 只剩坏行时，往前找到第一个能解析的
    with open(path, "w", encoding="utf-8") as f:
        f.write("{broken\n")
        f.write("not json at all\n")
    assert load_latest(path) is None


def test_load_latest_missing_file(tmp_path):
    assert load_latest(tmp_path / "nope.jsonl") is None


def test_session_path_creates_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = session_path("c1")
    assert p.name == "session.jsonl"
    assert p.parent.exists()                 # 目录已创建（文件本身还没写）
