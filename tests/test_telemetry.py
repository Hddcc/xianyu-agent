import json

from xianyu_agent.telemetry import JsonlEventSink, MemoryEventSink, emit


def test_memory_sink_copies_events():
    sink = MemoryEventSink()
    payload = {"values": [1]}
    emit(sink, "sample", run_id="r1", payload=payload)
    payload["values"].append(2)

    assert sink.for_run("r1")[0]["payload"] == {"values": [1]}


def test_jsonl_sink_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(path)
    emit(sink, "first", run_id="r1")
    emit(sink, "second", run_id="r2", text="中文")

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["first", "second"]
    assert events[1]["text"] == "中文"
