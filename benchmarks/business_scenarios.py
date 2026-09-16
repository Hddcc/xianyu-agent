"""Real-model evaluation on synthetic inputs; no Xianyu or notification writes.

Run from the project root:
  python -m benchmarks.business_scenarios --output tmp/business-evaluation.json

Notification results are simulated service outcomes, not real deliveries.
Reply semantics must be manually reviewed; structural success alone is insufficient.
"""
from __future__ import annotations

import argparse
import asyncio
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from uuid import uuid4

from dotenv import load_dotenv

from benchmarks.scenario_cases import NOTIFICATION_CASES, QUOTE_CASES, long_dialogue
from xianyu_agent import llm, notify
from xianyu_agent.agent import maybe_compact, run_agent
from xianyu_agent.cancel import CancellationToken
from xianyu_agent.deadline import Deadline
from xianyu_agent.experts import load_experts
from xianyu_agent.kf_tools import build_item_description
from xianyu_agent.session import append_snapshot, load_latest
from xianyu_agent.store import Store
from xianyu_agent.tools import ToolContext
from xianyu_agent.types import Context, Message, context_to_dict

_TRACE = ContextVar("business_evaluation_trace", default=None)
_REAL_STREAM = llm.stream


async def traced_stream(ctx, **kwargs):
    trace = _TRACE.get()
    if trace is not None:
        trace["model_calls"] += 1
    async for event in _REAL_STREAM(ctx, **kwargs):
        if trace is not None and event["type"] == "done":
            trace["model_stops"].append(event["stopReason"])
            if event["stopReason"] == "error":
                # Provider errors can contain sensitive request details.
                trace["errors"].append("model_stream_error")
        yield event


def new_trace():
    return {"model_calls": 0, "model_stops": [], "errors": []}


def pairing_valid(ctx):
    pending = []
    for message in llm.context_to_provider_messages(ctx):
        if pending and message["role"] != "tool":
            return False
        if message.get("tool_calls"):
            pending = [call["id"] for call in message["tool_calls"]]
            if not all(pending) or len(pending) != len(set(pending)):
                return False
        if message["role"] == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending:
                return False
            pending.remove(call_id)
    return not pending


async def agent_reply(ctx, profile, tctx, model):
    events = []
    async for event in run_agent(
            ctx, profile.tools, tctx, tctx.cancel, max_turns=4,
            temperature=profile.temperature_for(ctx.meta.get("bargain_count", 0)),
            max_tokens=profile.max_tokens, model=model, deadline=tctx.deadline):
        events.append(event)
    end = next((event for event in reversed(events) if event["type"] == "turn_end"), {})
    return end.get("text", "").strip(), end.get("stopReason", "missing"), events


async def evaluate_notification(case, state, repeat, store, experts, model):
    trace = new_trace()
    token = _TRACE.set(trace)
    chat_id = f"notify-{case['id']}-{state}-{repeat}"
    notifications = []
    result = {
        "id": chat_id, "case": case, "state": state, "repeat": repeat,
        "trace": trace, "notifications": notifications, "human_review": None,
        "structural_pass": False,
    }
    t0 = time.monotonic()

    async def notification(kind, detail):
        if state == "success":
            response = "已通过 Server酱 通知卖家"
        elif state == "failure":
            response = "error: Server酱 发送失败: 服务不可用"
        else:
            response = notify.send_notification("evaluation", "synthetic scenario")
        notifications.append({"kind": kind, "detail": detail, "result": response})
        return response

    try:
        item = build_item_description({"title": "蓝牙音箱", "desc": "二手音箱，包装齐全，支持包邮",
                                       "soldPrice": 399, "quantity": 1})
        profile = experts["price" if case["kind"] == "price" else "default"]
        history = [Message("user", "348元包邮可以吗？"),
                   Message("assistant", "可以，348元包邮，已经谈妥。")]
        for message in history:
            await store.add_message(chat_id, message.role, "item", message.role, message.content)
        ctx = Context(
            f"【商品信息】{json.dumps(item, ensure_ascii=False)}\n{profile.system_prompt}",
            [*history, Message("user", case["text"])],
            {"chat_id": chat_id, "bargain_count": 4},
        )
        tctx = ToolContext(chat_id=chat_id, item_id="item", item=item,
                           store=store, cancel=CancellationToken(), notify=notification,
                           deadline=Deadline(seconds=75, max_turns=4))
        reply, stop, events = await asyncio.wait_for(
            agent_reply(ctx, profile, tctx, model), timeout=90)
        matching = [entry for entry in notifications if entry["kind"] == case["kind"]]
        amount_ok = "amount" not in case or any(
            case["amount"] in re.findall(r"\d+(?:\.\d+)?", entry["detail"])
            or (case["amount"] == "0" and "包邮" in entry["detail"])
            for entry in matching)
        paired = pairing_valid(ctx)
        completed = stop == "end_turn" and bool(reply) and not trace["errors"]
        result.update({
            "reply": reply, "stop": stop, "events": events,
            "context": context_to_dict(ctx), "pairing_valid": paired,
            "expected_notification": bool(matching), "amount_preserved": amount_ok,
            "structural_pass": completed and paired and bool(matching) and amount_ok,
        })
    except Exception as exc:
        trace["errors"].append(type(exc).__name__)
        result.update({"reply": "", "stop": "exception"})
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        _TRACE.reset(token)
    return result


async def collect_text(ctx, model, max_tokens=800):
    text = ""
    stop = "missing"
    async for event in llm.stream(ctx, model=model, temperature=0.2, max_tokens=max_tokens):
        if event["type"] == "text_delta":
            text += event["delta"]
        elif event["type"] == "done":
            stop = event["stopReason"]
    return text, stop


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


async def evaluate_long_dialogue(case, repeat, store, experts, model, run_dir):
    trace = new_trace()
    token = _TRACE.set(trace)
    chat_id = f"long-{case['id']}-{repeat}"
    result = {"id": chat_id, "case": case, "repeat": repeat, "trace": trace,
              "structural_pass": False, "human_review": None}
    ctx = long_dialogue(case, rounds=40)
    original_recent = context_to_dict(ctx)["messages"][-20:]
    result["messages_before"] = len(ctx.messages)
    try:
        for message in ctx.messages:
            await store.add_message(chat_id, message.role, "item", message.role, message.content)
        for _ in range(4):
            await store.increment_bargain_count(chat_id)
        changed = await asyncio.wait_for(
            maybe_compact(ctx, CancellationToken(), model=model), timeout=60)
        result["summary"] = ctx.messages[0].content
        result["messages_after"] = len(ctx.messages)
        result["recent_preserved"] = context_to_dict(ctx)["messages"][1:] == original_recent
        snapshot = run_dir / f"{chat_id}.jsonl"
        append_snapshot(snapshot, ctx)
        result["snapshot_equal"] = context_to_dict(load_latest(snapshot)) == context_to_dict(ctx)
        archive = await store.get_context_by_chat(chat_id)
        result["archive_messages"] = len(archive)
        recall_ctx = Context(
            '仅依据给出的会话画像提取事实，缺失就用null，不得推测或补全。只输出JSON：'
            '{"quotes":[{"round":1,"buyer":数字,"seller":数字},...],'
            '"bargain_rounds":数字}。quotes按议价轮次排序，每轮一项。',
            [Message("user", result["summary"])],
        )
        raw, recall_stop = await asyncio.wait_for(collect_text(recall_ctx, model), timeout=60)
        result["recall_raw"] = raw
        result["recall_stop"] = recall_stop
        parsed = parse_json(raw)
        expected = [{"round": index, "buyer": buyer, "seller": seller}
                    for index, (buyer, seller) in enumerate(case["quotes"], 1)]
        result["all_quotes_recalled"] = parsed.get("quotes") == expected
        result["bargain_rounds_recalled"] = parsed.get("bargain_rounds") == 4
        ctx.system_prompt += "\n" + experts["price"].system_prompt
        ctx.messages.append(Message("user", "请确认你最后一次报价是多少元？只回答报价数字。"))
        tctx = ToolContext(chat_id=chat_id, item_id="item", store=store,
                           cancel=CancellationToken(), floor_note=f"{case['floor']}元以下不出",
                           deadline=Deadline(seconds=45, max_turns=4))
        reply, stop, events = await asyncio.wait_for(
            agent_reply(ctx, experts["price"], tctx, model), timeout=75)
        result.update({"continued_reply": reply, "continued_stop": stop,
                       "continued_events": events})
        result["final_quote_consistent"] = (
            re.findall(r"\d+(?:\.\d+)?", reply) == [str(case["final"])])
        result["pairing_valid"] = pairing_valid(ctx)
        result["structural_pass"] = all([
            changed, result["recent_preserved"], result["snapshot_equal"],
            len(archive) == 80, result["all_quotes_recalled"],
            result["bargain_rounds_recalled"], result["final_quote_consistent"],
            result["pairing_valid"], recall_stop == "end_turn", stop == "end_turn",
            not trace["errors"],
        ])
    except Exception as exc:
        trace["errors"].append(type(exc).__name__)
    finally:
        _TRACE.reset(token)
    return result


async def main(args):
    load_dotenv()
    models = llm.load_models_from_env()
    if not models.reply.api_key:
        raise SystemExit("API_KEY is not configured; no evaluation was performed")
    # Even the unconfigured-state scenario cannot reach an external notification service.
    import os
    os.environ["NOTIFY_METHOD"] = ""
    run_dir = args.output.resolve().parent / f"business-run-{uuid4().hex[:12]}"
    run_dir.mkdir(mode=0o777, parents=True)
    store = Store(str(run_dir / "history.db"), max_history=1000)
    experts = load_experts()
    llm.stream = traced_stream
    semaphore = asyncio.Semaphore(args.concurrency)
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": models.reply.name, "repeats": args.repeats,
        "notification_base_cases": len(NOTIFICATION_CASES),
        "notification_states": ["success", "failure", "unconfigured"],
        "long_dialogue_base_cases": len(QUOTE_CASES), "dialogue_rounds": 40,
        "storage_max_history": 1000,
        "scope": "Synthetic inputs, real reply model and Agent loop; simulated notifications; no platform delivery",
        "notifications": [], "long_dialogues": [],
    }

    def checkpoint():
        args.output.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    async def one_notification(case, state, repeat):
        async with semaphore:
            result = await evaluate_notification(case, state, repeat, store, experts, models.reply)
            results["notifications"].append(result)
            checkpoint()
            print(f"NOTIFY {result['id']} structural={result['structural_pass']} "
                  f"stop={result['stop']} reply={result['reply']}", flush=True)

    async def one_long(case, repeat):
        async with semaphore:
            result = await evaluate_long_dialogue(case, repeat, store, experts, models.reply, run_dir)
            results["long_dialogues"].append(result)
            checkpoint()
            print(f"LONG {result['id']} structural={result['structural_pass']} "
                  f"quotes={result.get('all_quotes_recalled')} "
                  f"reply={result.get('continued_reply', '')}", flush=True)

    try:
        await asyncio.gather(*[
            one_notification(case, state, repeat)
            for repeat in range(1, args.repeats + 1)
            for state in results["notification_states"] for case in NOTIFICATION_CASES
        ], *[
            one_long(case, repeat) for repeat in range(1, args.repeats + 1) for case in QUOTE_CASES
        ])
    finally:
        llm.stream = _REAL_STREAM
        checkpoint()
    print(f"Saved {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.concurrency < 1:
        parser.error("repeats and concurrency must be positive")
    asyncio.run(main(arguments))
