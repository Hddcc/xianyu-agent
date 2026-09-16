"""胶水。把通道、内核、能力层组装起来。

用法：
- python -m xianyu_agent.cli          # 正式值守：连闲鱼通道
- python -m xianyu_agent.cli --dev    # 终端调试：不连闲鱼，输入即买家消息
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv, set_key


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_and_complete_env() -> None:
    """交互式检查并补全关键环境变量（原项目逻辑）。"""
    critical_vars = {
        "API_KEY": "your_api_key_here",
        "COOKIES_STR": "your_cookies_here",
    }
    env_path = ".env"
    updated = False
    for key, placeholder in critical_vars.items():
        curr_val = os.getenv(key)
        if not curr_val or curr_val == placeholder:
            logging.warning("配置项 [%s] 未设置或为默认值，请输入", key)
            while True:
                val = input(f"请输入 {key}: ").strip()
                if val:
                    os.environ[key] = val
                    try:
                        if not os.path.exists(env_path):
                            with open(env_path, "w", encoding="utf-8"):
                                pass
                        set_key(env_path, key, val)
                        updated = True
                    except Exception as e:
                        logging.warning("无法自动写入 .env 文件，请手动保存: %s", e)
                    break
                print(f"{key} 不能为空，请重新输入")
    if updated:
        logging.info("新的配置已保存/更新至 .env 文件中")


def build_registry(dev_mode: bool = False):
    """组装内核：能力层 -> 路由 -> 专家 -> 注册表。"""
    from .experts import load_experts
    from .llm import load_models_from_env
    from .registry import SessionRegistry, load_config
    from .router import IntentRouter, make_classify_llm
    from .store import Store
    from .xianyu_api import XianyuApi

    store = Store()
    experts = load_experts()
    models = load_models_from_env()

    from .experts import _load_prompt
    classify_prompt = _load_prompt("classify_prompt", os.getenv("PROMPTS_DIR", "prompts"))
    router = IntentRouter(make_classify_llm(classify_prompt, models))

    api = XianyuApi(cookies_str=os.getenv("COOKIES_STR", "") or None)
    config = load_config(myid="me")
    registry = SessionRegistry(store=store, experts=experts, router=router,
                               models=models, config=config, api=api)

    if dev_mode:
        async def _print_sender(chat_id, to_user_id, text):
            print(f"\n>>> 已回复买家({to_user_id}): {text}\n")
        registry.set_sender(_print_sender)
        config.myid = "seller"
    return registry, api


async def run_dev() -> None:
    """终端调试模式：不连闲鱼，输入即买家消息，验证内核链路。"""
    from .types import IncomingChat

    registry, _ = build_registry(dev_mode=True)
    chat_id = "dev-chat"
    item_id = os.getenv("DEV_ITEM_ID", "")
    print("=== 终端调试模式 ===")
    print(f"会话: {chat_id}  商品: {item_id or '（未配置 DEV_ITEM_ID，商品信息不可用）'}")
    print("输入内容即模拟买家消息；Ctrl+C 退出。\n")
    while True:
        try:
            text = input("买家> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            return
        if not text:
            continue
        chat = IncomingChat(chat_id=chat_id, item_id=item_id,
                            sender_id="dev-buyer", sender_name="调试买家", text=text)
        await registry.handle(chat)


async def run_live() -> None:
    """正式值守模式。"""
    from .channel import XianyuChannel
    from .xianyu_api import XianyuCookieError

    registry, api = build_registry()
    cookies_str = os.getenv("COOKIES_STR", "")

    channel = XianyuChannel(cookies_str, api, dispatcher=registry)
    registry.config.myid = channel.myid        # 卖家ID，落库 assistant 消息时用
    registry.set_sender(channel.send)          # 发送入口只有这一个

    logging.info("启动闲鱼值守（卖家ID: %s）", channel.myid)
    try:
        await channel.run()
    except XianyuCookieError:
        sys.exit(1)


def check_cookie() -> None:
    """快速诊断：验证 Cookie 是否能获取 token（不进入值守循环）。"""
    from .xianyu_api import XianyuApi
    from .xianyu_utils import generate_device_id, trans_cookies

    cookies_str = os.getenv("COOKIES_STR", "")
    cookies = trans_cookies(cookies_str)
    if "unb" not in cookies or "_m_h5_tk" not in cookies:
        print("❌ Cookie 缺少关键字段 unb 或 _m_h5_tk，请重新抓取完整 Cookie")
        sys.exit(1)

    print(f"卖家ID(unb): {cookies['unb']}")
    print("正在获取 token...")
    api = XianyuApi(cookies_str)
    try:
        result = api.get_token(generate_device_id(cookies["unb"]))
    except Exception as e:
        print(f"❌ 获取 token 失败: {e}")
        sys.exit(1)

    if result and result.get("data", {}).get("accessToken"):
        print("✅ Cookie 有效，token 获取成功，可以正式值守")
    else:
        print(f"❌ token 获取失败: {str(result)[:300]}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 Pi 架构的闲鱼智能客服 Agent")
    parser.add_argument("--dev", action="store_true", help="终端调试模式（不连闲鱼）")
    parser.add_argument("--check", action="store_true", help="只检查 Cookie/token 是否有效，然后退出")
    args = parser.parse_args()

    if os.path.exists(".env"):
        load_dotenv()
    if os.path.exists(".env.example"):
        load_dotenv(".env.example")            # 不会覆盖已存在的变量

    setup_logging()
    logging.info("已加载 .env 配置")

    if args.check:
        check_cookie()
        return
    if args.dev:
        asyncio.run(run_dev())
    else:
        check_and_complete_env()
        asyncio.run(run_live())


if __name__ == "__main__":
    main()
