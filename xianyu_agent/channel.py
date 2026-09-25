"""闲鱼通道。只认识事件，不知道模型和工具的存在。

职责被严格限定为：维持长连接、收发原始消息、解密和过滤、心跳和 token、
人工接管（含切换关键词）、把回复编码成平台要求的格式。
它不知道"议价"这个词的意思——所有回复都来自 dispatcher（SessionRegistry）。

运维细节（心跳、token 刷新、断线重连、消息过期丢弃、订单消息识别）
沿用原项目 XianyuAutoAgent 打磨过的实现。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time

import websockets

from .types import IncomingChat
from .xianyu_api import XianyuApi, XianyuCookieError
from .xianyu_utils import decrypt, generate_device_id, generate_mid, generate_uuid, trans_cookies

logger = logging.getLogger(__name__)

BASE_URL = "wss://wss-goofish.dingtalk.com/"

WS_HEADERS = {
    "Host": "wss-goofish.dingtalk.com",
    "Connection": "Upgrade",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"),
    "Origin": "https://www.goofish.com",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


class XianyuChannel:
    def __init__(self, cookies_str: str, api: XianyuApi, dispatcher):
        self.api = api
        self.cookies_str = cookies_str
        self.cookies = trans_cookies(cookies_str)
        self.api.session.cookies.update(self.cookies)
        self.myid = self.cookies["unb"]
        self.device_id = generate_device_id(self.myid)
        self.dispatcher = dispatcher          # SessionRegistry：给它消息，它负责跑循环

        # 心跳
        self.heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", "15"))
        self.heartbeat_timeout = int(os.getenv("HEARTBEAT_TIMEOUT", "5"))
        self.last_heartbeat_time = 0.0
        self.last_heartbeat_response = 0.0
        self.heartbeat_task: asyncio.Task | None = None

        # token
        self.token_refresh_interval = int(os.getenv("TOKEN_REFRESH_INTERVAL", "3600"))
        self.token_retry_interval = int(os.getenv("TOKEN_RETRY_INTERVAL", "300"))
        self.last_token_refresh_time = 0.0
        self.current_token: str | None = None
        self.token_refresh_task: asyncio.Task | None = None
        self.connection_restart_flag = False

        # 人工接管
        self.manual_mode_conversations: set[str] = set()
        self.manual_mode_timeout = int(os.getenv("MANUAL_MODE_TIMEOUT", "3600"))
        self.manual_mode_timestamps: dict[str, float] = {}
        self.toggle_keywords = os.getenv("TOGGLE_KEYWORDS", "。")

        # 消息过期
        self.message_expire_time = int(os.getenv("MESSAGE_EXPIRE_TIME", "300000"))
        self.message_dedup_time = int(os.getenv("MESSAGE_DEDUP_TIME", "300000"))
        self._seen_message_keys: dict[str, float] = {}

        self.ws = None

    # ---------------------------------------------------------------- 连接与运维

    async def run(self):
        """外层重连循环。断线、token 过期，都靠它兜底。"""
        while True:
            self.connection_restart_flag = False
            try:
                logger.info("正在连接闲鱼长连接 %s ...", BASE_URL)
                async with websockets.connect(BASE_URL, extra_headers=WS_HEADERS,
                                              open_timeout=15) as ws:
                    self.ws = ws
                    logger.info("WebSocket 连接建立成功")
                    await self._register(ws)

                    self.last_heartbeat_time = time.time()
                    self.last_heartbeat_response = time.time()
                    self.heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
                    self.token_refresh_task = asyncio.create_task(self._token_refresh_loop())

                    async for raw in ws:
                        try:
                            if self.connection_restart_flag:
                                logger.info("检测到连接重启标志，准备重新建立连接")
                                break
                            message_data = json.loads(raw)
                            if self._is_heartbeat_response(message_data):
                                self.last_heartbeat_response = time.time()
                                continue
                            if ("headers" in message_data and "mid" in message_data["headers"]):
                                await self._send_ack(message_data, ws)
                            await self._handle_raw(message_data, ws)
                        except json.JSONDecodeError:
                            logger.error("消息解析失败")
                        except Exception:
                            logger.exception("处理消息时发生错误")
            except XianyuCookieError as e:
                logger.error("Cookie 失效/风控：%s", e)
                logger.error("程序退出，请更新 .env 中的 COOKIES_STR 后重新启动")
                raise
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket 连接已关闭")
            except Exception:
                logger.exception("连接发生错误")
            finally:
                for task in (self.heartbeat_task, self.token_refresh_task):
                    if task:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                self.heartbeat_task = None
                self.token_refresh_task = None
                if self.connection_restart_flag:
                    logger.info("主动重启连接，立即重连")
                else:
                    logger.info("等待 5 秒后重连")
                    await asyncio.sleep(5)

    async def _register(self, ws):
        """注册连接：token -> /reg -> ackDiff。"""
        if (not self.current_token
                or time.time() - self.last_token_refresh_time >= self.token_refresh_interval):
            logger.info("获取初始 token...")
            await self._refresh_token()
            if not self.current_token:
                raise RuntimeError("Token 获取失败")

        msg = {
            "lwp": "/reg",
            "headers": {
                "cache-header": "app-key token ua wv",
                "app-key": "444e9908a51d1cb236a27862abc769c9",
                "token": self.current_token,
                "ua": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 "
                       "DingTalk(2.1.5) OS(Windows/10) Browser(Chrome/133.0.0.0) "
                       "DingWeb/2.1.5 IMPaaS DingWeb/2.1.5"),
                "dt": "j",
                "wv": "im:3,au:3,sy:6",
                "sync": "0,0;0;0;",
                "did": self.device_id,
                "mid": generate_mid(),
            },
        }
        await ws.send(json.dumps(msg))
        await asyncio.sleep(1)
        msg = {
            "lwp": "/r/SyncStatus/ackDiff",
            "headers": {"mid": "5701741704675979 0"},
            "body": [{
                "pipeline": "sync", "tooLong2Tag": "PNM,1", "channel": "sync",
                "topic": "sync", "highPts": 0, "pts": int(time.time() * 1000) * 1000,
                "seq": 0, "timestamp": int(time.time() * 1000),
            }],
        }
        await ws.send(json.dumps(msg))
        logger.info("连接注册完成")

    async def _refresh_token(self):
        try:
            token_result = await asyncio.wait_for(
                asyncio.to_thread(self.api.get_token, self.device_id), timeout=45)
            data = token_result.get("data", {}) if token_result else {}
            if "accessToken" in data:
                self.current_token = data["accessToken"]
                self.last_token_refresh_time = time.time()
                logger.info("Token 刷新成功")
                return self.current_token
            logger.error("Token 刷新失败: %s", token_result)
            return None
        except XianyuCookieError:
            raise
        except Exception as e:
            logger.error("Token 刷新异常: %s", e)
            return None

    async def _token_refresh_loop(self):
        while True:
            try:
                if time.time() - self.last_token_refresh_time >= self.token_refresh_interval:
                    logger.info("Token 即将过期，准备刷新...")
                    new_token = await self._refresh_token()
                    if new_token:
                        self.connection_restart_flag = True
                        if self.ws:
                            await self.ws.close()
                        break
                    logger.error("Token 刷新失败，将在 %d 分钟后重试",
                                 self.token_retry_interval // 60)
                    await asyncio.sleep(self.token_retry_interval)
                    continue
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Token 刷新循环出错")
                await asyncio.sleep(60)

    async def _heartbeat_loop(self, ws):
        while True:
            try:
                now = time.time()
                if now - self.last_heartbeat_time >= self.heartbeat_interval:
                    await self._send_heartbeat(ws)
                if now - self.last_heartbeat_response > (self.heartbeat_interval
                                                         + self.heartbeat_timeout):
                    logger.warning("心跳响应超时，主动断开连接以触发重连")
                    try:
                        await ws.close()      # 让主循环退出，进入重连流程，避免假死
                    except Exception:
                        pass
                    break
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("心跳循环出错")
                break

    async def _send_heartbeat(self, ws):
        heartbeat_msg = {"lwp": "/!", "headers": {"mid": generate_mid()}}
        await ws.send(json.dumps(heartbeat_msg))
        self.last_heartbeat_time = time.time()
        logger.debug("心跳包已发送")

    def _is_heartbeat_response(self, message_data: dict) -> bool:
        return (isinstance(message_data, dict)
                and "headers" in message_data
                and "mid" in message_data["headers"]
                and "code" in message_data
                and message_data["code"] == 200)

    async def _send_ack(self, message_data: dict, ws):
        """平台要求逐条应答。"""
        ack = {
            "code": 200,
            "headers": {
                "mid": message_data["headers"].get("mid", generate_mid()),
                "sid": message_data["headers"].get("sid", ""),
            },
        }
        for key in ("app-key", "ua", "dt"):
            if key in message_data["headers"]:
                ack["headers"][key] = message_data["headers"][key]
        await ws.send(json.dumps(ack))

    # ---------------------------------------------------------------- 发送

    async def send(self, chat_id: str, to_user_id: str, text: str):
        """发送回复。全项目唯一的闲鱼消息发送入口，发给谁由通道决定。"""
        if self.ws is None:
            logger.error("WebSocket 未连接，无法发送消息")
            return
        content = {"contentType": 1, "text": {"text": text}}
        text_base64 = str(base64.b64encode(json.dumps(content).encode("utf-8")), "utf-8")
        msg = {
            "lwp": "/r/MessageSend/sendByReceiverScope",
            "headers": {"mid": generate_mid()},
            "body": [
                {
                    "uuid": generate_uuid(),
                    "cid": f"{chat_id}@goofish",
                    "conversationType": 1,
                    "content": {"contentType": 101,
                                "custom": {"type": 1, "data": text_base64}},
                    "redPointPolicy": 0,
                    "extension": {"extJson": "{}"},
                    "ctx": {"appVersion": "1.0", "platform": "web"},
                    "mtags": {},
                    "msgReadStatusSetting": 1,
                },
                {
                    "actualReceivers": [
                        f"{to_user_id}@goofish",
                        f"{self.myid}@goofish",
                    ]
                },
            ],
        }
        await self.ws.send(json.dumps(msg))

    # ---------------------------------------------------------------- 消息解析

    def _is_sync_package(self, message_data: dict) -> bool:
        try:
            return (isinstance(message_data, dict)
                    and "body" in message_data
                    and "syncPushPackage" in message_data["body"]
                    and "data" in message_data["body"]["syncPushPackage"]
                    and len(message_data["body"]["syncPushPackage"]["data"]) > 0)
        except Exception:
            return False

    def _is_chat_message(self, message: dict) -> bool:
        try:
            return ("1" in message and isinstance(message["1"], dict)
                    and "10" in message["1"] and isinstance(message["1"]["10"], dict)
                    and "reminderContent" in message["1"]["10"])
        except Exception:
            return False

    def _is_typing_status(self, message: dict) -> bool:
        try:
            return (isinstance(message["1"], list) and len(message["1"]) > 0
                    and isinstance(message["1"][0], dict)
                    and "1" in message["1"][0]
                    and isinstance(message["1"][0]["1"], str)
                    and "@goofish" in message["1"][0]["1"])
        except Exception:
            return False

    def _is_system_message(self, message: dict) -> bool:
        try:
            return ("3" in message and isinstance(message["3"], dict)
                    and "needPush" in message["3"]
                    and message["3"]["needPush"] == "false")
        except Exception:
            return False

    @staticmethod
    def _is_bracket_system_message(message: str) -> bool:
        if not message or not isinstance(message, str):
            return False
        clean = message.strip()
        return clean.startswith("[") and clean.endswith("]")

    def _check_toggle_keywords(self, message: str) -> bool:
        return message.strip() == self.toggle_keywords

    def _is_duplicate_message(self, key: str) -> bool:
        now = time.time() * 1000
        cutoff = now - self.message_dedup_time
        self._seen_message_keys = {
            seen_key: seen_at for seen_key, seen_at in self._seen_message_keys.items()
            if seen_at >= cutoff
        }
        if key in self._seen_message_keys:
            return True
        self._seen_message_keys[key] = now
        return False

    # ---------------------------------------------------------------- 人工接管

    def _is_manual_mode(self, chat_id: str) -> bool:
        if chat_id not in self.manual_mode_conversations:
            return False
        ts = self.manual_mode_timestamps.get(chat_id)
        if ts is not None and time.time() - ts > self.manual_mode_timeout:
            self._exit_manual_mode(chat_id)     # 超时自动交还
            return False
        return True

    def _enter_manual_mode(self, chat_id: str):
        self.manual_mode_conversations.add(chat_id)
        self.manual_mode_timestamps[chat_id] = time.time()

    def _exit_manual_mode(self, chat_id: str):
        self.manual_mode_conversations.discard(chat_id)
        self.manual_mode_timestamps.pop(chat_id, None)

    def _toggle_manual_mode(self, chat_id: str) -> str:
        if self._is_manual_mode(chat_id):
            self._exit_manual_mode(chat_id)
            return "auto"
        self._enter_manual_mode(chat_id)
        return "manual"

    # ---------------------------------------------------------------- 消息处理

    async def _handle_raw(self, message_data: dict, ws):
        """处理一条原始消息：解密、过滤、分拣。"""
        try:
            if not self._is_sync_package(message_data):
                return

            sync_data = message_data["body"]["syncPushPackage"]["data"][0]
            if "data" not in sync_data:
                return

            data = sync_data["data"]
            try:
                plain = base64.b64decode(data).decode("utf-8")
                json.loads(plain)
                return                  # 未加密的同步/心跳包，无需处理
            except Exception:
                try:
                    message = json.loads(decrypt(data))
                except Exception as e:
                    logger.error("消息解密失败: %s", e)
                    return

            # 订单消息：只记日志（需要自行扩展发货等逻辑的话，在这里加）
            try:
                red = message["3"]["redReminder"]
                user_id = str(message["1"]).split("@")[0]
                user_url = f"https://www.goofish.com/personal?userId={user_id}"
                if red == "等待买家付款":
                    logger.info("等待买家 %s 付款", user_url)
                    return
                if red == "交易关闭":
                    logger.info("买家 %s 交易关闭", user_url)
                    return
                if red == "等待卖家发货":
                    logger.info("交易成功 %s 等待卖家发货", user_url)
                    return
            except (KeyError, TypeError):
                pass

            # 输入状态 / 非聊天消息
            try:
                if self._is_typing_status(message):
                    logger.debug("买家正在输入")
                    return
            except (KeyError, TypeError):
                return
            if not self._is_chat_message(message):
                logger.debug("其他非聊天消息")
                return

            # 提取聊天内容
            create_time = int(message["1"]["5"])
            send_user_name = message["1"]["10"]["reminderTitle"]
            send_user_id = message["1"]["10"]["senderUserId"]
            send_message = message["1"]["10"]["reminderContent"]

            # 时效性验证（过滤过期消息）
            if time.time() * 1000 - create_time > self.message_expire_time:
                logger.debug("过期消息丢弃")
                return

            url_info = message["1"]["10"]["reminderUrl"]
            item_id = (url_info.split("itemId=")[1].split("&")[0]
                       if "itemId=" in url_info else None)
            chat_id = message["1"]["2"].split("@")[0]

            if not item_id:
                logger.warning("无法获取商品ID")
                return

            message_key = hashlib.sha256(
                "\x1f".join((chat_id, str(send_user_id), item_id,
                             str(create_time), send_message)).encode("utf-8")
            ).hexdigest()
            if self._is_duplicate_message(message_key):
                logger.info("重复聊天消息，跳过处理 (会话: %s, 时间: %s)",
                            chat_id, create_time)
                return

            chat = IncomingChat(chat_id=chat_id, item_id=item_id,
                                sender_id=send_user_id, sender_name=send_user_name,
                                text=send_message)

            # 卖家自己的消息：要么是切换命令，要么是人工回复
            if send_user_id == self.myid:
                if self._check_toggle_keywords(send_message):
                    mode = self._toggle_manual_mode(chat_id)
                    if mode == "manual":
                        self.dispatcher.abort(chat_id)     # 打断正在生成的回复
                        logger.info("🔴 已接管会话 %s (商品: %s)", chat_id, item_id)
                    else:
                        logger.info("🟢 已恢复会话 %s 的自动回复 (商品: %s)",
                                    chat_id, item_id)
                    return
                await self.dispatcher.record_manual_reply(chat, role="assistant")
                return

            logger.info("买家: %s (ID: %s), 商品: %s, 会话: %s, 消息: %s",
                        send_user_name, send_user_id, item_id, chat_id, send_message)

            # 人工模式：只记录，不回复
            if self._is_manual_mode(chat_id):
                logger.info("🔴 会话 %s 处于人工接管模式，跳过自动回复", chat_id)
                await self.dispatcher.record_manual_reply(chat, role="user")
                return

            # 系统消息过滤
            if self._is_bracket_system_message(send_message):
                logger.info("检测到系统消息 '%s'，跳过自动回复", send_message)
                return
            if self._is_system_message(message):
                logger.debug("系统消息，跳过处理")
                return

            # 进内核，通道到此为止
            await self.dispatcher.handle(chat)
        except Exception:
            logger.exception("处理消息时发生错误")
