"""Small, authenticated HTTP endpoint for updating the Xianyu Cookie.

The browser extension in ``cookie_sync_extension`` reads cookies locally and
posts them here.  This module deliberately has no third-party dependencies so
it can run inside the existing agent image.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Callable

from xianyu_agent.xianyu_api import XianyuApi, XianyuCookieError
from xianyu_agent.xianyu_utils import generate_device_id, trans_cookies

logger = logging.getLogger(__name__)

COOKIE_SYNC_PATH = "/api/cookie-sync"
REQUIRED_COOKIE_NAMES = ("unb", "_m_h5_tk")
MAX_BODY_BYTES = 64 * 1024
_ENV_WRITE_LOCK = Lock()


class CookieSyncError(ValueError):
    """The submitted Cookie cannot be used by the agent."""


def normalize_cookie(cookie: str) -> tuple[str, dict[str, str]]:
    """Normalize a browser Cookie header and verify required fields."""
    if not isinstance(cookie, str):
        raise CookieSyncError("Cookie 必须是文本")

    normalized = " ".join(cookie.replace("\r", " ").replace("\n", " ").split())
    if not normalized:
        raise CookieSyncError("Cookie 不能为空")

    try:
        cookies = trans_cookies(normalized)
    except Exception as exc:
        raise CookieSyncError("Cookie 格式无法解析") from exc

    missing = [name for name in REQUIRED_COOKIE_NAMES if not cookies.get(name)]
    if missing:
        raise CookieSyncError(f"Cookie 缺少必要字段: {', '.join(missing)}")
    return normalized, cookies


def validate_cookie(cookie: str) -> None:
    """Verify that the submitted Cookie can obtain a messaging token."""
    normalized, cookies = normalize_cookie(cookie)
    api = XianyuApi(normalized, persist_cookies=False)
    try:
        result = api.get_token(generate_device_id(cookies["unb"]))
    except XianyuCookieError as exc:
        raise CookieSyncError(str(exc)) from exc
    if not result or not result.get("data", {}).get("accessToken"):
        raise CookieSyncError("Cookie 无法获取闲鱼消息 token")


def replace_env_cookie(env_path: str | Path, cookie: str) -> Path:
    """Back up and replace COOKIES_STR in a bind-mount-safe way."""
    normalized, _ = normalize_cookie(cookie)
    path = Path(env_path)
    if not path.exists():
        raise CookieSyncError(f"配置文件不存在: {path}")

    backup = path.with_name(f"{path.name}.cookie-sync.bak")
    with _ENV_WRITE_LOCK:
        content = path.read_text(encoding="utf-8")
        shutil.copy2(path, backup)

        replacement = f"COOKIES_STR={normalized}"
        if re.search(r"^COOKIES_STR=.*$", content, flags=re.MULTILINE):
            updated = re.sub(r"^COOKIES_STR=.*$", lambda _: replacement, content,
                             count=1, flags=re.MULTILINE)
        else:
            separator = "" if not content or content.endswith("\n") else "\n"
            updated = f"{content}{separator}{replacement}\n"

        try:
            path.write_text(updated, encoding="utf-8")
        except Exception:
            shutil.copy2(backup, path)
            raise
    return backup


class _CookieSyncHandler(BaseHTTPRequestHandler):
    server: "CookieSyncHTTPServer"

    def log_message(self, format: str, *args) -> None:
        # Never let request details accidentally include a Cookie value.
        logger.info("Cookie 同步请求: %s", format.split(" ", 2)[0])

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = (b"" if status == HTTPStatus.NO_CONTENT
                else json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != COOKIE_SYNC_PATH:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        authorization = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.sync_token}"
        if not hmac.compare_digest(authorization, expected):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "授权失败"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "请求体大小无效"})
            return

        try:
            payload = json.loads(self.rfile.read(length))
            cookie = payload.get("cookie", "") if isinstance(payload, dict) else ""
            normalized, _ = normalize_cookie(cookie)
            self.server.cookie_validator(normalized)
            backup = replace_env_cookie(self.server.env_path, normalized)
            try:
                if self.server.on_applied:
                    self.server.on_applied(normalized)
            except Exception:
                with _ENV_WRITE_LOCK:
                    shutil.copy2(backup, self.server.env_path)
                raise
        except (CookieSyncError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception:
            logger.exception("Cookie 同步失败")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "同步失败"})
            return

        self._send_json(HTTPStatus.OK, {
            "status": "updated",
            "message": "Cookie 已更新，正在重新连接闲鱼",
            "backup": backup.name,
        })


class CookieSyncHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, sync_token: str, env_path: str | Path,
                 on_applied: Callable[[str], None] | None = None,
                 cookie_validator: Callable[[str], None] = validate_cookie):
        super().__init__(address, _CookieSyncHandler)
        self.sync_token = sync_token
        self.env_path = Path(env_path)
        self.on_applied = on_applied
        self.cookie_validator = cookie_validator


class CookieSyncServer:
    """Lifecycle wrapper used by the live agent process."""

    def __init__(self, host: str, port: int, sync_token: str,
                 env_path: str | Path, on_applied: Callable[[str], None] | None = None,
                 cookie_validator: Callable[[str], None] = validate_cookie):
        if not sync_token:
            raise ValueError("Cookie 同步接口必须配置 COOKIE_SYNC_TOKEN")
        self.httpd = CookieSyncHTTPServer(
            (host, port), sync_token, env_path, on_applied, cookie_validator)
        self.thread = Thread(target=self.httpd.serve_forever, name="cookie-sync", daemon=True)

    def start(self) -> None:
        self.thread.start()
        logger.info("Cookie 同步接口已启动: %s", self.httpd.server_address)

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


def configured_server(on_applied: Callable[[str], None] | None = None) -> CookieSyncServer | None:
    """Build the server only when an explicit sync token is configured."""
    token = os.getenv("COOKIE_SYNC_TOKEN", "").strip()
    if not token:
        return None
    return CookieSyncServer(
        host=os.getenv("COOKIE_SYNC_HOST", "0.0.0.0"),
        port=int(os.getenv("COOKIE_SYNC_PORT", "8765")),
        sync_token=token,
        env_path=os.getenv("COOKIE_SYNC_ENV_PATH", "/app/.env"),
        on_applied=on_applied,
    )
