"""通知能力：AI 做不了的事（发图、改价等），通知卖家人工介入。

渠道由 .env 配置，支持：
- NOTIFY_METHOD=serverchan  → Server酱（微信推送，最简单，只需一个 SendKey）
- NOTIFY_METHOD=smtp        → 邮件（QQ/163 邮箱 SMTP 授权码）
- 留空                      → notify_seller 工具返回错误，模型应诚实回复买家稍后处理

能力层：同步 HTTP / SMTP，async 调用方负责丢线程池。
"""
from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.header import Header
from email.mime.text import MIMEText

import httpx

logger = logging.getLogger(__name__)


def send_notification(title: str, content: str) -> str:
    """发送通知，返回结果描述（成功字符串或 error:...）。"""
    method = os.getenv("NOTIFY_METHOD", "").strip().lower()
    if method == "serverchan":
        return _send_serverchan(title, content)
    if method == "smtp":
        return _send_smtp(title, content)
    if not method:
        return "error: 未配置通知渠道（NOTIFY_METHOD），无法通知卖家"
    return f"error: 未知通知渠道 {method}"


def _send_serverchan(title: str, content: str) -> str:
    key = os.getenv("SERVERCHAN_SENDKEY", "").strip()
    if not key:
        return "error: 未配置 SERVERCHAN_SENDKEY"
    try:
        resp = httpx.post(f"https://sctapi.ftqq.com/{key}.send",
                          data={"title": title, "desp": content}, timeout=10)
        data = resp.json()
        if data.get("code") == 0:
            return "已通过 Server酱 通知卖家"
        return f"error: Server酱 发送失败: {data.get('message', data)}"
    except Exception as e:
        return f"error: 通知发送失败: {e}"


def _send_smtp(title: str, content: str) -> str:
    host = os.getenv("SMTP_HOST", "").strip()
    try:
        port = int(os.getenv("SMTP_PORT", "465"))
    except ValueError:
        port = 465
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    to = os.getenv("SMTP_TO", "").strip()
    if not (host and user and password and to):
        return "error: 邮件通知配置不完整（SMTP_HOST/USER/PASSWORD/TO）"

    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = Header(title, "utf-8")
    msg["From"] = user
    msg["To"] = to

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=15) as server:
                server.login(user, password)
                server.sendmail(user, [to], msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(user, password)
                server.sendmail(user, [to], msg.as_string())
        return "已通过邮件通知卖家"
    except Exception as e:
        return f"error: 邮件发送失败: {e}"
