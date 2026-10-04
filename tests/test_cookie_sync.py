import http.client
import json
from pathlib import Path
from threading import Thread

import pytest

from tools.cookie_sync import (
    CookieSyncError,
    CookieSyncHTTPServer,
    normalize_cookie,
    replace_env_cookie,
)


def test_normalize_cookie_requires_login_fields():
    normalized, cookies = normalize_cookie(" unb=seller;  _m_h5_tk=token_123_x; foo=bar ")

    assert normalized == "unb=seller; _m_h5_tk=token_123_x; foo=bar"
    assert cookies["unb"] == "seller"


def test_normalize_cookie_rejects_incomplete_cookie():
    with pytest.raises(CookieSyncError, match="缺少必要字段"):
        normalize_cookie("unb=seller")


def test_replace_env_cookie_keeps_backup(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("API_KEY=key\nCOOKIES_STR=old\n", encoding="utf-8")

    backup = replace_env_cookie(env_path, "unb=seller; _m_h5_tk=new_token_x")

    assert env_path.read_text(encoding="utf-8") == (
        "API_KEY=key\nCOOKIES_STR=unb=seller; _m_h5_tk=new_token_x\n"
    )
    assert backup.read_text(encoding="utf-8") == "API_KEY=key\nCOOKIES_STR=old\n"


def test_sync_endpoint_rejects_wrong_token_without_changing_env(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("COOKIES_STR=old\n", encoding="utf-8")
    server = CookieSyncHTTPServer(
        ("127.0.0.1", 0), "secret", env_path,
        cookie_validator=lambda cookie: None,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.request(
            "POST", "/api/cookie-sync",
            body=json.dumps({"cookie": "unb=seller; _m_h5_tk=new_token_x"}),
            headers={"Authorization": "Bearer wrong", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status == 401
    assert env_path.read_text(encoding="utf-8") == "COOKIES_STR=old\n"


def test_sync_endpoint_updates_env_after_validation(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("COOKIES_STR=old\n", encoding="utf-8")
    applied = []
    server = CookieSyncHTTPServer(
        ("127.0.0.1", 0), "secret", env_path,
        on_applied=applied.append,
        cookie_validator=lambda cookie: None,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cookie = "unb=seller; _m_h5_tk=new_token_x"
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.request(
            "POST", "/api/cookie-sync",
            body=json.dumps({"cookie": cookie}),
            headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status == 200
    assert env_path.read_text(encoding="utf-8") == f"COOKIES_STR={cookie}\n"
    assert applied == [cookie]


def test_sync_endpoint_restores_env_when_apply_fails(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("COOKIES_STR=old\n", encoding="utf-8")

    def fail_apply(cookie: str) -> None:
        raise RuntimeError("reconnect failed")

    server = CookieSyncHTTPServer(
        ("127.0.0.1", 0), "secret", env_path,
        on_applied=fail_apply,
        cookie_validator=lambda cookie: None,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.request(
            "POST", "/api/cookie-sync",
            body=json.dumps({"cookie": "unb=seller; _m_h5_tk=new_token_x"}),
            headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status == 500
    assert env_path.read_text(encoding="utf-8") == "COOKIES_STR=old\n"
