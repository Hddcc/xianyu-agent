from xianyu_agent.xianyu_api import XianyuApi


class TokenResponse:
    headers = {}

    def json(self):
        return {
            "ret": ["SUCCESS::调用成功"],
            "data": {"accessToken": "token"},
        }


def test_successful_token_refresh_resets_consecutive_attempts(monkeypatch):
    api = XianyuApi("unb=seller; _m_h5_tk=signing_token_x")
    api._token_attempts = 4
    monkeypatch.setattr(api.session, "post", lambda *args, **kwargs: TokenResponse())

    result = api.get_token("device")

    assert result["data"]["accessToken"] == "token"
    assert api._token_attempts == 0


def test_validation_api_does_not_write_cookie_back(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("COOKIES_STR=old\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    api = XianyuApi("unb=seller; _m_h5_tk=new_x", persist_cookies=False)

    api.update_env_cookies()

    assert env_path.read_text(encoding="utf-8") == "COOKIES_STR=old\n"
