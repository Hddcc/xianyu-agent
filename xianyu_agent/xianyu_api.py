"""闲鱼平台 API 封装（沿用原项目 XianyuAutoAgent 的 XianyuApis 实现）。

改造点：原项目在 Cookie 失效 / 触发风控时用 input() 交互 + sys.exit()，
无人值守服务里这样不合适——统一抛 XianyuCookieError，由入口层处理。
"""
from __future__ import annotations

import logging
import os
import re
import time

import requests

from .xianyu_utils import generate_sign

logger = logging.getLogger(__name__)


class XianyuCookieError(Exception):
    """Cookie 失效或触发风控，需要人工更新 .env 中的 COOKIES_STR。"""


class XianyuApi:
    def __init__(self, cookies_str: str | None = None):
        self.url = "https://h5api.m.goofish.com/h5/mtop.taobao.idlemessage.pc.login.token/1.0/"
        self.session = requests.Session()
        self.session.headers.update({
            "accept": "application/json",
            "accept-language": "zh-CN,zh;q=0.9",
            "cache-control": "no-cache",
            "origin": "https://www.goofish.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://www.goofish.com/",
            "sec-ch-ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"),
        })
        if cookies_str:
            from .xianyu_utils import trans_cookies
            self.session.cookies.update(trans_cookies(cookies_str))

    # ---------------------------------------------------------------- cookie 维护

    def clear_duplicate_cookies(self):
        """清理重复的 cookie（保留最新值）。"""
        new_jar = requests.cookies.RequestsCookieJar()
        added = set()
        cookie_list = list(self.session.cookies)
        cookie_list.reverse()
        for cookie in cookie_list:
            if cookie.name not in added:
                new_jar.set_cookie(cookie)
                added.add(cookie.name)
        self.session.cookies = new_jar
        self.update_env_cookies()

    def update_env_cookies(self):
        """把当前 cookie 同步写回 .env 的 COOKIES_STR。"""
        try:
            cookie_str = "; ".join(f"{c.name}={c.value}" for c in self.session.cookies)
            env_path = os.path.join(os.getcwd(), ".env")
            if not os.path.exists(env_path):
                return
            with open(env_path, "r", encoding="utf-8") as f:
                env_content = f.read()
            if "COOKIES_STR=" in env_content:
                new_content = re.sub(r"COOKIES_STR=.*", f"COOKIES_STR={cookie_str}", env_content)
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                logger.debug("已更新 .env 文件中的 COOKIES_STR")
        except Exception as e:
            logger.warning("更新 .env 文件失败: %s", e)

    def has_login(self, retry_count: int = 0) -> bool:
        """调用 hasLogin.do 检查登录状态。"""
        if retry_count >= 2:
            logger.error("Login 检查失败，重试次数过多")
            return False
        try:
            url = "https://passport.goofish.com/newlogin/hasLogin.do"
            params = {"appName": "xianyu", "fromSite": "77"}
            data = {
                "hid": self.session.cookies.get("unb", ""),
                "ltl": "true",
                "appName": "xianyu",
                "appEntrance": "web",
                "_csrf_token": self.session.cookies.get("XSRF-TOKEN", ""),
                "umidToken": "",
                "hsiz": self.session.cookies.get("cookie2", ""),
                "bizParams": "taobaoBizLoginFrom=web",
                "mainPage": "false",
                "isMobile": "false",
                "lang": "zh_CN",
                "returnUrl": "",
                "fromSite": "77",
                "isIframe": "true",
                "documentReferer": "https://www.goofish.com/",
                "defaultView": "hasLogin",
                "umidTag": "SERVER",
                "deviceId": self.session.cookies.get("cna", ""),
            }
            response = self.session.post(url, params=params, data=data, timeout=15)
            res_json = response.json()
            if res_json.get("content", {}).get("success"):
                logger.debug("Login 检查成功")
                self.clear_duplicate_cookies()
                return True
            logger.warning("Login 检查失败: %s", res_json)
            time.sleep(0.5)
            return self.has_login(retry_count + 1)
        except Exception as e:
            logger.error("Login 请求异常: %s", e)
            time.sleep(0.5)
            return self.has_login(retry_count + 1)

    # ---------------------------------------------------------------- token

    def get_token(self, device_id: str, retry_count: int = 0) -> dict | None:
        """获取消息通道 token。失败重试；Cookie 失效 / 风控抛 XianyuCookieError。"""
        # 总尝试上限：防止"token 接口一直失败但登录检查一直成功"导致的无限循环
        self._token_attempts = getattr(self, "_token_attempts", 0) + 1
        if self._token_attempts > 8:
            raise XianyuCookieError(
                "Token 获取多次失败（Cookie 可能已失效或触发风控），"
                "请更新 .env 中的 COOKIES_STR 后重启")
        if retry_count >= 2:
            logger.warning("获取 token 失败，尝试重新登录")
            if self.has_login():
                logger.info("重新登录成功，重新尝试获取 token")
                return self.get_token(device_id, 0)
            raise XianyuCookieError(
                "Cookie 已失效：请进入闲鱼网页版 - 点击消息 - 过滑块 - "
                "复制最新的 Cookie 更新 .env 中的 COOKIES_STR 后重启")

        params = {
            "jsv": "2.7.2",
            "appKey": "34839810",
            "t": str(int(time.time()) * 1000),
            "sign": "",
            "v": "1.0",
            "type": "originaljson",
            "accountSite": "xianyu",
            "dataType": "json",
            "timeout": "20000",
            "api": "mtop.taobao.idlemessage.pc.login.token",
            "sessionOption": "AutoLoginOnly",
            "spm_cnt": "a21ybx.im.0.0",
            "spm_pre": "a21ybx.item.want.1.14ad3da6ALVq3n",
            "log_id": "14ad3da6ALVq3n",
        }
        data_val = ('{"appKey":"444e9908a51d1cb236a27862abc769c9",'
                    f'"deviceId":"{device_id}"}}')
        data = {"data": data_val}
        headers = {
            "Host": "h5api.m.goofish.com",
            "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"),
            "accept": "application/json",
            "content-type": "application/x-www-form-urlencoded",
            "origin": "https://www.goofish.com",
            "referer": "https://www.goofish.com/",
        }

        token = self.session.cookies.get("_m_h5_tk", "").split("_")[0]
        params["sign"] = generate_sign(params["t"], token, data_val)

        try:
            response = self.session.post(self.url, headers=headers, params=params, data=data,
                                         timeout=15)
            res_json = response.json()

            if not isinstance(res_json, dict):
                logger.error("Token API 返回格式异常: %s", res_json)
                return self.get_token(device_id, retry_count + 1)

            ret_value = res_json.get("ret", [])
            if not any("SUCCESS::调用成功" in ret for ret in ret_value):
                error_msg = str(ret_value)
                if "RGV587_ERROR" in error_msg or "被挤爆啦" in error_msg:
                    logger.error("触发风控: %s", ret_value)
                    raise XianyuCookieError(
                        "触发闲鱼风控：请进入闲鱼网页版 - 点击消息 - 过滑块 - "
                        "复制最新的 Cookie 更新 .env 中的 COOKIES_STR 后重启")
                logger.warning("Token API 调用失败: %s", ret_value)
                if "Set-Cookie" in response.headers:
                    self.clear_duplicate_cookies()
                time.sleep(0.5)
                return self.get_token(device_id, retry_count + 1)

            logger.info("Token 获取成功")
            return res_json
        except XianyuCookieError:
            raise
        except Exception as e:
            logger.error("Token API 请求异常: %s", e)
            time.sleep(0.5)
            return self.get_token(device_id, retry_count + 1)

    # ---------------------------------------------------------------- 商品

    def get_item_info(self, item_id: str, retry_count: int = 0) -> dict:
        """获取商品信息（mtop.taobao.idle.pc.detail）。"""
        if retry_count >= 3:
            logger.error("获取商品信息失败，重试次数过多")
            return {"error": "获取商品信息失败，重试次数过多"}

        params = {
            "jsv": "2.7.2",
            "appKey": "34839810",
            "t": str(int(time.time()) * 1000),
            "sign": "",
            "v": "1.0",
            "type": "originaljson",
            "accountSite": "xianyu",
            "dataType": "json",
            "timeout": "20000",
            "api": "mtop.taobao.idle.pc.detail",
            "sessionOption": "AutoLoginOnly",
            "spm_cnt": "a21ybx.im.0.0",
        }
        data_val = '{"itemId":"' + item_id + '"}'
        data = {"data": data_val}

        token = self.session.cookies.get("_m_h5_tk", "").split("_")[0]
        params["sign"] = generate_sign(params["t"], token, data_val)

        try:
            response = self.session.post(
                "https://h5api.m.goofish.com/h5/mtop.taobao.idle.pc.detail/1.0/",
                params=params, data=data, timeout=15)
            res_json = response.json()
            if not isinstance(res_json, dict):
                logger.error("商品信息 API 返回格式异常: %s", res_json)
                return self.get_item_info(item_id, retry_count + 1)

            ret_value = res_json.get("ret", [])
            if not any("SUCCESS::调用成功" in ret for ret in ret_value):
                logger.warning("商品信息 API 调用失败: %s", ret_value)
                if "Set-Cookie" in response.headers:
                    self.clear_duplicate_cookies()
                time.sleep(0.5)
                return self.get_item_info(item_id, retry_count + 1)

            logger.debug("商品信息获取成功: %s", item_id)
            return res_json
        except Exception as e:
            logger.error("商品信息 API 请求异常: %s", e)
            time.sleep(0.5)
            return self.get_item_info(item_id, retry_count + 1)
