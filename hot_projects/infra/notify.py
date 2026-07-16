"""微信推送（Server酱）：把一条消息推到个人微信。可选功能——未配 key 即静默跳过。

设计约定：推送是"锦上添花"，任何失败只记 warning，绝不抛异常影响主流程（如定时周报）。
"""

import logging

import requests

from ..config import SERVERCHAN_SENDKEY

logger = logging.getLogger("hot_projects")

_API = "https://sctapi.ftqq.com/{key}.send"


def send(title: str, desp: str = "") -> bool:
    """推送一条消息；未配 SERVERCHAN_SENDKEY 或失败均返回 False（不抛异常）。"""
    key = SERVERCHAN_SENDKEY
    if not key:
        return False
    try:
        resp = requests.post(
            _API.format(key=key),
            data={"title": title[:32], "desp": desp},
            timeout=10,
        )
        if resp.status_code == 200 and resp.json().get("code", 0) == 0:
            logger.info("微信推送成功: %s", title)
            return True
        logger.warning("微信推送失败: status=%s, body=%s", resp.status_code, resp.text[:200])
    except Exception as e:  # noqa: BLE001 —— 推送失败绝不影响主流程
        logger.warning("微信推送异常: %s", e)
    return False
