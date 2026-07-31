"""微信推送(Server酱)。可选功能:没配 key 就静默跳过。

**推送失败绝不影响主流程** —— 不能让最后这一步的网络抖动把整轮判失败、报告不落盘。
所以这里吞掉一切异常,只记 warning。
"""

from __future__ import annotations

import logging

import requests

from .. import config

logger = logging.getLogger("hot_project")

_API = "https://sctapi.ftqq.com/{key}.send"
TITLE_LIMIT = 32        # Server酱 的标题上限,超了整条推送被拒


def send(title: str, body: str = "") -> bool:
    """推一条消息。没配 key、或推失败,都返回 False 且不抛异常。"""
    key = config.serverchan_sendkey()
    if not key:
        return False
    try:
        resp = requests.post(
            _API.format(key=key),
            data={"title": title[:TITLE_LIMIT], "desp": body},
            timeout=10,
        )
        if resp.status_code == 200 and resp.json().get("code", 0) == 0:
            logger.info("微信推送成功:%s", title)
            return True
        logger.warning("微信推送失败:HTTP %s %s", resp.status_code, resp.text[:200])
    except Exception as e:      # noqa: BLE001 —— 见模块文档:推送绝不影响主流程
        # SendKey 是 URL 的一段,而 requests 的连接类异常消息里带完整 URL —— 原样记就是
        # 把密钥写进 logs/,违反 config.py 里「密钥永不落文件」。脱敏后再记。
        logger.warning("微信推送异常:%s", str(e).replace(key, "***"))
    return False
