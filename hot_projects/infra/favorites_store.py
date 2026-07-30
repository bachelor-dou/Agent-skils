"""用户收藏持久化：按 user_id 存全局收藏清单。

存储文件 data/favorites.json（与 Github_DB.json 同目录、同款 fcntl 文件锁 + 原子替换）：
  {"users": {"<user_id>": [{"repo": "owner/name",
                            "favorited_at": "YYYY-MM-DDTHH:MM:SSZ",
                            "source_report": "2026-07-01.md"}]}}

收藏是「全局」的：某项目一旦收藏，在任何包含它的报告里都显示为已收藏。
"""

import fcntl
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone

from ..config import FAVORITES_FILE_PATH

logger = logging.getLogger("hot_projects")

_lock = threading.Lock()

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MAX_FAVORITES_PER_USER = 500
MAX_CATEGORY_LEN = 20


def valid_user_id(user_id: str) -> bool:
    return bool(user_id and USER_ID_RE.match(user_id))


def valid_repo(repo: str) -> bool:
    return bool(repo and REPO_RE.match(repo))


_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def clean_category(category: str) -> str:
    """规整分类标签：剔除控制字符、折叠空白、截断长度；空则返回 ''（未分类）。"""
    if not category:
        return ""
    # 先把非空白控制字符（\x00-\x08、\x0b\x0c、\x0e-\x1f、\x7f 等）换成空格，
    # 再用 split() 折叠所有空白，避免残留不可见字符污染标签。
    return " ".join(_CTRL_RE.sub(" ", str(category)).split())[:MAX_CATEGORY_LEN]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_all() -> dict:
    if not os.path.exists(FAVORITES_FILE_PATH):
        return {"users": {}}
    lock_fd = open(FAVORITES_FILE_PATH + ".lock", "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        with open(FAVORITES_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("收藏文件读取失败: %s，视为空。", e)
        return {"users": {}}
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
    if not isinstance(data.get("users"), dict):
        data = {"users": {}}
    return data


def get_favorites(user_id: str) -> list[dict]:
    """返回该用户收藏清单（按收藏时间倒序）。"""
    if not valid_user_id(user_id):
        return []
    with _lock:
        items = _read_all().get("users", {}).get(user_id, [])
    return items if isinstance(items, list) else []


def all_favorited_repos() -> set[str]:
    """所有用户收藏过的仓库全名，跨用户合并。DB 淘汰用它当保护名单。"""
    with _lock:
        users = _read_all().get("users", {})
    return {
        item["repo"]
        for items in users.values() if isinstance(items, list)
        for item in items if isinstance(item, dict) and item.get("repo")
    }


def _write_all(data: dict) -> None:
    lock_fd = open(FAVORITES_FILE_PATH + ".lock", "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = FAVORITES_FILE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, FAVORITES_FILE_PATH)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def set_favorite(user_id: str, repo: str, action: str,
                 source_report: str = "", short_desc: str | None = None,
                 category: str | None = None) -> list[dict]:
    """add / remove 单个收藏，返回更新后的清单。非法输入抛 ValueError。

    category 为单一分类标签：None 表示「不改动」（新增则存空串=未分类，已存在则保留原值），
    显式传字符串（含 ""）会覆盖，其中 "" 表示归到「未分类」。

    short_desc 与 category 同语义：None=不改动，字符串（含 ""）=覆盖（"" 即清空概要）。
    """
    if not valid_user_id(user_id):
        raise ValueError("invalid user_id")
    if not valid_repo(repo):
        raise ValueError("invalid repo")
    if action not in ("add", "remove"):
        raise ValueError("invalid action")

    with _lock:
        data = _read_all()
        users = data.setdefault("users", {})
        items = [x for x in users.get(user_id, []) if isinstance(x, dict)]

        if action == "remove":
            items = [x for x in items if x.get("repo") != repo]
        else:  # add：幂等去重，新收藏置顶；已存在则补写概要/分类
            existing = next((x for x in items if x.get("repo") == repo), None)
            if existing is not None:
                if short_desc is not None:
                    existing["short_desc"] = short_desc
                if category is not None:
                    existing["category"] = clean_category(category)
            else:
                if len(items) >= MAX_FAVORITES_PER_USER:
                    raise ValueError("favorites limit reached")
                items.insert(0, {
                    "repo": repo,
                    "favorited_at": _now(),
                    "source_report": source_report or "",
                    "short_desc": short_desc or "",
                    "category": clean_category(category or ""),
                })

        users[user_id] = items
        _write_all(data)
    return items
