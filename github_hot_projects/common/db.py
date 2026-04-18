"""
数据库读写模块
==============
管理 Github_DB.json 的加载、校验、更新与保存。

DB 结构：
  {
    "date": "YYYY-MM-DD",       # 上次更新日期
    "valid": true/false,        # 数据是否在有效期内（差值法是否可信）
    "projects": {
      "owner/repo": {
        "star": 12345,
        "desc": "LLM 生成的描述",
        "short_desc": "GitHub 原始 description",
        "language": "Python",
                "topics": ["ai", "llm"],
                "readme_url": "https://github.com/owner/repo/blob/HEAD/README.md",
                "refreshed_at": "2026-04-17T03:25:00Z"
      }
    }
  }

过期策略：
  距上次更新 > DATA_EXPIRE_DAYS → valid=false（差值不可信），
  但不清空 projects 数据，保留全部历史记录。
"""

import fcntl
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone

from .config import DATA_EXPIRE_DAYS, DB_FILE_PATH, GROWTH_CACHE_TTL_HOURS

logger = logging.getLogger("discover_hot")

_db_lock = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _lock_file_path() -> str:
    return DB_FILE_PATH + ".lock"


def _format_utc_timestamp(ts: datetime | None = None) -> str:
    return (ts or _utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_refresh_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _merge_project_records(disk_project: dict, memory_project: dict) -> dict:
    """按字段合并单个仓库记录，避免旧快照整条覆盖。"""
    merged = dict(disk_project)
    for key, value in memory_project.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def load_db() -> dict:
    """
    读取 Github_DB.json 并校验有效性。

    Returns:
        DB 字典，至少包含 "date"、"valid"、"projects" 三个键。
    """
    default_db: dict = {"date": "", "valid": False, "projects": {}}

    if not os.path.exists(DB_FILE_PATH):
        logger.info("DB 文件不存在，初始化空 DB。")
        return default_db

    try:
        with _db_lock:
            lock_fd = open(_lock_file_path(), "w")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_SH)  # 共享锁（允许并发读）
                with open(DB_FILE_PATH, "r", encoding="utf-8") as f:
                    db = json.load(f)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"DB 文件读取失败: {e}，重新初始化。")
        return default_db

    if "projects" not in db:
        db["projects"] = {}

    # 校验有效性
    date_str = db.get("date", "")
    if date_str:
        try:
            db_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_diff = (_utc_now() - db_date).days
            if days_diff > DATA_EXPIRE_DAYS:
                logger.warning(
                    f"DB 数据已过期（距上次更新 {days_diff} 天），"
                    f"标记 valid=false（保留 {len(db['projects'])} 条历史数据）。"
                )
                db["valid"] = False
            else:
                db["valid"] = True
                logger.info(f"DB 有效，距上次更新 {days_diff} 天。")
        except ValueError:
            logger.warning(f"DB date 格式异常: {date_str}，标记 valid=false。")
            db["valid"] = False
    else:
        db["valid"] = False

    return db


def save_db(db: dict) -> None:
    """保存 DB 到磁盘，自动更新 date 为今天。

    采用 read-merge-write 策略：在排他锁内先读取磁盘最新版本，
    将当前内存中的 projects 合并进去（内存侧优先），再写回。
    这样可以避免长会话持有旧快照时覆盖其他会话的新增数据。
    """
    db["date"] = _utc_now().strftime("%Y-%m-%d")
    try:
        with _db_lock:
            lock_fd = open(_lock_file_path(), "w")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)  # 排他锁（阻塞其他读写）

                # 读取磁盘最新版并合并 projects
                disk_db: dict = {}
                if os.path.exists(DB_FILE_PATH):
                    try:
                        with open(DB_FILE_PATH, "r", encoding="utf-8") as f:
                            disk_db = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        disk_db = {}

                disk_projects = disk_db.get("projects", {})
                mem_projects = db.get("projects", {})
                merged_projects = {
                    name: dict(info) if isinstance(info, dict) else info
                    for name, info in disk_projects.items()
                }
                for name, info in mem_projects.items():
                    if isinstance(info, dict) and isinstance(merged_projects.get(name), dict):
                        merged_projects[name] = _merge_project_records(merged_projects[name], info)
                    elif isinstance(info, dict):
                        merged_projects[name] = dict(info)
                    else:
                        merged_projects[name] = info

                merged_db = dict(disk_db)
                merged_db.update(db)
                merged_db["projects"] = merged_projects
                merged_db["valid"] = True
                db.clear()
                db.update(merged_db)

                temp_path = DB_FILE_PATH + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(db, f, ensure_ascii=False, indent=2)
                os.replace(temp_path, DB_FILE_PATH)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
        logger.info(f"DB 已保存: {len(db.get('projects', {}))} 个项目。")
    except IOError as e:
        logger.error(f"DB 保存失败: {e}")


def update_db_project(
    db_projects: dict, full_name: str, current_star: int, repo_item: dict
) -> None:
    """
    更新 DB 中指定仓库的 star 值及补充缺失字段。

    对已有仓库只更新 star 并补充空字段；
    对新仓库创建完整记录。

    Args:
        db_projects: db["projects"] 引用
        full_name:   "owner/repo"
        current_star: 当前 star 数
        repo_item:   GitHub API 返回的仓库字典
    """
    readme_url = f"https://github.com/{full_name}/blob/HEAD/README.md"
    description = repo_item.get("description") or ""
    language = repo_item.get("language") or ""
    topics = repo_item.get("topics") or []
    forks = repo_item.get("forks_count", 0)
    created_at = repo_item.get("created_at") or ""
    refreshed_at = _format_utc_timestamp()

    if full_name in db_projects:
        db_projects[full_name]["star"] = current_star
        db_projects[full_name]["forks"] = forks
        db_projects[full_name]["refreshed_at"] = refreshed_at
        if created_at and not db_projects[full_name].get("created_at"):
            db_projects[full_name]["created_at"] = created_at
        if "readme_url" not in db_projects[full_name]:
            db_projects[full_name]["readme_url"] = readme_url
        if description and not db_projects[full_name].get("short_desc"):
            db_projects[full_name]["short_desc"] = description[:500]
        if language and not db_projects[full_name].get("language"):
            db_projects[full_name]["language"] = language
        if topics and not db_projects[full_name].get("topics"):
            db_projects[full_name]["topics"] = topics
    else:
        db_projects[full_name] = {
            "star": current_star,
            "forks": forks,
            "created_at": created_at,
            "desc": "",
            "short_desc": description[:500],
            "language": language,
            "topics": topics,
            "readme_url": readme_url,
            "refreshed_at": refreshed_at,
        }


def is_project_refresh_fresh(
    project: dict,
    max_age_days: int = DATA_EXPIRE_DAYS,
) -> bool:
    """判断单仓库刷新时间是否仍在允许窗口内。"""
    refreshed_ts = _parse_refresh_timestamp(project.get("refreshed_at", ""))
    if refreshed_ts is None:
        return False
    return _utc_now() - refreshed_ts <= timedelta(days=max_age_days)


# ══════════════════════════════════════════════════════════════
# 增长缓存（跨会话复用，避免重复 API 调用）
# ══════════════════════════════════════════════════════════════


def get_cached_growth(
    db_projects: dict, full_name: str,
    ttl_hours: int = GROWTH_CACHE_TTL_HOURS,
) -> int | None:
    """
    获取缓存的增长值。在 TTL 内返回缓存的 growth，否则返回 None。
    """
    project = db_projects.get(full_name, {})
    gc = project.get("growth_cache")
    if not gc:
        return None
    computed_at = gc.get("computed_at", "")
    if not computed_at:
        return None
    try:
        ts = datetime.strptime(computed_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        hours_old = (_utc_now() - ts).total_seconds() / 3600
        if hours_old < ttl_hours:
            return gc.get("growth")
    except (ValueError, TypeError):
        pass
    return None


def set_growth_cache(
    db_projects: dict, full_name: str,
    growth: int, score: float | None = None,
) -> None:
    """
    将增长值（及可选的评分）写入 DB 缓存。
    """
    if full_name not in db_projects:
        return
    cache = {
        "growth": growth,
        "computed_at": _format_utc_timestamp(),
    }
    if score is not None:
        cache["score"] = score
    db_projects[full_name]["growth_cache"] = cache
