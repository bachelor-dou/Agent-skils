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
                "readme_url": "https://github.com/owner/repo/blob/HEAD/README.md"
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
from datetime import datetime, timezone

from .config import DATA_EXPIRE_DAYS, DB_FILE_PATH, GROWTH_CALC_DAYS

logger = logging.getLogger("discover_hot")

_db_lock = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _lock_file_path() -> str:
    return DB_FILE_PATH + ".lock"


def _format_utc_timestamp(ts: datetime | None = None) -> str:
    return (ts or _utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def save_db_desc_only(db: dict) -> int:
    """仅持久化 desc 字段，避免刷新快照基线字段。

    该函数用于实时/轻量场景：只合并 projects 下的 `desc`，
    不更新顶层 `date`/`valid`，从而避免影响增长差值窗口判断。

    Returns:
        实际发生 desc 变更的项目数量。
    """
    changed_projects = 0
    try:
        with _db_lock:
            lock_fd = open(_lock_file_path(), "w")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)  # 排他锁（阻塞其他读写）

                disk_db: dict = {}
                if os.path.exists(DB_FILE_PATH):
                    try:
                        with open(DB_FILE_PATH, "r", encoding="utf-8") as f:
                            disk_db = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        disk_db = {}

                disk_projects = disk_db.get("projects", {})
                if not isinstance(disk_projects, dict):
                    disk_projects = {}

                for name, info in db.get("projects", {}).items():
                    if not isinstance(info, dict):
                        continue

                    desc = info.get("desc", "")
                    if not desc:
                        continue

                    existing = disk_projects.get(name)
                    if isinstance(existing, dict):
                        if existing.get("desc") != desc:
                            existing["desc"] = desc
                            changed_projects += 1
                    else:
                        disk_projects[name] = {"desc": desc}
                        changed_projects += 1

                if changed_projects == 0:
                    return 0

                merged_db = dict(disk_db)
                merged_db["projects"] = disk_projects

                temp_path = DB_FILE_PATH + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(merged_db, f, ensure_ascii=False, indent=2)
                os.replace(temp_path, DB_FILE_PATH)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()

        logger.info(f"DB 仅描述字段已保存: {changed_projects} 个项目。")
        return changed_projects
    except IOError as e:
        logger.error(f"DB 描述字段保存失败: {e}")
        return 0


def update_db_project(
    db_projects: dict, full_name: str, current_star: int, repo_item: dict
) -> None:
    """
    更新 DB 中指定仓库的 star 值及补充缺失字段，并记录刷新时间。

    仅在 force_refresh / 周更新等批量刷新场景下调用。
    对已有仓库更新 star、refreshed_at 并补充空字段；
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

    if full_name in db_projects:
        db_projects[full_name]["star"] = current_star
        db_projects[full_name]["forks"] = forks
        db_projects[full_name]["refreshed_at"] = _format_utc_timestamp()
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
            "refreshed_at": _format_utc_timestamp(),
            "desc": "",
            "short_desc": description[:500],
            "language": language,
            "topics": topics,
            "readme_url": readme_url,
        }


def get_db_age_days(db: dict) -> int | None:
    """返回 DB 快照距今的天数（按 UTC 日期差），无有效日期则返回 None。"""
    db_date_str = db.get("date", "")
    if not db_date_str:
        return None
    try:
        db_date = datetime.strptime(db_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (_utc_now().date() - db_date.date()).days
    except ValueError:
        return None


def set_growth_cache(*_args, **_kwargs) -> None:
    """兼容旧版本接口：该函数已废弃，当前实现为 no-op。"""
    logger.debug("set_growth_cache 已废弃，调用已忽略。")


def is_db_diff_eligible(
    db: dict,
    growth_calc_days: int = GROWTH_CALC_DAYS,
) -> bool:
    """严格判断 DB 是否满足差值法前提（新项目榜 / 单查 使用）。

    同时满足以下三项才返回 True：
    1. db["valid"] — DB 未过期
    2. growth_calc_days < DATA_EXPIRE_DAYS — 窗口在有效期内
    3. DB 年龄 ≥ growth_calc_days − 1 — 差值接近请求窗口
    """
    if not db.get("valid", False):
        return False
    if growth_calc_days >= DATA_EXPIRE_DAYS:
        return False
    db_date_str = db.get("date", "")
    if not db_date_str:
        return False
    try:
        db_date = datetime.strptime(db_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        db_age_days = (_utc_now() - db_date).total_seconds() / 86400
        return db_age_days >= (growth_calc_days - 1)
    except ValueError:
        return False


def is_project_diff_eligible(
    project: dict,
    growth_calc_days: int = GROWTH_CALC_DAYS,
) -> bool:
    """严格判断单个仓库是否满足差值法条件（新项目榜 / 单查 使用）。

    同时满足：
    1. refreshed_at 距今 ≥ growth_calc_days − 1（足够旧）
    2. refreshed_at 距今 ≤ DATA_EXPIRE_DAYS（不超过有效期）
    """
    refreshed_at = project.get("refreshed_at", "")
    if not refreshed_at:
        return False
    try:
        refresh_dt = datetime.strptime(refreshed_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        age_days = (_utc_now() - refresh_dt).total_seconds() / 86400
        return (growth_calc_days - 1) <= age_days <= DATA_EXPIRE_DAYS
    except ValueError:
        return False


def is_project_same_batch(
    project: dict,
    db: dict,
) -> bool:
    """综合榜专用：判断仓库 refreshed_at 是否与 db["date"] 属于同一批次刷新。

    refreshed_at 与 db["date"] 差值 ≤ 1 天视为同批次。
    """
    refreshed_at = project.get("refreshed_at", "")
    db_date_str = db.get("date", "")
    if not refreshed_at or not db_date_str:
        return False
    try:
        refresh_dt = datetime.strptime(refreshed_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        db_date = datetime.strptime(db_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        diff_days = abs((refresh_dt - db_date).total_seconds()) / 86400
        return diff_days <= 1
    except ValueError:
        return False
