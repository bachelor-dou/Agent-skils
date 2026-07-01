"""
数据库读写模块
==============
管理 Github_DB.json 的加载、校验、更新与保存。

DB 结构：
  {
    "date": "YYYY-MM-DD",       # 上次更新日期（顶层唯一的新鲜度依据）
    "projects": {
      "owner/repo": {
        "star": 12345,
        "forks": 678,                            # 最近一次刷新的 fork 数
        "created_at": "YYYY-MM-DDTHH:MM:SSZ",     # 仓库创建时间（新项目判定）
        "refreshed_at": "YYYY-MM-DDTHH:MM:SSZ",  # 项目级快照时间，差值判定依据
        "desc": "LLM 生成的描述",
        "desc_updated_at": "YYYY-MM-DD",          # desc 生成日期，超过 DESC_REFRESH_DAYS 天重刷
        "short_desc": "GitHub 原始 description",
        "language": "Python",
                "topics": ["ai", "llm"],
                "readme_url": "https://github.com/owner/repo/blob/HEAD/README.md"
      }
    }
  }

有效性策略：
  不再维护顶层 valid 布尔。是否可用 DB 差值，在使用处按「项目级窗口」逐项实时判定
  （项目 refreshed_at 年龄与本次计算窗口相差 ≤ 容差小时数才走差值），过旧/过新自动回退实时。
  load_db 仅按 date 记录一条新鲜度提示日志，不影响逻辑。
"""

import fcntl
import json
import logging
import os
import threading
from datetime import datetime, timezone

from ..config import DB_FILE_PATH, GROWTH_CALC_DAYS

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
    读取 Github_DB.json。

    不再维护顶层 "valid" 布尔：DB 是否可用于差值，已改为在使用处按项目级窗口（refreshed_at
    与计算窗口相差 ≤ 容差）逐项实时判定。此处仅按 date 记录一条新鲜度提示日志。

    Returns:
        DB 字典，至少包含 "date"、"projects" 两个键。
    """
    default_db: dict = {"date": "", "projects": {}}

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

    # 仅记录新鲜度提示日志（不再据此设置/使用 valid 开关）。
    date_str = db.get("date", "")
    if date_str:
        try:
            db_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_diff = (_utc_now() - db_date).days
            if days_diff > GROWTH_CALC_DAYS:
                logger.warning(
                    f"DB 距上次更新 {days_diff} 天（> 默认窗口 {GROWTH_CALC_DAYS}），仅作提示；"
                    f"差值有效性运行时按项目级窗口逐项判定（保留 {len(db['projects'])} 条历史数据）。"
                )
            else:
                logger.info(f"DB 距上次更新 {days_diff} 天，共 {len(db['projects'])} 条记录。")
        except ValueError:
            logger.warning(f"DB date 格式异常: {date_str}。")

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
                merged_db.pop("valid", None)  # 不再写入顶层 valid（清理旧文件遗留字段）
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
                    desc_ts = info.get("desc_updated_at", "")

                    existing = disk_projects.get(name)
                    if isinstance(existing, dict):
                        touched = False
                        if existing.get("desc") != desc:
                            existing["desc"] = desc
                            touched = True
                        if desc_ts and existing.get("desc_updated_at") != desc_ts:
                            existing["desc_updated_at"] = desc_ts
                            touched = True
                        if touched:
                            changed_projects += 1
                    else:
                        new_record = {"desc": desc}
                        if desc_ts:
                            new_record["desc_updated_at"] = desc_ts
                        disk_projects[name] = new_record
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


def is_project_window_match(
    refreshed_at: str,
    growth_calc_days: int,
    tolerance_hours: float,
) -> bool:
    """判断某项目快照的年龄是否 ≈ 计算窗口（用于 DB 差值有效性）。

    项目年龄 = now − refreshed_at；当 |项目年龄 − growth_calc_days| ≤ tolerance_hours 时，
    current_star − DB旧star 才是有效的「近 growth_calc_days 天增长」。

    Args:
        refreshed_at: 项目快照时间（"YYYY-MM-DDTHH:MM:SSZ"）。
        growth_calc_days: 本次计算窗口（天）。
        tolerance_hours: 允许的最大偏差（小时）。
    """
    if not refreshed_at:
        return False
    try:
        refresh_dt = datetime.strptime(refreshed_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return False
    age_seconds = (_utc_now() - refresh_dt).total_seconds()
    return abs(age_seconds - growth_calc_days * 86400) <= tolerance_hours * 3600
