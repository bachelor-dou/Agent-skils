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

import json
import logging
import os
from datetime import datetime, timezone

from .config import DATA_EXPIRE_DAYS, DB_FILE_PATH

logger = logging.getLogger("discover_hot")


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
        with open(DB_FILE_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
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
            days_diff = (datetime.now(timezone.utc) - db_date).days
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
    """保存 DB 到磁盘，自动更新 date 为今天。"""
    db["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        with open(DB_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
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

    if full_name in db_projects:
        db_projects[full_name]["star"] = current_star
        db_projects[full_name]["forks"] = forks
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
        }
