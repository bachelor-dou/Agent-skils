#!/usr/bin/env python
"""
定时更新脚本：后台自动执行综合热度搜索、增长计算、排名并生成报告。

无需 LLM 对话，直接执行内置 DiscoveryPipeline 编排流程。

用法：
  python scheduled_update.py [--top-n 100] [--mode comprehensive]
"""
# ============================================================
# 部署为定时任务（cron）
# ============================================================
#
# 1. 手动运行测试：
#    export LLM_API_KEY="sk-xxx"
#    cd /home/openeuler/Agent-skils/github_hot_projects
#    python scheduled_update.py --top-n 100 --mode comprehensive
#
# 2. 编辑 crontab（每天凌晨 2 点自动执行）：
#    crontab -e
#    添加以下行：
#    0 2 * * * export LLM_API_KEY="sk-xxx" && cd /home/openeuler/Agent-skils/github_hot_projects && /usr/bin/python3 scheduled_update.py >> logs/scheduled.log 2>&1
#
# 3. 或使用 systemd timer：
#    sudo cp scheduled_update.service /etc/systemd/system/
#    sudo cp scheduled_update.timer   /etc/systemd/system/
#    sudo systemctl enable --now scheduled_update.timer
#
# 日志：logs/scheduled-YYYY-MM-DD.log
# ============================================================
import argparse
import logging
import logging.handlers
import os
import sys
from datetime import datetime

# 确保可以 import 项目包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from github_hot_projects.common.config import (
    HOT_PROJECT_COUNT,
    HOT_NEW_PROJECT_COUNT,
    LOG_DIR,
    STAR_GROWTH_THRESHOLD,
    TIME_WINDOW_DAYS,
)
from github_hot_projects.common.db import load_db, save_db
from github_hot_projects.common.token_manager import TokenManager
from github_hot_projects.agent_tools import (
    tool_batch_check_growth,
    tool_fetch_trending,
    tool_generate_report,
    tool_rank_candidates,
    tool_scan_star_range,
    tool_search_hot_projects,
    trending_repo_to_search_repo,
)


def setup_logging() -> str:
    """配置日志：同时输出到终端和文件，文件使用 RotatingFileHandler 防止过大。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"scheduled-{datetime.now().strftime('%Y-%m-%d')}.log",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            file_handler,
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


logger = logging.getLogger("scheduled_update")


class DiscoveryPipeline:
    """端到端项目发现管道。

    用于定时任务场景：search -> scan -> trending -> growth -> rank -> report。
    """

    def __init__(self, token_mgr: TokenManager, db: dict) -> None:
        self.token_mgr = token_mgr
        self.db = db

    def run(
        self,
        mode: str = "comprehensive",
        top_n: int | None = None,
        new_project_days: int | None = None,
        time_window_days: int = TIME_WINDOW_DAYS,
        growth_threshold: int = STAR_GROWTH_THRESHOLD,
        force_refresh: bool = False,
    ) -> dict:
        if top_n is None:
            top_n = HOT_NEW_PROJECT_COUNT if mode == "hot_new" else HOT_PROJECT_COUNT

        logger.info(
            "[Pipeline] 启动: mode=%s, top_n=%s, new_project_days=%s, "
            "time_window_days=%s, growth_threshold=%s, 数据源=search+scan+trending 三源合一",
            mode, top_n, new_project_days, time_window_days, growth_threshold,
        )

        all_repos, seen = self._collect_repos(new_project_days)
        if not all_repos:
            logger.error("[Pipeline] 搜索阶段未获取到任何仓库，终止。")
            return {"error": "搜索阶段无结果", "report_path": ""}

        logger.info("[Pipeline] Step 2: 批量增长计算 (%d 个仓库)", len(all_repos))
        growth_result = tool_batch_check_growth(
            self.token_mgr,
            all_repos,
            self.db,
            growth_threshold=growth_threshold,
            new_project_days=new_project_days,
            time_window_days=time_window_days,
            force_refresh=force_refresh,
        )
        candidates = growth_result.get("candidates", {})
        logger.info(
            "[Pipeline] 候选: %d / %d (阈值 >=%d)",
            len(candidates),
            growth_result.get("total_checked", 0),
            growth_threshold,
        )
        save_db(self.db)

        if not candidates:
            logger.warning("[Pipeline] 无候选项目，终止。")
            return {"error": "无候选项目", "report_path": "", "total_repos": len(all_repos)}

        logger.info("[Pipeline] Step 3: 排名 (mode=%s, top_n=%d)", mode, top_n)
        rank_result = tool_rank_candidates(
            candidates,
            top_n=top_n,
            mode=mode,
            db=self.db,
            new_project_days=new_project_days,
        )
        top_projects = rank_result.pop("_ordered_tuples", [])
        logger.info("[Pipeline] 排名完成: %d 个项目", len(top_projects))

        if not top_projects:
            logger.error("[Pipeline] 排名结果为空，终止。")
            return {"error": "排名结果为空", "report_path": "", "candidates_count": len(candidates)}

        logger.info("[Pipeline] Step 4: 生成报告")
        report_result = tool_generate_report(
            top_projects,
            self.db,
            mode=mode,
            new_project_days=new_project_days if mode == "hot_new" else None,
            time_window_days=time_window_days,
        )
        save_db(self.db)

        report_path = report_result.get("report_path", "")
        if report_path:
            logger.info("[Pipeline] 完成! 报告: %s", report_path)

        return {
            "report_path": report_path,
            "ranked_count": len(top_projects),
            "candidates_count": len(candidates),
            "total_repos": len(all_repos),
            "mode": mode,
        }

    def _collect_repos(self, new_project_days: int | None = None) -> tuple[list[dict], set[str]]:
        seen: set[str] = set()
        all_repos: list[dict] = []

        logger.info("[Pipeline] Step 1a: 关键词搜索（全类别）")
        search_result = tool_search_hot_projects(
            self.token_mgr,
            new_project_days=new_project_days,
        )
        raw_repos = search_result.pop("_raw_repos", [])
        all_repos.extend(raw_repos)
        seen.update(r["full_name"] for r in raw_repos)
        logger.info("[Pipeline]   关键词搜索: %d 个仓库", len(raw_repos))

        logger.info("[Pipeline] Step 1b: Star 范围扫描")
        scan_result = tool_scan_star_range(
            self.token_mgr,
            seen_repos=seen,
            new_project_days=new_project_days,
        )
        scan_repos = scan_result.pop("_raw_repos", [])
        for repo in scan_repos:
            if repo["full_name"] not in seen:
                seen.add(repo["full_name"])
                all_repos.append(repo)
        logger.info("[Pipeline]   范围扫描补充: %d 个, 累计 %d 个", len(scan_repos), len(all_repos))

        logger.info("[Pipeline] Step 1c: Trending 补源（daily+weekly+monthly）")
        trending_result = tool_fetch_trending(trending_range="all")
        trending_repos = trending_result.pop("_raw_repos", [])
        added = 0
        for repo in trending_repos:
            full_name = repo["full_name"]
            if full_name in seen:
                continue
            seen.add(full_name)
            all_repos.append(trending_repo_to_search_repo(repo))
            added += 1
        logger.info("[Pipeline]   Trending 补充: %d 个, 最终 %d 个", added, len(all_repos))

        return all_repos, seen


def run_update(top_n: int, mode: str) -> None:
    """执行完整的搜索→增长→排名→报告流程（委托给 DiscoveryPipeline）。"""
    token_mgr = TokenManager()
    db = load_db()

    logger.info(
        f"开始定时更新: mode={mode}, top_n={top_n}, "
        f"DB projects={len(db.get('projects', {}))}, valid={db.get('valid')}"
    )

    pipeline = DiscoveryPipeline(token_mgr, db)
    result = pipeline.run(mode=mode, top_n=top_n, force_refresh=True)

    report_path = result.get("report_path", "")
    if report_path:
        logger.info(f"定时更新完成! 报告: {report_path}")
    elif result.get("error"):
        logger.error(f"定时更新失败: {result['error']}")
    else:
        logger.error("报告生成失败。")


def main():
    parser = argparse.ArgumentParser(
        description="定时更新：自动搜索、计算增长、排名并生成 GitHub 热门项目报告"
    )
    parser.add_argument(
        "--top-n", type=int, default=HOT_PROJECT_COUNT,
        help=f"取前 N 个项目 (默认: {HOT_PROJECT_COUNT})",
    )
    parser.add_argument(
        "--mode", default="comprehensive",
        choices=["comprehensive", "hot_new"],
        help="排名模式 (默认: comprehensive)",
    )
    args = parser.parse_args()

    log_path = setup_logging()
    logger.info(f"日志: {log_path}")

    try:
        run_update(args.top_n, args.mode)
    except Exception:
        logger.exception("定时更新异常终止")
        sys.exit(1)


if __name__ == "__main__":
    main()


