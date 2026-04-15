#!/usr/bin/env python
"""
定时更新脚本：后台自动执行综合热度搜索、增长计算、排名并生成报告。

无需 LLM 对话，直接调用 Tool 函数完成完整流程：
  1. search_hot_projects — 全类别关键词搜索
  2. scan_star_range     — star 范围补充扫描
  3. fetch_trending      — GitHub Trending 三档补源
  4. batch_check_growth  — 批量计算增长并筛选候选
  5. rank_candidates     — 综合评分排序
  6. generate_report     — 生成 Markdown 报告

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
import os
import sys
from datetime import datetime

# 确保可以 import 项目包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from github_hot_projects.common.config import (
    HOT_PROJECT_COUNT,
    LOG_DIR,
    STAR_GROWTH_THRESHOLD,
)
from github_hot_projects.common.db import load_db, save_db
from github_hot_projects.common.token_manager import TokenManager
from github_hot_projects.agent_tools import (
    tool_search_hot_projects,
    tool_scan_star_range,
    tool_batch_check_growth,
    tool_rank_candidates,
    tool_generate_report,
    tool_fetch_trending,
)


def setup_logging() -> str:
    """配置日志：同时输出到终端和文件。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"scheduled-{datetime.now().strftime('%Y-%m-%d')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


logger = logging.getLogger("scheduled_update")


def run_update(top_n: int, mode: str) -> None:
    """执行完整的搜索→增长→排名→报告流程。"""
    token_mgr = TokenManager()
    db = load_db()

    logger.info(
        f"开始定时更新: mode={mode}, top_n={top_n}, "
        f"DB projects={len(db.get('projects', {}))}, valid={db.get('valid')}"
    )

    # ── 1. 搜索阶段（三源并行采集） ──
    # 1a. 全类别关键词搜索
    logger.info("Step 1a: 关键词搜索（全类别）")
    search_result = tool_search_hot_projects(token_mgr)
    all_repos = search_result.pop("_raw_repos", [])
    seen = {r["full_name"] for r in all_repos}
    logger.info(f"  关键词搜索: {len(all_repos)} 个仓库")

    # 1b. Star 范围扫描
    logger.info("Step 1b: Star 范围扫描")
    scan_result = tool_scan_star_range(token_mgr, seen_repos=seen)
    scan_repos = scan_result.pop("_raw_repos", [])
    for r in scan_repos:
        if r["full_name"] not in seen:
            seen.add(r["full_name"])
            all_repos.append(r)
    logger.info(f"  范围扫描补充: {len(scan_repos)} 个, 累计 {len(all_repos)} 个")

    # 1c. GitHub Trending 三档补源
    logger.info("Step 1c: Trending 补源（daily+weekly+monthly）")
    trending_result = tool_fetch_trending(include_all_periods=True)
    trending_repos = trending_result.pop("_raw_repos", [])
    added = 0
    for r in trending_repos:
        fn = r["full_name"]
        if fn not in seen:
            seen.add(fn)
            all_repos.append({
                "full_name": fn,
                "star": r["star"],
                "description": r.get("description", ""),
                "language": r.get("language", ""),
                "_raw": {
                    "full_name": fn,
                    "stargazers_count": r["star"],
                    "forks_count": r.get("forks", 0),
                    "description": r.get("description", ""),
                    "language": r.get("language", ""),
                    "topics": [],
                },
            })
            added += 1
    logger.info(f"  Trending 补充: {added} 个, 最终 {len(all_repos)} 个仓库")

    if not all_repos:
        logger.error("搜索阶段未获取到任何仓库，终止。")
        return

    # ── 2. 批量增长计算 ──
    logger.info("Step 2: 批量增长计算")
    growth_result = tool_batch_check_growth(
        token_mgr, all_repos, db,
        growth_threshold=STAR_GROWTH_THRESHOLD,
    )
    candidates = growth_result.get("candidates", {})
    logger.info(
        f"  候选: {len(candidates)} / {growth_result.get('total_checked', 0)} "
        f"(阈值: >={STAR_GROWTH_THRESHOLD})"
    )
    save_db(db)

    if not candidates:
        logger.warning("无候选项目（全部低于增长阈值），终止。")
        return

    # ── 3. 排序 ──
    logger.info(f"Step 3: 排名 (mode={mode}, top_n={top_n})")
    rank_result = tool_rank_candidates(
        candidates, top_n=top_n, mode=mode, db=db,
    )
    logger.info(f"  排名完成: {rank_result.get('returned', 0)} 个项目")

    top_projects = rank_result.pop("_ordered_tuples", [])
    if not top_projects:
        logger.error("排名结果为空，终止。")
        return

    # ── 4. 生成报告 ──
    logger.info("Step 4: 生成报告")
    report_result = tool_generate_report(top_projects, db, mode=mode)
    save_db(db)

    report_path = report_result.get("report_path", "")
    if report_path:
        logger.info(f"定时更新完成! 报告: {report_path}")
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


